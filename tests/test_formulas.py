"""Formula-level tests of the reader on tiny synthetic dictionaries: the gate threshold, the low-rank decode, group selection, the
write decompositions, the post-norm file and capture() on a toy model. No checkpoint, no language model, no GPU; fp64 reference
formulas written out independently of the reader."""
from __future__ import annotations

import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from statescope.reader import MoltOps, PublicMolt, StateVocab, capture, fvu, write_post_norm
from synthetic import POST_NORM_EPS, RMS_EPS, write_molt, write_public_molt, write_vocab


# ---------------------------------------------------------------------------------------------------------------------------------
# earlier design, MLP dictionary: MoltOps
# ---------------------------------------------------------------------------------------------------------------------------------

def _molt_reference(t: dict[str, torch.Tensor], x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """the docstring formulas of MoltOps, atom by atom, in float64: returns (gate values (n, T), y (n, d))"""
    t = {k: v.double() if v.is_floating_point() else v for k, v in t.items()}
    x = x.double()
    u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * t["calib_norm_w_eff"]
    z = (u - t["calib_mu_u"]) / t["calib_sd_u"]
    h = z @ t["e"].T - t["b"]
    a = torch.where((h / t["score_div"] > t["thr_atom"]) & (h > 0), h, torch.zeros_like(h))
    y = z @ t["W_skip"] + t["b_dec"]
    for key in [k for k in t if k.startswith("idx_r")]:
        r = key[len("idx_r"):]
        for p, atom in enumerate(t[key].tolist()):
            latent = (z @ t[f"V_r{r}"][p].T) * t[f"s_r{r}"][p]                # (n, r)
            y = y + a[:, atom:atom + 1] * (latent @ t[f"U_r{r}"][p])
    return a, y * t["calib_sd_y"] + t["calib_mu_y"]


def test_molt_forward_equals_the_written_out_formula(tmp_path):
    t = write_molt(tmp_path / "molt")
    molt = MoltOps(str(tmp_path / "molt"))
    x = torch.randn(64, molt.d, generator=torch.Generator().manual_seed(1))
    a, y = molt.forward(x)
    a_ref, y_ref = _molt_reference(t, x)
    assert torch.equal(a > 0, a_ref > 0)
    assert 0 < int((a > 0).sum()) < a.numel()                                 # the test has both on and off atoms
    assert torch.allclose(a.double(), a_ref, atol=1e-5)
    assert torch.allclose(y.double(), y_ref, atol=1e-4)


def test_gate_is_off_at_and_below_the_threshold_and_never_negative(tmp_path):
    write_molt(tmp_path / "molt")
    molt = MoltOps(str(tmp_path / "molt"))
    z = torch.randn(200, molt.d, generator=torch.Generator().manual_seed(2))
    h = z @ molt.e.T - molt.b
    a = molt.gate(z)
    on = a > 0
    assert torch.equal(on, (h / molt.score_div > molt.thr) & (h > 0))
    assert torch.equal(a[on], h[on])                                          # an atom that is on passes h unchanged
    assert (a >= 0).all()
    # exactly at the threshold the atom is off (strict inequality): make atom 0 sit at its threshold for token 0
    molt.thr[0] = h[0, 0] / molt.score_div[0]
    assert float(molt.gate(z)[0, 0]) == 0.0


def test_atom_writes_plus_skip_and_bias_sum_to_the_forward_output(tmp_path):
    write_molt(tmp_path / "molt")
    molt = MoltOps(str(tmp_path / "molt"))
    x = torch.randn(32, molt.d, generator=torch.Generator().manual_seed(3))
    a, y = molt.forward(x)
    z = molt.standardize_input(x)
    total = (z @ molt.W_skip + molt.b_dec) * molt.sd_y + molt.mu_y + sum(molt.atom_write(x, atom) for atom in range(molt.n_atoms))
    assert torch.allclose(total, y, atol=1e-4)
    assert int(molt.rank.sum()) == sum(r * len(idx) for r, idx in molt.idx.items())


def test_an_edited_gate_changes_the_output_by_that_atoms_write(tmp_path):
    write_molt(tmp_path / "molt")
    molt = MoltOps(str(tmp_path / "molt"))
    x = torch.randn(32, molt.d, generator=torch.Generator().manual_seed(4))
    a, y = molt.forward(x)
    atom = int(a.sum(0).argmax())
    edited = a.clone()
    edited[:, atom] = 0
    _, y_without = molt.forward(x, edited)
    assert torch.allclose(y - y_without, molt.atom_write(x, atom), atol=1e-4)


# ---------------------------------------------------------------------------------------------------------------------------------
# state dictionary reading the raw residual (earlier design's form); fvu
# ---------------------------------------------------------------------------------------------------------------------------------

def test_vocab_selects_the_k_groups_with_the_largest_written_norm(tmp_path):
    write_vocab(tmp_path / "vocab", layer=3)
    vocab = StateVocab(str(tmp_path / "vocab"))
    x = torch.randn(50, vocab.d, generator=torch.Generator().manual_seed(5))
    groups, codes = vocab.encode(x, "resid_post.3")
    all_codes = (vocab.standardize(x, "resid_post.3") @ vocab.F.view(vocab.G * vocab.b, vocab.d).T).view(-1, vocab.G, vocab.b)
    norms = all_codes.norm(dim=-1)
    kth = norms.sort(dim=-1, descending=True).values[:, vocab.k - 1]
    assert (norms.gather(1, groups) >= kth[:, None]).all()                    # every selected group is among the k largest
    assert (groups.sort(1).values.diff(dim=1) > 0).all()                      # and no group is selected twice
    # the frames are orthonormal, so a group's written norm equals its code norm
    frames = torch.einsum("gbd,gcd->gbc", vocab.Q, vocab.Q)
    assert torch.allclose(frames, torch.eye(vocab.b).expand_as(frames), atol=1e-5)


def test_vocab_reconstructs_exactly_what_lies_in_k_orthogonal_groups(tmp_path):
    t = write_vocab(tmp_path / "vocab", layer=3, d=12, groups=5, b=2, k=3, orthonormal_tied=True)
    vocab = StateVocab(str(tmp_path / "vocab"))
    g = torch.Generator().manual_seed(6)
    coeff = torch.zeros(20, 5, 2)
    coeff[:, [0, 2, 4]] = 1 + torch.rand(20, 3, 2, generator=g)               # a signal living in groups 0, 2, 4
    xt = torch.einsum("ngb,gbd->nd", coeff, t["D"])
    x = t["site_mu"][1] + t["site_scale"][1] * (xt + t["b_dec"])              # undo the standardization of resid_post
    groups, codes = vocab.encode(x, "resid_post.3")
    assert torch.equal(groups.sort(1).values, torch.tensor([0, 2, 4]).expand(20, 3))
    x_hat = vocab.decode(groups, codes, "resid_post.3")
    assert torch.allclose(x_hat, x, atol=1e-4)
    assert fvu(x, x_hat) < 1e-8


def test_group_writes_plus_mean_and_bias_sum_to_the_decode(tmp_path):
    write_vocab(tmp_path / "vocab", layer=3)
    vocab = StateVocab(str(tmp_path / "vocab"))
    x = torch.randn(8, vocab.d, generator=torch.Generator().manual_seed(7))
    for site in vocab.sites:
        groups, codes = vocab.encode(x, site)
        x_hat = vocab.decode(groups, codes, site)
        i = vocab.sites.index(site)
        for token in range(x.shape[0]):
            writes = sum(vocab.group_write(int(g), codes[token, j], site) for j, g in enumerate(groups[token].tolist()))
            assert torch.allclose(writes + vocab.mu[i] + vocab.scale[i] * vocab.b_dec, x_hat[token], atol=1e-4)


def test_dropping_a_group_removes_exactly_its_write(tmp_path):
    write_vocab(tmp_path / "vocab", layer=3)
    vocab = StateVocab(str(tmp_path / "vocab"))
    x = torch.randn(8, vocab.d, generator=torch.Generator().manual_seed(8))
    site = "resid_mid.3"
    groups, codes = vocab.encode(x, site)
    edited = codes.clone()
    edited[:, 0] = 0
    removed = vocab.decode(groups, codes, site) - vocab.decode(groups, edited, site)
    expected = torch.stack([vocab.group_write(int(groups[n, 0]), codes[n, 0], site) for n in range(x.shape[0])])
    assert torch.allclose(removed, expected, atol=1e-4)


def test_fvu_is_zero_for_a_perfect_copy_and_one_for_the_mean():
    x = torch.randn(100, 7, generator=torch.Generator().manual_seed(9))
    assert fvu(x, x) == 0.0
    assert abs(fvu(x, x.mean(0).expand_as(x)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------------------------------------------------------------
# published set, MLP dictionary: PublicMolt
# ---------------------------------------------------------------------------------------------------------------------------------

def _public_molt_reference(t: dict[str, torch.Tensor], layer: int, x: torch.Tensor, eps: float = POST_NORM_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """the docstring formulas of PublicMolt, transform by transform, in float64: returns (gate values (n, T), y (n, d))"""
    t = {k: v.double() for k, v in t.items()}
    x = x.double()
    u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * t["w_eff"]
    z = (u - t["model.input_standardizer.mean"][layer]) / t["model.input_standardizer.std"][layer]
    h = z @ t["model.e.weight"].T + t["model.e.bias"]
    theta = t["model.nonlinearity.theta"].reshape(-1)
    g = torch.where((h > theta) & (h > 0), h, torch.zeros_like(h))
    y = torch.zeros_like(x)
    transform = 0                                                             # transforms are numbered tier after tier
    for i in range(sum(k.startswith("model.Us.") for k in t)):
        U, V = t[f"model.Us.{i}"], t[f"model.Vs.{i}"]                         # (T_i, r_i, d), (T_i, d, r_i)
        for p in range(U.shape[0]):
            y = y + g[:, transform:transform + 1] * ((z @ V[p]) @ U[p])
            transform += 1
    assert transform == h.shape[1]
    return g, y * t["model.output_standardizer.std"][layer] + t["model.output_standardizer.mean"][layer]


def test_public_molt_forward_equals_the_written_out_formula(tmp_path):
    t = write_public_molt(tmp_path / "mlp", layer=1, n_layers=3)
    molt = PublicMolt(str(tmp_path / "mlp"))
    assert (molt.layer, molt.d, molt.n_transforms, molt.ranks, molt.eps) == (1, 12, 12, [4, 2, 1], POST_NORM_EPS)
    assert molt.rank.tolist() == [4] * 3 + [2] * 4 + [1] * 5
    x = torch.randn(64, molt.d, generator=torch.Generator().manual_seed(11))
    g, y = molt.forward(x)
    g_ref, y_ref = _public_molt_reference(t, 1, x)
    assert torch.equal(g > 0, g_ref > 0)
    assert 0 < int((g > 0).sum()) < g.numel()                                 # the test has both on and off transforms
    assert torch.allclose(g.double(), g_ref, atol=1e-5)
    assert torch.allclose(y.double(), y_ref, atol=1e-4)
    # the standardizers hold one row per layer: the formula with another layer's row gives another output
    assert not torch.allclose(y.double(), _public_molt_reference(t, 0, x)[1], atol=1e-2)


def test_public_gate_is_off_at_and_below_theta_and_never_negative(tmp_path):
    write_public_molt(tmp_path / "mlp")
    molt = PublicMolt(str(tmp_path / "mlp"))
    assert bool((molt.theta < 0).any()) and bool((molt.theta > 0).any())      # thresholds of both signs
    z = torch.randn(200, molt.d, generator=torch.Generator().manual_seed(12))
    h = torch.nn.functional.linear(z, molt.e, molt.b)                         # the gate's own operation: `at theta` needs the same bits
    g = molt.gate(z)
    on = g > 0
    assert torch.equal(on, (h > molt.theta) & (h > 0))
    assert torch.equal(g[on], h[on])                                          # a transform that is on passes h unchanged
    assert (g >= 0).all()
    below_zero = (h > molt.theta) & (h <= 0)                                  # above a negative theta, but not positive: off
    assert int(below_zero.sum()) > 0 and (g[below_zero] == 0).all()
    # exactly at theta the transform is off (strict inequality), and below it too: take a (token, transform) pair that is on
    token, transform = torch.nonzero(on)[0].tolist()
    molt.theta[transform] = h[token, transform]
    assert float(molt.gate(z)[token, transform]) == 0.0
    molt.theta[transform] = h[token, transform] + 0.25
    assert float(molt.gate(z)[token, transform]) == 0.0


def test_transform_writes_plus_the_output_mean_sum_to_the_forward_output(tmp_path):
    write_public_molt(tmp_path / "mlp")
    molt = PublicMolt(str(tmp_path / "mlp"))
    x = torch.randn(32, molt.d, generator=torch.Generator().manual_seed(13))
    g, y = molt.forward(x)
    firing = torch.nonzero(g.sum(0) > 0).reshape(-1).tolist()
    assert 0 < len(firing)
    total = sum(molt.transform_write(x, transform) for transform in firing) + molt.mu_out
    assert torch.allclose(total, y, atol=1e-4)
    for transform in set(range(molt.n_transforms)) - set(firing):             # a transform that never fires writes nothing
        assert float(molt.transform_write(x, transform).abs().max()) == 0.0


def test_an_edited_public_gate_changes_the_output_by_that_transforms_write(tmp_path):
    write_public_molt(tmp_path / "mlp")
    molt = PublicMolt(str(tmp_path / "mlp"))
    x = torch.randn(32, molt.d, generator=torch.Generator().manual_seed(14))
    g, y = molt.forward(x)
    transform = int(g.sum(0).argmax())
    edited = g.clone()
    edited[:, transform] = 0
    g_back, y_without = molt.forward(x, edited)
    assert torch.equal(g_back, edited)                                        # an edited gate is used as given
    assert torch.allclose(y - y_without, molt.transform_write(x, transform), atol=1e-4)


def test_public_molt_refuses_a_dictionary_trained_on_the_pre_norm_input(tmp_path):
    write_public_molt(tmp_path / "mlp", post_norm_stats=False)
    with pytest.raises(ValueError, match="post-norm"):
        PublicMolt(str(tmp_path / "mlp"))


def test_an_explicit_post_norm_file_gives_the_same_tensors_as_the_folders_own(tmp_path):
    t = write_public_molt(tmp_path / "mlp")
    x = torch.randn(32, 12, generator=torch.Generator().manual_seed(15))
    default = PublicMolt(str(tmp_path / "mlp"))
    shutil.copyfile(tmp_path / "mlp" / "postnorm.safetensors", tmp_path / "elsewhere.safetensors")
    explicit = PublicMolt(str(tmp_path / "mlp"), post_norm=str(tmp_path / "elsewhere.safetensors"))
    assert torch.equal(default.w_eff, explicit.w_eff) and default.eps == explicit.eps
    (g_default, y_default), (g_explicit, y_explicit) = default.forward(x), explicit.forward(x)
    assert torch.equal(g_default, g_explicit) and torch.equal(y_default, y_explicit)
    # the argument is what is read: a file with other coefficients gives another reader
    save_file({"w_eff": 2 * t["w_eff"]}, str(tmp_path / "other.safetensors"), metadata={"eps": "0.5"})
    other = PublicMolt(str(tmp_path / "mlp"), post_norm=str(tmp_path / "other.safetensors"))
    assert torch.equal(other.w_eff, 2 * t["w_eff"]) and other.eps == 0.5
    assert not torch.equal(other.forward(x)[1], y_default)


# ---------------------------------------------------------------------------------------------------------------------------------
# published set, state dictionary: StateVocab with per-token RMS input
# ---------------------------------------------------------------------------------------------------------------------------------

def test_token_rms_standardize_equals_the_written_out_formula(tmp_path):
    t = write_vocab(tmp_path / "state", layer=3, form="bsf", token_rms=True)
    vocab = StateVocab(str(tmp_path / "state"))
    assert vocab.rms_input and vocab.rms_eps == RMS_EPS and (vocab.cfg["form"], vocab.cfg["input"]) == ("bsf", "token_rms")
    x = torch.randn(40, vocab.d, generator=torch.Generator().manual_seed(16)) * (0.2 + 5 * torch.rand(40, 1, generator=torch.Generator().manual_seed(17)))
    xd = x.double()
    size = torch.sqrt(xd.pow(2).mean(-1, keepdim=True) + RMS_EPS)                # (n, 1)
    assert torch.allclose(vocab.token_rms(x).double(), size, atol=1e-6)
    for i, site in enumerate(vocab.sites):
        expected = (xd / size - t["site_mu_rms"][i].double()) / t["site_scale_rms"][i].double() - t["b_dec"].double()
        assert torch.allclose(vocab.standardize(x, site).double(), expected, atol=1e-5)


def test_a_dictionary_that_reads_the_raw_residual_has_no_token_rms(tmp_path):
    t = write_vocab(tmp_path / "vocab", layer=3)
    vocab = StateVocab(str(tmp_path / "vocab"))
    x = torch.randn(8, vocab.d, generator=torch.Generator().manual_seed(18))
    assert not vocab.rms_input and vocab.token_rms(x) is None
    expected = (x.double() - t["site_mu"][1].double()) / t["site_scale"][1].double() - t["b_dec"].double()
    assert torch.allclose(vocab.standardize(x, "resid_post.3").double(), expected, atol=1e-5)


def test_token_rms_reconstruct_is_encode_then_decode_with_the_tokens_rms(tmp_path):
    write_vocab(tmp_path / "state", layer=3, form="bsf", token_rms=True)
    vocab = StateVocab(str(tmp_path / "state"))
    x = torch.randn(20, vocab.d, generator=torch.Generator().manual_seed(19))
    for site in vocab.sites:
        groups, codes = vocab.encode(x, site)
        assert torch.equal(vocab.reconstruct(x, site), vocab.decode(groups, codes, site, rms=vocab.token_rms(x)))


def test_token_rms_reconstruction_is_exact_and_in_raw_units_for_k_orthogonal_groups(tmp_path):
    t = write_vocab(tmp_path / "state", layer=3, d=12, groups=5, b=2, k=3, orthonormal_tied=True, form="bsf", token_rms=True)
    vocab = StateVocab(str(tmp_path / "state"))
    g = torch.Generator().manual_seed(20)
    coeff = torch.zeros(20, 5, 2, dtype=torch.float64)
    coeff[:, [0, 2, 4]] = 1 + torch.rand(20, 3, 2, generator=g).double()      # a signal living in groups 0, 2, 4
    xt = torch.einsum("ngb,gbd->nd", coeff, t["D"].double())
    # The dictionary sees v = x / rms(x), which has mean square 1. Build such a v that also lies in the three groups after the
    # standardization: v = offset + alpha * direction, with alpha > 0 the root of |v|^2 = d.
    offset = t["site_mu_rms"][1].double() + t["site_scale_rms"][1].double() * t["b_dec"].double()     # (d,)
    direction = t["site_scale_rms"][1].double() * xt                                                  # (n, d)
    along, length_sq = direction @ offset, direction.pow(2).sum(-1)
    alpha = (-along + torch.sqrt(along ** 2 + length_sq * (12 - offset.pow(2).sum()))) / length_sq
    v = offset + alpha[:, None] * direction
    assert torch.allclose(v.pow(2).mean(-1), torch.ones(20, dtype=torch.float64), atol=1e-12)
    size = 0.5 + 4 * torch.rand(20, 1, generator=g).double()                  # tokens of different sizes
    x = (v * size).float()                                                    # the raw residual
    groups, codes = vocab.encode(x, "resid_post.3")
    assert torch.equal(groups.sort(1).values, torch.tensor([0, 2, 4]).expand(20, 3))
    x_hat = vocab.reconstruct(x, "resid_post.3")
    assert torch.allclose(x_hat, x, atol=1e-4)                                # raw units: every token back at its own size
    assert fvu(x, x_hat) < 1e-8
    assert torch.allclose(vocab.token_rms(x).double(), size, atol=1e-5)


def test_token_rms_decode_and_group_write_need_the_rms(tmp_path):
    write_vocab(tmp_path / "state", layer=3, form="bsf", token_rms=True)
    vocab = StateVocab(str(tmp_path / "state"))
    x = torch.randn(4, vocab.d, generator=torch.Generator().manual_seed(21))
    groups, codes = vocab.encode(x, "resid_mid.3")
    with pytest.raises(ValueError, match="decode needs rms"):
        vocab.decode(groups, codes, "resid_mid.3")
    with pytest.raises(ValueError, match="group_write needs the token's rms"):
        vocab.group_write(int(groups[0, 0]), codes[0, 0], "resid_mid.3")


def test_token_rms_group_writes_plus_mean_and_bias_sum_to_the_decode(tmp_path):
    write_vocab(tmp_path / "state", layer=3, form="bsf", token_rms=True)
    vocab = StateVocab(str(tmp_path / "state"))
    x = torch.randn(8, vocab.d, generator=torch.Generator().manual_seed(22)) * (0.2 + 5 * torch.rand(8, 1, generator=torch.Generator().manual_seed(23)))
    rms = vocab.token_rms(x)
    for i, site in enumerate(vocab.sites):
        groups, codes = vocab.encode(x, site)
        x_hat = vocab.decode(groups, codes, site, rms=rms)
        for token in range(x.shape[0]):
            writes = sum(vocab.group_write(int(g), codes[token, j], site, rms=rms[token]) for j, g in enumerate(groups[token].tolist()))
            assert torch.allclose(writes + (vocab.mu[i] + vocab.scale[i] * vocab.b_dec) * rms[token], x_hat[token], atol=1e-4)


# ---------------------------------------------------------------------------------------------------------------------------------
# the language model: write_post_norm and capture on toy modules
# ---------------------------------------------------------------------------------------------------------------------------------

class _ToyNorm(torch.nn.Module):
    """shaped like Qwen3.5's RMSNorm: `weight` is stored as an offset from 1, `eps` is a plain float"""

    def __init__(self, d: int, eps: float, seed: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(0.1 * torch.randn(d, generator=torch.Generator().manual_seed(seed)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)


class _ToyLayer(torch.nn.Module):
    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        self.post_attention_layernorm = _ToyNorm(d, eps=1e-6 * (seed + 1), seed=seed)
        self.mlp = torch.nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid_mid = x + 0.5 * x.roll(1, dims=1)                               # stands in for the mixer: every position gets another value
        return resid_mid + self.mlp(self.post_attention_layernorm(resid_mid))


class _ToyDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d)
        self.layers = torch.nn.ModuleList(_ToyLayer(d, seed) for seed in range(n_layers))


class _ToyLM(torch.nn.Module):
    """the parts of a transformers causal LM that the reader touches: model.model.layers[L].post_attention_layernorm and .mlp,
    called with input_ids and attention_mask. `fail_after_layer` makes the forward pass raise after that layer has run."""

    def __init__(self, vocab_size: int = 50, d: int = 6, n_layers: int = 3, fail_after_layer: int | None = None) -> None:
        super().__init__()
        self.model = _ToyDecoder(vocab_size, d, n_layers)
        self.fail_after_layer = fail_after_layer
        g = torch.Generator().manual_seed(0)                                  # fixed weights without reseeding the process-wide generator
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if not name.endswith("post_attention_layernorm.weight"):      # the norms have their own fixed weights
                    parameter.copy_(0.5 * torch.randn(parameter.shape, generator=g))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.model.embed(input_ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if i == self.fail_after_layer:
                raise RuntimeError("toy failure")
        return x

    def by_hand(self, input_ids: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """resid_mid and mlp_out of `layer`, (batch, positions, d) each, computed without calling any module's hooks"""
        x = self.model.embed.weight[input_ids]
        for i, block in enumerate(self.model.layers):
            resid_mid = x + 0.5 * x.roll(1, dims=1)
            norm = block.post_attention_layernorm
            u = resid_mid * torch.rsqrt(resid_mid.pow(2).mean(-1, keepdim=True) + norm.eps) * (1.0 + norm.weight)
            mlp_out = u @ block.mlp.weight.T + block.mlp.bias
            if i == layer:
                return resid_mid, mlp_out
            x = resid_mid + mlp_out
        raise IndexError(layer)


def _hook_count(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_write_post_norm_writes_one_plus_the_weight_and_the_eps(tmp_path):
    lm = _ToyLM()
    for layer in (0, 2):
        norm = lm.model.layers[layer].post_attention_layernorm
        path = tmp_path / f"postnorm_L{layer}.safetensors"
        write_post_norm(lm, layer, str(path))
        with safe_open(str(path), framework="pt") as f:
            w_eff, metadata = f.get_tensor("w_eff"), f.metadata()
        assert w_eff.dtype == torch.float32 and w_eff.shape == (6,)
        assert torch.equal(w_eff, (1.0 + norm.weight.detach().double()).float())
        assert float(metadata["eps"]) == norm.eps == 1e-6 * (layer + 1)       # each layer's own norm


def test_public_molt_reads_the_post_norm_file_that_write_post_norm_wrote(tmp_path):
    lm = _ToyLM(d=12)
    write_public_molt(tmp_path / "mlp", d=12)
    write_post_norm(lm, 2, str(tmp_path / "rebuilt.safetensors"))
    molt = PublicMolt(str(tmp_path / "mlp"), post_norm=str(tmp_path / "rebuilt.safetensors"))
    norm = lm.model.layers[2].post_attention_layernorm
    assert torch.equal(molt.w_eff, (1.0 + norm.weight.detach().double()).float()) and molt.eps == norm.eps


def test_capture_drops_position_0_and_returns_the_mlp_input_and_output(tmp_path):
    lm = _ToyLM()
    input_ids = torch.randint(0, 50, (3, 7), generator=torch.Generator().manual_seed(24))
    for layer in (0, 2):
        with torch.no_grad():
            resid_mid, mlp_out = lm.by_hand(input_ids, layer)
        acts = capture(lm, layer, input_ids)
        assert sorted(acts) == ["mlp_out", "resid_mid", "resid_post"]
        assert all(v.dtype == torch.float32 and v.shape == (3 * 6, 6) for v in acts.values())     # 3 sequences x (7 - 1) positions
        assert torch.allclose(acts["resid_mid"], resid_mid[:, 1:].reshape(-1, 6), atol=1e-6)
        assert torch.allclose(acts["mlp_out"], mlp_out[:, 1:].reshape(-1, 6), atol=1e-6)
        assert torch.equal(acts["resid_post"], acts["resid_mid"] + acts["mlp_out"])
        assert _hook_count(lm) == 0                                           # the hooks are removed afterwards


def test_capture_drops_masked_positions_too(tmp_path):
    lm = _ToyLM()
    input_ids = torch.randint(0, 50, (2, 6), generator=torch.Generator().manual_seed(25))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 1, 1]])
    before = mask.clone()
    with torch.no_grad():
        resid_mid, mlp_out = lm.by_hand(input_ids, 1)
    acts = capture(lm, 1, input_ids, mask)
    keep = torch.tensor([[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1]], dtype=torch.bool)   # the mask without position 0
    assert acts["resid_mid"].shape == (7, 6)
    assert torch.allclose(acts["resid_mid"], resid_mid[keep], atol=1e-6)
    assert torch.allclose(acts["mlp_out"], mlp_out[keep], atol=1e-6)
    assert torch.equal(acts["resid_post"], acts["resid_mid"] + acts["mlp_out"])
    assert torch.equal(mask, before)                                          # the caller's mask is not changed
    assert _hook_count(lm) == 0


def test_capture_removes_its_hooks_when_the_forward_pass_fails():
    lm = _ToyLM(fail_after_layer=1)
    input_ids = torch.randint(0, 50, (1, 5), generator=torch.Generator().manual_seed(26))
    with pytest.raises(RuntimeError, match="toy failure"):                    # the error is not swallowed
        capture(lm, 1, input_ids)
    assert _hook_count(lm) == 0
