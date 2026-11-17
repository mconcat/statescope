"""Formula-level tests of the reader on tiny synthetic dictionaries: the gate threshold, the low-rank decode, group selection, the
write decompositions. No checkpoint, no language model, no GPU; fp64 reference formulas written out independently of the reader."""
from __future__ import annotations

import torch

from statescope.reader import MoltOps, StateVocab, fvu
from synthetic import write_molt, write_vocab


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
