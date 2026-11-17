"""Reader for the per-layer dictionaries of a Qwen3.5 model. Self-contained: needs torch and safetensors (transformers only for `capture`
and `write_post_norm`, PyYAML only for `PublicMolt`).

This file imports nothing else from the package, so it can be copied next to the checkpoints and used on its own.

Two dictionaries per layer L, trained separately:

  StateVocab  ("vocab")  the residual stream at two sites of the layer,
                           resid_mid.L   the input of the MLP sub-block (what layers[L].post_attention_layernorm receives)
                           resid_post.L  the output of the layer (resid_mid.L + MLP output)
                         groups of `b` directions; the `k` groups with the largest written norm are active per token.
                         Reads both forms: untied ("comp", first design) and tied ("bsf", second design, per-token RMS input).
  the MLP of the layer as a sum of gated low-rank maps, x = resid_mid.L (raw)  ->  y ~ MLP output (raw), in one of two forms:
  MoltOps     ("molt")   first design: 41,484 "atoms" of ranks 1-16, a threshold gate per atom, plus a fixed linear part.
  PublicMolt             second design: the public penalty form (crosslayer-transcoder, branch MOLT): 3,968 "transforms" of ranks
                         8-128 with a JumpReLU gate, no linear part.

Position 0 of every sequence was dropped in training and evaluation of the first design; drop it when you measure anything with it.

    V = StateVocab("layers/L19/vocab", "cuda")
    M = MoltOps("layers/L19/molt", "cuda")
    acts = capture(model, 19, input_ids)                 # {"resid_mid", "mlp_out", "resid_post"}, each (tokens, d), position 0 dropped
    groups, codes = V.encode(acts["resid_post"], "resid_post.19")
    x_hat = V.decode(groups, codes, "resid_post.19")
    a, y_hat = M.forward(acts["resid_mid"])

Shapes in comments: n tokens, d model width, G groups, b directions per group, k active groups, T atoms, r an atom's rank.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

Device = str | torch.device


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """fraction of variance unexplained, pooled over tokens and dimensions (variance around the per-dimension mean of x)"""
    return float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def _qr_rows(W: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """W (G, b, d) -> Q (G, b, d) with orthonormal rows spanning the rows of W, R (G, b, b) with W = R Q"""
    Qs, Rs = [], []
    for c in W.split(chunk):
        Qt, Rt = torch.linalg.qr(c.transpose(1, 2).double())
        Qs.append(Qt.transpose(1, 2).float())
        Rs.append(Rt.transpose(1, 2).float())
    return torch.cat(Qs).contiguous(), torch.cat(Rs).contiguous()


class StateVocab:
    """State dictionary of one layer.

    A site's activation x is standardized, xt = (x - mu_site) / scale_site - b_dec, and written as a sum over the active groups g of
    c_g Q_g, where Q_g (b, d) is an orthonormal frame of the group's write subspace and c_g (b,) its code. Active = the k groups with
    the largest |c_g| (= the norm of what the group writes).

    A dictionary with cfg "input": "token_rms" (the tied form of the second design) reads the residual divided by its own size:
    x is replaced by x * rsqrt(mean(x^2) + eps) before the line above. Its reconstructions are then in those units, and the raw
    residual is the reconstruction times the token's own sqrt(mean(x^2) + eps): pass `rms=V.token_rms(x)` to `decode`, or call
    `reconstruct`, which does both steps.

      sites            ["resid_mid.L", "resid_post.L"]
      G, b, d, k       groups, directions per group, model width, active groups per token  (0.8B release: 16384, 4, 1024, 24)
      Q                (G, b, d) write frames, orthonormal rows
      F                (G, b, d) code functionals: c_g = F_g xt
    """

    def __init__(self, path: str, device: Device = "cpu") -> None:
        with open(os.path.join(path, "cfg.json")) as f:
            cfg = json.load(f)
        if (cfg["kind"], cfg["gate"]) != ("sasa_proper_segment", "topk") or cfg["form"] not in ("comp", "bsf"):
            raise ValueError(f"{path}: not a top-k state vocabulary export ({cfg['kind']}, {cfg['gate']}, {cfg['form']})")
        if cfg.get("input", "site") not in ("site", "token_rms"):
            raise ValueError(f"{path}: unknown input convention {cfg['input']!r}")
        weights = load_file(os.path.join(path, "weights.safetensors"))
        self.cfg: dict = cfg
        self.device: Device = device
        self.sites: list[str] = list(cfg["sites"])
        self.k: int = int(cfg["k_topk"])
        self.rms_input: bool = cfg.get("input", "site") == "token_rms"            # the dictionary reads x / rms(x), not x
        self.rms_eps: float = float(cfg["rms_eps"]) if self.rms_input else 0.0
        stats = "_rms" if self.rms_input else ""                                  # statistics of the normalised residual have their own names
        D = weights["D"].float().to(device)                                       # (G, b, d) decoder rows as trained (comp: not orthonormal)
        self.G, self.b, self.d = D.shape
        self.mu: torch.Tensor = weights[f"site_mu{stats}"].float().to(device)     # (sites, d)
        self.scale: torch.Tensor = weights[f"site_scale{stats}"].float().to(device)   # (sites,)
        self.b_dec: torch.Tensor = weights["b_dec"].float().to(device)            # (d,)
        self.Q, R = _qr_rows(D)                                                   # D_g = R_g Q_g
        self.F: torch.Tensor = torch.einsum("grs,grd->gsd", R, weights["E"].float().to(device)).contiguous()   # c_g = (E_g xt) R_g = F_g xt
        del D, R

    def token_rms(self, x: torch.Tensor) -> torch.Tensor | None:
        """x (n, d) raw -> (n, 1) sqrt(mean(x^2) + eps) of each token, or None for a dictionary that reads the raw residual"""
        if not self.rms_input:
            return None
        return torch.sqrt(x.to(self.device).float().pow(2).mean(-1, keepdim=True) + self.rms_eps)

    def standardize(self, x: torch.Tensor, site: str) -> torch.Tensor:
        i = self.sites.index(site)
        x = x.to(self.device).float()
        if self.rms_input:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_eps)
        return (x - self.mu[i]) / self.scale[i] - self.b_dec

    @torch.no_grad()
    def encode(self, x: torch.Tensor, site: str, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        """x (n, d) raw activations of `site` -> groups (n, k) long, codes (n, k, b)"""
        xt = self.standardize(x, site)
        F_flat = self.F.view(self.G * self.b, self.d)
        groups, codes = [], []
        for s in range(0, xt.shape[0], chunk):
            c = (xt[s:s + chunk] @ F_flat.T).view(-1, self.G, self.b)
            g = c.norm(dim=-1).topk(self.k, dim=-1).indices
            groups.append(g)
            codes.append(c.gather(1, g[:, :, None].expand(-1, -1, self.b)))
        return torch.cat(groups), torch.cat(codes)

    @torch.no_grad()
    def decode(self, groups: torch.Tensor, codes: torch.Tensor, site: str, chunk: int = 1024, rms: torch.Tensor | None = None) -> torch.Tensor:
        """(n, k), (n, k, b) -> reconstruction of the raw activation (n, d). Edit `codes` (or drop groups) before calling to steer.
        A dictionary with per-token RMS input needs `rms` = token_rms(x) (n, 1) of the tokens that were encoded."""
        i = self.sites.index(site)
        xt = torch.cat([torch.einsum("nkb,nkbd->nd", codes[s:s + chunk], self.Q[groups[s:s + chunk]]) for s in range(0, groups.shape[0], chunk)])
        x_hat = self.mu[i] + self.scale[i] * (xt + self.b_dec)
        if not self.rms_input:
            return x_hat
        if rms is None:
            raise ValueError("this dictionary reads x / rms(x): decode needs rms=token_rms(x) to return the raw activation")
        return x_hat * rms

    def reconstruct(self, x: torch.Tensor, site: str) -> torch.Tensor:
        """x (n, d) raw -> its reconstruction (n, d) in raw units: encode, then decode"""
        groups, codes = self.encode(x, site)
        return self.decode(groups, codes, site, rms=self.token_rms(x))

    def group_write(self, group: int, code: torch.Tensor, site: str, rms: torch.Tensor | None = None) -> torch.Tensor:
        """what one group with code (b,) adds to the raw activation of `site` (no mean, no bias): scale_site * code @ Q_group,
        times the token's `rms` for a dictionary with per-token RMS input"""
        write = self.scale[self.sites.index(site)] * (code.to(self.device).float() @ self.Q[group])
        if not self.rms_input:
            return write
        if rms is None:
            raise ValueError("this dictionary reads x / rms(x): group_write needs the token's rms to return a raw write")
        return write * rms


class MoltOps:
    """Operation dictionary of one layer's MLP.

      u = x * rsqrt(mean(x^2) + eps) * w_eff          the layer's own RMSNorm on raw resid_mid x
      z = (u - mu_u) / sd_u
      h_t = e_t . z - b_t ;  a_t = relu(h_t) if h_t / score_div_t > thr_t else 0              gate of atom t
      y_std = z W_skip + b_dec + sum over atoms t of  a_t * ((z V_t^T) * s_t) U_t             V_t, U_t (rank_t, d)
      y = y_std * sd_y + mu_y                                                                  ~ raw MLP output

    Atoms are stored by rank tier (1, 2, 4, 8, 16). `rank` (T,) gives each atom's rank. The gate has a hard threshold: an atom whose
    h sits within float rounding of its threshold can come out on or off depending on batch shape and precision (about one (token, atom)
    pair in 10^8), which changes that token's output by up to about 0.5 percent of the largest output.
    """

    def __init__(self, path: str, device: Device = "cpu", chunk: int = 512) -> None:
        with open(os.path.join(path, "cfg.json")) as f:
            cfg = json.load(f)
        rule = cfg["rule_cfg"]
        if rule.get("binary") or rule.get("read_shift") or rule.get("tied_read") or rule.get("offset"):
            raise ValueError(f"{path}: export variant not supported by this reader ({rule})")
        weights = load_file(os.path.join(path, "weights.safetensors"))
        self.cfg: dict = cfg
        self.device: Device = device
        self.chunk: int = chunk                                                    # tokens per decode step (memory, not results)
        self.d: int = int(cfg["d"])
        self.n_atoms: int = int(cfg["n_atoms"])
        with open(os.path.join(path, "input.json")) as f:
            self.eps: float = float(json.load(f)["eps"])

        def tensor(key: str) -> torch.Tensor:
            return weights[key].float().to(device)
        self.w_eff, self.mu_u, self.sd_u, self.mu_y, self.sd_y = (tensor(f"calib_{k}") for k in ("norm_w_eff", "mu_u", "sd_u", "mu_y", "sd_y"))
        self.e: torch.Tensor = tensor("e")                                         # (T, d) gate read directions
        self.b: torch.Tensor = tensor("b")                                         # (T,) gate biases
        self.W_skip: torch.Tensor = tensor("W_skip")                               # (d, d) linear part
        self.b_dec: torch.Tensor = tensor("b_dec")                                 # (d,)
        self.score_div: torch.Tensor = tensor("score_div")                         # (T,)
        self.thr: torch.Tensor = tensor("thr_atom")                                # (T,)
        self.ranks: list[int] = sorted(int(r) for r in cfg["rank_hist"])
        self.rank: torch.Tensor = torch.zeros(self.n_atoms, dtype=torch.long, device=device)
        self.pos: torch.Tensor = torch.zeros(self.n_atoms, dtype=torch.long, device=device)     # index of the atom inside its tier
        self.idx: dict[int, torch.Tensor] = {}                                     # rank -> (T_r,) atom ids of the tier
        self.Vs: dict[int, torch.Tensor] = {}                                      # rank -> (T_r, r, d) read directions times their scales
        self.U: dict[int, torch.Tensor] = {}                                       # rank -> (T_r, r, d) write directions
        for r in self.ranks:
            idx = weights[f"idx_r{r}"].long().to(device)
            self.idx[r], self.U[r] = idx, tensor(f"U_r{r}")
            self.Vs[r] = tensor(f"V_r{r}") * tensor(f"s_r{r}")[:, :, None]
            self.rank[idx], self.pos[idx] = r, torch.arange(len(idx), device=device)

    def standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w_eff
        return (u - self.mu_u) / self.sd_u

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        """z (n, d) standardized -> a (n, T), zero for atoms that are off"""
        h = z @ self.e.T - self.b
        return torch.relu(h) * ((h / self.score_div > self.thr) & (h > 0)).to(h.dtype)

    def _decode(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = z @ self.W_skip + self.b_dec
        for r in self.ranks:
            n_r = self.U[r].shape[0]
            lat = (z @ self.Vs[r].reshape(n_r * r, self.d).T).view(z.shape[0], n_r, r) * a[:, self.idx[r]][:, :, None]
            y = y + lat.reshape(z.shape[0], n_r * r) @ self.U[r].reshape(n_r * r, self.d)
        return y

    @torch.no_grad()
    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x (n, d) raw resid_mid -> a (n, T), y (n, d) raw MLP output estimate. Pass an edited `a` to steer (same shape)."""
        z = self.standardize_input(x)
        if a is None:
            a = self.gate(z)
        y = torch.cat([self._decode(zc, ac) for zc, ac in zip(z.split(self.chunk), a.split(self.chunk))])
        return a, y * self.sd_y + self.mu_y

    @torch.no_grad()
    def atom_write(self, x: torch.Tensor, atom: int) -> torch.Tensor:
        """what one atom adds to the raw MLP output for each token of x: (n, d), zero where the atom is off"""
        z = self.standardize_input(x)
        h = z @ self.e[atom] - self.b[atom]
        a = torch.relu(h) * ((h / self.score_div[atom] > self.thr[atom]) & (h > 0)).to(h.dtype)
        r, p = int(self.rank[atom]), int(self.pos[atom])
        return ((z @ self.Vs[r][p].T) * a[:, None]) @ self.U[r][p] * self.sd_y


class PublicMolt:
    """Operation dictionary of one layer's MLP in the public penalty form (crosslayer-transcoder, branch MOLT; second design).

      u = x * rsqrt(mean(x^2) + eps) * w_eff          the layer's own RMSNorm on raw resid_mid x
      z = (u - mu_in) / sd_in                         per-dimension statistics kept by the trainer
      h_t = e_t . z + b_t ;  g_t = h_t if h_t > theta_t and h_t > 0 else 0                    JumpReLU gate of transform t
      y_std = sum over transforms t of  g_t * (z V_t) U_t                                      V_t (d, rank_t), U_t (rank_t, d)
      y = y_std * sd_out + mu_out                                                              ~ raw MLP output

    Transforms are stored by rank tier, largest rank first (0.8B: ranks 128, 64, 32, 16, 8 with 128, 256, 512, 1024, 2048
    transforms). `rank` (T,) gives each transform's rank. The gate is a hard threshold, with the same rounding edge as MoltOps.

    The export folder holds the trainer's state dict (`weights.safetensors`) and its config (`config.yaml`). The RMSNorm coefficients
    (w_eff, eps) are not part of the state dict: they are read from `post_norm`, by default `postnorm.safetensors` in the export
    folder (tensor "w_eff", metadata "eps"; `write_post_norm` makes that file from the language model).
    """

    def __init__(self, path: str, device: Device = "cpu", post_norm: str | None = None, chunk: int = 512) -> None:
        import yaml
        from safetensors import safe_open
        with open(os.path.join(path, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        module = cfg["model"]["init_args"]
        model = module["model"]
        args = model["init_args"]
        classes = (model["class_path"], args["nonlinearity"]["class_path"], args["input_standardizer"]["class_path"],
                   args["output_standardizer"]["class_path"])
        expected = (("molt.Molt", "molt_lean.LeanMolt"), ("jumprelu.JumpReLU",), ("standardize.DimensionwiseInputStandardizer",),
                    ("standardize.DimensionwiseOutputStandardizer",))
        if not all(got.endswith(want) for got, want in zip(classes, expected)) or args.get("gate_whitening_path"):
            raise ValueError(f"{path}: export variant not supported by this reader ({classes}, gate whitening {args.get('gate_whitening_path')})")
        if not cfg["data"]["init_args"].get("post_norm_stats"):
            raise ValueError(f"{path}: this reader handles dictionaries trained on the post-norm MLP input only")
        weights = load_file(os.path.join(path, "weights.safetensors"))
        self.cfg: dict = cfg
        self.device: Device = device
        self.chunk: int = chunk                                                    # tokens per decode step (memory, not results)
        self.layer: int = int(module["layer"])
        self.d: int = int(args["d_acts"])
        with safe_open(post_norm or os.path.join(path, "postnorm.safetensors"), framework="pt") as f:
            self.w_eff: torch.Tensor = f.get_tensor("w_eff").float().to(device)    # (d,)
            self.eps: float = float(f.metadata()["eps"])

        def tensor(key: str) -> torch.Tensor:
            return weights[f"model.{key}"].float().to(device)
        # the standardizers hold one row per model layer; only this layer's row was filled
        self.mu_in, self.sd_in = tensor("input_standardizer.mean")[self.layer], tensor("input_standardizer.std")[self.layer]
        self.mu_out, self.sd_out = tensor("output_standardizer.mean")[self.layer], tensor("output_standardizer.std")[self.layer]
        self.e: torch.Tensor = tensor("e.weight")                                  # (T, d) gate read directions
        self.b: torch.Tensor = tensor("e.bias")                                    # (T,) gate biases
        self.theta: torch.Tensor = tensor("nonlinearity.theta").reshape(-1)        # (T,) learned thresholds
        self.n_transforms: int = self.e.shape[0]
        self.U: list[torch.Tensor] = [tensor(f"Us.{i}") for i in range(len(args["ranks"]))]    # per tier (T_i, r_i, d) write directions
        self.V: list[torch.Tensor] = [tensor(f"Vs.{i}") for i in range(len(args["ranks"]))]    # per tier (T_i, d, r_i) read directions
        self.ranks: list[int] = [U.shape[1] for U in self.U]
        if self.ranks != list(args["ranks"]) or sum(U.shape[0] for U in self.U) != self.n_transforms:
            raise ValueError(f"{path}: tiers {[tuple(U.shape) for U in self.U]} do not match the config (ranks {args['ranks']}, {self.n_transforms} gates)")
        self.rank: torch.Tensor = torch.cat([torch.full((U.shape[0],), U.shape[1], dtype=torch.long, device=device) for U in self.U])
        self.tier: torch.Tensor = torch.cat([torch.full((U.shape[0],), i, dtype=torch.long, device=device) for i, U in enumerate(self.U)])
        self.pos: torch.Tensor = torch.cat([torch.arange(U.shape[0], device=device) for U in self.U])   # index of the transform inside its tier

    def standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w_eff
        return (u - self.mu_in) / self.sd_in

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        """z (n, d) standardized -> g (n, T), zero for transforms that are off"""
        h = torch.nn.functional.linear(z, self.e, self.b)
        return ((h > self.theta) & (h > 0.0)) * h

    def _decode(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y, start = None, 0
        for U, V in zip(self.U, self.V):                                           # the gate multiplies the rank-r latents: no (n, T, d) tensor
            n_t = U.shape[0]
            latents = torch.einsum("bd,ndr->bnr", z, V) * g[:, start:start + n_t].unsqueeze(-1)
            part = torch.einsum("bnr,nrd->bd", latents, U)
            y = part if y is None else y + part
            start += n_t
        return y

    @torch.no_grad()
    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x (n, d) raw resid_mid -> g (n, T), y (n, d) raw MLP output estimate. Pass an edited `g` to steer (same shape)."""
        z = self.standardize_input(x)
        if g is None:
            g = self.gate(z)
        y = torch.cat([self._decode(zc, gc) for zc, gc in zip(z.split(self.chunk), g.split(self.chunk))])
        return g, y * self.sd_out + self.mu_out

    @torch.no_grad()
    def transform_write(self, x: torch.Tensor, transform: int) -> torch.Tensor:
        """what one transform adds to the raw MLP output for each token of x: (n, d), zero where the transform is off"""
        z = self.standardize_input(x)
        h = z @ self.e[transform] + self.b[transform]
        g = ((h > self.theta[transform]) & (h > 0.0)) * h
        i, p = int(self.tier[transform]), int(self.pos[transform])
        return ((z @ self.V[i][p]) * g[:, None]) @ self.U[i][p] * self.sd_out


def write_post_norm(model: torch.nn.Module, layer: int, path: str) -> None:
    """Write the RMSNorm coefficients in front of layer `layer`'s MLP as PublicMolt reads them: u = x * rsqrt(mean(x^2) + eps) * w_eff
    with w_eff = 1 + weight (Qwen3.5's RMSNorm stores its weight as an offset from 1)."""
    from safetensors.torch import save_file
    norm = decoder_layers(model)[layer].post_attention_layernorm
    save_file({"w_eff": (1.0 + norm.weight.detach().double()).float().cpu().contiguous()}, path,
              metadata={"eps": repr(float(norm.eps)), "definition": f"u = x * rsqrt(mean(x^2) + eps) * w_eff  (layer {layer} post_attention_layernorm)"})


def decoder_layers(model: torch.nn.Module):
    """the decoder layer list of a text-only Qwen3.5 causal LM loaded with transformers.AutoModelForCausalLM (model.model.layers)"""
    return model.model.layers


@torch.no_grad()
def capture(model: torch.nn.Module, layer: int, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """One forward pass. Returns fp32 tensors (tokens, d) with position 0 of every sequence dropped (and padded positions, if a mask is given):
    resid_mid = input of layers[layer].post_attention_layernorm, mlp_out = output of layers[layer].mlp, resid_post = resid_mid + mlp_out."""
    block = decoder_layers(model)[layer]
    got: dict[str, torch.Tensor] = {}
    h1 = block.post_attention_layernorm.register_forward_pre_hook(lambda m, args: got.__setitem__("resid_mid", args[0].detach()))
    h2 = block.mlp.register_forward_hook(lambda m, args, out: got.__setitem__("mlp_out", out.detach()))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:                                                                       # always remove the hooks; the exception propagates
        h1.remove()
        h2.remove()
    keep = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool().clone()
    keep[:, 0] = False
    out = {k: v[keep].float() for k, v in got.items()}
    out["resid_post"] = out["resid_mid"] + out["mlp_out"]
    return out
