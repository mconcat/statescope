"""Reader for the per-layer dictionaries of a Qwen3.5 model. Self-contained: needs torch and safetensors (transformers only for `capture`).

This file imports nothing else from the package, so it can be copied next to the checkpoints and used on its own.

Two dictionaries per layer L, trained separately:

  StateVocab  ("vocab")  the residual stream at two sites of the layer,
                           resid_mid.L   the input of the MLP sub-block (what layers[L].post_attention_layernorm receives)
                           resid_post.L  the output of the layer (resid_mid.L + MLP output)
                         groups of `b` directions; the `k` groups with the largest written norm are active per token.
  MoltOps     ("molt")   the MLP of the layer as a sum of gated low-rank maps ("atoms", ranks 1-16):
                           x = resid_mid.L (raw)  ->  y ~ MLP output (raw).

Position 0 of every sequence was dropped in training and evaluation; drop it when you measure anything.

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

      sites            ["resid_mid.L", "resid_post.L"]
      G, b, d, k       groups, directions per group, model width, active groups per token  (0.8B release: 16384, 4, 1024, 24)
      Q                (G, b, d) write frames, orthonormal rows
      F                (G, b, d) code functionals: c_g = F_g xt
    """

    def __init__(self, path: str, device: Device = "cpu") -> None:
        with open(os.path.join(path, "cfg.json")) as f:
            cfg = json.load(f)
        if (cfg["kind"], cfg["gate"], cfg["form"]) != ("sasa_proper_segment", "topk", "comp"):
            raise ValueError(f"{path}: not a top-k state vocabulary export ({cfg['kind']}, {cfg['gate']}, {cfg['form']})")
        weights = load_file(os.path.join(path, "weights.safetensors"))
        self.cfg: dict = cfg
        self.device: Device = device
        self.sites: list[str] = list(cfg["sites"])
        self.k: int = int(cfg["k_topk"])
        D = weights["D"].float().to(device)                                       # (G, b, d) decoder rows as trained (not orthonormal)
        self.G, self.b, self.d = D.shape
        self.mu: torch.Tensor = weights["site_mu"].float().to(device)             # (sites, d)
        self.scale: torch.Tensor = weights["site_scale"].float().to(device)       # (sites,)
        self.b_dec: torch.Tensor = weights["b_dec"].float().to(device)            # (d,)
        self.Q, R = _qr_rows(D)                                                   # D_g = R_g Q_g
        self.F: torch.Tensor = torch.einsum("grs,grd->gsd", R, weights["E"].float().to(device)).contiguous()   # c_g = (E_g xt) R_g = F_g xt
        del D, R

    def standardize(self, x: torch.Tensor, site: str) -> torch.Tensor:
        i = self.sites.index(site)
        return (x.to(self.device).float() - self.mu[i]) / self.scale[i] - self.b_dec

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
    def decode(self, groups: torch.Tensor, codes: torch.Tensor, site: str, chunk: int = 1024) -> torch.Tensor:
        """(n, k), (n, k, b) -> reconstruction of the raw activation (n, d). Edit `codes` (or drop groups) before calling to steer."""
        i = self.sites.index(site)
        xt = torch.cat([torch.einsum("nkb,nkbd->nd", codes[s:s + chunk], self.Q[groups[s:s + chunk]]) for s in range(0, groups.shape[0], chunk)])
        return self.mu[i] + self.scale[i] * (xt + self.b_dec)

    def group_write(self, group: int, code: torch.Tensor, site: str) -> torch.Tensor:
        """what one group with code (b,) adds to the raw activation of `site` (no mean, no bias): scale_site * code @ Q_group"""
        return self.scale[self.sites.index(site)] * (code.to(self.device).float() @ self.Q[group])


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
