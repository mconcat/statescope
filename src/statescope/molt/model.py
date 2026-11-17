"""The MLP dictionary while it trains: parameters, initial weights, the gate rule and the whitener.

Everything is in standardized coordinates:  u (B, d) = the layer's RMSNorm output, standardized  ->  y_std (B, d) ~ standardized MLP output.

    h_t   = e_t . (u W) - b_t                                       gate input of atom t; W (d, d) is the whitener
    a_t   = relu(h_t) if atom t is switched on, else 0              "on" is decided per tier, see select() and infer_gate()
    y_std = u W_skip + b_dec + sum_t (a_t / m_t) ((u V_t^T) * s_t) U_t

    e (T, d)          gate read directions, unit rows
    V_t, U_t (r, d)   read and write directions of atom t (unit rows), r its rank;   s_t (r,) its scales
    m_t               running mean of a_t over the tokens where the atom fires (not trained; it keeps a / m near 1)
    W_skip (d, d)     the frozen linear part

Atoms are stored by rank tier: tier i holds the T_i atoms of rank ranks[i] as U[i], V[i] (T_i, r, d) and s[i] (T_i, r);
`tier_index(i)` (T_i,) gives their atom ids, `pos` (T,) an atom's position inside its tier.

Shapes in comments: B tokens, d model width, T atoms, r rank.
"""
from __future__ import annotations

import torch
from torch import nn

from statescope.calibration import CalibStats
from statescope.molt.recipe import MoltRecipe


# ------------------------------------------------------------------------------------------------ whitener
def standardized_mlp_input(stats: CalibStats, x: torch.Tensor) -> torch.Tensor:
    """raw resid_mid x (n, d) -> the dictionary's input u: the layer's RMSNorm, then standardized. `stats` on x's device."""
    u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + stats.norm_eps) * stats.norm_w_eff
    return (u - stats.mu_u) / stats.sd_u


@torch.no_grad()
def whitener_from(u_chunks, eig_floor: float) -> tuple[torch.Tensor, dict]:
    """Sigma^(-1/2) of the given chunks of u (each (n, d)), accumulated in float64 on the CPU. Eigenvalues below eig_floor times
    their mean are raised to it. Returns (W (d, d) float64, a record of the spectrum)."""
    n, sum1, sum2 = 0, None, None
    for u in u_chunks:
        u64 = u.double()
        sum1 = u64.sum(0) if sum1 is None else sum1 + u64.sum(0)
        sum2 = u64.T @ u64 if sum2 is None else sum2 + u64.T @ u64
        n += u64.shape[0]
    mean = sum1 / n
    cov = sum2 / n - torch.outer(mean, mean)
    eigval, eigvec = torch.linalg.eigh(cov)
    floor = eig_floor * float(eigval.mean())
    W = (eigvec * eigval.clamp_min(floor).pow(-0.5)[None, :]) @ eigvec.T
    record = {"n_tokens": n, "eig_min": float(eigval.min()), "eig_max": float(eigval.max()), "eig_mean": float(eigval.mean()),
              "floor": floor, "n_floored": int((eigval < floor).sum())}
    return W, record


# ------------------------------------------------------------------------------------------------ model
class MoltModel(nn.Module):
    def __init__(self, recipe: MoltRecipe, tier: torch.Tensor, e: torch.Tensor, U: list[torch.Tensor], V: list[torch.Tensor],
                 s: list[torch.Tensor], whitener: torch.Tensor, skip_map: torch.Tensor) -> None:
        """tier (T,) long: each atom's rank. U[i], V[i] (T_i, r_i, d), s[i] (T_i, r_i) for the tiers in increasing rank."""
        super().__init__()
        self.recipe = recipe
        self.n_atoms, self.d = e.shape
        self.ranks: list[int] = tier.unique().tolist()                       # increasing
        self.e = nn.Parameter(e)                                             # (T, d)
        self.b = nn.Parameter(torch.zeros(self.n_atoms))                     # (T,)
        self.b_dec = nn.Parameter(torch.zeros(self.d))                       # (d,)
        self.U, self.V, self.s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        self.register_buffer("W_skip", skip_map.clone())                     # (d, d) frozen
        self.register_buffer("whitener", whitener.clone())                   # (d, d)
        self.register_buffer("tier", tier.clone())                           # (T,)
        self.register_buffer("pos", torch.zeros(self.n_atoms, dtype=torch.long))
        for i, r in enumerate(self.ranks):
            idx = (tier == r).nonzero().squeeze(1)
            self.register_buffer(f"idx_{i}", idx)                            # (T_i,) atom ids of the tier
            self.register_buffer(f"frozen_{i}", torch.zeros(idx.numel(), r, dtype=torch.bool))    # scales held at zero (debias phase)
            self.pos[idx] = torch.arange(idx.numel())
            self.U.append(nn.Parameter(U[i]))
            self.V.append(nn.Parameter(V[i]))
            self.s.append(nn.Parameter(s[i]))
        self.register_buffer("gate_mean", torch.ones(self.n_atoms))          # m; 1 until the atom first fires
        self.register_buffer("gate_mean_set", torch.zeros(self.n_atoms, dtype=torch.bool))
        self.register_buffer("tier_thr", torch.zeros(len(self.ranks)))       # inference threshold of each tier
        self.register_buffer("tier_thr_set", torch.zeros(len(self.ranks), dtype=torch.bool))
        # The checkpoint format divides the gate input by a per-atom number before comparing it with the threshold; this recipe
        # does not scale by rank, so it is 1 everywhere.
        self.register_buffer("score_div", torch.ones(self.n_atoms))

    def tier_index(self, i: int) -> torch.Tensor:
        return getattr(self, f"idx_{i}")

    def frozen(self, i: int) -> torch.Tensor:
        return getattr(self, f"frozen_{i}")

    # ---- forward
    def whiten(self, u: torch.Tensor) -> torch.Tensor:
        return u @ self.whitener

    def gate_input(self, u: torch.Tensor) -> torch.Tensor:
        """u (B, d) -> h (B, T)"""
        return self.whiten(u) @ self.e.T - self.b

    def decode(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """g (B, T) gate values already divided by gate_mean, u (B, d) -> y_std (B, d)"""
        B, d = u.shape[0], self.d
        out = u @ self.W_skip + self.b_dec
        for i, r in enumerate(self.ranks):
            n_r = self.U[i].shape[0]
            lat = u @ (self.V[i] * self.s[i][:, :, None]).reshape(n_r * r, d).T          # (B, T_i r): s_tj <V_tj, u>
            lat = (lat.view(B, n_r, r) * g[:, self.tier_index(i)][:, :, None]).reshape(B, n_r * r)
            out = out + lat @ self.U[i].reshape(n_r * r, d)
        return out

    @torch.no_grad()
    def select(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The training rule on a whole batch h (B, T): in each tier the k_r * B largest entries are on.
        Returns (mask (B, T) bool, per tier the smallest positive selected h (+inf if none))."""
        B = h.shape[0]
        mask = torch.zeros_like(h, dtype=torch.bool)
        smallest = torch.full((len(self.ranks),), float("inf"), dtype=h.dtype, device=h.device)
        for i, r in enumerate(self.ranks):
            idx = self.tier_index(i)
            n_r = idx.numel()
            take = min(self.recipe.k_by_rank[r] * B, B * n_r)
            values, flat_index = h[:, idx].reshape(-1).topk(take)
            flat = torch.zeros(B * n_r, dtype=torch.bool, device=h.device)
            flat[flat_index] = True
            mask[:, idx] = flat.view(B, n_r)
            positive = values[values > 0]
            if positive.numel():
                smallest[i] = positive.min()
        return mask, smallest

    def atom_thr(self) -> torch.Tensor:
        """(T,) each atom's inference threshold = its tier's"""
        out = torch.zeros(self.n_atoms, dtype=self.e.dtype, device=self.e.device)
        for i in range(len(self.ranks)):
            out[self.tier_index(i)] = self.tier_thr[i]
        return out

    @torch.no_grad()
    def infer_mask(self, h: torch.Tensor) -> torch.Tensor:
        return (h / self.score_div > self.atom_thr()) & (h > 0)

    def forward_infer(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The inference rule: u (B, d) -> (a (B, T), y_std (B, d)). One forward for the whole batch (no chunking)."""
        h = self.gate_input(u)
        a = torch.relu(h) * self.infer_mask(h).to(h.dtype)
        return a, self.decode(a / self.gate_mean, u)

    # ---- state that is not trained by the optimizer
    @torch.no_grad()
    def update_gate_mean_(self, a: torch.Tensor) -> None:
        """a (B, T) of the training rule on the whole batch. Atoms that fire: first time m = the batch mean of their firing a,
        afterwards a running mean of it."""
        decay = self.recipe.gate_mean_decay
        count = (a > 0).sum(0)
        on = count > 0
        batch_mean = (a.sum(0) / count.clamp_min(1)).to(self.gate_mean.dtype)
        first = on & ~self.gate_mean_set
        later = on & self.gate_mean_set
        self.gate_mean[first] = batch_mean[first]
        self.gate_mean[later] = decay * self.gate_mean[later] + (1 - decay) * batch_mean[later]
        self.gate_mean_set |= first

    @torch.no_grad()
    def update_tier_thr_(self, smallest: torch.Tensor) -> None:
        """smallest (tiers,) from select(): the first finite value sets the tier's threshold, later ones move its running mean."""
        decay = self.recipe.threshold_decay
        ok = torch.isfinite(smallest)
        first = ok & ~self.tier_thr_set
        later = ok & self.tier_thr_set
        self.tier_thr.copy_(torch.where(first, smallest, torch.where(later, decay * self.tier_thr + (1 - decay) * smallest, self.tier_thr)))
        self.tier_thr_set |= first

    # ---- constraints: unit rows for e, U, V
    @torch.no_grad()
    def project_grads_(self) -> None:
        """before the optimizer step: remove from each gradient row its component along the (unit) row it belongs to"""
        for w in (*self.U, *self.V, self.e):
            if w.grad is not None:
                w.grad.sub_((w.grad * w).sum(-1, keepdim=True) * w)
        for i, s in enumerate(self.s):
            if s.grad is not None:
                s.grad.masked_fill_(self.frozen(i), 0.0)

    @torch.no_grad()
    def renormalize_(self) -> None:
        """after the optimizer step"""
        for w in (*self.U, *self.V, self.e):
            w.div_(w.norm(dim=-1, keepdim=True))
        for i, s in enumerate(self.s):
            s.masked_fill_(self.frozen(i), 0.0)

    @torch.no_grad()
    def freeze_zero_scales_(self) -> None:
        for i, s in enumerate(self.s):
            self.frozen(i).copy_(s == 0)

    # ---- views used by the export
    @torch.no_grad()
    def folded_e(self) -> torch.Tensor:
        """(T, d) gate directions with the whitener folded in: h = u . folded_e - b"""
        return self.e.detach() @ self.whitener

    @torch.no_grad()
    def scales_rect(self) -> torch.Tensor:
        """(T, max_rank) all scales in one table, zero beyond an atom's rank"""
        out = torch.zeros(self.n_atoms, self.recipe.max_rank, dtype=self.e.dtype, device=self.e.device)
        for i, r in enumerate(self.ranks):
            out[self.tier_index(i), :r] = self.s[i].detach()
        return out


# ------------------------------------------------------------------------------------------------ initial weights
def assign_tiers(recipe: MoltRecipe, d: int, reserved_rank1: int) -> torch.Tensor:
    """(T,) long: the rank of every atom. Atoms 0 .. reserved_rank1 - 1 are rank 1 (one slot per MLP neuron, a convention of the
    release so that all dictionaries of a model share the assignment); the others get their rank by a seeded permutation."""
    rest = recipe.tier_counts(d)
    rest[1] -= reserved_rank1
    if rest[1] < 0:
        raise ValueError(f"{reserved_rank1} reserved rank-1 atoms exceed the rank-1 tier ({recipe.tier_counts(d)[1]})")
    pool = torch.cat([torch.full((count,), rank, dtype=torch.long) for rank, count in sorted(rest.items())])
    pool = pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(recipe.tier_seed))]
    return torch.cat([torch.ones(reserved_rank1, dtype=torch.long), pool])


@torch.no_grad()
def init_model(recipe: MoltRecipe, d: int, reserved_rank1: int, whitener: torch.Tensor, skip_map: torch.Tensor,
               draw_atoms: int = 1024) -> MoltModel:
    """The initial model, on the CPU. One generator draws e (T, d), then max_rank write directions U for every atom, then as many
    read directions V; each row is normalized and an atom of rank r keeps the first r of its max_rank rows.

    The original drew U and V as two (T, max_rank, d) tensors. Here they are drawn `draw_atoms` atoms at a time, which gives the same
    numbers (the CPU generator fills a tensor in order, 16 values at a time, so d must be a multiple of 16) without ever holding the
    two full tensors (2 x 2.7 GB at d = 1,024; 2 x 16 GB at d = 2,560)."""
    if d % 16:
        raise ValueError(f"d = {d}: the chunked initial draw equals the one-tensor draw only when d is a multiple of 16")
    tier = assign_tiers(recipe, d, reserved_rank1)
    n, max_rank = tier.numel(), recipe.max_rank
    ranks = tier.unique().tolist()
    gen = torch.Generator().manual_seed(recipe.init_seed(n))
    e = torch.randn(n, d, generator=gen)
    e = e / e.norm(dim=-1, keepdim=True)
    directions: dict[str, list[torch.Tensor]] = {}
    for name in ("U", "V"):
        parts: dict[int, list[torch.Tensor]] = {r: [] for r in ranks}
        for start in range(0, n, draw_atoms):
            rows = torch.randn(min(n, start + draw_atoms) - start, max_rank, d, generator=gen)
            rows = rows / rows.norm(dim=-1, keepdim=True)
            tier_here = tier[start:start + draw_atoms]
            for r in ranks:
                parts[r].append(rows[tier_here == r, :r].clone())
        directions[name] = [torch.cat(parts[r]) for r in ranks]
    scale = recipe.init_scale / tier.to(torch.float32).sqrt()                # (T,) init_scale / sqrt(rank)
    s = [scale[tier == r][:, None].expand(-1, r).clone() for r in ranks]
    return MoltModel(recipe, tier, e, directions["U"], directions["V"], s, whitener.float(), skip_map.float())
