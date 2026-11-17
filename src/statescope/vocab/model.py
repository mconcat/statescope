"""The state dictionary as it is trained.

A site's raw activation x (d,) is standardized with the site's own mean and scalar scale and centred by a shared bias:

    xs = (x - mu_site) / scale_site            scale_site^2 = E||x - mu_site||^2 / d, so E||xs||^2 = d at every site
    xt = xs - b_dec
    z_g = E_g xt                               (b,)   code of group g; E (G, b, d) is the encoder
    A   = the k groups with the largest ||z_g D_g||   (the norm of what the group writes; D (G, b, d) is the decoder)
    xt ~ sum over g in A of z_g D_g

Loss per site: L_rec = E||xt - sum_A z_g D_g||^2 / d, plus an auxiliary term that lets dead groups reconstruct the residual.
E and D are free (not tied, not normalized). Every `balance_every` steps they are balanced: E_g -> A_g E_g, D_g -> A_g^-1 D_g with A_g
chosen so that E_g E_g^T = D_g D_g^T; the map E_g^T D_g of the group does not change.

Precision: only the dense encode runs under bf16 autocast; selection, decode and every reduction are fp32.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from statescope.vocab.recipe import VocabRecipe

# numerical guards of the balancing pass (fp64)
_EIG_FLOOR_REL = 1e-12            # eigenvalues are floored at this fraction of the group's largest one before a matrix power
_EIG_FLOOR_ABS = 1e-300
_BALANCE_CLIP = (1e-6, 1e6)       # spectrum of A_g^2; A_g itself is then within [1e-3, 1e3]


def initial_frames(groups: int, group_size: int, d: int, seed: int) -> torch.Tensor:
    """(G, b, d) orthonormal rows per group, fp32 on the CPU, from a CPU generator of its own (independent of the global generators
    and of the device)."""
    generator = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(groups, d, group_size, generator=generator))      # (G, d, b) orthonormal columns
    return Q.transpose(1, 2).contiguous()


@torch.no_grad()
def balance_encoder_decoder(E: torch.Tensor, D: torch.Tensor, chunk: int) -> None:
    """In place on (G, b, d) tensors: E_g -> A E_g, D_g -> A^-1 D_g with A symmetric positive definite and A^2 = P, the solution of
    P G_E P = G_D (G_E = E_g E_g^T, G_D = D_g D_g^T). Afterwards E_g E_g^T = D_g D_g^T, and E_g^T D_g is unchanged.
    Only b x b eigendecompositions, in fp64, `chunk` groups at a time."""
    def sym_pow(M: torch.Tensor, p: float) -> torch.Tensor:
        lam, V = torch.linalg.eigh(M)
        lam = lam.clamp_min(_EIG_FLOOR_REL * lam.amax(-1, keepdim=True).clamp_min(_EIG_FLOOR_ABS))
        return (V * lam.pow(p)[:, None, :]) @ V.transpose(1, 2)

    for i in range(0, E.shape[0], chunk):
        Ec, Dc = E[i:i + chunk].double(), D[i:i + chunk].double()
        gram_e, gram_d = Ec @ Ec.transpose(1, 2), Dc @ Dc.transpose(1, 2)                # (chunk, b, b)
        root, inv_root = sym_pow(gram_e, 0.5), sym_pow(gram_e, -0.5)
        P = inv_root @ sym_pow(root @ gram_d @ root, 0.5) @ inv_root
        P = 0.5 * (P + P.transpose(1, 2))
        lam, V = torch.linalg.eigh(P)
        a = lam.clamp(*_BALANCE_CLIP).sqrt()
        A = (V * a[:, None, :]) @ V.transpose(1, 2)
        A_inv = (V * (1.0 / a)[:, None, :]) @ V.transpose(1, 2)
        E[i:i + chunk] = (A @ Ec).to(E.dtype)
        D[i:i + chunk] = (A_inv @ Dc).to(D.dtype)


@dataclass
class SiteLosses:
    """what one site's batch gives"""
    loss: torch.Tensor      # () L_rec + L_aux, with grad
    L_rec: torch.Tensor     # ()
    L_aux: torch.Tensor     # () zero (no grad) when no group is dead
    mask: torch.Tensor      # (n, G) bool: the selected groups
    cut: torch.Tensor       # () the smallest selected norm of the batch


class StateDictionary(nn.Module):
    def __init__(self, d: int, sites: list[str], recipe: VocabRecipe, device: torch.device | str) -> None:
        super().__init__()
        self.d, self.sites, self.recipe = d, list(sites), recipe
        frames = initial_frames(recipe.groups, recipe.group_size, d, recipe.seed)
        self.init_hash: str = hashlib.md5(frames.numpy().tobytes()).hexdigest()
        # Parameter order is part of the numerics: gradient clipping takes the norm of the per-tensor norms in this order.
        self.D = nn.Parameter(frames.to(device))                                 # (G, b, d) decoder
        self.b_dec = nn.Parameter(torch.zeros(d, device=device))                 # (d,)
        self.E = nn.Parameter(self.D.detach().clone())                           # (G, b, d) encoder, initialised to the decoder
        self.register_buffer("site_mu", torch.zeros(len(self.sites), d, device=device))       # (S, d), set by calibration
        self.register_buffer("site_scale", torch.ones(len(self.sites), device=device))        # (S,)
        self.register_buffer("thr", torch.zeros((), device=device))              # EMA of the batch's smallest selected norm
        self.register_buffer("var", torch.full((), float(d), device=device))     # E||xs - mean||^2 = d: the loss is divided by it
        self.thr_started = False
        self.steps_done = 0

    @property
    def aux_groups(self) -> int:
        return self.recipe.aux_groups(self.d)

    # ---- coordinates
    def standardize(self, x: torch.Tensor, site: int) -> torch.Tensor:
        """x (n, d) raw activations of site number `site` -> xs (n, d)"""
        return (x.float() - self.site_mu[site]) / self.site_scale[site]

    def encode(self, xt: torch.Tensor) -> torch.Tensor:
        """xt (n, d) -> z (n, G, b). The matmul is the only operation left to the caller's autocast."""
        G, b, d = self.D.shape
        return (xt @ self.E.reshape(G * b, d).T).view(-1, G, b)

    def write(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (n, G, b), zero for groups that are off -> sum_g codes_g D_g (n, d)"""
        G, b, d = self.D.shape
        return codes.reshape(-1, G * b) @ self.D.view(G * b, d)

    def written_norm(self, z: torch.Tensor) -> torch.Tensor:
        """z (n, G, b) -> ||z_g D_g|| (n, G), fp32, no grad: sqrt(z_g (D_g D_g^T) z_g^T)"""
        with torch.no_grad(), torch.autocast(z.device.type, enabled=False):
            gram = self.D.detach().float() @ self.D.detach().float().transpose(1, 2)        # (G, b, b)
            return torch.cat([torch.einsum("nkr,krs,nks->nk", zc, gram, zc).clamp_min(0).sqrt()
                              for zc in z.detach().float().split(self.recipe.norm_chunk)])

    @torch.no_grad()
    def select(self, norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """norm (n, G) -> mask (n, G) bool of each token's k largest, and the smallest selected value of the batch"""
        values, groups = norm.topk(self.recipe.active_groups, dim=-1, sorted=False)
        return torch.zeros_like(norm, dtype=torch.bool).scatter_(-1, groups, True), values.min()

    # ---- training
    def losses(self, xs: torch.Tensor, dead: torch.Tensor) -> SiteLosses:
        """xs (n, d) standardized activations of one site; dead (G,) bool"""
        device_type = xs.device.type
        xt = xs - self.b_dec
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=self.recipe.encode_in_bf16):
            z = self.encode(xt)
        with torch.autocast(device_type, enabled=False):
            z = z.float()
            norm = self.written_norm(z)
            mask, cut = self.select(norm)
            codes = z * mask[..., None].to(z.dtype)
            residual = xt - self.write(codes)
            L_rec = (residual ** 2).sum(-1).mean() / self.var
            L_aux = self._aux_loss(residual.detach(), z, norm, dead) / self.var
        return SiteLosses(L_rec + L_aux, L_rec, L_aux, mask, cut)

    def _aux_loss(self, residual: torch.Tensor, z: torch.Tensor, norm: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
        """Per token, the `aux_groups` dead groups with the largest written norm reconstruct the (detached) residual of the live
        groups. Scaled down while fewer than `aux_groups` groups are dead."""
        n_dead = int(dead.sum())
        if n_dead == 0:
            return residual.new_tensor(0.0)
        k_aux = self.aux_groups
        masked = torch.where(dead[None, :], norm, torch.full_like(norm, -float("inf")))
        chosen = masked.topk(min(k_aux, n_dead), dim=-1).indices                          # (n, min(k_aux, n_dead))
        on = torch.zeros_like(norm, dtype=z.dtype).scatter_(-1, chosen, 1.0)
        reconstruction = self.write(z * on[..., None])
        scale = min(n_dead / k_aux, 1.0)
        return self.recipe.aux_coefficient * scale * ((reconstruction - residual) ** 2).sum(-1).mean()

    @torch.no_grad()
    def after_step(self) -> None:
        """call after every optimizer step: balances encoder and decoder every `balance_every`-th step"""
        self.steps_done += 1
        if self.steps_done % self.recipe.balance_every == 0:
            balance_encoder_decoder(self.E.data, self.D.data, self.recipe.balance_chunk)

    @torch.no_grad()
    def update_threshold(self, cut: torch.Tensor) -> None:
        if not self.thr_started:
            self.thr.copy_(cut)
            self.thr_started = True
        else:
            self.thr.mul_(self.recipe.threshold_decay).add_(cut, alpha=1 - self.recipe.threshold_decay)

    # ---- inference (fp32, no autocast)
    @torch.no_grad()
    def write_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Q (G, b, d) orthonormal rows spanning each group's decoder rows, R (G, b, b) with D_g = R_g Q_g (fp32 QR on the device)"""
        Qs, Rs = [], []
        for c in self.D.float().split(self.recipe.basis_chunk):
            Qt, Rt = torch.linalg.qr(c.transpose(1, 2))
            Qs.append(Qt.transpose(1, 2))
            Rs.append(Rt.transpose(1, 2))
        return torch.cat(Qs), torch.cat(Rs)

    @torch.no_grad()
    def reconstruct(self, xs: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """xs (n, d) standardized -> its reconstruction (n, d) in the same units, from each token's k selected groups.
        (Q, R) = write_basis(). n x G x b floats are allocated: call with about a thousand tokens at a time.

        The sum over a token's groups is an index_add_ with repeated rows. On a GPU the order of those additions is not fixed, so two
        calls on the same input can differ in the last bits (about 1e-7 relative)."""
        xt = xs - self.b_dec
        z = self.encode(xt).float()
        mask = self.select(self.written_norm(z))[0]
        rows, groups = mask.nonzero(as_tuple=True)
        codes = torch.einsum("nr,nrs->ns", z[rows, groups], R[groups])                    # codes in the orthonormal basis
        x_hat = torch.zeros_like(xs)
        x_hat.index_add_(0, rows, (codes[:, None, :] @ Q.index_select(0, groups)).squeeze(1))
        return x_hat + self.b_dec
