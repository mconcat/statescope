"""One optimizer step of the MLP dictionary, and the per-atom counters kept across steps.

A step, on a batch u (B, d), y_std (B, d):
  1. the training gate is decided on the whole batch without gradient (so splitting the batch cannot change the selection), and
     the atoms' gate means are updated from it;
  2. the batch goes through forward and backward in `micro_batches` pieces with that selection:
         loss = mean over tokens of |y_hat - y_std|^2  +  preact_coef * mean over tokens of sum_t relu(-h_t)
  3. if there are dead atoms, the auxiliary loss is added (gradient to the dead atoms only):
         aux_coef * sum |residual - y_aux|^2 / sum |residual|^2,   residual = y_std - y_hat of step 2 (no gradient)
  4. gradients of unit rows are projected, Adam steps, rows are renormalized, and each tier's inference threshold moves
     toward this batch's cut-off.

Shapes in comments: B tokens, d model width, T atoms, D dead atoms.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from statescope.earlier_design.molt.model import MoltModel
from statescope.earlier_design.molt.recipe import MoltRecipe


@dataclass
class StepResult:
    loss: float
    mse: float
    aux_loss: float            # 0.0 when no atom is dead
    n_aux_selected: int        # dead atoms per token in the auxiliary output (0 when no atom is dead)
    lr: float
    active: torch.Tensor       # (B, T) bool: atom fired on the token, from the micro-batch forwards


def aux_output(model: MoltModel, u: torch.Tensor, dead: torch.Tensor, k: int) -> tuple[torch.Tensor, int]:
    """What the dead atoms would write: per token the k dead atoms with the largest gate input, each with gate value h / m
    (sign kept, no relu). u (B, d), dead (D,) atom ids -> (y_aux (B, d), atoms used per token)."""
    h = model.whiten(u) @ model.e[dead].T - model.b[dead]                    # (B, D)
    n_selected = min(k, dead.numel())
    top = h.detach().topk(n_selected, dim=1).indices
    selected = torch.zeros_like(h, dtype=torch.bool).scatter_(1, top, True)
    g = h * selected / model.gate_mean[dead].detach()[None]                  # (B, D)
    B, d = u.shape[0], model.d
    tier = model.tier[dead]
    out = None
    for i, r in enumerate(model.ranks):
        where = (tier == r).nonzero().squeeze(1)                             # positions in `dead` of this tier's atoms
        if where.numel() == 0:
            continue
        p = model.pos[dead[where]]
        U_d, V_d, s_d = model.U[i][p], model.V[i][p], model.s[i][p]
        n_d = p.numel()
        lat = u @ (V_d * s_d[:, :, None]).reshape(n_d * r, d).T
        lat = (lat.view(B, n_d, r) * g[:, where][:, :, None]).reshape(B, n_d * r)
        part = lat @ U_d.reshape(n_d * r, d)
        out = part if out is None else out + part
    return out, n_selected


def train_step(model: MoltModel, optimizer: torch.optim.Optimizer, u: torch.Tensor, y_std: torch.Tensor, dead: torch.Tensor,
               recipe: MoltRecipe) -> StepResult:
    """dead (D,) long: ids of the atoms that are dead before this step."""
    B, micro = u.shape[0], recipe.micro_batches
    if B % micro:
        raise ValueError(f"batch {B} not divisible into {micro} micro-batches")
    with torch.no_grad():
        h_full = model.gate_input(u)
        mask, smallest = model.select(h_full)
        model.update_gate_mean_(torch.relu(h_full) * mask.to(h_full.dtype))
        del h_full
    losses, mses, actives, y_hats = [], [], [], []
    for i, (uc, yc) in enumerate(zip(u.chunk(micro), y_std.chunk(micro))):
        h = model.gate_input(uc)
        a = torch.relu(h) * mask.chunk(micro)[i].to(h.dtype)
        y_hat = model.decode(a / model.gate_mean, uc)
        mse = ((y_hat - yc) ** 2).sum(-1).mean()
        loss = mse + recipe.preact_coef * torch.relu(-h).sum(-1).mean()
        (loss * (uc.shape[0] / B)).backward()
        losses.append(loss.detach() * (uc.shape[0] / B))
        mses.append(mse.detach() * (uc.shape[0] / B))
        actives.append(a.detach() != 0)
        y_hats.append(y_hat.detach())
        del h, a, y_hat, mse, loss
    aux_total, n_aux_selected = 0.0, 0
    if recipe.aux_k and dead.numel() > 0:
        residuals = [yc - y_hat for yc, y_hat in zip(y_std.chunk(micro), y_hats)]
        denominator = sum(float((res ** 2).sum()) for res in residuals)
        for uc, res in zip(u.chunk(micro), residuals):
            y_aux, n_aux_selected = aux_output(model, uc, dead, recipe.aux_k)
            aux = recipe.aux_coef * ((res - y_aux) ** 2).sum() / denominator
            aux.backward()
            aux_total += float(aux.detach())
    model.project_grads_()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.renormalize_()
    model.update_tier_thr_(smallest)
    return StepResult(loss=float(sum(losses)), mse=float(sum(mses)), aux_loss=aux_total, n_aux_selected=n_aux_selected,
                      lr=optimizer.param_groups[0]["lr"], active=torch.cat(actives))


@dataclass
class Counters:
    """Per-atom firing bookkeeping of a run."""
    since: torch.Tensor            # (T,) long: tokens since the atom last fired
    segment_fires: torch.Tensor    # (segments, T) long: (token, atom) firings in each segment of the run
    segment_tokens: list[int]      # tokens trained in each segment

    @staticmethod
    def new(n_atoms: int, segments: int, device) -> "Counters":
        return Counters(torch.zeros(n_atoms, dtype=torch.long, device=device), torch.zeros(segments, n_atoms, dtype=torch.long, device=device),
                        [0] * segments)

    def dead_mask(self, dead_tokens: int) -> torch.Tensor:
        return self.since >= dead_tokens

    @torch.no_grad()
    def update(self, active: torch.Tensor, step: int, run_steps: int) -> None:
        """active (B, T) bool of the step just taken"""
        n_tokens, segments = active.shape[0], len(self.segment_tokens)
        self.since += n_tokens
        self.since[active.any(0)] = 0
        segment = min(segments - 1, step * segments // run_steps)
        self.segment_fires[segment] += active.sum(0)
        self.segment_tokens[segment] += n_tokens
