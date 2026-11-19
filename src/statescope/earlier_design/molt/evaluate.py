"""Held-out numbers of the MLP dictionary trainer.

  QuickEval    at every logged step: the training model under its inference rule on a few thousand tokens of each held-out split.
  final_eval   at the end: the exported folder, read back with the release reader (statescope.reader.MoltOps), on all held-out tokens,
               plus a check that the export computes what the training model computes ("fold error").

FVU = fraction of variance unexplained of the raw MLP output; L0 = atoms active per token.

One detail both evaluations share with training: the model's input u is made from the STANDARDIZED resid_mid, z = (x - mu_x) / sd_x
then u = u_from_z(z), because that is what the shuffle buffer hands to the trainer. The reader goes from raw x to u directly; the two
differ by float rounding, and the fold error measures that together with the export's own rounding.
"""
from __future__ import annotations

import torch

from statescope.earlier_design.calibration import CalibStats
from statescope.earlier_design.molt.model import MoltModel
from statescope.heldout import HELDOUT_SPLITS
from statescope.metrics import molt_heldout_numbers
from statescope.reader import MoltOps

Heldout = dict[str, tuple[torch.Tensor, torch.Tensor]]     # split -> (x raw resid_mid (n, d), y raw MLP output (n, d)), fp32


class QuickEval:
    """Fixed evaluation tokens (the first `n_tokens` of each split), kept on the device for the whole run."""

    def __init__(self, heldout: Heldout, stats: CalibStats, n_tokens: int, device) -> None:
        """`stats` on `device`"""
        self.stats = stats
        self.u: dict[str, torch.Tensor] = {}
        self.y: dict[str, torch.Tensor] = {}
        self.variance: dict[str, float] = {}
        for split, (x, y) in heldout.items():
            x, y = x[:n_tokens].to(device), y[:n_tokens].to(device)
            z, _ = stats.standardize(x, y)
            self.u[split], self.y[split] = stats.u_from_z(z), y
            self.variance[split] = float(((y - y.mean(0)) ** 2).sum(-1).mean())

    @torch.no_grad()
    def __call__(self, model: MoltModel) -> dict[str, float]:
        """{"q_fvu_<split>", "q_l0_<split>"}"""
        out = {}
        for split, u in self.u.items():
            a, y_std = model.forward_infer(u)
            y_hat = y_std * self.stats.sd_y + self.stats.mu_y
            out[f"q_fvu_{split}"] = float(((y_hat - self.y[split]) ** 2).sum(-1).mean() / self.variance[split])
            out[f"q_l0_{split}"] = float((a != 0).sum(-1).float().mean())
        return out


@torch.no_grad()
def fold_error(model: MoltModel, exported: MoltOps, stats: CalibStats, x: torch.Tensor, y: torch.Tensor) -> float:
    """max |export(x) - training model(x)| / max |training model(x)| on raw tokens x (n, d), all in one forward"""
    z, _ = stats.standardize(x, y)
    y_model = model.forward_infer(stats.u_from_z(z))[1] * stats.sd_y + stats.mu_y
    y_export = exported.forward(x)[1]
    return float((y_export - y_model).abs().max() / y_model.abs().max())




@torch.no_grad()
def final_eval(model: MoltModel, export_dir: str, heldout: Heldout, stats: CalibStats, device) -> dict[str, float]:
    """`heldout`: all tokens of every split, on the CPU. `stats` on `device`. Returns the numbers of final.json except the dead count."""
    exported = MoltOps(export_dir, device)
    x_first, y_first = heldout[HELDOUT_SPLITS[0]]            # the fold check uses the first tokens of the first split (web text)
    n = model.recipe.fold_check_tokens
    out = {"fold_max_rel_err": fold_error(model, exported, stats, x_first[:n].to(device), y_first[:n].to(device))}
    out.update(molt_heldout_numbers(exported, heldout, model.recipe.eval_chunk))
    return out
