"""Writing a trained MLP dictionary in the release format (the folder `statescope.reader.MoltOps` reads).

    cfg.json                 sizes, the gate rule's settings, atoms per rank
    input.json               how the dictionary's input is made from raw resid_mid
    weights.safetensors      the folded dictionary (below) and the calibration statistics it needs
    freq_train.safetensors   how often each atom fired during training, per tenth of the run

"Folded" = ready for inference: the whitener is multiplied into the gate directions (e <- e W), and each atom's scales are divided by
its gate mean (s <- s / m), so the reader computes  a = relu(h) [h / score_div > thr_atom],  y_std = u W_skip + b_dec + sum a ((u V^T) s) U.
The training-space values are kept next to them as v2_m, v2_s, v2_m_init (names of the format).

The format is older than this recipe and carries a few fields the recipe does not use (log_theta, thr, thr_init, and theta0 / bandwidth /
ste_to_input / gate in cfg.json). They are written with their fixed values so that the checkpoint layout stays what it was.
"""
from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import save_file

from statescope.calibration import CalibStats
from statescope.molt.model import MoltModel
from statescope.molt.step import Counters

CALIB_KEYS = ("mu_u", "sd_u", "mu_y", "sd_y", "norm_w_eff")
SOURCE = "statescope.molt.export (per-tier export)"
RULE_NOTE = "per-tier rule: s = s_train / m; a = relu(h) * [h / score_div > thr_atom]"


def export_cfg(model: MoltModel) -> dict:
    recipe = model.recipe
    n, R, d = model.n_atoms, recipe.max_rank, model.d
    # "rect_cfg" describes the same dictionary stored as rectangular (n, r_max, d) tensors, an older layout that tools can rebuild
    # from this export; n_params counts that layout: e, b, log_theta, U, V, s, W_skip, b_dec.
    rect = {"kind": "gate_rule_folded", "name": recipe.name, "d": d, "n_atoms": n, "r_max": R, "tau": recipe.init_scale,
            "theta0": recipe.unused_theta0, "bandwidth": recipe.unused_bandwidth, "ste_to_input": recipe.unused_ste_to_input,
            "n_params": n * d + 2 * n + 2 * n * R * d + n * R + d * d + d,
            "space": "standardized u in, standardized y out (wrap with RawAdaptive)",      # format text; RawAdaptive was the original wrapper class
            "note": RULE_NOTE, "gate": "batchtopk", "k": recipe.k_total,
            "rule_cfg": {"rule": "q", "alpha": 0.0, "budget": recipe.budget, "k_by_rank": {str(r): k for r, k in recipe.k_by_rank.items()}}}
    rank_hist = {str(r): int(model.tier_index(i).numel()) for i, r in enumerate(model.ranks)}
    return dict(rect, kind="gate_rule_stripped", rank_hist=rank_hist, tiered_full=True, rect_cfg=rect, source=SOURCE)


@torch.no_grad()
def export_tensors(model: MoltModel, stats: CalibStats) -> dict[str, torch.Tensor]:
    """every tensor of weights.safetensors, on the CPU"""
    def cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().contiguous().cpu()
    n = model.n_atoms
    out = {"e": cpu(model.folded_e()), "b": cpu(model.b), "log_theta": torch.full((n,), math.log(model.recipe.unused_theta0)),
           "W_skip": cpu(model.W_skip), "b_dec": cpu(model.b_dec), "thr": torch.zeros(1), "thr_init": torch.zeros(1, dtype=torch.bool),
           "score_div": cpu(model.score_div), "thr_atom": cpu(model.atom_thr())}
    for key in CALIB_KEYS:
        out[f"calib_{key}"] = cpu(getattr(stats, key))
    out["v2_m"], out["v2_s"], out["v2_m_init"] = cpu(model.gate_mean), cpu(model.scales_rect()), cpu(model.gate_mean_set)
    for i, r in enumerate(model.ranks):
        idx = model.tier_index(i)
        out[f"idx_r{r}"] = cpu(idx)
        out[f"s_r{r}"] = cpu(model.s[i] / model.gate_mean[idx][:, None])
        out[f"U_r{r}"] = cpu(model.U[i])
        out[f"V_r{r}"] = cpu(model.V[i])
    return out


def write_export(path: str, model: MoltModel, stats: CalibStats, counters: Counters) -> None:
    os.makedirs(path, exist_ok=True)
    save_file(export_tensors(model, stats), os.path.join(path, "weights.safetensors"))
    with open(os.path.join(path, "cfg.json"), "w") as f:
        json.dump(export_cfg(model), f, indent=1)
    with open(os.path.join(path, "input.json"), "w") as f:
        json.dump({"input": "u", "eps": stats.norm_eps, "definition": "inner((n(x) - mu_u) / sd_u), n(x) = x rsqrt(mean x^2 + eps) w_eff"}, f, indent=1)
    save_file({"seg": counters.segment_fires.cpu().contiguous(), "seg_tokens": torch.tensor(counters.segment_tokens, dtype=torch.long)},
              os.path.join(path, "freq_train.safetensors"))
