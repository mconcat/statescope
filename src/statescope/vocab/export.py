"""The released file format of a state dictionary (read by statescope.reader.StateVocab). The format is frozen: published
checkpoints use it, so nothing here may be renamed or dropped.

  weights.safetensors   D, E (G, b, d); b_dec (d,); site_mu (S, d); site_scale (S,); thr (1,); log_gamma (G,) all zeros
  cfg.json              sizes, sites, calibration record, training counters (see vocab_cfg)
  usage.safetensors     per group: fire (G, S) times selected per site, since (G,) tokens since last selected at any site,
                        use (G, S) selection-rate EMA; n_tokens (S,) tokens that `fire` counts over
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from statescope.vocab.model import StateDictionary

TRAINER = "statescope.vocab.train"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# files whose md5 is recorded in cfg.json as code_md5 (which code wrote the export)
_CODE_FILES = ("vocab/recipe.py", "vocab/model.py", "vocab/calibrate.py", "vocab/evaluate.py", "vocab/export.py", "vocab/train.py",
               "train/loop.py", "data/mix.py", "lm.py", "calibration.py")
# Fields of the original cfg.json that describe options this trainer does not have. They are written with the values every released
# dictionary has, so that tools reading the format keep working.
_FORMAT_CONSTANTS_A = {"lam_dr": 0.0, "lam_pc": 0.0, "kappa": 1.0, "pc_p": 1.0, "pc_eps": 0.01, "pc_ramp_frac": 0.05, "lam_nest": 0.0,
                       "nest_lmin_dims": 256, "nest_alpha": 1.0, "n_dense": 0}
_FORMAT_CONSTANTS_B = {"form": "comp", "lam": 0.0, "warm": 1000}


def code_md5() -> dict[str, str]:
    return {f"statescope/{rel}": hashlib.md5((_PACKAGE_ROOT / rel).read_bytes()).hexdigest() for rel in _CODE_FILES}


def weight_tensors(model: StateDictionary) -> dict[str, torch.Tensor]:
    """the tensors of weights.safetensors (fp32, on the model's device)"""
    return {"D": model.D.data.float(), "log_gamma": torch.zeros(model.recipe.groups, device=model.D.device), "b_dec": model.b_dec.data.float(),
            "thr": model.thr.float().reshape(1), "site_mu": model.site_mu.float(), "site_scale": model.site_scale.float(),
            "E": model.E.data.float()}


def vocab_cfg(model: StateDictionary, calib_record: dict, steps: int, tokens: int, clip_hits: int, thr_site: torch.Tensor) -> dict:
    """cfg.json. `name`, `arm`, `spec` and the *_dims / unit fields repeat the recipe in the original's vocabulary."""
    recipe = model.recipe
    G, b, k = recipe.groups, recipe.group_size, recipe.active_groups
    return {
        "kind": "sasa_proper_segment", "name": f"segment_b{b}", "d_in": model.d, "n_units": G, "unit_size": b, "k_units": k,
        "b": b, "total_dims": G * b, "active_dims": k * b, **_FORMAT_CONSTANTS_A,
        "k_topk": k, "init_seed": recipe.seed, "init_hash": model.init_hash, **_FORMAT_CONSTANTS_B,
        "sites": list(model.sites), "thr": float(model.thr), "gate": "topk", "score": "output_norm", "E_is_encoder": True,
        "trainer": TRAINER, "arm": f"V0s{recipe.seed}",
        "spec": {"sites": list(model.sites), "seed": recipe.seed, "tf": "none", "sel": "norm", "balance_every": recipe.balance_every},
        "steps": steps, "clip_hits": clip_hits, "thr_site": {site: float(thr_site[j]) for j, site in enumerate(model.sites)},
        "train_gate": "topk", "train_score": "sigma", "beta_log_tail": [],
        "calib": calib_record, "code_md5": code_md5(), "tokens": tokens,
    }


@torch.no_grad()
def export_vocab(path: str, model: StateDictionary, cfg: dict, since: torch.Tensor, use: torch.Tensor, fire: torch.Tensor,
                 fire_tokens: int) -> None:
    """since (G,) int64; use (S, G) float64; fire (S, G) int64; fire_tokens: tokens seen while `fire` was counted"""
    os.makedirs(path, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in weight_tensors(model).items()}, os.path.join(path, "weights.safetensors"))
    with open(os.path.join(path, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    save_file({"fire": fire.T.contiguous().cpu(), "n_tokens": torch.full((len(model.sites),), fire_tokens, dtype=torch.long),
               "since": since.cpu(), "use": use.T.contiguous().float().cpu()}, os.path.join(path, "usage.safetensors"))
