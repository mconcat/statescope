"""The gate-threshold check reproduces the original fold_check results for the release (rule R6 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import json

import pytest
import torch

from conftest import TEST_OUT, Settings
from statescope.earlier_design.foldcheck import fold_error_verdict, gate_flip_check
from statescope.heldout import load_heldout
from statescope.reader import MoltOps


@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.parametrize("layer", [15, 19, 22, 23])
def test_gate_flip_check_reproduces_the_recorded_results(settings: Settings, device: str, layer: int):
    recorded = json.loads((settings.need_legacy_snapshot() / "results" / "mlp_track" / "fold_check" / "fold_check.json").read_text())[str(layer)]
    x, _ = load_heldout(str(settings.calib_dir_08b(layer)), "pile", 4096)
    molt = MoltOps(str(settings.layer_dir(layer, "molt")), device)
    got = gate_flip_check(molt, x.to(device), recorded["recorded_fold_max_rel_err"])
    del molt
    torch.cuda.empty_cache()
    (TEST_OUT / "measured").mkdir(parents=True, exist_ok=True)
    (TEST_OUT / "measured" / f"foldcheck_L{layer}.json").write_text(json.dumps({"new": got, "recorded": {k: v for k, v in recorded.items() if k != "nearest_pairs"}}, indent=1))
    assert got["gate_decisions_differing_fp32_vs_fp64"] == recorded["gate_decisions_differing_fp32_vs_fp64"]
    old_err = recorded["output_rel_err_from_fp32_vs_fp64_gate_alone"]
    assert abs(got["output_rel_err_from_fp32_vs_fp64_gate_alone"] - old_err) <= 1e-6 * max(old_err, 1e-12)
    assert abs(got["rounding_step_h_q99"] - recorded["abs_h32_minus_h64"]["0.99"]) <= 0.1 * recorded["abs_h32_minus_h64"]["0.99"]
    if recorded["recorded_fold_max_rel_err"] > 1e-4:
        new_best, old_best = got["best_match_within_100x_q99"], recorded["best_match_within_100x_q99"]
        assert (new_best["token"], new_best["atom"]) == (old_best["token"], old_best["atom"])
        assert abs(new_best["ratio_to_recorded"] - old_best["ratio_to_recorded"]) <= 1e-6
        assert fold_error_verdict(recorded["recorded_fold_max_rel_err"], got) == "fold: one flipped pair"
    else:
        assert fold_error_verdict(recorded["recorded_fold_max_rel_err"], got) is None
