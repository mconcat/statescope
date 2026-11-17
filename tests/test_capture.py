"""capture() against the legacy capture on the same loaded model (rule R1.5 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import json

import pytest

import statescope.reader
from conftest import Settings
from legacy_bridge import run_legacy


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("model, layers", [("qwen3.5-0.8b", [0, 19])])
def test_capture_equals_the_legacy_capture(settings: Settings, model: str, layers: list[int]):
    out = run_legacy("capture_reference.py", ["--reader", statescope.reader.__file__, "--model", model,
                                              "--layers", ",".join(str(layer) for layer in layers)], settings)
    info = json.loads((out / "info.json").read_text())
    assert info["layers_object_is_same"] is True
    for layer in layers:
        got = info["layers"][str(layer)]
        assert got["n_tokens"] == info["n_ids"] - 1                     # position 0 dropped
        assert got["max_abs_diff_resid_mid"] == 0.0, got
        assert got["max_abs_diff_mlp_out"] == 0.0, got
        assert got["max_rel_diff_mid_plus_mlp_vs_layer_output"] <= 5e-3, got
