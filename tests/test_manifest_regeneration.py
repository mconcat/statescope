"""The manifest builder regenerates the published manifest from the release files and the original result files (rule R7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import REPO, Settings
from statescope.earlier_design.foldcheck import fold_error_verdict
from statescope.earlier_design.release import LayerNumbers, build_manifest


def _last_record(path: Path) -> dict:
    return json.loads([line for line in path.read_text().splitlines() if line.strip()][-1])


def _numbers_from_original_results(results: Path, root: Path, layer: int, fold: dict, splice: dict) -> LayerNumbers:
    """the original run-folder layout (what the original collect_layers / release_manifest read)"""
    run = results / "mlp_track" / "train" / f"L{layer}"
    vocab_log = _last_record(run / "V0" / "vocab_V0s0.jsonl")
    vocab_cfg = json.loads((root / "layers" / f"L{layer:02d}" / "vocab" / "cfg.json").read_text())
    budget = json.loads((run / "M1w" / "mlp_M1w" / "s0" / "M1w" / "budget.json").read_text())
    extras = json.loads((run / "M1w" / "mlp_M1w" / "s0" / "extras" / "extras_final.json").read_text())
    final = budget["final"]
    check = fold.get(str(layer))
    if check is not None and abs(check["recorded_fold_max_rel_err"] - final["fold_max_rel_err"]) >= 1e-12:
        check = None                                              # a check of another export of this layer
    s = splice.get(str(layer))
    return LayerNumbers(
        vocab_heldout_fvu={site: vocab_log["heldout_fvu"][site] for site in (f"resid_mid.{layer}", f"resid_post.{layer}")},
        vocab_groups=vocab_cfg["n_units"], vocab_dead_groups=vocab_log["n_dead"], vocab_train_tokens=vocab_cfg["tokens"],
        molt_heldout_fvu=final["fvu_pooled"], molt_n_atoms=final["n_m_init"], molt_train_tokens=budget["tokens"],
        molt_silent_fraction=1 - extras["n_live_heldout"] / final["n_m_init"], molt_fold_err=final["fold_max_rel_err"],
        molt_gate_flips=None if check is None else check["gate_decisions_differing_fp32_vs_fp64"],
        molt_fold_verdict=fold_error_verdict(final["fold_max_rel_err"], check),
        splice=s if s is not None and s["n_docs"] == 100 else None)


@pytest.mark.release
@pytest.mark.legacy
def test_manifest_is_regenerated_exactly(settings: Settings):
    snapshot, results, root = settings.need_legacy_snapshot(), settings.need_legacy_results(), settings.need_release_root()
    published = json.loads((snapshot / "release" / "qwen35-08b-layer-dicts" / "manifest.json").read_text())
    fold = json.loads((snapshot / "results" / "mlp_track" / "fold_check" / "fold_check.json").read_text())
    splice = json.loads((snapshot / "results" / "mlp_track" / "splice_loss" / "splice.json").read_text())
    header = json.loads((REPO / "release" / "qwen35-08b" / "header.json").read_text())
    numbers = {layer: _numbers_from_original_results(results, root, layer, fold, splice) for layer in range(24)}
    regenerated = build_manifest(header, str(root), numbers)

    assert {k: v for k, v in regenerated.items() if k != "splice"} == {k: v for k, v in published.items() if k != "splice"}
    assert list(regenerated) == list(published)                   # same key order at the top level
    committed = json.loads((REPO / "release" / "qwen35-08b" / "manifest.json").read_text())
    assert regenerated == committed                               # the manifest in this repository is the regenerated one
