"""Short training of the MLP dictionary (recipe M1w): the new trainer against two legacy runs (rules R4.1 - R4.4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TEST_OUT, Settings
from legacy_bridge import run_legacy
from new_bridge import run_new
from test_stream import data_paths_08b
from train_args import DEAD_VARIANT, LOG_EVERY, SHORT_STEPS, legacy_molt_run
from train_compare import RunFiles, compare, read_jsonl

LAYER = 19
STEP_NUMBERS = ("loss", "mse", "l_aux", "n_aux_candidates", "n_aux_selected", "l0_train", "n_dead_1M", "lr",
                "q_fvu_pile", "q_fvu_ultra", "q_fvu_cont", "q_fvu_chat", "q_l0_pile", "q_l0_ultra", "q_l0_cont", "q_l0_chat")
# cfg.json keys that only say which code wrote the file (fixed before the first comparison); every other key must be equal
CFG_PROVENANCE = {"source", "note"}


def _legacy_files(run: Path) -> RunFiles:
    arm = run / "run" / "mlp_M1w"
    export = arm / "s0" / "M1w"
    final = json.loads((export / "budget.json").read_text())["final"]
    return RunFiles(
        tensors={"weights": export / "weights.safetensors", "whitener": arm / "whitener.safetensors", "freq_train": export / "freq_train.safetensors"},
        jsons={"cfg.json": json.loads((export / "cfg.json").read_text()), "input.json": json.loads((export / "input.json").read_text())},
        steps=[{"step": r["step"], **{k: r[k] for k in STEP_NUMBERS}} for r in read_jsonl(arm / "curves" / "M1w_s0.jsonl")],
        final={"fold_max_rel_err": final["fold_max_rel_err"], "fvu_pooled": final["fvu_pooled"], "l0": final["all"]["l0"],
               "n_dead_1M_end": final["n_dead_1M_end"], **{f"fvu_{s}": final[s]["fvu"] for s in ("pile", "ultra", "cont", "chat")}},
        provenance={"cfg.json": CFG_PROVENANCE})


def _new_files(run: Path) -> RunFiles:
    """layout written by statescope's MLP dictionary trainer: export/ (the release format), whitener.safetensors, curve.jsonl, final.json"""
    export = run / "run" / "export"
    return RunFiles(
        tensors={"weights": export / "weights.safetensors", "whitener": run / "run" / "whitener.safetensors", "freq_train": export / "freq_train.safetensors"},
        jsons={"cfg.json": json.loads((export / "cfg.json").read_text()), "input.json": json.loads((export / "input.json").read_text())},
        steps=[{"step": r["step"], **{k: r[k] for k in STEP_NUMBERS}} for r in read_jsonl(run / "run" / "curve.jsonl")],
        final=json.loads((run / "run" / "final.json").read_text()),
        provenance={"cfg.json": CFG_PROVENANCE})


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("variant", ["base", "dead"])
def test_molt_training_equals_legacy(settings: Settings, variant: str):
    extra = tuple(DEAD_VARIANT) if variant == "dead" else ()
    a, b = (_legacy_files(run_legacy("train_reference.py", legacy_molt_run(settings, LAYER, rep, extra=extra), settings, timeout_s=6 * 3600, training=True))
            for rep in ("A", "B"))
    if variant == "dead":
        assert any(s["n_aux_selected"] > 0 for s in a.steps), "the legacy 'dead' reference never ran the auxiliary loss; the configuration is not a test of it"
    paths = data_paths_08b(settings)
    results = settings.need_legacy_results()
    args = ["--layer", str(LAYER), "--calib-dir", str(settings.calib_dir_08b(LAYER)), "--fixmap", str(results / "mlp_track" / "fixedmap" / f"L{LAYER}.safetensors"),
            "--steps", str(SHORT_STEPS), "--log-every", str(LOG_EVERY), "--mem-gb", str(settings.train_gb),
            "--pile-dir", paths.pile_dir, "--ultrachat-dir", paths.ultrachat_dir, "--view-corpus", paths.view_corpus, "--full-corpus", paths.full_corpus]
    if variant == "dead":
        args += ["--dead-tokens", DEAD_VARIANT[1]]
    new = _new_files(run_new("train_molt.py", args, settings, timeout_s=6 * 3600))
    failures, report = compare(new, a, b)
    (TEST_OUT / "measured").mkdir(parents=True, exist_ok=True)
    (TEST_OUT / "measured" / f"train_molt_{variant}.json").write_text(json.dumps(report, indent=1))
    assert failures == [], "\n".join(failures[:40])
