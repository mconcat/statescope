"""Short training of the state dictionary (V0, seed 0): the new trainer against two legacy runs (rules R4.1 - R4.4).
The dead-group auxiliary loss is not reached in 300 steps; it is covered by the state-injection test of rule R4.5."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TEST_OUT, Settings
from legacy_bridge import run_legacy
from new_bridge import run_new
from test_stream import data_paths_08b
from train_args import LOG_EVERY, SHORT_STEPS, legacy_vocab_run
from train_compare import RunFiles, compare, read_jsonl

LAYER = 19
SITES = (f"resid_mid.{LAYER}", f"resid_post.{LAYER}")
STEP_NUMBERS = (*(f"L_rec__{s}" for s in SITES), "L_aux", "grad_norm", "n_dead", "n_fired", "lr", "clip_hits", "n_always")
# cfg.json keys that only say which code wrote the file (fixed before the first comparison); every other key must be equal
CFG_PROVENANCE = {"trainer", "code_md5"}


def _steps(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        row = {"step": r["step"], **{k: r[k] for k in STEP_NUMBERS}}
        for site, value in r.get("heldout_fvu", {}).items():       # logged every fourth log step and at the last step
            row[f"heldout_fvu.{site}"] = value
        out.append(row)
    return out


def _legacy_files(run: Path) -> RunFiles:
    export = run / "run" / "vocab_V0s0" / "final"
    done = json.loads((run / "run" / "layer_recipe_done.json").read_text())
    return RunFiles(
        tensors={"weights": export / "weights.safetensors", "usage": export / "usage.safetensors"},
        jsons={"cfg.json": json.loads((export / "cfg.json").read_text())},
        steps=_steps(read_jsonl(run / "run" / "vocab_V0s0.jsonl")),
        final={f"heldout_fvu.{site}": value for site, value in done["vocab_heldout_fvu"]["V0s0"].items()},
        provenance={"cfg.json": CFG_PROVENANCE})


def _new_files(run: Path) -> RunFiles:
    """layout written by statescope's state dictionary trainer: export/ (the release format), log.jsonl, final.json"""
    export = run / "run" / "export"
    final = json.loads((run / "run" / "final.json").read_text())
    return RunFiles(
        tensors={"weights": export / "weights.safetensors", "usage": export / "usage.safetensors"},
        jsons={"cfg.json": json.loads((export / "cfg.json").read_text())},
        steps=_steps(read_jsonl(run / "run" / "log.jsonl")),
        final={f"heldout_fvu.{site}": value for site, value in final["heldout_fvu"].items()},
        provenance={"cfg.json": CFG_PROVENANCE})


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
def test_vocab_training_equals_legacy(settings: Settings):
    a, b = (_legacy_files(run_legacy("train_reference.py", legacy_vocab_run(settings, LAYER, rep), settings, timeout_s=6 * 3600, training=True))
            for rep in ("A", "B"))
    paths = data_paths_08b(settings)
    args = ["--layer", str(LAYER), "--calib-dir", str(settings.calib_dir_08b(LAYER)), "--seed", "0",
            "--steps", str(SHORT_STEPS), "--log-every", str(LOG_EVERY), "--mem-gb", str(settings.train_gb),
            "--pile-dir", paths.pile_dir, "--ultrachat-dir", paths.ultrachat_dir, "--view-corpus", paths.view_corpus, "--full-corpus", paths.full_corpus]
    new = _new_files(run_new("train_vocab.py", args, settings, timeout_s=6 * 3600))
    failures, report = compare(new, a, b)
    (TEST_OUT / "measured").mkdir(parents=True, exist_ok=True)
    (TEST_OUT / "measured" / "train_vocab.json").write_text(json.dumps(report, indent=1))
    assert failures == [], "\n".join(failures[:40])
