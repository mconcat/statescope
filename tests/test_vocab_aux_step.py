"""The state dictionary's dead-group auxiliary loss: ONE training step from an injected state, the new code against two legacy runs
(rule R4.5 of EQUIVALENCE_RULES.md).

Each process builds and calibrates the dictionary as its trainer does, marks every 97th group as dead, takes one step on the first 4,096
held-out pile tokens of the calibration folder and saves the step's numbers and every parameter and counter after it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TEST_OUT, Settings
from legacy_bridge import run_legacy
from new_bridge import run_new
from test_stream import data_paths_08b
from train_compare import RunFiles, _number_diffs, _tensor_diffs, noise_floor

LAYER = 19
SITES = (f"resid_mid.{LAYER}", f"resid_post.{LAYER}")
STEP_NUMBERS = (*(f"L_rec__{s}" for s in SITES), "L_aux", "grad_norm", "n_dead", "n_fired", "lr", "clip_hits", "n_always")
# what must be equal before the step is worth comparing: the initial frames, the site calibration, the parameters going into the step
SETUP_KEYS = ("init_hash", "calib", "params_before_sha256", "n_tokens", "n_dead_injected", "sites")


def _files(run: Path) -> RunFiles:
    info = json.loads((run / "step.json").read_text())
    record = info["record"]
    return RunFiles(tensors={"step": run / "step.safetensors"}, jsons={"setup": {k: info[k] for k in SETUP_KEYS}},
                    steps=[{"step": record["step"], **{k: record[k] for k in STEP_NUMBERS}}], final={})


def step_args(settings: Settings) -> list[str]:
    paths = data_paths_08b(settings)
    return ["--layer", str(LAYER), "--calib-dir", str(settings.calib_dir_08b(LAYER)), "--view-corpus", paths.view_corpus,
            "--full-corpus", paths.full_corpus, "--mem-gb", str(settings.train_gb)]


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
def test_vocab_aux_step_equals_legacy(settings: Settings):
    a, b = (_files(run_legacy("vocab_step_reference.py", [*step_args(settings), "--rep", rep], settings, timeout_s=6 * 3600, training=True))
            for rep in ("A", "B"))
    for name, run in (("A", a), ("B", b)):
        record = run.steps[0]
        if not (record["n_dead"] > 0 and record["L_aux"] != 0.0):
            pytest.fail(f"rule R4.5 is void: legacy run {name} reports n_dead {record['n_dead']}, L_aux {record['L_aux']}", pytrace=False)

    paths = data_paths_08b(settings)
    new = _files(run_new("vocab_step.py", [*step_args(settings), "--pile-dir", paths.pile_dir, "--ultrachat-dir", paths.ultrachat_dir],
                         settings, timeout_s=6 * 3600, exclude=("molt",)))

    floor = noise_floor(a, b)                       # every difference between the two legacy results
    exact = floor["identical"]
    failures = list(floor["problems"])
    report = {"legacy_runs_bit_identical": exact, "legacy_record": a.steps[0], "max_new_vs_a": {}, "noise": {}}

    def check(name: str, diff: float, noise: float) -> None:
        """bit-identical if the legacy runs are; otherwise R4.3's 3x rule, with the difference reported"""
        report["max_new_vs_a"][name], report["noise"][name] = diff, noise
        if (diff != 0.0) if exact else (diff > 3 * noise):
            failures.append(f"{name}: |new - A| = {diff:.3g}, |A - B| = {noise:.3g}")

    problems, diffs = _tensor_diffs(new.tensors["step"], a.tensors["step"])
    failures += [f"new vs A: {p}" for p in problems]
    for key, diff in diffs.items():
        check(f"tensor {key}", diff, floor["tensors"]["step"].get(key, 0.0))
    for key, diff in _number_diffs({k: new.steps[0][k] for k in a.steps[0]}, a.steps[0]).items():
        check(f"number {key}", diff, floor["steps"][0][key])
    if new.jsons["setup"] != a.jsons["setup"]:
        differing = sorted(k for k in SETUP_KEYS if new.jsons["setup"][k] != a.jsons["setup"][k])
        failures.append(f"the state before the step differs from legacy in {differing}")
    (TEST_OUT / "measured").mkdir(parents=True, exist_ok=True)
    (TEST_OUT / "measured" / "vocab_aux_step.json").write_text(json.dumps(report, indent=1))
    assert failures == [], "\n".join(failures[:40])
