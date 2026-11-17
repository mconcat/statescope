"""The shared training loop yields the legacy loop's batches (rule R4.0 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import json

import pytest

from conftest import Settings
from legacy_bridge import run_legacy
from new_bridge import run_new
from test_stream import data_paths_08b
from train_args import SHORT_STEPS, legacy_vocab_run

LAYER = 19


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
def test_loop_batches_equal_the_legacy_batches(settings: Settings):
    legacy = run_legacy("train_reference.py", legacy_vocab_run(settings, LAYER, "hash", hash_batches=True), settings, timeout_s=6 * 3600, training=True)
    paths = data_paths_08b(settings)
    new = run_new("loop_hashes.py", ["--layer", str(LAYER), "--calib-dir", str(settings.calib_dir_08b(LAYER)), "--steps", str(SHORT_STEPS),
                                     "--pile-dir", paths.pile_dir, "--ultrachat-dir", paths.ultrachat_dir,
                                     "--view-corpus", paths.view_corpus, "--full-corpus", paths.full_corpus], settings, timeout_s=6 * 3600)
    old_batches, new_batches = _jsonl(legacy / "batches.jsonl"), _jsonl(new / "batches.jsonl")
    assert len(old_batches) == len(new_batches) == SHORT_STEPS
    differing = [o["step"] for o, n in zip(old_batches, new_batches) if o != n]
    assert differing == [], f"first differing step {differing[0]}"
    old_units = [json.loads(json.dumps(r)) for r in _jsonl(legacy / "run" / "mix_.jsonl") if "shard" in r]
    new_units = json.loads((new / "units.json").read_text())
    for record in old_units:
        record.pop("x_head_md5_match", None)          # the legacy --ref-index comparison; not part of the loop
    assert new_units == old_units
