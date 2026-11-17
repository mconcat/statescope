"""Spliced-in loss: new code vs the legacy code and vs the recorded numbers (rules R5.1 - R5.3 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch

from conftest import TEST_OUT, Settings
from legacy_bridge import run_legacy
from statescope.reader import MoltOps, StateVocab
from statescope.splice import ARMS, load_documents, splice_layer

LAYERS = [9, 19]
N_DOCS = 100
MODEL_ID = "Qwen/Qwen3.5-0.8B"
FIRST_ROW = 131500          # fineweb-edu rows before this one are not used; the dictionaries' training mixture has no fineweb at all


def _quantities(result: dict) -> dict[str, float]:
    flat = {"ce_clean": result["ce_clean"]}
    for arm in ARMS:
        for key in ("dce", "kl", "fvu_batchwise"):
            flat[f"{arm}.{key}"] = result[arm][key]
    return flat


def _legacy(settings: Settings, rep: str) -> dict:
    out = run_legacy("splice_reference.py", ["--root", str(settings.need_release_root()), "--layers", ",".join(map(str, LAYERS)),
                                             "--docs", str(N_DOCS), "--rep", rep], settings)
    return json.loads((out / "splice.json").read_text())


@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
def test_spliced_loss_matches_legacy_and_the_recorded_numbers(settings: Settings, device: str):
    corpus = os.environ.get("STATESCOPE_SPLICE_CORPUS", "")
    if not os.path.isfile(corpus):
        pytest.fail("STATESCOPE_SPLICE_CORPUS (fineweb-edu JSON lines) is not set or missing (see tests/local.env.example)", pytrace=False)
    legacy_a, legacy_b = _legacy(settings, "A"), _legacy(settings, "B")
    recorded = json.loads((settings.need_legacy_snapshot() / "results" / "mlp_track" / "splice_loss" / "splice.json").read_text())

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    ids = load_documents(tokenizer, corpus, N_DOCS, n_tokens=256, first_row=FIRST_ROW)
    assert hashlib.sha256(ids.numpy().tobytes()).hexdigest() == legacy_a["ids_sha256"]        # same documents, token for token

    report = {}
    failures = []
    for layer in LAYERS:
        vocab = StateVocab(str(settings.layer_dir(layer, "vocab")), device)
        molt = MoltOps(str(settings.layer_dir(layer, "molt")), device)
        new = splice_layer(model, layer, ids, vocab, molt, batch=2)
        del vocab, molt
        torch.cuda.empty_cache()
        a, b, rec = legacy_a[str(layer)], legacy_b[str(layer)], recorded[str(layer)]
        assert new["n_positions"] == a["n_positions"] == rec["n_positions"]
        assert max(abs(v) for v in new["identity_splice_kl_first_batch"].values()) <= 1e-4
        qn, qa, qb, qr = _quantities(new), _quantities(a), _quantities(b), _quantities(rec)
        rows = {}
        for q in qn:
            noise = abs(qa[q] - qb[q])
            rows[q] = {"new": qn[q], "legacy_a": qa[q], "legacy_b": qb[q], "recorded": qr[q], "noise_floor": noise}
            if abs(qn[q] - qa[q]) > max(3 * noise, 1e-6):                                      # R5.2
                failures.append(f"L{layer} {q}: new {qn[q]!r} vs legacy {qa[q]!r} (noise floor {noise:.3g})")
            if not q.endswith("fvu_batchwise") and abs(qn[q] - qr[q]) > 0.002:                 # R5.3
                failures.append(f"L{layer} {q}: new {qn[q]!r} vs recorded {qr[q]!r}")
        report[str(layer)] = rows
    os.makedirs(TEST_OUT / "measured", exist_ok=True)
    (TEST_OUT / "measured" / "splice.json").write_text(json.dumps(report, indent=1))
    assert failures == []
