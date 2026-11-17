"""The example reproduces the table of the original release notes (rule R1.7 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import importlib.util

import pytest
import torch

from conftest import REPO, Settings
from statescope.reader import MoltOps, StateVocab

MODEL_ID = "Qwen/Qwen3.5-0.8B"
# notes/mlp_track/release.md, "example.py only" table: (layer, text) -> tokens, FVU resid_mid, FVU resid_post, FVU MLP, atoms on
RELEASE_NOTES_TABLE = {
    (19, "english"): (67, .211, .207, .138, 56.9),
    (19, "code"): (89, .302, .294, .153, 45.3),
    (19, "korean"): (64, .524, .520, .385, 108.1),
    (9, "english"): (67, .263, .250, .292, 57.2),
    (9, "code"): (89, .318, .306, .281, 61.6),
    (9, "korean"): (64, .454, .437, .366, 74.6),
}


@pytest.mark.release
@pytest.mark.gpu
@pytest.mark.model
def test_example_reproduces_the_release_notes_table(settings: Settings, device: str):
    spec = importlib.util.spec_from_file_location("reconstruct_texts", REPO / "examples" / "reconstruct_texts.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    problems = []
    for layer in (19, 9):
        vocab = StateVocab(str(settings.layer_dir(layer, "vocab")), device)
        molt = MoltOps(str(settings.layer_dir(layer, "molt")), device)
        for name, text in example.TEXTS.items():
            row = example.reconstruct(model, tokenizer, vocab, molt, layer, text)
            tokens, mid, post, mlp, atoms = RELEASE_NOTES_TABLE[(layer, name)]
            got = (row["tokens"], row["fvu_resid_mid"], row["fvu_resid_post"], row["fvu_mlp"], row["atoms_on"])
            ok = got[0] == tokens and all(abs(g - w) <= 0.002 for g, w in zip(got[1:4], (mid, post, mlp))) and abs(got[4] - atoms) <= 0.2
            if not ok:
                problems.append((layer, name, got, RELEASE_NOTES_TABLE[(layer, name)]))
        del vocab, molt
        torch.cuda.empty_cache()
    assert problems == []
