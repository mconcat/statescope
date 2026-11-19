"""The two readers on the Qwen3.5-4B set (32 layers, width 2,560; same forms as the 0.8B set).
The pass/fail rules are R10.1 - R10.5 of tests/EQUIVALENCE_RULES.md (written before these tests were first run): R8's checks and limits
on other folders, read with R11.4 (the reference reads the original files, this repository's reader the published copy). The checks
themselves are the ones of test_reader2.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from conftest import TEST_OUT, Settings
from statescope.config import MODELS
from statescope.reader import write_post_norm
from test_reader2 import LayerFiles, check_molt_reader, check_state_reader

pytestmark = [pytest.mark.legacy, pytest.mark.gpu, pytest.mark.model]

MODEL = MODELS["qwen3.5-4b"]
LAYERS = range(MODEL.n_layers)
MEASURED = TEST_OUT / "measured" / "reader4b"
# tokens per public forward in the reference run: it builds (chunk, transforms, width) tensors, here 16 x 9,920 x 2,560 fp32 = 1.6 GB
REFERENCE_CHUNK = 16


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    torch.cuda.empty_cache()


def files_4b(settings: Settings, layer: int) -> LayerFiles:
    """the reference reads the 4B layer's training exports; the MLP export holds its own post-norm file (R10.4 compares it with the
    language model)"""
    molt = settings.molt4b_dir(layer)
    return LayerFiles(settings.calib4b_dir(layer), settings.vocab4b_dir(layer), molt, molt / "postnorm.safetensors", settings.need_release4b(),
                      MEASURED, REFERENCE_CHUNK)


def test_post_norm_files_come_from_the_language_model(settings: Settings) -> None:
    """R10.4: every export's postnorm.safetensors holds the layer's RMSNorm coefficients of the language model, bit for bit"""
    from statescope.lm import load_language_model
    out = TEST_OUT / "design4b" / "postnorm"
    out.mkdir(parents=True, exist_ok=True)
    lm = load_language_model(MODEL, "cpu")
    for layer in LAYERS:
        mine = out / f"L{layer}.safetensors"
        write_post_norm(lm.model, layer, str(mine))
        theirs = settings.molt4b_dir(layer) / "postnorm.safetensors"
        with safe_open(str(mine), framework="pt") as a, safe_open(str(theirs), framework="pt") as b:
            assert torch.equal(a.get_tensor("w_eff"), b.get_tensor("w_eff")), f"layer {layer}: w_eff differs from {theirs}"
            assert float(a.metadata()["eps"]) == float(b.metadata()["eps"]), f"layer {layer}: eps differs from {theirs}"
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "post_norm.json").write_text(json.dumps({"layers_compared_with_the_language_model": list(LAYERS)}) + "\n")


@pytest.mark.parametrize("layer", LAYERS)
def test_state_reader(layer: int, settings: Settings, device: str) -> None:
    """R10.1, R10.2, R10.5"""
    check_state_reader(layer, settings, device, files_4b(settings, layer))


@pytest.mark.parametrize("layer", LAYERS)
def test_molt_reader(layer: int, settings: Settings, device: str) -> None:
    """R10.3, R10.5"""
    check_molt_reader(layer, settings, device, files_4b(settings, layer), "molt")
