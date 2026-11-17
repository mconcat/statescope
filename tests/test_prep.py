"""Preparation: the new calibration and fixed map against the files the released dictionaries were trained from (rules R3.1, R3.2)."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from conftest import TEST_OUT, Settings
from statescope.config import MIX_08B, MODELS
from statescope.data.mix import training_stream
from statescope.heldout import HELDOUT_SPLITS
from statescope.lm import load_language_model
from statescope.prep.calibrate import calibrate
from statescope.prep.fixedmap import fixed_map
from test_stream import data_paths_08b

LAYER = 19


def _same_tensors(new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> list[str]:
    problems = sorted(set(new) ^ set(old))
    for key in set(new) & set(old):
        if new[key].dtype != old[key].dtype or new[key].shape != old[key].shape:
            problems.append(f"{key}: {new[key].dtype} {tuple(new[key].shape)} vs {old[key].dtype} {tuple(old[key].shape)}")
        elif not torch.equal(new[key], old[key]):
            problems.append(f"{key}: values differ (max abs {float((new[key].float() - old[key].float()).abs().max()):.3g})")
    return problems


@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.model
def test_calibration_and_fixed_map_equal_the_files_on_disk(settings: Settings, device: str):
    reference_calib = settings.calib_dir_08b(LAYER)
    reference_map = settings.need_legacy_results() / "mlp_track" / "fixedmap" / f"L{LAYER}.safetensors"
    recorded = json.loads((settings.need_legacy_results() / "mlp_track" / "train" / f"L{LAYER}" / "M1w" / "layer_recipe_config.json").read_text())
    paths = data_paths_08b(settings)
    lm = load_language_model(MODELS["qwen3.5-0.8b"], device)
    out = TEST_OUT / "prep" / f"calib_L{LAYER}"
    if out.exists():
        shutil.rmtree(out)
    metadata = calibrate(lm, LAYER, paths, MIX_08B, str(out))

    problems = []
    # R3.1 statistics
    with safe_open(str(reference_calib / "calib_stats.safetensors"), framework="pt") as f:
        old_meta = f.metadata()
    problems += [f"calib_stats {p}" for p in _same_tensors(load_file(str(out / "calib_stats.safetensors")), load_file(str(reference_calib / "calib_stats.safetensors")))]
    if metadata != old_meta:
        problems.append(f"calib_stats metadata: {metadata} vs {old_meta}")
    # R3.1 held-out shards
    for split in HELDOUT_SPLITS:
        name = f"heldout_{split}"
        problems += [f"{name} {p}" for p in _same_tensors(load_file(str(out / name / "shard_00000.safetensors")), load_file(str(reference_calib / name / "shard_00000.safetensors")))]
        if json.loads((out / name / "index.json").read_text()) != json.loads((reference_calib / name / "index.json").read_text()):
            problems.append(f"{name} index.json differs")
    # R3.2 fixed map (computed from the new calibration folder)
    new_map = TEST_OUT / "prep" / f"fixedmap_L{LAYER}.safetensors"
    fixed_map(lm, LAYER, training_stream(lm.tokenizer, paths, MIX_08B), str(out), str(new_map))
    problems += [f"fixedmap {p}" for p in _same_tensors(load_file(str(new_map)), load_file(str(reference_map)))]
    md5 = hashlib.md5(new_map.read_bytes()).hexdigest()
    print("fixed map md5", md5, "recorded", recorded["molt_info"]["M1w"]["fixmap_md5"], "equal" if md5 == recorded["molt_info"]["M1w"]["fixmap_md5"] else "DIFFERENT CONTAINER BYTES")
    assert problems == []
