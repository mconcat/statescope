"""The frozen 0.8B set of the second design, laid out as published (state/L<L>/, mlp/L<L>/, MANIFEST.json).
The pass/fail rule is R8.7 of tests/EQUIVALENCE_RULES.md (written before these checks were first run)."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from conftest import REPO, TEST_OUT, Settings
from hashing import sha256_file
from statescope.config import MODELS
from statescope.reader import PublicMolt, write_post_norm
from test_reader2 import check_molt_reader
from test_repo_rules import FORBIDDEN

pytestmark = [pytest.mark.legacy]

LAYERS = range(24)
STATE_FILES = ("cfg.json", "weights.safetensors", "usage.safetensors")
MLP_FILES = ("config.yaml", "weights.safetensors", "postnorm.safetensors")
# R8.7 (d): nothing of the machines the set was made on. The repository's own list, plus folders of rented machines and any IPv4 address.
PRIVATE = re.compile("|".join(FORBIDDEN + [r"/root/", r"/workspace", r"\bdesktop\b", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"]))


def _header(path: Path) -> dict:
    """the JSON header of a safetensors file (tensor table and metadata), without reading the tensors"""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


@pytest.mark.parametrize("layer", LAYERS)
def test_manifest_hashes(layer: int, settings: Settings) -> None:
    """R8.7 (a)"""
    root = settings.need_release2()
    entry = json.loads((root / "MANIFEST.json").read_text())["layers"][str(layer)]
    assert {name: sha256_file(root / "state" / f"L{layer}" / name) for name in STATE_FILES} == entry["state"]["sha256"]
    assert {name: sha256_file(root / "mlp" / f"L{layer}" / name) for name in MLP_FILES} == entry["mlp"]["sha256"]


@pytest.mark.parametrize("layer", range(23))
def test_mlp_tensors_are_the_tested_exports(layer: int, settings: Settings) -> None:
    """R8.7 (b): layers 0-22 hold the tensors R8 read, bit for bit (CPU)"""
    released = PublicMolt(str(settings.need_release2() / "mlp" / f"L{layer}"))
    export = settings.molt2_dir(layer)
    tested = PublicMolt(str(export / "export"), post_norm=str(export / "postnorm_rebuilt" / "postnorm.safetensors"))
    assert (released.layer, released.eps, released.ranks) == (tested.layer, tested.eps, tested.ranks)
    for name in ("w_eff", "mu_in", "sd_in", "mu_out", "sd_out", "e", "b", "theta"):
        assert torch.equal(getattr(released, name), getattr(tested, name)), name
    for a, b in zip(released.U + released.V, tested.U + tested.V):
        assert torch.equal(a, b)


@pytest.mark.gpu
@pytest.mark.model
def test_layer_23(settings: Settings, device: str) -> None:
    """R8.7 (c): layer 23 comes from another training run than the export R8 read, so it gets its own reference run"""
    from statescope.lm import load_language_model
    folder = settings.need_release2() / "mlp" / "L23"
    rebuilt = TEST_OUT / "release2" / "postnorm_L23.safetensors"
    rebuilt.parent.mkdir(parents=True, exist_ok=True)
    write_post_norm(load_language_model(MODELS["qwen3.5-0.8b"], "cpu").model, 23, str(rebuilt))
    with safe_open(str(rebuilt), framework="pt") as a, safe_open(str(folder / "postnorm.safetensors"), framework="pt") as b:
        assert torch.equal(a.get_tensor("w_eff"), b.get_tensor("w_eff"))                 # R8.4
        assert float(a.metadata()["eps"]) == float(b.metadata()["eps"])
    check_molt_reader(23, settings, device, folder, folder / "postnorm.safetensors", "molt_release")   # R8.3, R8.5


def test_configs_hold_the_documented_training_facts(settings: Settings) -> None:
    """R8.8 (c): what the documents say about "each layer" is in every layer's config"""
    import copy

    import yaml
    root = settings.need_release2()
    reference = None
    for layer in LAYERS:
        cfg = json.loads((root / "state" / f"L{layer}" / "cfg.json").read_text())
        facts = (cfg["tokens"], cfg["steps"], cfg["n_units"], cfg["unit_size"], cfg["k_topk"], cfg["form"], cfg["input"], cfg["rms_eps"], cfg["init_seed"])
        assert facts == (30003200, 7325, 16384, 4, 24, "bsf", "token_rms", 1e-06, 0), layer
        config = yaml.safe_load((root / "mlp" / f"L{layer}" / "config.yaml").read_text())
        assert config["model"]["init_args"]["layer"] == layer and config["trainer"]["logger"]["init_args"]["name"] == f"L{layer}"
        assert config["model"]["init_args"]["lambda_sparsity"] == (2.5e-05 if layer == 23 else 7.5e-05), layer
        rest = copy.deepcopy(config)
        del rest["model"]["init_args"]["layer"], rest["model"]["init_args"]["lambda_sparsity"], rest["trainer"]["logger"]["init_args"]["name"]
        reference = reference or rest
        assert rest == reference, f"layer {layer}: the MLP config differs from layer 0 in more than layer, logger name and penalty"
    assert (reference["trainer"]["max_steps"], reference["data"]["init_args"]["batch_size"], reference["model"]["init_args"]["learning_rate"]) == (58600, 500, 4.0e-05)


def test_reader_fvu_on_all_heldout_tokens(settings: Settings) -> None:
    """R8.8 (a), (b): the numbers of release/qwen35-08b-v2/reader_fvu.json (devtools/measure_release2.py) against the original's table"""
    measured = json.loads((REPO / "release" / "qwen35-08b-v2" / "reader_fvu.json").read_text())["layers"]
    table = json.loads((settings.need_legacy_results() / "mlp_track" / "q08c" / "train1" / "table.json").read_text())
    worst = 0.0
    for layer in LAYERS:
        mine, theirs = measured[str(layer)], table[str(layer)]
        for site, key in ((f"resid_mid.{layer}", "mid"), (f"resid_post.{layer}", "post")):
            for which, column in (("position_0_dropped", f"raw_{key}"), ("with_position_0", f"raw_with_pos0_{key}")):
                diff = abs(mine[site][which] - theirs[column])
                worst = max(worst, diff)
                assert diff <= 1e-4, (layer, site, which, mine[site][which], theirs[column])
        for which in ("position_0_dropped", "with_position_0"):
            assert 0.0 < mine["mlp"][which] < 1.0, (layer, which)
    MEASURED = TEST_OUT / "measured" / "release2"
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "reader_fvu_vs_table.json").write_text(json.dumps({"largest_difference": worst, "limit": 1e-4}) + "\n")


def test_no_private_paths(settings: Settings) -> None:
    """R8.7 (d)"""
    root = settings.need_release2()
    hits = []
    for layer in LAYERS:
        for path in (root / "state" / f"L{layer}" / "cfg.json", root / "mlp" / f"L{layer}" / "config.yaml"):
            hits += [(str(path), line.strip()) for line in path.read_text().splitlines() if PRIVATE.search(line)]
        for path in (root / "state" / f"L{layer}" / "weights.safetensors", root / "state" / f"L{layer}" / "usage.safetensors",
                     root / "mlp" / f"L{layer}" / "weights.safetensors", root / "mlp" / f"L{layer}" / "postnorm.safetensors"):
            metadata = _header(path).get("__metadata__") or {}
            hits += [(str(path), f"{k}: {v}") for k, v in metadata.items() if PRIVATE.search(str(v))]
    assert not hits, hits
    assert not PRIVATE.search((root / "MANIFEST.json").read_text())
