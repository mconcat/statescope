"""The frozen 0.8B set of the second design, laid out as published (state/L<L>/, mlp/L<L>/, MANIFEST.json).
The pass/fail rule is R8.7 of tests/EQUIVALENCE_RULES.md (written before these checks were first run)."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from conftest import TEST_OUT, Settings
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


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
    assert {name: _sha256(root / "state" / f"L{layer}" / name) for name in STATE_FILES} == entry["state"]["sha256"]
    assert {name: _sha256(root / "mlp" / f"L{layer}" / name) for name in MLP_FILES} == entry["mlp"]["sha256"]


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
