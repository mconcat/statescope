"""Checks of a set laid out as published (state/L<L>/, mlp/L<L>/, MANIFEST.json), shared by the 0.8B and the 4B tests.
Rules R8.7 (a), (d), R10.6 (a), (d) and R11.1 - R11.3 of tests/EQUIVALENCE_RULES.md."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from hashing import sha256_file
from test_repo_rules import FORBIDDEN

STATE_FILES = ("cfg.json", "weights.safetensors", "usage.safetensors")
MLP_FILES = ("config.yaml", "weights.safetensors", "postnorm.safetensors")
# R8.7 (d): nothing of the machines the set was made on. The repository's own list, plus folders of rented machines and any IPv4 address.
PRIVATE = re.compile("|".join(FORBIDDEN + [r"/root/", r"/workspace", r"\bdesktop\b", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"]))
# R11.3: names of the original, unpublished project
INTERNAL_NAMES = re.compile("sasa|segment_b4|B1s0|layer_recipe|molt_stream|cache_acts|shard_data|public_molt|code_md5", re.IGNORECASE)
# R11: the edit of the two configs
ORIGINAL_KIND, PUBLISHED_KIND = "sasa_proper_segment", "state_dictionary"
REMOVED_CFG_KEYS = ("name", "trainer", "arm", "code_md5")
REMOVED_CONFIG_LINE = b"  class_path: measure.public_molt.shard_data.ShardDataModule\n"


def safetensors_header(path: Path) -> dict:
    """the JSON header of a safetensors file (tensor table and metadata), without reading the tensors"""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def check_manifest_hashes(root: Path, layer: int) -> None:
    """every published file of the layer has the manifest's sha256"""
    entry = json.loads((root / "MANIFEST.json").read_text())["layers"][str(layer)]
    assert {name: sha256_file(root / "state" / f"L{layer}" / name) for name in STATE_FILES} == entry["state"]["sha256"]
    assert {name: sha256_file(root / "mlp" / f"L{layer}" / name) for name in MLP_FILES} == entry["mlp"]["sha256"]


def check_config_edits(root: Path, layer: int, original_state: Path, mlp_copy: Path) -> None:
    """R11.1 and R11.2: the two published configs are the originals with the stated edit and nothing else; the state dictionary's
    tensor files are the original export's files.
    original_state: the original export folder of the state dictionary; mlp_copy: the path-free copy of the MLP dictionary"""
    entry = json.loads((root / "MANIFEST.json").read_text())["layers"][str(layer)]
    state, mlp = root / "state" / f"L{layer}", root / "mlp" / f"L{layer}"
    expected = json.loads((original_state / "cfg.json").read_text())
    assert expected["kind"] == ORIGINAL_KIND
    expected["kind"] = PUBLISHED_KIND
    for key in REMOVED_CFG_KEYS:
        del expected[key]
    assert json.loads((state / "cfg.json").read_text()) == expected
    original_config = (mlp_copy / "config.yaml").read_bytes()
    assert original_config.count(REMOVED_CONFIG_LINE) == 1
    assert (mlp / "config.yaml").read_bytes() == original_config.replace(REMOVED_CONFIG_LINE, b"")
    assert entry["state"]["original_sha256"] == {"cfg.json": sha256_file(original_state / "cfg.json")}
    assert entry["mlp"]["original_sha256"] == {"config.yaml": sha256_file(mlp_copy / "config.yaml")}
    for name in STATE_FILES[1:]:
        assert (state / name).samefile(original_state / name), name


def _scan(root: Path, layers: range, text_files: tuple[str, ...], pattern: re.Pattern, tensor_names: bool) -> list[tuple[str, str]]:
    """lines of the text files and entries of the safetensors headers that match `pattern`"""
    paths = [root / name for name in text_files]
    headers = []
    for layer in layers:
        paths += [root / "state" / f"L{layer}" / "cfg.json", root / "mlp" / f"L{layer}" / "config.yaml"]
        headers += [root / "state" / f"L{layer}" / name for name in STATE_FILES[1:]] + [root / "mlp" / f"L{layer}" / name for name in MLP_FILES[1:]]
    hits = [(str(path), line.strip()) for path in paths for line in path.read_text().splitlines() if pattern.search(line)]
    for path in headers:
        header = safetensors_header(path)
        entries = [f"{k}: {v}" for k, v in (header.get("__metadata__") or {}).items()]
        if tensor_names:
            entries += [name for name in header if name != "__metadata__"]
        hits += [(str(path), entry) for entry in entries if pattern.search(entry)]
    return hits


def check_no_private_paths(root: Path, layers: range, text_files: tuple[str, ...]) -> None:
    """R8.7 (d): no machine path, lab host name or IPv4 address in a config, a header's metadata or one of the named text files"""
    assert _scan(root, layers, text_files, PRIVATE, tensor_names=False) == []


def check_no_internal_names(root: Path, layers: range, text_files: tuple[str, ...]) -> None:
    """R11.3: none of the original project's internal names in a config, a header (metadata and tensor names) or a named text file"""
    assert _scan(root, layers, text_files, INTERNAL_NAMES, tensor_names=True) == []
