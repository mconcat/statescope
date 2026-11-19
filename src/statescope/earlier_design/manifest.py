"""The release manifest: which files make up each layer's two dictionaries, their sizes and sha256, and the measured numbers.

    manifest = load_manifest("release/qwen35-08b/manifest.json")
    problems = verify_files(manifest, "/path/to/release")        # [] when every file is present with the listed size and hash

This is the earlier design's manifest (manifest.json, layers/Lxx/{vocab,molt}/), checked by `statescope verify`; the published
set's MANIFEST.json (state/L<L>/, mlp/L<L>/) is checked by demo/verify_release.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

DICT_KINDS = ("vocab", "molt")     # state dictionary, MLP operation dictionary


@dataclass(frozen=True)
class FileEntry:
    path: str       # relative to the release root, e.g. layers/L19/molt/weights.safetensors
    bytes: int
    sha256: str


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def layer_dir(root: str, layer: int, kind: str) -> str:
    """folder of one dictionary of one layer under the release root"""
    if kind not in DICT_KINDS:
        raise ValueError(f"kind must be one of {DICT_KINDS}, got {kind!r}")
    return os.path.join(root, "layers", f"L{layer:02d}", kind)


def file_entries(manifest: dict) -> list[FileEntry]:
    """every file of every listed layer, in manifest order"""
    entries: list[FileEntry] = []
    for layer in manifest["layers"].values():
        for kind in DICT_KINDS:
            entries += [FileEntry(f["path"], int(f["bytes"]), f["sha256"]) for f in layer[kind]["files"]]
    return entries


def sha256_file(path: str, block_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _check(root: str, entry: FileEntry) -> str | None:
    path = os.path.join(root, entry.path)
    if not os.path.isfile(path):
        return f"{entry.path}: missing"
    size = os.path.getsize(path)
    if size != entry.bytes:
        return f"{entry.path}: {size} bytes, manifest says {entry.bytes}"
    digest = sha256_file(path)
    if digest != entry.sha256:
        return f"{entry.path}: sha256 {digest}, manifest says {entry.sha256}"
    return None


def verify_files(manifest: dict, root: str, workers: int = 4) -> list[str]:
    """Compare every listed file under `root` with the manifest. Returns one message per problem (empty list = all good)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:        # hashing releases the GIL; reading dominates on a network share
        results = pool.map(lambda entry: _check(root, entry), file_entries(manifest))
    return [message for message in results if message is not None]
