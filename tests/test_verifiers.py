"""The check of a downloaded weight set against its manifest (demo/verify_release.py: MANIFEST.json, state/L<L>/, mlp/L<L>/; sha256),
on tiny folders made here and run as a script, as a user runs it. CPU, no private data.

The expected hashes come from tests/hashing.py, not from the verifier's own sha256 function.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import REPO
from hashing import sha256_file, sha256_tensor

VERIFY_RELEASE = REPO / "demo" / "verify_release.py"
SHA256_OF_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA256_OF_NOTHING = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _flip_one_byte(path: Path) -> None:
    """same size, other content"""
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def test_the_tests_own_sha256_helpers_give_the_known_digests(tmp_path):
    (tmp_path / "abc").write_bytes(b"abc")
    (tmp_path / "empty").write_bytes(b"")
    assert sha256_file(tmp_path / "abc") == SHA256_OF_ABC
    assert sha256_file(tmp_path / "empty") == SHA256_OF_NOTHING
    abc = torch.tensor([ord("a"), ord("b"), ord("c")], dtype=torch.uint8)
    assert sha256_tensor(abc) == SHA256_OF_ABC
    column = torch.stack([abc, torch.zeros_like(abc)], dim=1)[:, 0]        # the same values as a view that is not contiguous
    assert not column.is_contiguous() and sha256_tensor(column) == SHA256_OF_ABC


PUBLISHED_FILES = {"state": ("cfg.json", "weights.safetensors", "usage.safetensors"), "mlp": ("config.yaml", "weights.safetensors", "postnorm.safetensors")}


def _write_published_set(root: Path) -> list[Path]:
    """a folder in the published layout with one layer of small files and a MANIFEST.json that lists their sha256; returns the files"""
    files: list[Path] = []
    entry: dict = {}
    for kind, names in PUBLISHED_FILES.items():
        folder = root / kind / "L0"
        folder.mkdir(parents=True)
        for i, name in enumerate(names):
            (folder / name).write_bytes(f"{kind} {name} ".encode() * (50 + i))
            files.append(folder / name)
        entry[kind] = {"sha256": {name: sha256_file(folder / name) for name in names}}
    (root / "MANIFEST.json").write_text(json.dumps({"layers": {"0": entry}}))
    return files


def _run_verify_release(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(VERIFY_RELEASE), "--release", str(root)], capture_output=True, text=True, timeout=120)


def test_verify_release_accepts_a_set_that_matches_its_manifest(tmp_path):
    files = _write_published_set(tmp_path / "set")
    done = _run_verify_release(tmp_path / "set")
    assert done.returncode == 0, done.stderr
    assert f"all {len(files)} files match the manifest" in done.stdout
    assert "layer 0: checked" in done.stdout


@pytest.mark.parametrize("index", range(6))
def test_verify_release_names_a_file_with_one_flipped_byte(tmp_path, index: int):
    files = _write_published_set(tmp_path / "set")
    _flip_one_byte(files[index])
    done = _run_verify_release(tmp_path / "set")
    assert done.returncode != 0
    assert "1 of 6 files do not match the manifest" in done.stderr
    assert str(files[index]) in done.stderr
    assert "all 6 files" not in done.stdout


def test_verify_release_fails_when_a_file_is_missing(tmp_path):
    files = _write_published_set(tmp_path / "set")
    files[4].unlink()
    done = _run_verify_release(tmp_path / "set")
    assert done.returncode != 0                                    # only the exit code is pinned, not how the failure is worded
    assert "all 6 files" not in done.stdout
