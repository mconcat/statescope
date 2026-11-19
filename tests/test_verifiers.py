"""The two checks of a downloaded weight set against its manifest, on tiny folders made here. CPU, no private data.

  published set    demo/verify_release.py   MANIFEST.json, state/L<L>/, mlp/L<L>/; sha256 only; run as a script, as a user runs it
  earlier design   `statescope verify`      manifest.json, layers/Lxx/{vocab,molt}/; size and sha256

The expected hashes come from tests/hashing.py, not from either verifier's own sha256 function.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import statescope.cli
from conftest import REPO
from hashing import sha256_file, sha256_tensor
from statescope.earlier_design.manifest import load_manifest, verify_files

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


# ---------------------------------------------------------------------------------------------------------------------------------
# published set: demo/verify_release.py
# ---------------------------------------------------------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------------------------------------------------------
# earlier design: statescope.earlier_design.manifest.verify_files and `statescope verify`
# ---------------------------------------------------------------------------------------------------------------------------------

EARLIER_FILES = {"vocab": ("cfg.json", "weights.safetensors"), "molt": ("cfg.json", "input.json", "weights.safetensors")}


def _write_earlier_set(root: Path) -> tuple[Path, list[str]]:
    """a folder in the earlier design's layout with one layer of small files; returns (manifest path, relative file paths)"""
    paths: list[str] = []
    layer: dict = {}
    for kind, names in EARLIER_FILES.items():
        folder = root / "layers" / "L00" / kind
        folder.mkdir(parents=True)
        entries = []
        for i, name in enumerate(names):
            (folder / name).write_bytes(f"{kind} {name} ".encode() * (40 + i))
            rel = f"layers/L00/{kind}/{name}"
            entries.append({"path": rel, "bytes": (folder / name).stat().st_size, "sha256": sha256_file(folder / name)})
            paths.append(rel)
        layer[kind] = {"files": entries}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"layers": {"0": layer}}))
    return manifest_path, paths


def test_verify_files_returns_nothing_for_a_set_that_matches_its_manifest(tmp_path):
    manifest_path, paths = _write_earlier_set(tmp_path / "set")
    assert len(paths) == 5
    assert verify_files(load_manifest(str(manifest_path)), str(tmp_path / "set")) == []


def test_verify_files_gives_one_message_per_problem(tmp_path):
    root = tmp_path / "set"
    manifest_path, paths = _write_earlier_set(root)
    flipped, resized, missing = paths[0], paths[2], paths[4]
    _flip_one_byte(root / flipped)
    listed_bytes = (root / resized).stat().st_size
    (root / resized).write_bytes((root / resized).read_bytes() + b"x")
    (root / missing).unlink()
    problems = verify_files(load_manifest(str(manifest_path)), str(root))
    assert len(problems) == 3
    by_file = {message.split(":")[0]: message for message in problems}
    assert sorted(by_file) == sorted([flipped, resized, missing])
    assert "sha256" in by_file[flipped] and sha256_file(root / flipped) in by_file[flipped]
    assert by_file[resized] == f"{resized}: {listed_bytes + 1} bytes, manifest says {listed_bytes}"
    assert by_file[missing] == f"{missing}: missing"


def _statescope_verify(monkeypatch: pytest.MonkeyPatch, manifest_path: Path, root: Path) -> None:
    """`statescope verify --manifest ... --root ...` in this process (main reads sys.argv)"""
    monkeypatch.setattr(sys, "argv", ["statescope", "verify", "--manifest", str(manifest_path), "--root", str(root)])
    statescope.cli.main()


def test_statescope_verify_passes_on_a_set_that_matches_its_manifest(tmp_path, monkeypatch, capsys):
    manifest_path, _ = _write_earlier_set(tmp_path / "set")
    _statescope_verify(monkeypatch, manifest_path, tmp_path / "set")
    assert "every listed file has the listed size and sha256" in capsys.readouterr().out


def test_statescope_verify_exits_with_the_problems_listed(tmp_path, monkeypatch, capsys):
    root = tmp_path / "set"
    manifest_path, paths = _write_earlier_set(root)
    _flip_one_byte(root / paths[1])
    (root / paths[3]).unlink()
    with pytest.raises(SystemExit) as raised:
        _statescope_verify(monkeypatch, manifest_path, root)
    assert raised.value.code not in (None, 0)
    assert "2 file(s) do not match the manifest" in str(raised.value.code)
    out = capsys.readouterr().out
    assert f"{paths[1]}: sha256" in out and f"{paths[3]}: missing" in out
    assert "every listed file" not in out
