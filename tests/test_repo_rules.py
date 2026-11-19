"""Rules of this repository that a reviewer should not have to check by eye."""
from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess

import torch

import statescope.reader
from conftest import REPO
from synthetic import write_public_molt, write_vocab

SOURCE_DIRS = ("src", "demo")
LEGACY_PACKAGES = {"train", "measure", "gpt2", "tools"}
# `except` clauses that are allowed, each for a stated reason. Everything else in the package must let errors propagate.
ALLOWED_EXCEPT: set[tuple[str, str]] = set()
# machine-specific strings that must not be committed
# (file, matched text) pairs that are allowed, each for a stated reason
ALLOWED_STRINGS = {
    ("README.md", "mconcat"),                                     # the public address of the weights repository
    ("release/qwen35-08b-v2/MODEL_CARD.md", "mconcat"),           # the public addresses of the code and weights repositories
    ("release/qwen35-4b/MODEL_CARD.md", "mconcat"),               # the same, in the 4B set's card
}
FORBIDDEN = [r"/home/", r"/mnt/", r"\b10\.42\.\d", r"\b192\.168\.\d", r"\bprocessor\b", r"\bworkstation\b", r"\bmconcat\b"]
READER = REPO / "src" / "statescope" / "reader.py"
# everything the single-file reader may import (first component of the module name); nothing of this package is in the list
READER_IMPORTS = {"__future__", "json", "os", "torch", "safetensors", "yaml"}


def _python_files() -> list:
    return [p for d in SOURCE_DIRS for p in sorted((REPO / d).rglob("*.py"))]


def test_package_never_imports_the_legacy_code():
    offenders = []
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            offenders += [f"{path.relative_to(REPO)}: {n}" for n in names if n.split(".")[0] in LEGACY_PACKAGES]
    assert offenders == []


def test_no_exception_is_swallowed_or_turned_into_a_fallback():
    offenders = []
    for path in _python_files():
        rel = str(path.relative_to(REPO))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ExceptHandler):
                caught = ast.unparse(node.type) if node.type is not None else "<bare>"
                if (rel, caught) not in ALLOWED_EXCEPT:
                    offenders.append(f"{rel}:{node.lineno}: except {caught}")
    assert offenders == []


def test_no_machine_specific_string_is_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    offenders = []
    for rel in tracked:
        if rel in ("tests/test_repo_rules.py", "devtools/legacy-env-freeze.txt", "LICENSE"):   # the licence names the copyright holder
            continue
        text = (REPO / rel).read_text(errors="replace")
        for pattern in FORBIDDEN:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                if (rel, match.group(0)) not in ALLOWED_STRINGS:
                    offenders.append(f"{rel}:{line}: {match.group(0)}")
    assert offenders == []


def test_the_reader_imports_only_the_standard_library_torch_safetensors_and_yaml():
    """The reader is one file that can be copied next to the weights: it imports nothing of this package and no further dependency.
    Imports inside functions count too."""
    offenders = []
    n_imports = 0
    for node in ast.walk(ast.parse(READER.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = ["<relative import>"] if node.level else [node.module]
        else:
            continue
        n_imports += len(names)
        offenders += [f"line {node.lineno}: {name}" for name in names if name.split(".")[0] not in READER_IMPORTS]
    assert n_imports > 0, "no import found in the reader: this test would pass without checking anything"
    assert offenders == []


def test_a_copy_of_the_reader_file_works_alone_and_gives_identical_outputs(tmp_path):
    """reader.py copied to another folder and loaded by file path under another module name (as a user who copies it next to the
    weights would, and as tests/legacy_scripts/capture_reference.py loads it) reads the same dictionaries to the same bits."""
    copy_path = tmp_path / "elsewhere" / "reader.py"
    copy_path.parent.mkdir()
    shutil.copyfile(READER, copy_path)
    spec = importlib.util.spec_from_file_location("reader_copy_under_test", copy_path)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy.__file__ == str(copy_path) and copy is not statescope.reader

    write_public_molt(tmp_path / "mlp")
    write_vocab(tmp_path / "state", layer=3)
    x = torch.randn(48, 12, generator=torch.Generator().manual_seed(0))
    gate, y = statescope.reader.PublicMolt(str(tmp_path / "mlp")).forward(x)
    gate_copy, y_copy = copy.PublicMolt(str(tmp_path / "mlp")).forward(x)
    assert int((gate > 0).sum()) > 0
    assert torch.equal(gate, gate_copy) and torch.equal(y, y_copy)
    mine, theirs = statescope.reader.StateVocab(str(tmp_path / "state")), copy.StateVocab(str(tmp_path / "state"))
    for site in mine.sites:
        groups, codes = mine.encode(x, site)
        groups_copy, codes_copy = theirs.encode(x, site)
        assert torch.equal(groups, groups_copy) and torch.equal(codes, codes_copy), site
        assert torch.equal(mine.reconstruct(x, site), theirs.reconstruct(x, site)), site
    assert copy.fvu(x, 0.5 * x) == statescope.reader.fvu(x, 0.5 * x)
