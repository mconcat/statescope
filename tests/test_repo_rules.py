"""Rules of this repository that a reviewer should not have to check by eye."""
from __future__ import annotations

import ast
import re
import subprocess

from conftest import REPO

SOURCE_DIRS = ("src", "examples", "demo")
LEGACY_PACKAGES = {"train", "measure", "gpt2", "tools"}
# `except` clauses that are allowed, each for a stated reason. Everything else in the package must let errors propagate.
ALLOWED_EXCEPT = {
    ("src/statescope/lm.py", "_Reached"),     # control flow: the hook ends the forward pass once both tensors are captured
}
# machine-specific strings that must not be committed
# (file, matched text) pairs that are allowed, each for a stated reason
ALLOWED_STRINGS = {
    ("src/statescope/earlier_design/release.py", "/home/"),      # the manifest builder's own check that no published JSON holds a home path
}
FORBIDDEN = [r"/home/", r"/mnt/", r"\b10\.42\.\d", r"\b192\.168\.\d", r"\bprocessor\b", r"\bworkstation\b", r"\bmconcat\b"]


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


def test_the_two_dictionary_packages_do_not_import_each_other():
    """The two trainers of the earlier design stay independent of each other.

    This rule was first needed by the cache keys of tests/new_bridge.py, which left the other dictionary's source out of a training
    run's key. Since both packages moved under earlier_design/ the `exclude` arguments match nothing and every key covers all of
    src/statescope, so the keys no longer depend on this. The independence itself is kept as a rule of the layout."""
    offenders = []
    for package, other in (("molt", "vocab"), ("vocab", "molt")):
        files = sorted((REPO / "src" / "statescope" / "earlier_design" / package).rglob("*.py"))
        assert files != [], f"no source files found for the package {package}: this test would pass without checking anything"
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                offenders += [f"{path.relative_to(REPO)}: {n}" for n in names if n.startswith(f"statescope.earlier_design.{other}")]
    assert offenders == []
