"""Every `statescope` import written anywhere in the repository resolves, and the command line keeps its entry point.

Guard for moving modules inside the package: an import left on an old path fails here, also when it sits inside a function that no
other test calls (the command line imports its modules inside the subcommands) or in a script that is only run by hand (demo/,
examples/, devtools/, tests/new_scripts/). The files are parsed, never executed; only the `statescope` modules they name are imported.
CPU only, no private data.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

from conftest import REPO

PACKAGE = "statescope"
PACKAGE_DIR = REPO / "src" / PACKAGE
SCANNED_DIRS = ("src", "demo", "examples", "devtools", "tests")
NOT_SCANNED = REPO / "tests" / "legacy_scripts"      # these run with the frozen original code on the path, without this package
SUBCOMMANDS = ("verify", "calibrate", "fixedmap", "train-vocab", "train-molt", "foldcheck", "splice")


def _scanned_files() -> list[Path]:
    files = [path for name in SCANNED_DIRS for path in sorted((REPO / name).rglob("*.py"))]
    return [path for path in files if NOT_SCANNED not in path.parents]


def _is_package_module(module: str) -> bool:
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def _first_missing(module: str) -> str | None:
    """The shortest prefix of a dotted module name that cannot be found (`a`, then `a.b`, then `a.b.c`), or None if all exist.
    Prefix by prefix, because importlib's find_spec raises instead of answering when the parent of the name it is asked for is missing."""
    parts = module.split(".")
    for end in range(1, len(parts) + 1):
        prefix = ".".join(parts[:end])
        if importlib.util.find_spec(prefix) is None:
            return prefix
    return None


def _has_submodule(module_name: str, name: str) -> bool:
    module = sys.modules[module_name]
    return hasattr(module, "__path__") and importlib.util.find_spec(f"{module_name}.{name}") is not None


def unresolved_imports(source: str, label: str) -> list[str]:
    """One line per `statescope` import in `source` that does not resolve; imports inside functions and classes are included.

    `import statescope.a.b`          the module statescope.a.b must import.
    `from statescope.a import b, c`  statescope.a must import, and each name must be an attribute of it or a submodule of it.
    """
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            wanted = [(alias.name, []) for alias in node.names if _is_package_module(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_package_module(node.module or ""):
            wanted = [(node.module, [alias.name for alias in node.names])]
        else:
            continue
        for module_name, names in wanted:
            missing = _first_missing(module_name)
            if missing is not None:
                problems.append(f"{label}:{node.lineno}: no module {missing}")
                continue
            module = importlib.import_module(module_name)
            for name in names:
                if not hasattr(module, name) and not _has_submodule(module_name, name):
                    problems.append(f"{label}:{node.lineno}: {module_name} has no name or submodule {name}")
    return problems


def relative_imports(source: str, label: str) -> list[str]:
    """One line per relative import (`from . import x`, `from .a import b`) in `source`."""
    return [f"{label}:{node.lineno}: relative import" for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.level > 0]


def test_the_checker_reports_a_missing_module_and_a_missing_name():
    """without this, the test below could pass because the checker finds nothing"""
    source = ("import json\n"
              "import statescope.no_such_module\n"
              "from statescope.config import MODELS\n"
              "def f():\n"
              "    from statescope.no_such_package.mix import x\n"
              "    from statescope.config import no_such_name\n"
              "    from statescope import reader, no_such_submodule\n")
    assert unresolved_imports(source, "sample") == [
        "sample:2: no module statescope.no_such_module",
        "sample:5: no module statescope.no_such_package",
        "sample:6: statescope.config has no name or submodule no_such_name",
        "sample:7: statescope has no name or submodule no_such_submodule",
    ]
    assert relative_imports("from . import a\nfrom .b import c\nimport d\n", "sample") == ["sample:1: relative import", "sample:2: relative import"]


def test_every_statescope_import_in_the_repository_resolves():
    files = _scanned_files()
    assert len(files) > 50, "the scan found too few files to be looking at the repository"
    problems = []
    for path in files:
        problems += unresolved_imports(path.read_text(), str(path.relative_to(REPO)))
    assert problems == []


def test_the_package_uses_absolute_imports_only():
    """the checker above reads absolute imports; a relative one would escape it (and the rule against importing the original code)"""
    problems = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        problems += relative_imports(path.read_text(), str(path.relative_to(REPO)))
    assert problems == []


def test_every_source_folder_of_the_package_is_a_package():
    """a folder of modules without __init__.py would import as a namespace package and be left out of an installed copy"""
    folders = sorted({path.parent for path in PACKAGE_DIR.rglob("*.py")})
    assert PACKAGE_DIR in folders
    assert [str(folder.relative_to(REPO)) for folder in folders if not (folder / "__init__.py").is_file()] == []


def test_the_console_script_still_points_at_the_command_line():
    """the installed `statescope` command was generated from this entry; moving the function would break it until a reinstall"""
    from statescope.cli import main
    with open(REPO / "pyproject.toml", "rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    assert scripts == {"statescope": "statescope.cli:main"}
    assert callable(main)


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_answers_help(subcommand: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    from statescope.cli import main
    monkeypatch.setattr(sys, "argv", ["statescope", subcommand, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert f"statescope {subcommand}" in capsys.readouterr().out
