"""Freeze a copy of the original research code that the equivalence tests compare against.

The original repository keeps changing (and is imported by running jobs), so the tests never read it. This script copies the static
import closure of the release entry points, plus a fixed list of side files, into a snapshot folder and writes a hash list.

    python devtools/freeze_legacy.py --source <original repo> --dest legacy_snapshot

It only READS the source: files are parsed with `ast`, never imported or executed, and nothing is written outside --dest.
The snapshot is not committed (see .gitignore); `SHA256SUMS.json` inside it is the record of what was frozen.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

# Python entry points of the 0.8B path (what produced and checked the released checkpoints).
ENTRIES_08B: list[str] = [
    "measure.molt_4b_infra.calib_cache",   # calibration statistics + held-out activations
    "measure.layer_recipe.fixedmap",       # fixed mean-gate map (linear part of the MLP)
    "measure.mlp_track.train_fixwhite",    # MLP dictionary trainer (M1w)
    "train.layer_recipe",                  # state dictionary trainer (V0)
    "measure.mlp_track.collect_layers",    # per-layer table
    "measure.mlp_track.fold_check",        # fold-error verdict
    "measure.mlp_track.release_manifest",  # manifest.json
    "measure.mlp_track.release_parity",    # standalone reader vs repository readers
    "measure.mlp_track.splice_loss",       # spliced-in next-token loss
    "measure.mlp_track.mix_scripts",       # script census of the training mixture
    "tools.capmem",                        # CUDA memory cap wrapper
    "release/qwen35-08b-layer-dicts/reader.py",
    "release/qwen35-08b-layer-dicts/example.py",
]
# Wrappers that run the same trainers on the 4B model (stage 2 folds them into configuration).
ENTRIES_4B: list[str] = [
    "measure.mlp_track.q4b_stream",
    "measure.mlp_track.fixedmap_model",
    "measure.mlp_track.q4b_corpus_merge",
    "measure.mlp_track.q4b_gen",
]
# Non-Python files: launchers, specs, the release manifest, the fact notes and the measured results that the tests assert against.
SIDE_FILES: list[str] = [
    "release/qwen35-08b-layer-dicts/manifest.json",
    "logs/mlp_track/run.sh",
    "logs/mlp_track/prep_layer.sh",
    "logs/mlp_track/prep_many.sh",
    "logs/mlp_track/lane_m1w_layers.sh",
    "logs/mlp_track/lane_vocab.sh",
    "logs/mlp_track/lane_m1w.sh",
    "logs/mlp_track/lane_after.sh",
    "logs/mlp_track/checks_m1w.sh",
    "logs/mlp_track/q4b_job.sh",
    "logs/mlp_track/q4b2_job.sh",
    "logs/mlp_track/q4b2_chain.sh",
    "logs/mlp_track/q4b2_desk_chain.sh",
    "logs/whole_structure_v1/train_run.sh",
    "logs/whole_structure_v1/prep_layer.sh",
    "results/mlp_track/specs/v0.json",
    "results/layer_recipe/specs/p1a.json",
    "results/whole_structure_v1/specs/v0.json",
    "results/mlp_track/release_parity/parity.json",
    "results/mlp_track/release_parity/mix_scripts.json",
    "results/mlp_track/fold_check/fold_check.json",
    "results/mlp_track/splice_loss/splice.json",
    "results/mlp_track/all_layers_table.json",
    "notes/mlp_track/release.md",
    "notes/mlp_track/RELEASE_CODE_MAP.md",
    "notes/mlp_track/fold_check.md",
    "notes/mlp_track/all_layers_registration.md",
    "notes/mlp_track/splice_loss_registration.md",
]


def module_file(root: Path, module: str) -> Path | None:
    """`a.b.c` -> root/a/b/c.py or root/a/b/c/__init__.py, if one exists"""
    base = root / module.replace(".", "/")
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_names(tree: ast.AST, package: str) -> list[str]:
    """every dotted name an import statement of this module could resolve to (absolute form)"""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package.split(".") if package else []
                up = up[: len(up) - (node.level - 1)]
                base = ".".join(up + ([base] if base else []))
            names.append(base)
            names += [f"{base}.{alias.name}" for alias in node.names]
    return names


def import_closure(root: Path, entries: list[str]) -> tuple[dict[Path, list[str]], dict[str, list[str]]]:
    """Transitive closure of in-repo imports. Returns ({file: sorted in-repo modules it imports}, {module: lines hinting at dynamic imports})."""
    stack: list[Path] = []
    for entry in entries:
        path = root / entry if entry.endswith(".py") else module_file(root, entry)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"entry point {entry!r} not found under {root}")
        stack.append(path)
    edges: dict[Path, list[str]] = {}
    dynamic: dict[str, list[str]] = {}
    while stack:
        path = stack.pop()
        if path in edges:
            continue
        source = path.read_text()
        me = module_name(root, path)
        package = me if path.name == "__init__.py" else me.rpartition(".")[0]
        deps: set[Path] = set()
        for name in imported_names(ast.parse(source), package):
            parts = name.split(".")
            found = False
            for i in range(1, len(parts) + 1):          # `import a.b.c` also loads a/__init__ and a/b/__init__
                dep = module_file(root, ".".join(parts[:i]))
                if dep is not None:
                    deps.add(dep)
                    found = True
            if not found:                               # a script importing its sibling (script dir on sys.path)
                sibling = path.parent / f"{parts[0]}.py"
                if sibling.is_file():
                    deps.add(sibling)
        hints = [ln.strip() for ln in source.splitlines()
                 if any(key in ln for key in ("importlib", "__import__", "runpy", "exec(", "sys.path", "subprocess"))]
        if hints:
            dynamic[me] = hints
        edges[path] = sorted(module_name(root, d) for d in deps)
        stack.extend(deps)
    return edges, dynamic


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),     # the trainers recorded md5 of their own source in each run's config
        "bytes": len(data),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="the original repository (read only)")
    ap.add_argument("--dest", required=True, type=Path, help="snapshot folder to create (must not exist)")
    args = ap.parse_args()
    source: Path = args.source.resolve()
    dest: Path = args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; a snapshot is written once and never updated in place")
    if source == dest or source in dest.parents:
        raise ValueError("--dest must lie outside --source")

    edges_08b, dyn_08b = import_closure(source, ENTRIES_08B)
    edges_4b, dyn_4b = import_closure(source, ENTRIES_4B)
    py_files = sorted(set(edges_08b) | set(edges_4b))
    side = [source / rel for rel in SIDE_FILES]
    missing = [str(p) for p in side if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"side files missing: {missing}")

    files: dict[str, dict] = {}
    for path in py_files + side:
        rel = path.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rec = file_record(target)
        if rec != file_record(path):
            raise RuntimeError(f"{rel} changed while it was being copied")
        if path in edges_08b or path in edges_4b:
            rec["in_closure"] = [tag for tag, edges in (("0.8b", edges_08b), ("4b", edges_4b)) if path in edges]
            rec["lines"] = len(target.read_text().splitlines())
            rec["imports"] = (edges_08b.get(path) or edges_4b.get(path) or [])
        files[str(rel)] = rec

    record = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries_08b": ENTRIES_08B,
        "entries_4b": ENTRIES_4B,
        "n_python_files": len(py_files),
        "n_python_lines": sum(files[str(p.relative_to(source))]["lines"] for p in py_files),
        "dynamic_import_hints": {**dyn_08b, **dyn_4b},
        "files": files,
    }
    (dest / "SHA256SUMS.json").write_text(json.dumps(record, indent=1) + "\n")
    with open(dest / "SHA256SUMS", "w") as f:           # `sha256sum -c SHA256SUMS` from inside the snapshot
        for rel, rec in sorted(files.items()):
            f.write(f"{rec['sha256']}  {rel}\n")
    for root_dir, _dirs, names in os.walk(dest):        # read-only files: an accidental edit of the reference fails loudly
        for name in names:
            p = Path(root_dir) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"froze {len(py_files)} python files ({record['n_python_lines']} lines) and {len(side)} side files into {dest}")


if __name__ == "__main__":
    main()
