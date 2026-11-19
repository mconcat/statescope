"""Freeze the reference code of the line-count demonstration (Qwen3.5-0.8B: L8 MLP -> L9 GDN head 10 -> L9 MLP), as a third snapshot
next to the first two and without touching them.

One source, only read: the original repository. Copied:
  - the static import closure of the demonstration's scripts (as in freeze_legacy.py);
  - the fork of the public MOLT trainer, whole: the demonstration loads the MLP dictionaries through the fork's classes, by path, at run time;
  - the note, the registrations and the result files the demonstration recorded: the recorded numbers the port is compared with.

    python devtools/freeze_demo.py --source <original repo> --dest legacy_snapshot3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path

from freeze_legacy import file_record, import_closure

ENTRIES: list[str] = [
    "measure.demo_line_count.lib",           # batches of the registered cells, patches, state dictionary at a site
    "measure.demo_line_count.s0_gates",      # the demonstration's gates (baselines, MLP dictionary against the model's forward)
    "measure.demo_line_count.s2_mlp9",       # L9 MLP: transforms' shares
    "measure.demo_line_count.s3_mlp8",       # L8 MLP: transforms through the true L9 mixer
    "measure.demo_line_count.s4_amend",      # single groups, writes restricted to a group's frame
    "measure.public_molt.checks_public",     # answer-key check of an MLP dictionary on the line-count items
]
WHOLE_DIRS: list[str] = ["external/crosslayer-transcoder-molt-fork"]
SIDE_FILES: list[str] = [
    "notes/demo_line_count_L8_L9.md",
    "results/demo_line_count/REGISTRATION.md",
    "results/demo_line_count/REGISTRATION_2.md",
    "results/demo_line_count/registration.md5",
    "results/demo_line_count/gates.json",
    "results/demo_line_count/s2.json",
    "results/demo_line_count/s3.json",
    "results/demo_line_count/s4.json",
]
SKIP_DIRS = {"__pycache__", ".venv", ".git", "results", "wandb"}


def tree_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        out += [Path(folder) / n for n in sorted(names) if not n.endswith(".pyc")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="the original repository (read only)")
    ap.add_argument("--dest", required=True, type=Path, help="snapshot folder to create (must not exist)")
    args = ap.parse_args()
    source, dest = args.source.resolve(), args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; a snapshot is written once and never updated in place")
    if source == dest or source in dest.parents:
        raise ValueError("--dest must lie outside --source")

    edges, dynamic = import_closure(source, ENTRIES)
    extra = [p for rel in WHOLE_DIRS for p in tree_files(source / rel)] + [source / rel for rel in SIDE_FILES]
    missing = [str(p) for p in extra if not p.is_file()] + [rel for rel in WHOLE_DIRS if not (source / rel).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing in the source: {missing}")

    files: dict[str, dict] = {}
    for path in sorted(set(edges) | set(extra)):
        rel = path.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rec = file_record(target)
        if rec != file_record(path):
            raise RuntimeError(f"{rel} changed while it was being copied")
        if path in edges:
            rec["imports"] = edges[path]
        files[str(rel)] = rec

    record = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": ENTRIES, "whole_dirs": WHOLE_DIRS, "n_files": len(files),
              "dynamic_import_hints": dynamic, "files": files}
    (dest / "SHA256SUMS.json").write_text(json.dumps(record, indent=1) + "\n")
    with open(dest / "SHA256SUMS", "w") as f:           # `sha256sum -c SHA256SUMS` from inside the snapshot
        for rel, rec in sorted(files.items()):
            f.write(f"{rec['sha256']}  {rel}\n")
    for folder, _dirs, names in os.walk(dest):          # read-only files: an accidental edit of the reference fails loudly
        for name in names:
            p = Path(folder) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"froze {len(files)} files into {dest}")


if __name__ == "__main__":
    main()
