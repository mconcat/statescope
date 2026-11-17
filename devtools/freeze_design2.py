"""Freeze the reference code of the second 0.8B design (tied state dictionary with per-token RMS input; MLP dictionary in the public
penalty form), next to the first snapshot and without touching it.

Two sources, both only read:

  --bundle  the extracted training bundle that was shipped to the rented machines for the MLP dictionary (the public trainer's fork, the
            unmodified public code, capture, data module, loader). Copied whole. Two bundles were shipped during the training night;
            the second is frozen. They differ in the driver script, the data module (shard_data.py), the runbook and added GDN files;
            the model, trainer, loader and export files are identical. Which layers ran from which bundle is not recorded here.
  --source  the original repository, for the state dictionary's launcher and the separately written reader (static import closure,
            as in freeze_legacy.py) and a few side files.

A file that both sources hold is taken from the bundle; `live_differs` in the record says whether the repository's copy has moved on.

    python devtools/freeze_design2.py --source <original repo> --bundle <extracted bundle> --dest legacy_snapshot2
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

# Entry points read from the original repository.
ENTRIES: list[str] = [
    "measure.mlp_track.train_vocab_bsf",     # launcher: tied form + per-token RMS input around the shared state-dictionary trainer
    "measure.mlp_track.vocab_heldout_fvu",   # reader written apart from the trainer (the reference of the new state reader)
    "measure.mlp_track.check_vocab_bsf",     # form checks of a tied export
    "measure.mlp_track.q4b_stream",          # token stream over the self-answered corpus
]
SIDE_FILES: list[str] = [
    "results/mlp_track/specs/b1.json",
    "logs/mlp_track/q08b_launch.sh",
    "logs/mlp_track/q08b_checks.sh",
    "notes/public_molt_RUNBOOK.md",
]
SKIP_DIRS = {"__pycache__", ".venv", ".git"}


def bundle_files(bundle: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, names in os.walk(bundle):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        out += [Path(root) / n for n in sorted(names) if not n.endswith(".pyc")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="the original repository (read only)")
    ap.add_argument("--bundle", required=True, type=Path, help="the extracted training bundle (read only)")
    ap.add_argument("--dest", required=True, type=Path, help="snapshot folder to create (must not exist)")
    args = ap.parse_args()
    source, bundle, dest = args.source.resolve(), args.bundle.resolve(), args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"{dest} exists; a snapshot is written once and never updated in place")
    if any(root == dest or root in dest.parents for root in (source, bundle)):
        raise ValueError("--dest must lie outside --source and --bundle")

    edges, dynamic = import_closure(source, ENTRIES)
    side = [source / rel for rel in SIDE_FILES]
    missing = [str(p) for p in side if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"side files missing: {missing}")

    files: dict[str, dict] = {}

    def copy(path: Path, root: Path, origin: str) -> dict:
        rel = path.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rec = file_record(target)
        if rec != file_record(path):
            raise RuntimeError(f"{rel} changed while it was being copied")
        rec["from"] = origin
        files[str(rel)] = rec
        return rec

    for path in bundle_files(bundle):
        rec = copy(path, bundle, "bundle")
        live = source / path.relative_to(bundle)
        rec["live_differs"] = (not live.is_file()) or file_record(live)["sha256"] != rec["sha256"]
    for path in sorted(edges) + side:
        rel = str(path.relative_to(source))
        if rel in files:                                 # the bundle's copy stays
            continue
        rec = copy(path, source, "repository")
        if path in edges:
            rec["imports"] = edges[path]

    record = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": ENTRIES,
        "n_files": len(files),
        "n_from_bundle": sum(r["from"] == "bundle" for r in files.values()),
        "bundle_files_the_repository_has_changed_since": sorted(k for k, r in files.items() if r.get("live_differs")),
        "dynamic_import_hints": dynamic,
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
    print(f"froze {len(files)} files ({record['n_from_bundle']} from the bundle) into {dest}")


if __name__ == "__main__":
    main()
