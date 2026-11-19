"""Run a script of the NEW code in a separate process on the training GPU (same device, lock and thread settings as the legacy
reference runs of tests/legacy_bridge.py), cached by a key over the package source, the script and its arguments."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from conftest import REPO, TEST_OUT, Settings

SCRIPTS = REPO / "tests" / "new_scripts"


def _source_digest(exclude: tuple[str, ...]) -> bytes:
    """sha256 over the package source, leaving out the sub-packages named in `exclude` (ones the script provably does not use)"""
    h = hashlib.sha256()
    for path in sorted((REPO / "src" / "statescope").rglob("*.py")):
        if path.relative_to(REPO / "src" / "statescope").parts[0] in exclude:
            continue
        h.update(str(path.relative_to(REPO)).encode())
        h.update(path.read_bytes())
    return h.digest()


def run_new(script: str, args: list[str], settings: Settings, timeout_s: int = 3600, exclude: tuple[str, ...] = ()) -> Path:
    """Run tests/new_scripts/<script> with `--out <dir>` + args on settings.train_device; returns <dir> (cached). Raises on failure.
    exclude: sub-packages of statescope whose source is left out of the cache key because the script does not import them
    (tests/test_repo_rules.py checks that the two dictionary packages do not import each other)."""
    script_path = SCRIPTS / script
    key = hashlib.sha256()
    for part in (_source_digest(exclude), script_path.read_bytes(), json.dumps(args).encode(), settings.train_device.encode()):
        key.update(hashlib.sha256(part).digest())
    out = TEST_OUT / "new" / f"{script_path.stem}-{key.hexdigest()[:16]}"
    if os.environ.get("STATESCOPE_USED_RUNS"):       # cleanup aid: append every cached run folder this session resolves to that file
        with open(os.environ["STATESCOPE_USED_RUNS"], "a") as f:
            f.write(f"{out}\n")
    if (out / "DONE").is_file():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(torch.device(settings.train_device).index or 0),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
    command = [sys.executable, str(script_path), "--out", str(out), *args]
    if settings.train_lock is not None:
        command = ["flock", str(settings.train_lock), *command]
    with open(out / "log.txt", "w") as log:
        proc = subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    if proc.returncode != 0:
        tail = "\n".join((out / "log.txt").read_text().splitlines()[-30:])
        raise RuntimeError(f"new-code script {script} failed (exit {proc.returncode}); last lines of {out / 'log.txt'}:\n{tail}")
    (out / "DONE").write_text(json.dumps({"script": script, "args": args}) + "\n")
    return out
