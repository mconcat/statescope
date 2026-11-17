"""Run the frozen original code in a separate process and hand its outputs to a test.

Why a separate process: the original modules are named `train.*` / `measure.*`, change the environment when imported, and read and write
paths relative to their repository root. Each run gets
  * PYTHONPATH = the snapshot only, so no research code other than the frozen copy can be imported (the script checks this itself with
    `_hermetic.assert_hermetic`),
  * its working directory = a sandbox under test_out/, so relative `results/...` writes can never land in the original repository,
  * no bytecode writing (the snapshot stays exactly the hashed files).

Outputs are cached under test_out/legacy/ by a key over the snapshot's hash list, the script text and its arguments: the snapshot is
frozen, so a reference is computed once.
"""
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

SCRIPTS = REPO / "tests" / "legacy_scripts"
SANDBOX = TEST_OUT / "legacy_sandbox"
_verified: set[Path] = set()


def verify_snapshot(snapshot: Path) -> None:
    """every file of the snapshot still has the sha256 recorded when it was frozen (checked once per session)"""
    if snapshot in _verified:
        return
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if hashlib.sha256((snapshot / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"legacy snapshot file changed since it was frozen: {rel}")
    _verified.add(snapshot)


def run_legacy(script: str, args: list[str], settings: Settings, timeout_s: int = 3600, *, training: bool = False,
               coverage: bool = False) -> Path:
    """Run tests/legacy_scripts/<script> with `--out <dir>` + args; returns <dir> (cached). Raises if the script fails.

    training: a trainer run. It goes to settings.train_device (queued on settings.train_lock if set) and caps its own memory through
              its --mem-gb argument, so this process must not be using that GPU's budget at the same time.
    coverage: record executed lines of the snapshot (and of files the script writes into <dir>) in <dir>/.coverage."""
    snapshot = settings.need_legacy_snapshot().resolve()
    verify_snapshot(snapshot)
    device = settings.train_device if training else settings.device
    script_path = SCRIPTS / script
    key = hashlib.sha256()
    for part in ((snapshot / "SHA256SUMS").read_bytes(), script_path.read_bytes(), (SCRIPTS / "_hermetic.py").read_bytes(),
                 json.dumps(args).encode(), device.encode(), str(coverage).encode()):
        key.update(hashlib.sha256(part).digest())
    out = TEST_OUT / "legacy" / f"{script_path.stem}-{key.hexdigest()[:16]}"
    if (out / "DONE").is_file():
        return out
    if out.exists():                 # an earlier attempt that did not finish: the legacy trainers refuse to write into a used folder
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (SANDBOX / "results").mkdir(parents=True, exist_ok=True)
    legacy_gb = settings.gpu_gb - 2.0
    if not training and torch.cuda.is_initialized():
        # the legacy process shares this process's GPU budget: this process must hold (almost) nothing while it runs
        torch.cuda.empty_cache()
        held_gb = torch.cuda.memory_reserved(torch.device(device)) / 2**30
        if held_gb > 1.0:
            raise RuntimeError(f"this process still holds {held_gb:.1f} GiB on {device}; free it before a legacy reference run")
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(snapshot),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LEGACY_SNAPSHOT": str(snapshot),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(torch.device(device).index or 0),
        "LEGACY_GPU_GB": str(legacy_gb),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    })
    command = [sys.executable]
    if coverage:
        command += ["-m", "coverage", "run", f"--data-file={out / '.coverage'}", f"--include={snapshot}/*,{out}/*"]
    command += [str(script_path), "--out", str(out), *args]
    if training and settings.train_lock is not None:
        command = ["flock", str(settings.train_lock), *command]
    with open(out / "log.txt", "w") as log:
        proc = subprocess.run(command, cwd=SANDBOX, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    if proc.returncode != 0:
        tail = "\n".join((out / "log.txt").read_text().splitlines()[-30:])
        raise RuntimeError(f"legacy script {script} failed (exit {proc.returncode}); last lines of {out / 'log.txt'}:\n{tail}")
    (out / "DONE").write_text(json.dumps({"script": script, "args": args}) + "\n")
    return out
