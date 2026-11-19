"""sha256 of a file and of a tensor's bytes, for the tests. One copy, so that two tests that compare hashes hash the same way."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    """hex sha256 of a file's bytes, read in 16 MiB blocks"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    """hex sha256 of a tensor's values as contiguous bytes on the CPU (shape and dtype are not part of the hash)"""
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
