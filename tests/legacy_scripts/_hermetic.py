"""Shared by the scripts that run inside the frozen original code (see tests/legacy_bridge.py)."""
from __future__ import annotations

import os
import sys

import torch

RESEARCH_PACKAGES = ("train", "measure", "gpt2", "tools")


def assert_hermetic() -> list[str]:
    """Every imported research module must come from the frozen snapshot, and the new package must not be loaded at all.
    Returns the sorted names of the research modules that were imported (the run-time dependency list of this script)."""
    snapshot = os.path.realpath(os.environ["LEGACY_SNAPSHOT"]) + os.sep
    used = []
    for name, module in list(sys.modules.items()):
        top = name.split(".")[0]
        if top == "layer_dicts":
            raise RuntimeError("the refactored package was imported inside a legacy reference run")
        if top not in RESEARCH_PACKAGES:
            continue
        file = getattr(module, "__file__", None)
        if file is None:                                    # namespace package: check its search path instead
            paths = [os.path.realpath(p) + os.sep for p in module.__path__]
            if not all(p.startswith(snapshot) for p in paths):
                raise RuntimeError(f"{name} resolves outside the snapshot: {paths}")
            continue
        if not os.path.realpath(file).startswith(snapshot):
            raise RuntimeError(f"{name} was imported from {file}, not from the frozen snapshot")
        used.append(name)
    return sorted(used)


def cap_gpu() -> str:
    """cap this process on its (only visible) GPU at LEGACY_GPU_GB; returns the device string"""
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(float(os.environ["LEGACY_GPU_GB"]) / total_gb, 0)
    return "cuda:0"
