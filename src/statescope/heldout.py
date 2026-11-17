"""Held-out activations of a calibration folder.

A calibration folder holds, for one layer, `calib_stats.safetensors` and four folders `heldout_<split>/` (index.json + shards). A shard
stores, for the tokens of the held-out sequences in order: x = raw resid_mid, y = raw MLP output, tok = token ids.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

HELDOUT_SPLITS = ("pile", "ultra", "cont", "chat")    # web text, user prompts, model continuations, model chat answers


def load_heldout(calib_dir: str, split: str, max_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """-> x (n, d) raw resid_mid, y (n, d) raw MLP output, fp32 on the CPU; the first `max_tokens` tokens if given"""
    folder = os.path.join(calib_dir, f"heldout_{split}")
    with open(os.path.join(folder, "index.json")) as f:
        index = json.load(f)
    xs, ys = [], []
    for shard in index["shards"]:
        tensors = load_file(os.path.join(folder, shard["file"]))
        xs.append(tensors["x"])
        ys.append(tensors["y"])
    x, y = torch.cat(xs), torch.cat(ys)
    if max_tokens:
        x, y = x[:max_tokens], y[:max_tokens]
    return x.float(), y.float()
