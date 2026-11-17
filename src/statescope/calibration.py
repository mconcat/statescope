"""Calibration statistics of one layer (written by statescope.prep.calibrate) and the two changes of coordinates built on them.

  z     = (x - mu_x) / sd_x                     standardized resid_mid; what the shuffle buffer stores (as bf16)
  y_std = (y - mu_y) / sd_y                     standardized MLP output
  u_std = (rmsnorm(x) - mu_u) / sd_u            the MLP dictionary's input: the layer's own RMSNorm applied to x, then standardized
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class CalibStats:
    mu_x: torch.Tensor
    sd_x: torch.Tensor
    mu_y: torch.Tensor
    sd_y: torch.Tensor
    mu_u: torch.Tensor
    sd_u: torch.Tensor
    norm_w_eff: torch.Tensor      # 1 + the RMSNorm weight
    norm_eps: float
    sha256: str                   # over mu_x, sd_x, mu_y, sd_y (identifies the calibration)
    file_metadata: dict           # the safetensors header as written
    folder_meta: dict             # meta.json of the calibration folder

    @property
    def d(self) -> int:
        return int(self.mu_x.shape[0])

    def to(self, device) -> "CalibStats":
        moved = {k: getattr(self, k).to(device) for k in ("mu_x", "sd_x", "mu_y", "sd_y", "mu_u", "sd_u", "norm_w_eff")}
        return CalibStats(**moved, norm_eps=self.norm_eps, sha256=self.sha256, file_metadata=self.file_metadata, folder_meta=self.folder_meta)

    def standardize(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (x - self.mu_x) / self.sd_x, (y - self.mu_y) / self.sd_y

    def u_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """standardized resid_mid -> standardized RMSNorm output (the MLP dictionary's input)"""
        x = z * self.sd_x + self.mu_x
        u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.norm_w_eff
        return (u - self.mu_u) / self.sd_u


def load_calibration(calib_dir: str, device) -> CalibStats:
    path = os.path.join(calib_dir, "calib_stats.safetensors")
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k).to(device) for k in ("mu_x", "sd_x", "mu_y", "sd_y", "mu_u", "sd_u", "norm_w_eff")}
    h = hashlib.sha256()
    for key in ("mu_x", "sd_x", "mu_y", "sd_y"):
        h.update(tensors[key].float().cpu().contiguous().numpy().tobytes())
    if h.hexdigest() != metadata["sha256"]:
        raise ValueError(f"{path}: the tensors do not match the sha256 in the file's metadata")
    with open(os.path.join(calib_dir, "meta.json")) as f:
        folder_meta = json.load(f)
    return CalibStats(**tensors, norm_eps=float(metadata["norm_eps"]), sha256=metadata["sha256"], file_metadata=dict(metadata), folder_meta=folder_meta)
