"""The fixed mean-gate map: the linear part of a layer's SwiGLU MLP, used as the (frozen) skip matrix of the MLP dictionary.

  y = W_down [SiLU(W_gate u) * (W_up u)],   u = x * w / rms(x)   (the layer's RMSNorm, w = 1 + its weight)
  with every gate frozen at its mean over the calibration tokens, Gbar_j = E[SiLU(w_gate,j . u)]:
      y ~ M0 u + c,   M0 = W_down diag(Gbar) W_up
  in the dictionary's standardized coordinates (u~ = (u - mu_u) / sd_u, y~ = (y - mu_y) / sd_y, prediction u~ @ W_skip):
      W_skip = diag(sd_u) M0^T diag(1 / sd_y)

Written file: {W_skip_std, M0, Gbar, c_raw}, all fp32. Everything is computed in float64 on the bf16 model's activations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from statescope.earlier_design.data.mix import MixStream, reference_sequences
from statescope.lm import PAD_TOKEN, LanguageModel, forward_capture

SEQUENCES_PER_BATCH = 8


@torch.no_grad()
def mean_gates(lm: LanguageModel, layer: int, stream: MixStream, w_eff: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    """Gbar (hidden,) float64: the mean of SiLU(W_gate u) over every position of the calibration sequences; and the token count"""
    W_gate = lm.mlp(layer).gate_proj.weight.detach().double()
    reference = reference_sequences(stream)
    pad = int(lm.tokenizer.convert_tokens_to_ids(PAD_TOKEN))
    total = torch.zeros(W_gate.shape[0], dtype=torch.float64, device=lm.device)
    n = 0
    for name in sorted(reference):
        if not name.startswith("calib_"):
            continue
        seqs = reference[name]
        for i in range(0, len(seqs), SEQUENCES_PER_BATCH):
            batch = seqs[i:i + SEQUENCES_PER_BATCH]
            padded = ((max(len(s) for s in batch) + 63) // 64) * 64
            ids = torch.full((len(batch), padded), pad, dtype=torch.long)
            keep = torch.zeros((len(batch), padded), dtype=torch.bool)
            for r, s in enumerate(batch):
                ids[r, :len(s)] = s
                keep[r, :len(s)] = True
            x = forward_capture(lm, layer, ids)[0][keep.to(lm.device)].double()
            u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w_eff
            total += F.silu(u @ W_gate.T).sum(0)
            n += x.shape[0]
    return total / n, n


@torch.no_grad()
def fixed_map(lm: LanguageModel, layer: int, stream: MixStream, calib_dir: str, out_path: str) -> dict:
    """Compute and write the fixed mean-gate map of `layer`. `stream` supplies the calibration sequences. Returns {"n_calib_tokens": ...}."""
    with safe_open(f"{calib_dir}/calib_stats.safetensors", framework="pt") as f:
        stats = {k: f.get_tensor(k).to(lm.device).double() for k in ("mu_u", "sd_u", "mu_y", "sd_y", "norm_w_eff")}
        eps = float(f.metadata()["norm_eps"])
    mlp = lm.mlp(layer)
    W_up, W_down = mlp.up_proj.weight.detach().double(), mlp.down_proj.weight.detach().double()
    Gbar, n = mean_gates(lm, layer, stream, stats["norm_w_eff"], eps)
    M0 = W_down @ torch.diag(Gbar) @ W_up                                    # (d, d): y ~ M0 u + c
    W_skip = (stats["sd_u"][:, None] * M0.T) / stats["sd_y"][None, :]
    c = stats["mu_y"] - M0 @ stats["mu_u"]
    save_file({"W_skip_std": W_skip.float().cpu().contiguous(), "M0": M0.float().cpu().contiguous(), "Gbar": Gbar.float().cpu(), "c_raw": c.float().cpu()}, out_path)
    return {"layer": layer, "n_calib_tokens": n}
