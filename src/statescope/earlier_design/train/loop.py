"""The training loop shared by both dictionaries: token stream -> activations of one layer -> shuffle buffer -> batches.

    stream unit (262k tokens, four sources in order)
      -> one forward pass per source part: x = resid_mid, y = MLP output (bf16)
      -> standardized with the calibration statistics and appended to a buffer of `buffer_tokens` rows (bf16, pinned host memory)
      -> when the buffer is full: a random permutation (drawn on the GPU); the first half is trained on in batches, the second half
         stays in the buffer and is topped up from the stream.

`training_batches` is a generator of `Batch`; a trainer is just a loop over it. Random draws: only the permutation, from the CUDA
generator seeded with `seed` when the generator starts (so the batches do not depend on anything drawn earlier, e.g. model init).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from statescope.earlier_design.calibration import CalibStats
from statescope.config import SOURCES
from statescope.earlier_design.data.mix import MixStream, Part
from statescope.lm import Capturer


@dataclass(frozen=True)
class LoopSettings:
    tokens: int                         # training budget; the run is ceil(tokens / batch) steps
    batch: int = 4096
    buffer_tokens: int = 1 << 20
    seed: int = 0

    @property
    def steps(self) -> int:
        return int(math.ceil(self.tokens / self.batch))


@dataclass
class Batch:
    step: int
    z: torch.Tensor         # (B, d) fp32 on the GPU: standardized resid_mid as it left the bf16 buffer
    y_std: torch.Tensor     # (B, d) fp32: standardized MLP output
    pos0: torch.Tensor      # (B,) bool: the token is the first of its sequence
    tok: torch.Tensor       # (B,) int32 token ids


@dataclass
class Unit:
    """one stream unit after the forward passes (host tensors)"""
    index: int
    x: torch.Tensor         # (n, d) bf16 raw resid_mid
    y: torch.Tensor         # (n, d) bf16 raw MLP output
    pos0: torch.Tensor      # (n,) bool
    tok: torch.Tensor       # (n,) int32
    tokens_by_source: dict[str, int]
    info: dict              # the stream's record of the unit (rows used per source)


def capture_unit(capturer: Capturer, index: int, parts: list[Part], info: dict) -> Unit:
    """one Capturer.run per source part, concatenated in the parts' order"""
    xs, ys, p0, toks, n = [], [], [], [], {}
    for name, seqs in parts:
        if not seqs:
            n[name] = 0
            continue
        got = capturer.run(seqs)
        xs.append(got["x"])
        ys.append(got["y"])
        p0.append(got["pos"] == 0)
        toks.append(got["tok"])
        n[name] = int(got["x"].shape[0])
    return Unit(index, torch.cat(xs), torch.cat(ys), torch.cat(p0), torch.cat(toks), n, info)


def first_raw_inputs(capturer: Capturer, stream: MixStream, n_tokens: int, chunk: int = 4096) -> Iterator[torch.Tensor]:
    """The first n_tokens raw resid_mid rows of the stream, in stream order, as fp32 host chunks. The stream is left where it was
    (snapshot / restore), so the training loop afterwards sees the same tokens again."""
    state = stream.snapshot()
    seen = 0
    while seen < n_tokens:
        index, _, parts, info = stream.next_unit()
        x = capture_unit(capturer, index, parts, info).x.float()
        for i in range(0, x.shape[0], chunk):
            if seen >= n_tokens:
                break
            block = x[i:i + chunk][: n_tokens - seen]
            yield block
            seen += block.shape[0]
    stream.restore(state)


def unit_record(unit: Unit, stats: CalibStats, device) -> dict:
    """what is logged per unit: token counts, the rows used, and a calibration sanity check on the first 8,192 tokens"""
    record = {"shard": unit.index, "tokens": int(unit.x.shape[0]), "per_src": unit.tokens_by_source,
              **{k: unit.info[k] for k in ("pile_rows", "ultra_k", "ultra_rows", "cont_idx", "chat_idx")}}
    if "exhausted" in unit.info:
        record["exhausted"] = unit.info["exhausted"]
    with torch.no_grad():
        z_head = (unit.x[:8192].to(device).float() - stats.mu_x) / stats.sd_x
        record["z_head_mean_abs"], record["z_head_sd_mean"] = float(z_head.mean(0).abs().mean()), float(z_head.std(0).mean())
    return record


def training_batches(capturer: Capturer, stream: MixStream, stats: CalibStats, settings: LoopSettings, device: torch.device,
                     on_unit: Callable[[dict], None] | None = None) -> Iterator[Batch]:
    """Yield settings.steps batches. `stats` must be on `device`. `on_unit` receives each unit's record (see unit_record)."""
    B, N = settings.batch, settings.buffer_tokens
    half = N // 2
    d = stats.d
    buf_z = torch.empty(N, d, dtype=torch.bfloat16, device="cpu", pin_memory=True)
    buf_y = torch.empty(N, d, dtype=torch.bfloat16, device="cpu", pin_memory=True)
    buf_pos0 = torch.zeros(N, dtype=torch.bool, device="cpu", pin_memory=True)
    buf_tok = torch.zeros(N, dtype=torch.int32, device="cpu", pin_memory=True)
    fill, step, tokens_seen = 0, 0, 0
    torch.cuda.manual_seed(settings.seed)
    rng_before = torch.cuda.get_rng_state()
    while True:
        index, _, parts, info = stream.next_unit()
        unit = capture_unit(capturer, index, parts, info)
        if index == 0 and not torch.equal(rng_before, torch.cuda.get_rng_state()):
            raise RuntimeError("the language model's forward pass consumed the CUDA generator (the buffer permutation would change)")
        if on_unit is not None:
            on_unit(unit_record(unit, stats, device))
        pos, n_unit = 0, unit.x.shape[0]
        while pos < n_unit:
            n_new = min(n_unit - pos, N - fill)
            with torch.no_grad():
                z, y_std = stats.standardize(unit.x[pos:pos + n_new].to(device).float(), unit.y[pos:pos + n_new].to(device).float())
                buf_z[fill:fill + n_new] = z.to(torch.bfloat16)
                buf_y[fill:fill + n_new] = y_std.to(torch.bfloat16)
                buf_pos0[fill:fill + n_new] = unit.pos0[pos:pos + n_new]
                buf_tok[fill:fill + n_new] = unit.tok[pos:pos + n_new]
                del z, y_std
            fill += n_new
            pos += n_new
            if fill < N:
                continue
            perm = torch.randperm(N, device=device)
            for j in range(0, half - B + 1, B):
                sel = perm[j:j + B].cpu()
                yield Batch(step, buf_z[sel].to(device).float(), buf_y[sel].to(device).float(), buf_pos0[sel].to(device), buf_tok[sel].to(device))
                step += 1
                tokens_seen += B
                if tokens_seen >= settings.tokens:
                    return
            keep = perm[half:].cpu()
            buf_z[:N - half].copy_(buf_z[keep])
            buf_y[:N - half].copy_(buf_y[keep])
            buf_pos0[:N - half].copy_(buf_pos0[keep])
            buf_tok[:N - half].copy_(buf_tok[keep])
            fill = N - half
