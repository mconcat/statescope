"""The state dictionary's held-out reconstruction number, with the definition its trainer used.

FVU = fraction of variance unexplained = SSE / SST, where SST is the squared deviation from a per-dimension mean.
"""
from __future__ import annotations

import torch

from statescope.reader import StateVocab


def _sst(y: torch.Tensor) -> float:
    """squared deviation from the per-dimension mean, in float64"""
    y64 = y.double()
    return float(((y64 - y64.mean(0)) ** 2).sum())


@torch.no_grad()
def vocab_heldout_fvu(vocab: StateVocab, site: str, splits: list[torch.Tensor]) -> float:
    """splits: raw activations of `site`, one tensor (n, d) per held-out split. Sum of the splits' SSE over the sum of the splits' SST
    (each split around its own mean), in the units of the normalised residual x / rms(x): the units the dictionary reads and its
    trainer reports."""
    sse, sst = 0.0, 0.0
    for x in splits:
        x = x.to(vocab.device)
        groups, codes = vocab.encode(x, site)
        rms = vocab.token_rms(x)
        x, x_hat = x / rms, vocab.decode(groups, codes, site, rms=torch.ones_like(rms))
        sse += float(((x - x_hat).double() ** 2).sum())
        sst += _sst(x)
    return sse / sst
