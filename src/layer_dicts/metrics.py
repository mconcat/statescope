"""Held-out reconstruction numbers, with the definitions the trainers used (so the manifest's values can be reproduced from the reader).

FVU = fraction of variance unexplained = SSE / SST, where SST is the squared deviation from a per-dimension mean.
"""
from __future__ import annotations

import torch

from layer_dicts.reader import MoltOps, StateVocab


def _sst(y: torch.Tensor) -> float:
    """squared deviation from the per-dimension mean, in float64"""
    y64 = y.double()
    return float(((y64 - y64.mean(0)) ** 2).sum())


@torch.no_grad()
def molt_pooled_fvu(molt: MoltOps, heldout: dict[str, tuple[torch.Tensor, torch.Tensor]], chunk: int = 2048) -> dict[str, float]:
    """heldout: {split: (x raw resid_mid, y raw MLP output)}. Returns {"<split>": fvu of the split, ..., "pooled": pooled fvu}.

    Per split: SSE over SST around the split's own mean. Pooled: sum of the splits' SSE (= fvu_s * SST_s) over SST around the grand mean.
    `chunk` = tokens per forward (the trainer's evaluation used 2048)."""
    out: dict[str, float] = {}
    sse_total, n_all = 0.0, 0
    sum1: torch.Tensor | None = None
    sum2: torch.Tensor | None = None
    for split, (x, y) in heldout.items():
        x, y = x.to(molt.device), y.to(molt.device)
        sse = 0.0
        for i in range(0, x.shape[0], chunk):
            _, y_hat = molt.forward(x[i:i + chunk])
            sse += float(((y_hat - y[i:i + chunk]).double() ** 2).sum())
        sst = _sst(y)
        out[split] = sse / sst
        sse_total += out[split] * sst
        y64 = y.double()
        n_all += y64.shape[0]
        sum1 = y64.sum(0) if sum1 is None else sum1 + y64.sum(0)
        sum2 = (y64 * y64).sum(0) if sum2 is None else sum2 + (y64 * y64).sum(0)
    assert sum1 is not None and sum2 is not None, "no held-out split given"
    sst_all = float((sum2 - sum1 * sum1 / n_all).sum())
    out["pooled"] = sse_total / sst_all
    return out


@torch.no_grad()
def vocab_heldout_fvu(vocab: StateVocab, site: str, splits: list[torch.Tensor]) -> float:
    """splits: raw activations of `site`, one tensor (n, d) per held-out split. Sum of the splits' SSE over the sum of the splits' SST
    (each split around its own mean)."""
    sse, sst = 0.0, 0.0
    for x in splits:
        x = x.to(vocab.device)
        groups, codes = vocab.encode(x, site)
        x_hat = vocab.decode(groups, codes, site)
        sse += float(((x - x_hat).double() ** 2).sum())
        sst += _sst(x)
    return sse / sst
