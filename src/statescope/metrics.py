"""Held-out reconstruction numbers, with the definitions the trainers used (so the manifest's values can be reproduced from the reader).

FVU = fraction of variance unexplained = SSE / SST, where SST is the squared deviation from a per-dimension mean.
"""
from __future__ import annotations

import torch

from statescope.reader import MoltOps, StateVocab


def _sst(y: torch.Tensor) -> float:
    """squared deviation from the per-dimension mean, in float64"""
    y64 = y.double()
    return float(((y64 - y64.mean(0)) ** 2).sum())


@torch.no_grad()
def molt_heldout_numbers(exported: MoltOps, heldout: dict[str, tuple[torch.Tensor, torch.Tensor]], chunk: int = 2048) -> dict[str, float]:
    """FVU of every split, the pooled FVU and the mean number of active atoms of an exported dictionary; `chunk` tokens per forward
    (the chunk size decides how the gate inputs are rounded, so it is part of the numbers' definition).

      fvu_<split>   SSE over SST around the split's own per-dimension mean
      fvu_pooled    sum of the splits' SSE (as fvu x SST) over SST around the grand mean
      l0            (token, atom) pairs with a nonzero gate, per token, over all splits

    heldout: {split: (x raw resid_mid, y raw MLP output)}. Sums over tokens are float64 on the device. The four per-split terms of the
    pooled numerator are added with Python's built-in sum(), as the trainer that produced the released numbers did: since Python 3.12
    that is a compensated sum, and it can differ from a running `+=` by one unit in the last place."""
    fvu: dict[str, float] = {}
    sst: dict[str, float] = {}
    n_active, n_tokens = 0.0, 0
    sum1: torch.Tensor | float = 0.0
    sum2: torch.Tensor | float = 0.0
    for split, (x, y) in heldout.items():
        x, y = x.to(exported.device), y.to(exported.device)
        sse = 0.0
        for i in range(0, x.shape[0], chunk):
            a, y_hat = exported.forward(x[i:i + chunk])
            sse += float(((y_hat - y[i:i + chunk]).double() ** 2).sum())
            n_active += float((a != 0).sum())
        y64 = y.double()
        sst[split] = float(((y64 - y64.mean(0)) ** 2).sum())
        fvu[split] = sse / sst[split]
        n_tokens += y64.shape[0]
        sum1, sum2 = sum1 + y64.sum(0), sum2 + (y64 * y64).sum(0)
    sst_all = float((sum2 - sum1 * sum1 / n_tokens).sum())
    out = {"fvu_pooled": sum(fvu[split] * sst[split] for split in heldout) / sst_all}
    out.update({f"fvu_{split}": value for split, value in fvu.items()})
    out["l0"] = n_active / n_tokens
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
