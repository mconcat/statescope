"""Held-out FVU of the state dictionary during and after training.

Data: the first `tokens` tokens of each held-out split of the calibration folder (position 0 of the sequences included, as the split
stores it). resid_mid = x; resid_post = x + y computed here in fp32 from the stored bf16 x and MLP output y.

FVU of a site = sum over splits of SSE / sum over splits of SST, in the dictionary's standardized units, SST around each split's own
per-dimension mean. Sums are fp32 on the device, accumulated across chunks and splits as Python floats.
"""
from __future__ import annotations

import torch

from statescope.heldout import HELDOUT_SPLITS, load_heldout
from statescope.vocab.model import StateDictionary
from statescope.vocab.recipe import site_names


def load_heldout_sites(calib_dir: str, layer: int, device: torch.device | str, tokens: int) -> dict[str, list[torch.Tensor]]:
    """{site: [raw activations (n, d) fp32 on `device`, one tensor per held-out split]}"""
    site_mid, site_post = site_names(layer)
    held: dict[str, list[torch.Tensor]] = {site_mid: [], site_post: []}
    for split in HELDOUT_SPLITS:
        x, y = load_heldout(calib_dir, split, tokens)
        x = x.to(device)
        held[site_mid].append(x)
        held[site_post].append(x + y.to(device))
    return held


@torch.no_grad()
def heldout_fvu(model: StateDictionary, held: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    """held: {site: [raw activations per split]} -> {site: FVU}"""
    chunk = model.recipe.heldout_chunk
    Q, R = model.write_basis()
    out: dict[str, float] = {}
    for j, site in enumerate(model.sites):
        sse, sst = 0.0, 0.0
        for x in held[site]:
            xs = model.standardize(x, j)
            for i in range(0, xs.shape[0], chunk):
                x_hat = model.reconstruct(xs[i:i + chunk], Q, R)
                sse += float(((xs[i:i + chunk] - x_hat) ** 2).sum())
            sst += float(((xs - xs.mean(0)) ** 2).sum())
        out[site] = sse / sst
    return out
