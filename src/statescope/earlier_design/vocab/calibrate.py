"""Per-site mean and scale of the state dictionary, from the calibration sequences.

    mu_site    (d,)  mean activation of the site
    scale_site ()    sqrt(E||x - mu_site||^2 / d): one scalar per site, so that standardized activations have E||xs||^2 = d

Both are float64 sums over every calibration token except the first of each sequence (position 0 has no context and a norm several
times the median; the dictionary never sees it). resid_post is the layer's own output as the model computed it (a bf16 sum), not
resid_mid + MLP output recomputed in fp32.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import torch

from statescope.earlier_design.vocab.recipe import site_names
from statescope.lm import PAD_TOKEN, LanguageModel


@dataclass(frozen=True)
class SiteCalibration:
    mu: torch.Tensor        # (S, d) fp32
    scale: torch.Tensor     # (S,) fp32
    record: dict            # written to cfg.json: tokens per site, the scales (float64) and a sha256 of the float64 mean and variance


@torch.no_grad()
def capture_sites(lm: LanguageModel, layer: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass. ids (rows, T) -> (resid_mid, resid_post), each (rows, T, d) in the model's dtype on the model's device:
    the input of the layer's MLP norm and the output of the layer."""
    got: dict[str, torch.Tensor] = {}

    def take_mid(module, args, kwargs):
        if args:
            got["resid_mid"] = args[0]
        elif "hidden_states" in kwargs:
            got["resid_mid"] = kwargs["hidden_states"]
        else:
            raise RuntimeError("the MLP norm was called with no positional input and no hidden_states keyword")

    def take_post(module, args, out):
        got["resid_post"] = out[0] if isinstance(out, (tuple, list)) else out

    h1 = lm.mlp_norm(layer).register_forward_pre_hook(take_mid, with_kwargs=True)
    h2 = lm.layers[layer].register_forward_hook(take_post)
    try:
        lm.model(input_ids=ids.to(lm.device), attention_mask=None, use_cache=False, logits_to_keep=1)
    finally:                                    # always remove the hooks; an exception propagates
        h1.remove()
        h2.remove()
    if set(got) != {"resid_mid", "resid_post"}:
        raise RuntimeError(f"the forward pass did not reach layer {layer} (captured {sorted(got)})")
    return got["resid_mid"], got["resid_post"]


def site_batches(lm: LanguageModel, layer: int, seqs: list[torch.Tensor], batch_tokens: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (resid_mid, resid_post), each (kept tokens, d) in the model's dtype, one forward pass at a time. Sequences are batched
    greedily (as many following sequences as fit in batch_tokens when padded on the right to the longest, a multiple of 64; no
    attention mask). Kept = every position of a sequence except the first."""
    pad = int(lm.tokenizer.convert_tokens_to_ids(PAD_TOKEN))
    i = 0
    while i < len(seqs):
        longest = len(seqs[i])
        j = i
        while j < len(seqs) and (j - i + 1) * max(longest, len(seqs[j])) <= batch_tokens:
            longest = max(longest, len(seqs[j]))
            j += 1
        j = max(j, i + 1)
        padded = ((longest + 63) // 64) * 64
        ids = torch.full((j - i, padded), pad, dtype=torch.long)
        keep = torch.zeros((j - i, padded), dtype=torch.bool)
        for r, s in enumerate(seqs[i:j]):
            ids[r, :len(s)] = s
            keep[r, 1:len(s)] = True
        resid_mid, resid_post = capture_sites(lm, layer, ids)
        rows, positions = keep.to(lm.device).nonzero(as_tuple=True)
        yield resid_mid[rows, positions], resid_post[rows, positions]
        i = j


@torch.no_grad()
def calibrate_sites(lm: LanguageModel, layer: int, calib_seqs: dict[str, list[torch.Tensor]], batch_tokens: int) -> SiteCalibration:
    """calib_seqs: {name: token sequences}; every sequence of every entry is used, in the dict's order."""
    sites = site_names(layer)
    d = lm.spec.d_model
    sum1 = torch.zeros(len(sites), d, dtype=torch.float64, device=lm.device)       # per site: sum of x
    sum2 = torch.zeros(len(sites), dtype=torch.float64, device=lm.device)          # per site: sum of ||x||^2
    n_tokens = 0
    for seqs in calib_seqs.values():
        for acts in site_batches(lm, layer, seqs, batch_tokens):
            n_tokens += acts[0].shape[0]
            for j, x in enumerate(acts):
                x = x.double()
                sum1[j] += x.sum(0)
                sum2[j] += (x * x).sum()
    mu = sum1 / n_tokens
    var = (sum2 / n_tokens - (mu * mu).sum(-1)) / d
    record = {"n_tokens_per_site": n_tokens, "site_scale": {site: float(v) for site, v in zip(sites, var.sqrt())},
              "sha256": hashlib.sha256(mu.cpu().numpy().tobytes() + var.cpu().numpy().tobytes()).hexdigest()}
    return SiteCalibration(mu.float(), var.sqrt().float(), record)
