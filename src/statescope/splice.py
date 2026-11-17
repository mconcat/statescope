"""Next-token loss with one layer's dictionary spliced into the forward pass of the language model.

Each arm makes one replacement at layer L in an otherwise unchanged forward pass (reconstructions are computed in fp32 and cast back to
the model's dtype). Position 0 is never replaced (it was dropped in training); loss and KL are means over the positions whose stream was
replaced and that have a next token.

  molt        output of layers[L].mlp          <- MoltOps.forward(resid_mid)
  mlp_zero    output of layers[L].mlp          <- 0
  mlp_mean    output of layers[L].mlp          <- its mean over the replaced positions of the batch
  vocab_post  output of layer L                <- StateVocab reconstruction of resid_post.L (resid_mid + MLP output)
  vocab_mid   input of the MLP sub-block       <- StateVocab reconstruction of resid_mid.L (the MLP reads it and the skip path carries it)
  post_mean   output of layer L                <- mean of resid_post over the replaced positions of the batch

Per arm: dce = CE(arm) - CE(clean) in nats, kl = KL(clean || arm) in nats, fvu_batchwise = FVU of the replacement on these tokens
(variance around each batch's own mean). The recovered fractions use different baselines (MLP: zero ablation; state: mean ablation),
so the two are not comparable with each other.
"""
from __future__ import annotations

import json
from typing import Callable

import torch
import torch.nn.functional as F

from statescope.reader import MoltOps, StateVocab, decoder_layers

ARMS = ("molt", "mlp_zero", "mlp_mean", "vocab_post", "vocab_mid", "post_mean")
IDENTITY_ARMS = ("ident_mlp", "ident_mid")      # replace a tensor by itself: must change nothing (hook self-check)
MAX_IDENTITY_KL = 1e-4


def load_documents(tokenizer, jsonl_path: str, n_docs: int, n_tokens: int = 256, first_row: int = 0) -> torch.Tensor:
    """The first `n_docs` rows at or after `first_row` of a JSON-lines file ({"text": ...}) that have at least `n_tokens` tokens,
    cut to `n_tokens`. -> (n_docs, n_tokens) long"""
    rows: list[list[int]] = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i < first_row:
                continue
            ids = tokenizer(json.loads(line)["text"], add_special_tokens=True)["input_ids"]
            if len(ids) >= n_tokens:
                rows.append(ids[:n_tokens])
            if len(rows) == n_docs:
                break
    if len(rows) != n_docs:
        raise ValueError(f"only {len(rows)} documents of {n_tokens} tokens at or after row {first_row} of {jsonl_path}")
    return torch.tensor(rows)


def _with_positions_replaced(original: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """original (B, T, d) in the model's dtype, new (B, T-1, d) fp32 for positions 1.. -> original with positions 1.. replaced"""
    out = original.clone()
    out[:, 1:] = new.to(original.dtype)
    return out


class _Splice:
    """The hooks of one arm at one layer for one batch. `clean` = this batch's clean {"resid_mid", "mlp_out"} (B, T, d) fp32.
    After the forward pass, `sse` and `sst` hold the replacement's squared error and the replaced tensor's variance on this batch."""

    def __init__(self, block: torch.nn.Module, arm: str, vocab: StateVocab, molt: MoltOps, layer: int, clean: dict[str, torch.Tensor]) -> None:
        self.block, self.arm, self.vocab, self.molt, self.layer, self.clean = block, arm, vocab, molt, layer, clean
        self.sse, self.sst = 0.0, 0.0
        self._handles: list = []

    def _count(self, x: torch.Tensor, x_hat: torch.Tensor) -> None:
        self.sse += float(((x - x_hat) ** 2).sum())
        self.sst += float(((x - x.mean(0)) ** 2).sum())

    def __enter__(self) -> "_Splice":
        arm, layer, vocab, molt = self.arm, self.layer, self.vocab, self.molt
        mid, mlp = self.clean["resid_mid"][:, 1:], self.clean["mlp_out"][:, 1:]
        B, n, d = mid.shape
        mid2, mlp2 = mid.reshape(-1, d), mlp.reshape(-1, d)
        post2 = mid2 + mlp2
        if arm in ("molt", "mlp_zero", "mlp_mean", "ident_mlp"):
            make: dict[str, Callable[[], torch.Tensor]] = {
                "ident_mlp": lambda: mlp2,
                "molt": lambda: molt.forward(mid2)[1],
                "mlp_zero": lambda: torch.zeros_like(mlp2),
                "mlp_mean": lambda: mlp2.mean(0, keepdim=True).expand_as(mlp2),
            }
            new = make[arm]()
            self._count(mlp2, new)
            self._handles.append(self.block.mlp.register_forward_hook(lambda m, args, out: _with_positions_replaced(out, new.view(B, n, d))))
        elif arm in ("vocab_post", "post_mean"):
            if arm == "vocab_post":
                groups, codes = vocab.encode(post2, f"resid_post.{layer}")
                new = vocab.decode(groups, codes, f"resid_post.{layer}")
            else:
                new = post2.mean(0, keepdim=True).expand_as(post2)
            self._count(post2, new)

            def replace_layer_output(m, args, out):
                if isinstance(out, tuple):
                    return (_with_positions_replaced(out[0], new.view(B, n, d)),) + tuple(out[1:])
                return _with_positions_replaced(out, new.view(B, n, d))
            self._handles.append(self.block.register_forward_hook(replace_layer_output))
        elif arm in ("vocab_mid", "ident_mid"):
            if arm == "vocab_mid":
                groups, codes = vocab.encode(mid2, f"resid_mid.{layer}")
                new = vocab.decode(groups, codes, f"resid_mid.{layer}")
            else:
                new = mid2
            self._count(mid2, new)
            delta = (new - mid2).view(B, n, d)
            self._handles.append(self.block.post_attention_layernorm.register_forward_pre_hook(
                lambda m, args: (_with_positions_replaced(args[0], new.view(B, n, d)),) + tuple(args[1:])))

            def shift_skip_path(m, args, out):          # layer output = resid_mid + mlp(...), so the skip path gets (new - old) too
                o = out[0] if isinstance(out, tuple) else out
                o2 = o.clone()
                o2[:, 1:] = (o[:, 1:].float() + delta).to(o.dtype)
                return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
            self._handles.append(self.block.register_forward_hook(shift_skip_path))
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _log_probs(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """log-probabilities at positions 1..T-2 (the replaced positions that have a next token), fp32"""
    return F.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 1:-1].float(), -1)


@torch.no_grad()
def splice_layer(model: torch.nn.Module, layer: int, ids: torch.Tensor, vocab: StateVocab, molt: MoltOps, batch: int = 2) -> dict:
    """ids (n_docs, T) long. Returns {"ce_clean", "<arm>": {"dce", "kl", "fvu_batchwise"}, "molt_loss_recovered", ...} for one layer."""
    block = decoder_layers(model)[layer]
    device = next(model.parameters()).device
    acc = {arm: {"ce": 0.0, "kl": 0.0, "sse": 0.0, "sst": 0.0} for arm in ARMS}
    ce_clean, n_pos = 0.0, 0
    identity_kl: dict[str, float] = {}
    for start in range(0, ids.shape[0], batch):
        b = ids[start:start + batch].to(device)
        clean: dict[str, torch.Tensor] = {}
        h1 = block.post_attention_layernorm.register_forward_pre_hook(lambda m, args: clean.__setitem__("resid_mid", args[0].detach().float()))
        h2 = block.mlp.register_forward_hook(lambda m, args, out: clean.__setitem__("mlp_out", out.detach().float()))
        lp_clean = _log_probs(model, b)
        h1.remove()
        h2.remove()
        target = b[:, 2:]
        ce_clean += float(-lp_clean.gather(-1, target[..., None]).sum())
        n_pos += target.numel()
        if start == 0:
            for arm in IDENTITY_ARMS:
                with _Splice(block, arm, vocab, molt, layer, clean):
                    lp = _log_probs(model, b)
                identity_kl[arm] = float((lp_clean.exp() * (lp_clean - lp)).sum()) / target.numel()
                del lp
            if max(abs(v) for v in identity_kl.values()) > MAX_IDENTITY_KL:
                raise RuntimeError(f"layer {layer}: replacing a tensor by itself changes the output: {identity_kl}")
        for arm in ARMS:
            splice = _Splice(block, arm, vocab, molt, layer, clean)
            with splice:
                lp = _log_probs(model, b)
            acc[arm]["ce"] += float(-lp.gather(-1, target[..., None]).sum())
            acc[arm]["kl"] += float((lp_clean.exp() * (lp_clean - lp)).sum())
            acc[arm]["sse"] += splice.sse
            acc[arm]["sst"] += splice.sst
            del lp
        del lp_clean
    result: dict = {"layer": layer, "n_docs": int(ids.shape[0]), "n_positions": n_pos, "ce_clean": ce_clean / n_pos,
                    "identity_splice_kl_first_batch": identity_kl}
    for arm in ARMS:
        result[arm] = {"dce": acc[arm]["ce"] / n_pos - ce_clean / n_pos, "kl": acc[arm]["kl"] / n_pos,
                       "fvu_batchwise": acc[arm]["sse"] / acc[arm]["sst"]}
    result["molt_loss_recovered"] = 1 - result["molt"]["dce"] / result["mlp_zero"]["dce"]
    result["vocab_post_loss_recovered"] = 1 - result["vocab_post"]["dce"] / result["post_mean"]["dce"]
    return result
