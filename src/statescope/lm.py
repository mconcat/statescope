"""The language model: loading, and capturing one layer's MLP input and output for lists of token sequences."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from statescope.config import ModelSpec
from statescope.reader import decoder_layers

PAD_TOKEN = "<|endoftext|>"


@dataclass
class LanguageModel:
    spec: ModelSpec
    model: nn.Module
    tokenizer: object
    layers: nn.ModuleList
    device: torch.device

    def mlp_norm(self, layer: int) -> nn.Module:
        """the RMSNorm in front of the layer's MLP; its input is resid_mid"""
        return self.layers[layer].post_attention_layernorm

    def mlp(self, layer: int) -> nn.Module:
        return self.layers[layer].mlp


def load_language_model(spec: ModelSpec, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16) -> LanguageModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=dtype, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = decoder_layers(model)
    if len(layers) != spec.n_layers:
        raise RuntimeError(f"{spec.hf_id}: found {len(layers)} decoder layers, the spec says {spec.n_layers}")
    return LanguageModel(spec, model, tokenizer, layers, next(model.parameters()).device)


class _Reached(Exception):
    """raised inside the last hook to end the forward pass once everything needed has been captured"""


@torch.no_grad()
def forward_capture(lm: LanguageModel, layer: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass up to layer `layer`'s MLP. ids (rows, T) on any device -> (resid_mid, mlp_out), each (rows, T, d) in the model's
    dtype on the model's device. The pass stops right after the MLP (the layers above are not run)."""
    got: dict[str, torch.Tensor] = {}

    def take_input(module, args, kwargs):
        if args:
            got["resid_mid"] = args[0]
        elif "hidden_states" in kwargs:
            got["resid_mid"] = kwargs["hidden_states"]
        else:
            raise RuntimeError("the MLP norm was called with no positional input and no hidden_states keyword")

    def take_output(module, args, out):
        got["mlp_out"] = out[0] if isinstance(out, (tuple, list)) else out
        raise _Reached

    h1 = lm.mlp_norm(layer).register_forward_pre_hook(take_input, with_kwargs=True)
    h2 = lm.mlp(layer).register_forward_hook(take_output)
    try:
        lm.model(input_ids=ids.to(lm.device), attention_mask=None, use_cache=False, logits_to_keep=1)
    except _Reached:
        pass                                    # the expected way out: both tensors are in `got`
    finally:
        h1.remove()
        h2.remove()
    if set(got) != {"resid_mid", "mlp_out"}:
        raise RuntimeError(f"the forward pass ended without reaching layer {layer}'s MLP (captured {sorted(got)})")
    return got["resid_mid"], got["mlp_out"]


class Capturer:
    """Activations of one layer for lists of token sequences, batched by length.

    run(seqs) returns, for every kept position in sequence order (CPU tensors):
      x    (n, d) bf16   raw resid_mid (input of the MLP sub-block)
      y    (n, d) bf16   raw MLP output
      gen  (n,)   bool   position is at or after the sequence's prompt length (False everywhere without prompt lengths)
      seq  (n,)   int64  index of the sequence in `seqs`
      pos  (n,)   int16  position in the sequence
      tok  (n,)   int32  token id
    Kept = positions min_pos .. len-1 of each sequence (padding is never kept).
    """

    def __init__(self, lm: LanguageModel, layer: int, min_pos: int = 0, batch_tokens: int = 32768) -> None:
        self.lm, self.layer, self.min_pos, self.batch_tokens = lm, layer, min_pos, batch_tokens
        self.pad = int(lm.tokenizer.convert_tokens_to_ids(PAD_TOKEN))

    @torch.no_grad()
    def run(self, seqs: list[torch.Tensor], prompt_lens: list[int] | None = None) -> dict[str, torch.Tensor]:
        out: dict[str, list[torch.Tensor]] = {k: [] for k in ("x", "y", "gen", "seq", "pos", "tok")}
        i = 0
        while i < len(seqs):
            # greedy batch: as many following sequences as fit in batch_tokens when padded to the longest (a multiple of 64)
            longest = len(seqs[i])
            j = i
            while j < len(seqs) and (j - i + 1) * max(longest, len(seqs[j])) <= self.batch_tokens:
                longest = max(longest, len(seqs[j]))
                j += 1
            j = max(j, i + 1)
            padded = ((longest + 63) // 64) * 64
            ids = torch.full((j - i, padded), self.pad, dtype=torch.long)
            keep = torch.zeros((j - i, padded), dtype=torch.bool)
            gen = torch.zeros((j - i, padded), dtype=torch.bool)
            for r, s in enumerate(seqs[i:j]):
                ids[r, :len(s)] = s
                keep[r, self.min_pos:len(s)] = True
                if prompt_lens is not None:
                    gen[r, prompt_lens[i + r]:len(s)] = True
            resid_mid, mlp_out = forward_capture(self.lm, self.layer, ids)
            keep_dev = keep.to(self.lm.device)
            out["x"].append(resid_mid[keep_dev].to(torch.bfloat16).cpu())
            out["y"].append(mlp_out[keep_dev].to(torch.bfloat16).cpu())
            rows, positions = keep.nonzero(as_tuple=True)
            out["gen"].append(gen[rows, positions])
            out["seq"].append(rows + i)
            out["pos"].append(positions.to(torch.int16))
            out["tok"].append(ids[rows, positions].to(torch.int32))
            i = j
        return {k: torch.cat(v) for k, v in out.items()}
