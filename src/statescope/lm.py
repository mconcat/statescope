"""Loading the language model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from statescope.config import ModelSpec
from statescope.reader import decoder_layers


@dataclass
class LanguageModel:
    spec: ModelSpec
    model: nn.Module
    tokenizer: object
    layers: nn.ModuleList
    device: torch.device


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
