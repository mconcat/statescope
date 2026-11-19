"""The models the published dictionaries exist for. Nothing else in the package hard-codes a model name or size."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str         # short name, e.g. "qwen3.5-0.8b"
    hf_id: str        # Hugging Face id
    d_model: int
    n_layers: int


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-0.8b": ModelSpec("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", 1024, 24),
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", 2560, 32),
}
