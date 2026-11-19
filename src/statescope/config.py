"""Every setting of the pipeline in one place: which model, where the token sources are, how the training mixture is cut.

Nothing else in the package hard-codes a model name, a corpus path or a reserved row range.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str         # short name used in run records, e.g. "qwen3.5-0.8b"
    hf_id: str        # Hugging Face id
    d_model: int
    n_layers: int


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-0.8b": ModelSpec("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", 1024, 24),
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", 2560, 32),
}


@dataclass(frozen=True)
class DataPaths:
    """Where the four token sources live."""
    pile_dir: str        # the 64 parquet shards `train-*.parquet` of the pre-tokenized Pile (rows of 1,024 token ids, column input_ids)
    ultrachat_dir: str   # UltraChat 200k parquet files (column messages)
    view_corpus: str     # the model's own continuations / chat answers as they were when the calibration folder was made
    full_corpus: str     # the same corpus, later state (its first sequences equal the view's); the stream reads this one


@dataclass(frozen=True)
class MixSpec:
    """The training mixture and the rows reserved for calibration and held-out evaluation.

    A unit is `unit_tokens` tokens: Pile, UltraChat, on-policy continuations, on-policy chat, in that order, with the shares below.
    The defaults are the Qwen3.5-0.8B runs."""
    unit_tokens: int = 1 << 18                       # 262,144
    # shares are given as token counts per unit by `quota()`: a third Pile, a third UltraChat, the rest split between cont and chat
    pile_row_tokens: int = 1024                      # tokens per Pile row
    ultra_max_tokens: int = 1024                     # UltraChat conversations are cut to this many tokens
    pile_seed: int = 0                               # order of the Pile (shard, row group) units
    # reserved for calibration and held-out evaluation
    pile_calib_shard: int = 60                       # Pile shards from this one on are never training data
    pile_heldout_shard: int = 62
    calib_pile_rows: int = 64
    heldout_pile_rows: int = 64
    onpol_calib_start: int = 10400                   # on-policy sequences [start, start + calib_onpol_seqs) are the calibration rows
    calib_onpol_seqs: int = 64
    ultra_calib_rows: tuple[int, int] = (30000, 30072)   # rows of the first UltraChat training file used for calibration
    heldout_ultra_convs: int = 128                   # conversations from the end of the UltraChat test file
    # exclusion of training sequences that share their first tokens with a calibration / held-out sequence
    prefix_tokens: int = 64
    exclude_from_unit: int = 114                     # the exclusion applies from this unit on (the units before it replay the original cache)
    # UltraChat files
    ultra_train_file: str = "train_sft-00000-of-00003-a3ecf92756993583.parquet"
    ultra_test_file: str = "test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"
    ultra_extension_files: tuple[str, ...] = ("train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
                                              "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet")
    min_pos: int = 0                                 # positions before this one are not kept (0: every position)

    def quota(self) -> dict[str, int]:
        """tokens per unit for each source while the on-policy pools last"""
        third = self.unit_tokens // 3
        rest = self.unit_tokens - 2 * third
        return {"pile": third, "ultra": third, "cont": rest // 2, "chat": rest - rest // 2}

    def quota_after_onpolicy(self) -> dict[str, int]:
        """tokens per unit once an on-policy pool has run out: Pile and UltraChat, half each"""
        half = self.unit_tokens // 2
        return {"pile": half, "ultra": self.unit_tokens - half, "cont": 0, "chat": 0}


SOURCES = ("pile", "ultra", "cont", "chat")
MIX_08B = MixSpec()
# the training mixture and reserved rows of each model; a model without an entry cannot be calibrated or trained from the command line
MIXES: dict[str, MixSpec] = {"qwen3.5-0.8b": MIX_08B}
