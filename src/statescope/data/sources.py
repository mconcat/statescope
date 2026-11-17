"""Readers of the four token sources: pre-tokenized Pile shards, UltraChat conversations, and the model's own text (on-policy corpus)."""
from __future__ import annotations

import glob
import os
import random
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

N_PILE_SHARDS = 64
N_PILE_HOLDOUT_SHARDS = 2        # the last two shards are never in the training order


def pile_shard_files(pile_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(pile_dir, "train-*.parquet")))
    if len(files) != N_PILE_SHARDS:
        raise FileNotFoundError(f"expected {N_PILE_SHARDS} Pile shards in {pile_dir}, found {len(files)}")
    return files


def pile_dir_in_hf_cache(hf_home: str, repo: str = "chanind/pile-uncopyrighted-qwen-3.5-1024-abbrv-2B") -> str:
    """the data folder of the pre-tokenized Pile dataset inside a Hugging Face cache"""
    found = glob.glob(os.path.join(hf_home, "hub", "datasets--" + repo.replace("/", "--"), "snapshots", "*", "data"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one snapshot of {repo} under {hf_home}, found {len(found)}")
    return found[0]


def _rows(table, row_tokens: int) -> torch.Tensor:
    """a parquet table with a list column input_ids -> (rows, row_tokens) long"""
    column = table.column("input_ids").combine_chunks()
    lengths = np.asarray(column.value_lengths())
    if not (lengths == row_tokens).all():
        raise ValueError(f"Pile row lengths != {row_tokens}: {np.unique(lengths)[:5]}")
    flat = np.asarray(column.values).astype(np.int64)
    return torch.from_numpy(flat.reshape(-1, row_tokens))


class PileTrainStream:
    """A fixed, seeded order over the (shard, row group) units of the training shards. `pos` = units already consumed (for snapshots)."""

    def __init__(self, pile_dir: str, seed: int, row_tokens: int = 1024, pos: int = 0) -> None:
        self.files = pile_shard_files(pile_dir)[: N_PILE_SHARDS - N_PILE_HOLDOUT_SHARDS]
        self.row_tokens = row_tokens
        self.order: list[tuple[int, int]] = []
        for i, path in enumerate(self.files):
            self.order += [(i, g) for g in range(pq.ParquetFile(path).metadata.num_row_groups)]
        random.Random(seed).shuffle(self.order)
        self.pos = pos
        self._open: dict[int, pq.ParquetFile] = {}

    def __iter__(self) -> Iterator[torch.Tensor]:
        while self.pos < len(self.order):
            shard, group = self.order[self.pos]
            if shard not in self._open:
                self._open[shard] = pq.ParquetFile(self.files[shard])
            self.pos += 1
            yield _rows(self._open[shard].read_row_group(group, columns=["input_ids"]), self.row_tokens)


def pile_first_rows(pile_dir: str, shard: int, n_rows: int, row_tokens: int = 1024) -> list[torch.Tensor]:
    """the first n_rows rows of one Pile shard (calibration and held-out rows)"""
    pf = pq.ParquetFile(pile_shard_files(pile_dir)[shard])
    rows, have = [], 0
    for group in range(pf.metadata.num_row_groups):
        r = _rows(pf.read_row_group(group, columns=["input_ids"]), row_tokens)
        rows.append(r)
        have += len(r)
        if have >= n_rows:
            break
    return list(torch.cat(rows)[:n_rows])


class UltraChat:
    """Whole UltraChat conversations through the tokenizer's own chat template, cut to max_tokens tokens."""

    def __init__(self, path: str, tokenizer, max_tokens: int) -> None:
        self.messages = pq.ParquetFile(path).read(columns=["messages"]).column("messages")
        self.n = len(self.messages)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def get(self, i: int) -> torch.Tensor:
        ids = self.tokenizer.apply_chat_template(self.messages[i].as_py(), tokenize=True)
        if hasattr(ids, "keys"):            # some tokenizer versions return an encoding object
            ids = ids["input_ids"]
        return torch.tensor(ids[: self.max_tokens], dtype=torch.long)


def _corpus_parts(corpus: str, split: str) -> list[dict[str, torch.Tensor]]:
    folder = os.path.join(corpus, split)
    names = sorted(f for f in os.listdir(folder) if f.startswith("part_") and f.endswith(".safetensors"))
    return [load_file(os.path.join(folder, name)) for name in names]


def corpus_tokens(corpus: str, split: str) -> list[torch.Tensor]:
    """all sequences of an on-policy corpus split in generation order"""
    seqs: list[torch.Tensor] = []
    for t in _corpus_parts(corpus, split):
        offsets = t["offsets"]
        seqs += [t["tokens"][offsets[i]:offsets[i + 1]].long() for i in range(len(offsets) - 1)]
    return seqs


def corpus_sequences(corpus: str, split: str) -> tuple[list[torch.Tensor], list[int]]:
    """All sequences of an on-policy corpus split in generation order, and for each its source row (Pile row / UltraChat prompt row)."""
    seqs: list[torch.Tensor] = []
    src: list[int] = []
    for t in _corpus_parts(corpus, split):
        offsets = t["offsets"]
        for i in range(len(offsets) - 1):
            seqs.append(t["tokens"][offsets[i]:offsets[i + 1]].long())
            src.append(int(t["src"][i]))
    return seqs, src
