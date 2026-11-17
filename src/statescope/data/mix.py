"""The training token stream: units of `unit_tokens` tokens mixed from four sources in a fixed order.

Per unit, in this order: Pile rows, UltraChat conversations, the model's own continuations ("cont"), the model's own chat answers
("chat"). The trainer's shuffle buffer mixes them afterwards. Nothing here touches the GPU or torch's random generator, so the stream is
a pure function of the files and the `MixSpec`.

When an on-policy pool runs out, later units are Pile and UltraChat only (half each); nothing is ever repeated.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import torch

from statescope.config import SOURCES, DataPaths, MixSpec
from statescope.data.sources import PileTrainStream, UltraChat, corpus_sequences, corpus_tokens, pile_first_rows

Part = tuple[str, list[torch.Tensor]]      # (source name, token sequences)


def sequence_md5(tokens: torch.Tensor, n: int | None = None) -> str:
    """md5 of the first n token ids (all if n is None) as int64 bytes"""
    tokens = tokens if n is None else tokens[:n]
    return hashlib.md5(tokens.to(torch.int64).numpy().tobytes()).hexdigest()


@dataclass
class StreamState:
    """Everything needed to continue the stream from a unit boundary."""
    pile_pos: int = 0                                   # Pile (shard, row group) units consumed
    pile_buf: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1024, dtype=torch.long))   # Pile rows read but not yet used (rows, row tokens)
    pile_rows_used: int = 0
    cursor: dict[str, int] = field(default_factory=lambda: {"cont": 0, "chat": 0, "ultra": 0})
    unit: int = 0
    onpolicy_exhausted: bool = False

    def copy(self) -> "StreamState":
        return StreamState(self.pile_pos, self.pile_buf.clone(), self.pile_rows_used, dict(self.cursor), self.unit, self.onpolicy_exhausted)


class MixStream:
    """`next_unit()` returns (unit index, state before the unit, parts, info); parts = [(source, [token tensors])] in SOURCES order."""

    def __init__(self, tokenizer, paths: DataPaths, spec: MixSpec, extend: bool = True) -> None:
        """extend=False restricts the stream to what the original 30M-token cache used (it fails where those pools end)."""
        self.tokenizer, self.paths, self.spec, self.extend = tokenizer, paths, spec, bool(extend)
        self.excluded_prefixes: set[str] = set()
        self.pile = PileTrainStream(paths.pile_dir, spec.pile_seed, spec.pile_row_tokens)
        self.pile.order = [u for u in self.pile.order if u[0] < spec.pile_calib_shard]
        self._pile_iter = None
        # on-policy pools: the full corpus; its first sequences are the view's (see check_view_prefix)
        self.pool = {k: corpus_sequences(paths.full_corpus if extend else paths.view_corpus, f"{k}_train") for k in ("cont", "chat")}
        self.reserve_start, self.reserve_end = spec.onpol_calib_start, spec.onpol_calib_start + spec.calib_onpol_seqs
        self.ultra_calib_rows = set(range(*spec.ultra_calib_rows))
        self.ultra_train = UltraChat(os.path.join(paths.ultrachat_dir, spec.ultra_train_file), tokenizer, spec.ultra_max_tokens)
        self.ultra_extension: list[UltraChat] | None = None
        self.ultra_floor = spec.ultra_calib_rows[1]
        self.quota, self.quota_after = spec.quota(), spec.quota_after_onpolicy()
        self.state = StreamState(pile_buf=torch.zeros(0, spec.pile_row_tokens, dtype=torch.long))

    # ------------------------------------------------------------------------------------------------ state
    def snapshot(self) -> StreamState:
        return self.state.copy()

    def restore(self, state: StreamState) -> None:
        self.state = state.copy()
        self._pile_iter = None

    def check_view_prefix(self) -> dict:
        """The full corpus must start with the view's sequences, so that the early part of the stream is what calibration saw."""
        out = {}
        for kind in ("cont", "chat"):
            view, view_src = corpus_sequences(self.paths.view_corpus, f"{kind}_train")
            full, full_src = self.pool[kind]
            n = min(len(view), len(full))
            same = all(torch.equal(view[i], full[i]) for i in range(n)) and view_src[:n] == full_src[:n]
            out[kind] = {"view_seqs": len(view), "full_seqs": len(full), "compared": n, "identical": bool(same)}
            if not same:
                raise RuntimeError(f"{kind}: the full corpus does not start with the view's sequences")
        return out

    # ------------------------------------------------------------------------------------------------ sources
    def _pile_rows(self, n_rows: int) -> list[torch.Tensor]:
        s = self.state
        if self._pile_iter is None:
            self.pile.pos = s.pile_pos
            self._pile_iter = iter(self.pile)
        buf = s.pile_buf
        while buf.shape[0] < n_rows:
            buf = torch.cat([buf, next(self._pile_iter)])
            s.pile_pos = self.pile.pos
        rows, s.pile_buf = buf[:n_rows], buf[n_rows:].clone()
        s.pile_rows_used += n_rows
        return list(rows)

    def ultra_row(self, k: int) -> tuple[str, int]:
        """(file, row) of the k-th UltraChat training conversation: the first training file from its end down to the calibration rows,
        then the extension files in file order"""
        n0 = self.ultra_train.n
        row = n0 - 1 - k
        if row >= self.ultra_floor:
            return self.spec.ultra_train_file, row
        if not self.extend:
            raise RuntimeError("UltraChat train rows reached the calibration rows")
        j = k - (n0 - self.ultra_floor)
        if self.ultra_extension is None:
            self.ultra_extension = [UltraChat(os.path.join(self.paths.ultrachat_dir, f), self.tokenizer, self.spec.ultra_max_tokens)
                                    for f in self.spec.ultra_extension_files]
        for name, convs in zip(self.spec.ultra_extension_files, self.ultra_extension):
            if j < convs.n:
                return name, j
            j -= convs.n
        raise RuntimeError("UltraChat training files exhausted")

    def ultra_get(self, k: int) -> torch.Tensor:
        name, row = self.ultra_row(k)
        if name == self.spec.ultra_train_file:
            return self.ultra_train.get(row)
        assert self.ultra_extension is not None
        return self.ultra_extension[self.spec.ultra_extension_files.index(name)].get(row)

    def onpolicy_index(self, kind: str, i: int) -> int | None:
        """The next usable on-policy index at or after i (skips the calibration reserve and, for chat, sequences whose prompt is an
        UltraChat calibration row), or None when the pool is exhausted."""
        seqs, src = self.pool[kind]
        while True:
            if self.reserve_start <= i < self.reserve_end:
                if not self.extend:
                    return None
                i = self.reserve_end
            if i >= len(seqs):
                return None
            if i >= self.reserve_end and kind == "chat" and src[i] in self.ultra_calib_rows:
                i += 1
                continue
            return i

    def excluded(self, tokens: torch.Tensor) -> bool:
        """From unit `exclude_from_unit` on, a sequence sharing its first `prefix_tokens` tokens with a calibration or held-out sequence is dropped."""
        n = self.spec.prefix_tokens
        return self.state.unit >= self.spec.exclude_from_unit and len(tokens) >= n and sequence_md5(tokens, n) in self.excluded_prefixes

    # ------------------------------------------------------------------------------------------------ one unit
    def next_unit(self) -> tuple[int, StreamState, list[Part], dict]:
        before = self.snapshot()
        s, spec = self.state, self.spec
        quota = self.quota_after if s.onpolicy_exhausted else self.quota
        parts: list[Part] = []
        info: dict = {"shard": s.unit}
        n_excluded = {k: 0 for k in SOURCES}
        n_rows = -(-quota["pile"] // (spec.pile_row_tokens - spec.min_pos))
        rows = self._pile_rows(n_rows)
        if self.excluded_prefixes:
            kept = [r for r in rows if not self.excluded(r)]
            n_excluded["pile"] = len(rows) - len(kept)
            rows = kept
        parts.append(("pile", rows))
        info["pile_rows"] = [s.pile_rows_used - n_rows, s.pile_rows_used]
        k0 = k = s.cursor["ultra"]
        got, convs = 0, []
        while got < quota["ultra"]:
            c = self.ultra_get(k)
            k += 1
            if self.excluded_prefixes and self.excluded(c):
                n_excluded["ultra"] += 1
                continue
            convs.append(c)
            got += max(0, len(c) - spec.min_pos)
        parts.append(("ultra", convs))
        info["ultra_k"] = [k0, k]
        info["ultra_rows"] = [self.ultra_row(k0), self.ultra_row(k - 1)]
        s.cursor["ultra"] = k
        for kind in ("cont", "chat"):
            seqs, _src = self.pool[kind]
            got, take = 0, []
            i = s.cursor[kind]
            while got < quota[kind]:
                j = self.onpolicy_index(kind, i)
                if j is None:
                    if not self.extend:
                        raise RuntimeError(f"{kind} sequences exhausted at {i} (the calibration reserve starts there)")
                    s.onpolicy_exhausted = True
                    info.setdefault("exhausted", []).append(kind)
                    break
                i = j + 1
                if self.excluded_prefixes and self.excluded(seqs[j]):
                    n_excluded[kind] += 1
                    continue
                take.append(j)
                got += max(0, len(seqs[j]) - spec.min_pos)
            parts.append((kind, [seqs[j] for j in take]))
            info[f"{kind}_idx"] = [take[0], take[-1] + 1] if take else None
            s.cursor[kind] = i
        info["per_src_tokens"] = {name: int(sum(len(t) for t in seqs_)) for name, seqs_ in parts}
        if any(n_excluded.values()):
            info["excluded_prefix64"] = n_excluded
        s.unit += 1
        return before.unit, before, parts, info


def reference_sequences(stream: MixStream) -> dict[str, list[torch.Tensor]]:
    """Every calibration and held-out sequence (the rows the training stream must stay away from), as token tensors."""
    spec, paths = stream.spec, stream.paths
    ultra_test = UltraChat(os.path.join(paths.ultrachat_dir, spec.ultra_test_file), stream.tokenizer, spec.ultra_max_tokens)
    return {
        "calib_pile": pile_first_rows(paths.pile_dir, spec.pile_calib_shard, spec.calib_pile_rows, spec.pile_row_tokens),
        "calib_ultra": [stream.ultra_train.get(i) for i in range(*spec.ultra_calib_rows)],
        "calib_cont": stream.pool["cont"][0][stream.reserve_start:stream.reserve_end],
        "calib_chat": stream.pool["chat"][0][stream.reserve_start:stream.reserve_end],
        "heldout_pile": pile_first_rows(paths.pile_dir, spec.pile_heldout_shard, spec.heldout_pile_rows, spec.pile_row_tokens),
        "heldout_ultra": [ultra_test.get(ultra_test.n - 1 - k) for k in range(spec.heldout_ultra_convs)],
        "heldout_cont": corpus_tokens(paths.full_corpus, "cont_heldout"),
        "heldout_chat": corpus_tokens(paths.full_corpus, "chat_heldout"),
    }


def prefix_set(reference: dict[str, list[torch.Tensor]], prefix_tokens: int) -> set[str]:
    return {sequence_md5(s, prefix_tokens) for seqs in reference.values() for s in seqs if len(s) >= prefix_tokens}


def training_stream(tokenizer, paths: DataPaths, spec: MixSpec) -> MixStream:
    """The stream as the trainers use it: full corpus, view-prefix check, and the calibration / held-out prefixes excluded."""
    stream = MixStream(tokenizer, paths, spec, extend=True)
    stream.check_view_prefix()
    stream.excluded_prefixes = prefix_set(reference_sequences(stream), spec.prefix_tokens)
    return stream
