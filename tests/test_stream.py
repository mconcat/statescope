"""The training token stream equals the legacy stream token for token (rule R2 of EQUIVALENCE_RULES.md). CPU only."""
from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch

from conftest import Settings
from legacy_bridge import run_legacy
from statescope.config import MIX_08B, DataPaths
from statescope.earlier_design.data.mix import MixStream, StreamState, prefix_set, reference_sequences
from statescope.earlier_design.data.sources import pile_dir_in_hf_cache

N_UNITS, REPLAY_FROM, REPLAY_UNITS = 200, 120, 5


def data_paths_08b(settings: Settings) -> DataPaths:
    results = settings.need_legacy_results()
    ultrachat = os.environ.get("STATESCOPE_ULTRACHAT_DIR", "")
    if not os.path.isdir(ultrachat):
        pytest.fail("STATESCOPE_ULTRACHAT_DIR is not set or missing (see tests/local.env.example)", pytrace=False)
    return DataPaths(pile_dir=pile_dir_in_hf_cache(os.environ["HF_HOME"]), ultrachat_dir=ultrachat,
                     view_corpus=str(results / "site_train" / "corpus_as_of_L13"), full_corpus=str(results / "molt_qwen" / "corpus"))


def _digest(t: torch.Tensor) -> list:
    return [int(len(t)), hashlib.sha256(t.to(torch.int64).numpy().tobytes()).hexdigest()]


def _state_record(st: StreamState) -> dict:
    return {"pile_pos": st.pile_pos, "pile_buf": _digest(st.pile_buf.reshape(-1)), "pile_rows_used": st.pile_rows_used,
            "cursor": dict(st.cursor), "unit": st.unit, "onpolicy_exhausted": st.onpolicy_exhausted}


def _unit_record(unit, state, parts, info) -> dict:
    return {"unit": unit, "state": _state_record(state), "parts": [[name, [_digest(s) for s in seqs]] for name, seqs in parts],
            "info": json.loads(json.dumps(info, default=str))}


@pytest.mark.legacy
def test_stream_equals_the_legacy_stream_token_for_token(settings: Settings):
    paths = data_paths_08b(settings)
    out = run_legacy("stream_reference.py", ["--view-corpus", paths.view_corpus, "--full-corpus", paths.full_corpus, "--units", str(N_UNITS),
                                             "--replay-from", str(REPLAY_FROM), "--replay-units", str(REPLAY_UNITS)], settings)
    legacy = json.loads((out / "stream.json").read_text())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    stream = MixStream(tokenizer, paths, MIX_08B, extend=True)
    assert stream.check_view_prefix() == legacy["view_check"]
    reference = reference_sequences(stream)
    assert {k: [_digest(s) for s in v] for k, v in reference.items()} == legacy["reference"]
    stream.excluded_prefixes = prefix_set(reference, MIX_08B.prefix_tokens)
    assert sorted(stream.excluded_prefixes) == legacy["excluded_prefixes"]

    replay_state = None
    for i in range(N_UNITS):
        unit, state, parts, info = stream.next_unit()
        if unit == REPLAY_FROM:
            replay_state = state
        assert _unit_record(unit, state, parts, info) == legacy["units"][i], f"unit {i} differs"
    assert replay_state is not None
    stream.restore(replay_state)
    for i in range(REPLAY_UNITS):
        assert _unit_record(*stream.next_unit()) == legacy["replay"][i], f"replayed unit {REPLAY_FROM + i} differs"
