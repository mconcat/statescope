"""Width 2,560: the reader on the existing Qwen3.5-4B layer-16 pilot exports (rule R1.6 of EQUIVALENCE_RULES.md)."""
from __future__ import annotations

import pytest
import torch

from conftest import Settings
from reader_checks import check_molt_against_legacy, check_vocab_against_legacy, legacy_reference
from statescope.reader import MoltOps, StateVocab

LAYER = 16


@pytest.mark.legacy
@pytest.mark.gpu
def test_reader_reads_the_4b_pilot_and_matches_the_legacy_readers(settings: Settings, device: str):
    pilot = settings.need_legacy_results() / "mlp_track" / "q4b"
    calib_dir = pilot / "calib" / f"calib_q4b_L{LAYER}"
    molt_dir = pilot / "train" / f"L{LAYER}" / "M1w" / "mlp_M1w" / "s0" / "M1w"
    vocab_dir = pilot / "train" / f"L{LAYER}" / "V0" / "vocab_V0s0" / "final"
    ref = legacy_reference(settings, LAYER, calib_dir, molt_dir, vocab_dir)

    molt = MoltOps(str(molt_dir), device)
    assert (molt.d, molt.n_atoms) == (2560, 41484)
    measured = {"molt": check_molt_against_legacy(molt, ref, device)}
    del molt
    torch.cuda.empty_cache()

    vocab = StateVocab(str(vocab_dir), device)
    assert (vocab.G, vocab.b, vocab.d, vocab.k) == (16384, 4, 2560, 24)
    measured["vocab"] = check_vocab_against_legacy(vocab, ref, LAYER, device)
    print(measured)
