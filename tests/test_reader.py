"""Reader tests. The pass/fail rules are R1.1 - R1.4 of tests/EQUIVALENCE_RULES.md (written before these tests were first run)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from conftest import REPO, Settings
from statescope.heldout import HELDOUT_SPLITS, load_heldout
from statescope.manifest import load_manifest, verify_files
from statescope.metrics import molt_pooled_fvu, vocab_heldout_fvu
from statescope.reader import MoltOps, StateVocab, fvu
from legacy_bridge import run_legacy
from reader_checks import N_TOKENS, check_molt_against_legacy, check_vocab_against_legacy, legacy_reference

MANIFEST = REPO / "release" / "qwen35-08b" / "manifest.json"

@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    torch.cuda.empty_cache()


def _load_frozen_reader(snapshot: Path):
    """the standalone reader exactly as it was released next to the checkpoints"""
    spec = importlib.util.spec_from_file_location("frozen_release_reader", snapshot / "release" / "qwen35-08b-layer-dicts" / "reader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pile_tokens(settings: Settings, layer: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_heldout(str(settings.calib_dir_08b(layer)), "pile", N_TOKENS)
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------------------------------------------- R1.1
@pytest.mark.release
def test_every_manifest_file_has_the_listed_size_and_sha256(settings: Settings):
    manifest = load_manifest(str(MANIFEST))
    assert len(manifest["layers"]) == manifest["n_layers"] == 24
    problems = verify_files(manifest, str(settings.need_release_root()), workers=4)
    assert problems == []


# ---------------------------------------------------------------------------------------------------------------- R1.2
@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.parametrize("layer", [0, 19])
def test_new_reader_is_bit_identical_to_the_frozen_standalone_reader(settings: Settings, device: str, layer: int):
    frozen = _load_frozen_reader(settings.need_legacy_snapshot())
    x, y = _pile_tokens(settings, layer, device)

    molt_dir = str(settings.layer_dir(layer, "molt"))
    new_m, old_m = MoltOps(molt_dir, device), frozen.MoltOps(molt_dir, device)
    a_new, y_new = new_m.forward(x)
    a_old, y_old = old_m.forward(x)
    assert torch.equal(a_new, a_old)
    assert torch.equal(y_new, y_old)
    firing = torch.nonzero(a_new[:64].sum(0) > 0).reshape(-1)
    for atom in firing[:: max(1, len(firing) // 24)].tolist():                 # about 24 firing atoms spread over the id range
        assert torch.equal(new_m.atom_write(x[:64], atom), old_m.atom_write(x[:64], atom))
    assert fvu(y, y_new) == frozen.fvu(y, y_old)
    del new_m, old_m, a_new, a_old, y_new, y_old
    torch.cuda.empty_cache()

    vocab_dir = str(settings.layer_dir(layer, "vocab"))
    new_v, old_v = StateVocab(vocab_dir, device), frozen.StateVocab(vocab_dir, device)
    assert torch.equal(new_v.Q, old_v.Q) and torch.equal(new_v.F, old_v.F)
    for kind, acts in (("resid_mid", x), ("resid_post", x + y)):
        site = f"{kind}.{layer}"
        g_new, c_new = new_v.encode(acts, site)
        g_old, c_old = old_v.encode(acts, site)
        assert torch.equal(g_new, g_old)
        assert torch.equal(c_new, c_old)
        assert torch.equal(new_v.decode(g_new, c_new, site), old_v.decode(g_old, c_old, site))
        group = int(g_new[0, 0])
        assert torch.equal(new_v.group_write(group, c_new[0, 0], site), old_v.group_write(group, c_old[0, 0], site))


# ---------------------------------------------------------------------------------------------------------------- R1.3
def _recorded_parity(settings: Settings, layer: int) -> dict:
    """what the original release_parity.py measured for this layer (processor, RTX 5090, 2026-11-16)"""
    path = settings.need_legacy_snapshot() / "results" / "mlp_track" / "release_parity" / "parity.json"
    return json.loads(path.read_text())[str(layer)]


@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.parametrize("layer", [0, 19])
def test_new_reader_matches_the_legacy_repository_readers(settings: Settings, device: str, layer: int):
    ref = legacy_reference(settings, layer, settings.calib_dir_08b(layer), settings.layer_dir(layer, "molt"), settings.layer_dir(layer, "vocab"))
    recorded = _recorded_parity(settings, layer)

    molt = MoltOps(str(settings.layer_dir(layer, "molt")), device)
    got = check_molt_against_legacy(molt, ref, device)
    # the numbers the original measured on the same card model
    assert got["gate_decisions_differing"] == recorded["molt"]["gate_decisions_differing"] == 0
    assert got["max_abs_gate_value_diff"] == recorded["molt"]["max_abs_gate_value_diff"] == 0.0
    assert got["max_rel_output_diff"] == recorded["molt"]["max_rel_output_diff"] == 0.0
    assert abs(got["fvu"] - recorded["molt"]["fvu_release"]) <= 1e-7
    del molt
    torch.cuda.empty_cache()

    vocab = StateVocab(str(settings.layer_dir(layer, "vocab")), device)
    got_v = check_vocab_against_legacy(vocab, ref, layer, device)
    for site, measured in got_v.items():
        if "v0" in recorded:                                                   # the original had no L0 state dictionary yet when it measured
            assert measured["tokens_with_different_group_set"] == recorded["v0"][site]["tokens_with_different_group_set"] == 0
            assert abs(measured["fvu"] - recorded["v0"][site]["fvu_release"]) <= 1e-7


# ---------------------------------------------------------------------------------------------------------------- R1.4
@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.parametrize("layer", [0, 9, 19, 23])
def test_reader_reproduces_the_manifest_fvu(settings: Settings, device: str, layer: int):
    listed = load_manifest(str(MANIFEST))["layers"][str(layer)]
    calib = str(settings.calib_dir_08b(layer))
    heldout = {split: load_heldout(calib, split) for split in HELDOUT_SPLITS}

    molt = MoltOps(str(settings.layer_dir(layer, "molt")), device)
    got = molt_pooled_fvu(molt, heldout, chunk=2048)
    assert abs(got["pooled"] - listed["molt"]["heldout_fvu"]) <= 1e-5, (got, listed["molt"]["heldout_fvu"])
    del molt
    torch.cuda.empty_cache()

    vocab = StateVocab(str(settings.layer_dir(layer, "vocab")), device)
    first = {split: (x[:N_TOKENS], y[:N_TOKENS]) for split, (x, y) in heldout.items()}
    for kind in ("resid_mid", "resid_post"):
        site = f"{kind}.{layer}"
        acts = [x if kind == "resid_mid" else x.to(device) + y.to(device) for x, y in first.values()]
        got_site = vocab_heldout_fvu(vocab, site, acts)
        assert abs(got_site - listed["vocab"]["heldout_fvu"][site]) <= 1e-5, (site, got_site, listed["vocab"]["heldout_fvu"][site])
