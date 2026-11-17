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

MANIFEST = REPO / "release" / "qwen35-08b" / "manifest.json"
N_TOKENS = 4096
# rule R0: per 4,096 tokens
MAX_DIFFERING = 5
MAX_REL_DIFF_ELSEWHERE = 1e-4


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


def _dense_gate(ref: dict[str, torch.Tensor], n_atoms: int, device: str) -> torch.Tensor:
    gate = torch.zeros(ref["x"].shape[0], n_atoms, device=device)
    gate[ref["molt_gate_rows"].to(device), ref["molt_gate_atoms"].to(device)] = ref["molt_gate_values"].to(device)
    return gate


def _legacy_reference(settings: Settings, layer: int, calib_dir: Path, molt_dir: Path, vocab_dir: Path) -> dict[str, torch.Tensor]:
    out = run_legacy("reader_reference.py", ["--layer", str(layer), "--calib-dir", str(calib_dir), "--molt-dir", str(molt_dir),
                                             "--vocab-dir", str(vocab_dir), "--n", str(N_TOKENS)], settings)
    return load_file(str(out / "reference.safetensors"))


def check_molt_against_legacy(molt: MoltOps, ref: dict[str, torch.Tensor], device: str) -> dict[str, float]:
    """rule R0 (a)+(b) and the write-sum identity for the MLP dictionary; returns the measured numbers"""
    x = ref["x"].to(device)
    a_old, y_old = _dense_gate(ref, molt.n_atoms, device), ref["molt_y"].to(device)
    a_new, y_new = molt.forward(x)
    differing = (a_old > 0) != (a_new > 0)
    clean_tokens = ~differing.any(1)
    measured = {
        "gate_decisions_differing": int(differing.sum()),
        "max_abs_gate_value_diff": float((a_old - a_new).abs().max()),
        "max_rel_output_diff": float((y_old - y_new).abs().max() / y_old.abs().max()),
        "max_rel_output_diff_clean_tokens": float((y_old - y_new)[clean_tokens].abs().max() / y_old.abs().max()),
    }
    assert measured["gate_decisions_differing"] <= MAX_DIFFERING, measured
    assert measured["max_rel_output_diff_clean_tokens"] <= MAX_REL_DIFF_ELSEWHERE, measured
    # every firing atom's write + skip + bias = forward (first 64 tokens)
    firing = torch.nonzero(a_new[:64].sum(0) > 0).reshape(-1).tolist()
    total = sum(molt.atom_write(x[:64], atom) for atom in firing)
    z = molt.standardize_input(x[:64])
    full = (z @ molt.W_skip + molt.b_dec) * molt.sd_y + molt.mu_y + total
    measured["max_rel_diff_sum_of_atom_writes_vs_forward"] = float((full - y_new[:64]).abs().max() / y_new[:64].abs().max())
    assert measured["max_rel_diff_sum_of_atom_writes_vs_forward"] <= 1.5e-6, measured
    measured["fvu"] = fvu(ref["y"].to(device), y_new)
    return measured


def check_vocab_against_legacy(vocab: StateVocab, ref: dict[str, torch.Tensor], layer: int, device: str) -> dict[str, dict[str, float]]:
    """rule R0 (a)+(b) for the state dictionary, the float limits of R1.3 and the write-sum identity; returns the measured numbers per site"""
    x, y = ref["x"].to(device), ref["y"].to(device)
    result = {}
    for kind, acts in (("resid_mid", x), ("resid_post", x + y)):
        site = f"{kind}.{layer}"
        g_old, c_old, r_old = (ref[f"vocab_{kind}_{name}"].to(device) for name in ("groups", "codes", "recon"))
        g_new, c_new = vocab.encode(acts, site)
        r_new = vocab.decode(g_new, c_new, site)
        same_set = (g_old.sort(1).values == g_new.sort(1).values).all(1)
        same_order = (g_old == g_new).all(1)
        i = vocab.sites.index(site)
        writes = sum(vocab.group_write(int(g), c_new[0, j], site) for j, g in enumerate(g_new[0].tolist()))
        whole = writes + vocab.mu[i] + vocab.scale[i] * vocab.b_dec
        measured = {
            "tokens_with_different_group_set": int((~same_set).sum()),
            "max_abs_code_diff_same_order": float((c_old - c_new)[same_order].abs().max()),
            "max_rel_recon_diff_same_set": float((r_old - r_new)[same_set].abs().max() / acts.abs().max()),
            "max_rel_diff_sum_of_group_writes_vs_decode_token0": float((whole - r_new[0]).abs().max() / r_new[0].abs().max()),
            "fvu": fvu(acts, r_new),
        }
        assert measured["tokens_with_different_group_set"] <= MAX_DIFFERING, (site, measured)
        assert measured["max_rel_recon_diff_same_set"] <= MAX_REL_DIFF_ELSEWHERE, (site, measured)
        assert measured["max_abs_code_diff_same_order"] <= 1.2e-5, (site, measured)
        assert measured["max_rel_recon_diff_same_set"] <= 5e-7, (site, measured)
        assert measured["max_rel_diff_sum_of_group_writes_vs_decode_token0"] <= 1e-6, (site, measured)
        result[site] = measured
    return result


@pytest.mark.release
@pytest.mark.legacy
@pytest.mark.gpu
@pytest.mark.parametrize("layer", [0, 19])
def test_new_reader_matches_the_legacy_repository_readers(settings: Settings, device: str, layer: int):
    ref = _legacy_reference(settings, layer, settings.calib_dir_08b(layer), settings.layer_dir(layer, "molt"), settings.layer_dir(layer, "vocab"))
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
