"""Comparisons of the new reader with reference outputs of the legacy repository readers (rules R0 and R1.3 of EQUIVALENCE_RULES.md).
Shared by the 0.8B tests and the width-2,560 test."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from conftest import Settings
from legacy_bridge import run_legacy
from statescope.reader import MoltOps, StateVocab, fvu

N_TOKENS = 4096
# rule R0: per 4,096 tokens
MAX_DIFFERING = 5
MAX_REL_DIFF_ELSEWHERE = 1e-4



def _dense_gate(ref: dict[str, torch.Tensor], n_atoms: int, device: str) -> torch.Tensor:
    gate = torch.zeros(ref["x"].shape[0], n_atoms, device=device)
    gate[ref["molt_gate_rows"].to(device), ref["molt_gate_atoms"].to(device)] = ref["molt_gate_values"].to(device)
    return gate


def legacy_reference(settings: Settings, layer: int, calib_dir: Path, molt_dir: Path, vocab_dir: Path) -> dict[str, torch.Tensor]:
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
