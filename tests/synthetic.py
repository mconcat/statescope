"""Tiny random dictionaries in the release file format, for tests that need no checkpoint, no model and no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_molt(path: Path, d: int = 12, atoms_by_rank: dict[int, int] | None = None, seed: int = 0) -> dict[str, torch.Tensor]:
    """an MLP dictionary export with random weights; returns the tensors written"""
    atoms_by_rank = atoms_by_rank or {1: 5, 2: 4, 4: 3}
    g = torch.Generator().manual_seed(seed)
    n_atoms = sum(atoms_by_rank.values())
    order = torch.randperm(n_atoms, generator=g)                     # atom ids are interleaved across the rank tiers
    t: dict[str, torch.Tensor] = {
        "calib_norm_w_eff": 1 + 0.1 * torch.randn(d, generator=g), "calib_mu_u": 0.1 * torch.randn(d, generator=g),
        "calib_sd_u": 0.5 + torch.rand(d, generator=g), "calib_mu_y": 0.1 * torch.randn(d, generator=g), "calib_sd_y": 0.5 + torch.rand(d, generator=g),
        "e": torch.randn(n_atoms, d, generator=g), "b": 0.5 * torch.randn(n_atoms, generator=g),
        "W_skip": 0.1 * torch.randn(d, d, generator=g), "b_dec": 0.1 * torch.randn(d, generator=g),
        "score_div": 0.5 + torch.rand(n_atoms, generator=g), "thr_atom": 0.5 * torch.rand(n_atoms, generator=g),
    }
    start = 0
    for r, n_r in atoms_by_rank.items():
        t[f"idx_r{r}"] = order[start:start + n_r].sort().values
        t[f"U_r{r}"] = torch.randn(n_r, r, d, generator=g)
        t[f"V_r{r}"] = torch.randn(n_r, r, d, generator=g)
        t[f"s_r{r}"] = torch.randn(n_r, r, generator=g)
        start += n_r
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "weights.safetensors"))
    (path / "cfg.json").write_text(json.dumps({"d": d, "n_atoms": n_atoms, "rule_cfg": {"rule": "q"}, "rank_hist": {str(r): n for r, n in atoms_by_rank.items()}}))
    (path / "input.json").write_text(json.dumps({"input": "u", "eps": 1e-6}))
    return t


def write_vocab(path: Path, layer: int = 3, d: int = 12, groups: int = 10, b: int = 2, k: int = 3, seed: int = 0, orthonormal_tied: bool = False) -> dict[str, torch.Tensor]:
    """a state dictionary export with random weights. orthonormal_tied: D has orthonormal rows spanning mutually orthogonal group subspaces
    and E = D (needs groups * b <= d); then a signal lying in k groups is reconstructed exactly."""
    g = torch.Generator().manual_seed(seed)
    if orthonormal_tied:
        basis = torch.linalg.qr(torch.randn(d, d, generator=g).double())[0].float()
        D = basis[: groups * b].reshape(groups, b, d).contiguous()
        E = D.clone()
    else:
        D, E = torch.randn(groups, b, d, generator=g), torch.randn(groups, b, d, generator=g)
    t = {"D": D, "E": E, "site_mu": 0.1 * torch.randn(2, d, generator=g), "site_scale": 0.5 + torch.rand(2, generator=g), "b_dec": 0.1 * torch.randn(d, generator=g)}
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "weights.safetensors"))
    (path / "cfg.json").write_text(json.dumps({"kind": "sasa_proper_segment", "gate": "topk", "form": "comp", "k_topk": k,
                                               "sites": [f"resid_mid.{layer}", f"resid_post.{layer}"]}))
    return t
