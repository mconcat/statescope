"""Tiny random dictionaries in the release file format, for tests that need no checkpoint, no model and no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


RMS_EPS = 1e-6         # eps of the state dictionary's per-token RMS input


def write_vocab(path: Path, layer: int = 3, d: int = 12, groups: int = 10, b: int = 2, k: int = 3, seed: int = 0,
                orthonormal_tied: bool = False) -> dict[str, torch.Tensor]:
    """a state dictionary export with random weights, in the published form (cfg "form": "bsf", "input": "token_rms"; the site
    statistics stored as site_mu_rms / site_scale_rms); returns the tensors written (under the names they have in the file).
    orthonormal_tied: D has orthonormal rows spanning mutually orthogonal group subspaces and E = D (needs groups * b <= d); then a
                      signal lying in k groups is reconstructed exactly. Off: random D and E, which the reader must still read
                      by its formulas (it orthonormalises the frames itself)."""
    g = torch.Generator().manual_seed(seed)
    if orthonormal_tied:
        basis = torch.linalg.qr(torch.randn(d, d, generator=g).double())[0].float()
        D = basis[: groups * b].reshape(groups, b, d).contiguous()
        E = D.clone()
    else:
        D, E = torch.randn(groups, b, d, generator=g), torch.randn(groups, b, d, generator=g)
    t = {"D": D, "E": E, "site_mu_rms": 0.1 * torch.randn(2, d, generator=g), "site_scale_rms": 0.5 + torch.rand(2, generator=g),
         "b_dec": 0.1 * torch.randn(d, generator=g)}
    cfg: dict = {"kind": "state_dictionary", "gate": "topk", "form": "bsf", "input": "token_rms", "rms_eps": RMS_EPS, "k_topk": k,
                 "sites": [f"resid_mid.{layer}", f"resid_post.{layer}"]}
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "weights.safetensors"))
    (path / "cfg.json").write_text(json.dumps(cfg))
    return t


POST_NORM_EPS = 1e-6    # eps of Qwen3.5's RMSNorm


def write_public_molt(path: Path, layer: int = 1, n_layers: int = 3, d: int = 12, transforms_by_rank: dict[int, int] | None = None, seed: int = 0,
                      *, post_norm_stats: bool = True) -> dict[str, torch.Tensor]:
    """an MLP dictionary export of the published set (the public penalty form) with random weights: config.yaml, weights.safetensors
    with the trainer's state-dict names, and postnorm.safetensors. Returns the tensors written, "w_eff" of the post-norm file included.

    transforms_by_rank: rank -> number of transforms, largest rank first (as the published tiers are stored).
    The standardizers hold one row per model layer, as in the published files, and every row is different, so reading another
    layer's row gives other numbers. Thresholds of both signs: a negative theta leaves the `h > 0` clause of the gate to decide.
    post_norm_stats=False writes the config of a dictionary trained on the pre-norm MLP input, which the reader refuses."""
    transforms_by_rank = transforms_by_rank or {4: 3, 2: 4, 1: 5}
    g = torch.Generator().manual_seed(seed)
    n_transforms = sum(transforms_by_rank.values())
    t: dict[str, torch.Tensor] = {
        "model.e.weight": torch.randn(n_transforms, d, generator=g), "model.e.bias": 0.5 * torch.randn(n_transforms, generator=g),
        "model.nonlinearity.theta": 0.5 * torch.randn(1, n_transforms, generator=g),
        "model.input_standardizer.mean": (0.1 * torch.randn(n_layers, d, generator=g)).bfloat16(),
        "model.input_standardizer.std": (0.5 + torch.rand(n_layers, d, generator=g)).bfloat16(),
        "model.output_standardizer.mean": (0.1 * torch.randn(n_layers, d, generator=g)).bfloat16(),
        "model.output_standardizer.std": (0.5 + torch.rand(n_layers, d, generator=g)).bfloat16(),
    }
    for i, (r, n_r) in enumerate(transforms_by_rank.items()):
        t[f"model.Us.{i}"] = torch.randn(n_r, r, d, generator=g)             # write directions
        t[f"model.Vs.{i}"] = torch.randn(n_r, d, r, generator=g)             # read directions
    w_eff = 1 + 0.1 * torch.randn(d, generator=g)
    package = "crosslayer_transcoder.model"
    config = {
        "model": {"class_path": f"{package}.clt_lightning.MoltModule", "init_args": {"layer": layer, "model": {
            "class_path": f"{package}.molt.Molt", "init_args": {
                "d_acts": d, "ranks": list(transforms_by_rank),
                "nonlinearity": {"class_path": f"{package}.jumprelu.JumpReLU", "init_args": {"n_layers": 1, "d_features": n_transforms}},
                "input_standardizer": {"class_path": f"{package}.standardize.DimensionwiseInputStandardizer",
                                       "init_args": {"n_layers": n_layers, "activation_dim": d}},
                "output_standardizer": {"class_path": f"{package}.standardize.DimensionwiseOutputStandardizer",
                                        "init_args": {"n_layers": n_layers, "activation_dim": d}}}}}},
        "data": {"init_args": {"batch_size": 500}},
    }
    if post_norm_stats:
        config["data"]["init_args"]["post_norm_stats"] = True
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "weights.safetensors"))
    save_file({"w_eff": w_eff}, str(path / "postnorm.safetensors"), metadata={"eps": repr(POST_NORM_EPS)})
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {**t, "w_eff": w_eff}
