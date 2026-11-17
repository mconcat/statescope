"""Readers of the second 0.8B design (tied state dictionary with per-token RMS input; MLP dictionary in the public penalty form).
The pass/fail rules are R8.1 - R8.5 of tests/EQUIVALENCE_RULES.md (written before these tests were first run)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from conftest import TEST_OUT, Settings
from legacy_bridge import run_legacy
from reader_checks import MAX_DIFFERING, MAX_REL_DIFF_ELSEWHERE
from statescope.config import MODELS
from statescope.heldout import HELDOUT_SPLITS, load_heldout
from statescope.metrics import vocab_heldout_fvu
from statescope.reader import PublicMolt, StateVocab, write_post_norm

pytestmark = [pytest.mark.legacy, pytest.mark.gpu, pytest.mark.model]

LAYERS = range(24)
N_TOKENS = 4096        # per held-out split: decisions
N_TENSOR = 1024        # per held-out split: reconstructed tensors
MEASURED = TEST_OUT / "measured" / "reader2"


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def post_norm_dir(settings: Settings) -> Path:
    """R8.4: the RMSNorm coefficients of every layer, rebuilt from the language model; bit-identical to the original project's rebuilt
    copies where those exist."""
    from statescope.lm import load_language_model
    out = TEST_OUT / "design2" / "postnorm"
    out.mkdir(parents=True, exist_ok=True)
    lm = load_language_model(MODELS["qwen3.5-0.8b"], "cpu")
    compared = []
    for layer in LAYERS:
        mine = out / f"L{layer}.safetensors"
        write_post_norm(lm.model, layer, str(mine))
        theirs = settings.molt2_dir(layer) / "postnorm_rebuilt" / "postnorm.safetensors"
        if theirs.is_file():
            with safe_open(str(mine), framework="pt") as a, safe_open(str(theirs), framework="pt") as b:
                assert torch.equal(a.get_tensor("w_eff"), b.get_tensor("w_eff")), f"layer {layer}: w_eff differs from {theirs}"
                assert float(a.metadata()["eps"]) == float(b.metadata()["eps"]), f"layer {layer}: eps differs from {theirs}"
            compared.append(layer)
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "post_norm.json").write_text(json.dumps({"layers_compared_with_the_original_rebuilt_copy": compared}) + "\n")
    return out


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def _reference(settings: Settings, layer: int, post_norm_dir: Path) -> tuple[dict[str, torch.Tensor], dict, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """reference tensors and info of one layer (cached run inside the second snapshot), and the same held-out tokens loaded by the new code"""
    out = run_legacy("reader2_reference.py", [
        "--layer", str(layer), "--calib-dir", str(settings.calib2_dir(layer)), "--vocab-dir", str(settings.vocab2_dir(layer)),
        "--molt-dir", str(settings.molt2_dir(layer) / "export"), "--post-norm", str(post_norm_dir / f"L{layer}.safetensors"),
        "--n", str(N_TOKENS), "--n-tensor", str(N_TENSOR)], settings, snapshot=settings.need_snapshot2())
    info = json.loads((out / "info.json").read_text())
    held = {s: load_heldout(str(settings.calib2_dir(layer)), s, N_TOKENS) for s in HELDOUT_SPLITS}
    assert list(info["splits"]) == list(HELDOUT_SPLITS)
    for s, (x, y) in held.items():
        assert {"x": _sha256(x), "y": _sha256(y)} == info["input_sha256"][s], f"layer {layer}, split {s}: the two sides read different tokens"
    return load_file(str(out / "reference.safetensors")), info, held


def _save(layer: int, name: str, measured: dict) -> None:
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / f"L{layer:02d}_{name}.json").write_text(json.dumps(measured, indent=1) + "\n")


@pytest.mark.parametrize("layer", LAYERS)
def test_state_reader(layer: int, settings: Settings, device: str, post_norm_dir: Path) -> None:
    ref, info, held = _reference(settings, layer, post_norm_dir)
    vocab = StateVocab(str(settings.vocab2_dir(layer)), device)
    assert (vocab.cfg["form"], vocab.cfg["input"]) == ("bsf", "token_rms")
    trainer = json.loads((settings.vocab2_dir(layer).parents[1] / "layer_recipe_done.json").read_text())["vocab_heldout_fvu"]["B1s0"]
    measured: dict = {}
    for j, site in enumerate(vocab.sites):
        kind = site.split(".")[0]
        acts = [(x if j == 0 else x + y).to(device) for x, y in held.values()]
        per_split = {}
        sse_raw = sst_raw = 0.0
        for s, x in zip(held, acts):
            groups, codes = vocab.encode(x, site)
            raw = vocab.decode(groups, codes, site, rms=vocab.token_rms(x))
            ref_groups = ref[f"vocab_{kind}_{s}_groups"].to(device).long()
            ref_raw = ref[f"vocab_{kind}_{s}_raw"].to(device)
            differing = (groups.sort(1).values != ref_groups.sort(1).values).any(1)
            clean = ~differing[:N_TENSOR]
            per_split[s] = {
                "tokens_with_different_group_set": int(differing.sum()),
                "max_rel_raw_recon_diff_same_set": float((raw[:N_TENSOR] - ref_raw)[clean].abs().max() / ref_raw.abs().max()),
            }
            assert per_split[s]["tokens_with_different_group_set"] <= MAX_DIFFERING, (site, s, per_split[s])            # R8.1 (a)
            assert per_split[s]["max_rel_raw_recon_diff_same_set"] <= MAX_REL_DIFF_ELSEWHERE, (site, s, per_split[s])  # R8.1 (b)
            sse_raw += float(((x - raw).double() ** 2).sum())
            sst_raw += float(((x - x.mean(0)).double() ** 2).sum())
        own = vocab_heldout_fvu(vocab, site, acts)
        measured[site] = {"per_split": per_split, "fvu_own_units": own, "fvu_own_units_trainer": trainer[site],
                          "fvu_own_units_reference_reader": info["vocab_fvu_own_units"][site], "fvu_raw_units": sse_raw / sst_raw}
        assert abs(own - trainer[site]) <= 1e-5, (site, measured[site])                                               # R8.2
    # R8.5: one token's group writes + mean + bias = decode
    site = vocab.sites[0]
    x0 = held["pile"][0][:1].to(device)
    groups, codes = vocab.encode(x0, site)
    rms = vocab.token_rms(x0)
    whole = sum(vocab.group_write(int(g), codes[0, k], site, rms=rms[0]) for k, g in enumerate(groups[0].tolist())) \
        + (vocab.mu[0] + vocab.scale[0] * vocab.b_dec) * rms[0]
    decoded = vocab.decode(groups, codes, site, rms=rms)[0]
    measured["max_rel_diff_sum_of_group_writes_vs_decode_token0"] = float((whole - decoded).abs().max() / decoded.abs().max())
    _save(layer, "vocab", measured)
    assert measured["max_rel_diff_sum_of_group_writes_vs_decode_token0"] <= 1e-6, measured


@pytest.mark.parametrize("layer", LAYERS)
def test_molt_reader(layer: int, settings: Settings, device: str, post_norm_dir: Path) -> None:
    ref, info, held = _reference(settings, layer, post_norm_dir)
    molt = PublicMolt(str(settings.molt2_dir(layer) / "export"), device, post_norm=str(post_norm_dir / f"L{layer}.safetensors"))
    assert (molt.layer, molt.n_transforms) == (layer, info["molt_n_transforms"])
    measured: dict = {"per_split": {}}
    sse = sst = n_active = 0.0
    for s, (x, y) in held.items():
        x, y = x.to(device), y.to(device)
        g, y_hat = molt.forward(x)
        g_ref = torch.zeros_like(g)
        g_ref[ref[f"molt_{s}_rows"].to(device).long(), ref[f"molt_{s}_transforms"].to(device).long()] = ref[f"molt_{s}_values"].to(device)
        y_ref = ref[f"molt_{s}_y"].to(device)
        differing = (g > 0) != (g_ref > 0)
        clean = ~differing.any(1)[:N_TENSOR]
        measured["per_split"][s] = {
            "gate_decisions_differing": int(differing.sum()),
            "max_rel_gate_value_diff": float((g - g_ref).abs().max() / g_ref.abs().max()),
            "max_rel_output_diff_clean_tokens": float((y_hat[:N_TENSOR] - y_ref)[clean].abs().max() / y_ref.abs().max()),
        }
        assert measured["per_split"][s]["gate_decisions_differing"] <= MAX_DIFFERING, (s, measured["per_split"][s])                  # R8.3 (a)
        assert measured["per_split"][s]["max_rel_output_diff_clean_tokens"] <= MAX_REL_DIFF_ELSEWHERE, (s, measured["per_split"][s])  # R8.3 (b)
        sse += float(((y - y_hat).double() ** 2).sum())
        sst += float(((y - y.mean(0)).double() ** 2).sum())
        n_active += float((g > 0).sum())
    measured["fvu_pooled_over_split_variances"] = sse / sst      # a new measurement on the calibration folder's held-out tokens (see R8)
    measured["transforms_on_per_token"] = n_active / (N_TOKENS * len(held))
    # R8.5: the firing transforms' writes + the output mean = forward (first 64 pile tokens)
    x64 = held["pile"][0][:64].to(device)
    g64, y64 = molt.forward(x64)
    firing = torch.nonzero(g64.sum(0) > 0).reshape(-1).tolist()
    total = sum(molt.transform_write(x64, t) for t in firing) + molt.mu_out
    measured["max_rel_diff_sum_of_transform_writes_vs_forward"] = float((total - y64).abs().max() / y64.abs().max())
    _save(layer, "molt", measured)
    assert measured["max_rel_diff_sum_of_transform_writes_vs_forward"] <= 1.5e-6, measured
