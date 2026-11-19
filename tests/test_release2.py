"""The 0.8B set laid out as published (state/L<L>/, mlp/L<L>/, MANIFEST.json).
The pass/fail rules are R8.7, R8.8 and R11.1 - R11.3 of tests/EQUIVALENCE_RULES.md (each written before its checks were first run)."""
from __future__ import annotations

import copy
import json

import pytest
import torch
import yaml
from safetensors import safe_open

from conftest import REPO, TEST_OUT, Settings
from published_checks import check_config_edits, check_manifest_hashes, check_no_internal_names, check_no_private_paths
from statescope.config import MODELS
from statescope.reader import PublicMolt, write_post_norm

pytestmark = [pytest.mark.legacy]

LAYERS = range(24)


@pytest.mark.parametrize("layer", LAYERS)
def test_manifest_hashes(layer: int, settings: Settings) -> None:
    """R8.7 (a)"""
    check_manifest_hashes(settings.need_release2(), layer)


@pytest.mark.parametrize("layer", LAYERS)
def test_published_configs_are_the_originals_with_the_stated_edit(layer: int, settings: Settings) -> None:
    """R11.1, R11.2"""
    check_config_edits(settings.need_release2(), layer, settings.vocab2_dir(layer), settings.mlp_copy_08b_dir(layer))


@pytest.mark.parametrize("layer", range(23))
def test_mlp_tensors_are_the_tested_exports(layer: int, settings: Settings) -> None:
    """R8.7 (b): layers 0-22 hold the tensors of the training exports that the reference of R8.3 reads, bit for bit (CPU)"""
    released = PublicMolt(str(settings.need_release2() / "mlp" / f"L{layer}"))
    export = settings.molt2_dir(layer)
    tested = PublicMolt(str(export / "export"), post_norm=str(export / "postnorm_rebuilt" / "postnorm.safetensors"))
    assert (released.layer, released.eps, released.ranks) == (tested.layer, tested.eps, tested.ranks)
    for name in ("w_eff", "mu_in", "sd_in", "mu_out", "sd_out", "e", "b", "theta"):
        assert torch.equal(getattr(released, name), getattr(tested, name)), name
    for a, b in zip(released.U + released.V, tested.U + tested.V):
        assert torch.equal(a, b)


@pytest.mark.model
def test_layer_23_post_norm_file_comes_from_the_language_model(settings: Settings) -> None:
    """R8.7 (c), the part R8.4 states: layer 23 comes from another training run than the export R8 read, and its published
    post-norm file holds the language model's coefficients. (R8.3 and R8.5 for layer 23 are in test_reader2.py.)"""
    from statescope.lm import load_language_model
    folder = settings.need_release2() / "mlp" / "L23"
    rebuilt = TEST_OUT / "release2" / "postnorm_L23.safetensors"
    rebuilt.parent.mkdir(parents=True, exist_ok=True)
    write_post_norm(load_language_model(MODELS["qwen3.5-0.8b"], "cpu").model, 23, str(rebuilt))
    with safe_open(str(rebuilt), framework="pt") as a, safe_open(str(folder / "postnorm.safetensors"), framework="pt") as b:
        assert torch.equal(a.get_tensor("w_eff"), b.get_tensor("w_eff"))
        assert float(a.metadata()["eps"]) == float(b.metadata()["eps"])


def test_configs_hold_the_documented_training_facts(settings: Settings) -> None:
    """R8.8 (c): what the documents say about "each layer" is in every layer's config"""
    root = settings.need_release2()
    reference = None
    for layer in LAYERS:
        cfg = json.loads((root / "state" / f"L{layer}" / "cfg.json").read_text())
        facts = (cfg["tokens"], cfg["steps"], cfg["n_units"], cfg["unit_size"], cfg["k_topk"], cfg["form"], cfg["input"], cfg["rms_eps"], cfg["init_seed"])
        assert facts == (30003200, 7325, 16384, 4, 24, "bsf", "token_rms", 1e-06, 0), layer
        config = yaml.safe_load((root / "mlp" / f"L{layer}" / "config.yaml").read_text())
        assert config["model"]["init_args"]["layer"] == layer and config["trainer"]["logger"]["init_args"]["name"] == f"L{layer}"
        assert config["model"]["init_args"]["lambda_sparsity"] == (2.5e-05 if layer == 23 else 7.5e-05), layer
        rest = copy.deepcopy(config)
        del rest["model"]["init_args"]["layer"], rest["model"]["init_args"]["lambda_sparsity"], rest["trainer"]["logger"]["init_args"]["name"]
        reference = reference or rest
        assert rest == reference, f"layer {layer}: the MLP config differs from layer 0 in more than layer, logger name and penalty"
    assert (reference["trainer"]["max_steps"], reference["data"]["init_args"]["batch_size"], reference["model"]["init_args"]["learning_rate"]) == (58600, 500, 4.0e-05)


def test_reader_fvu_on_all_heldout_tokens(settings: Settings) -> None:
    """R8.8 (a), (b): the numbers of release/qwen35-08b-v2/reader_fvu.json (devtools/measure_release2.py) against the original's table"""
    measured = json.loads((REPO / "release" / "qwen35-08b-v2" / "reader_fvu.json").read_text())["layers"]
    table = json.loads((settings.need_legacy_results() / "mlp_track" / "q08c" / "train1" / "table.json").read_text())
    worst = 0.0
    for layer in LAYERS:
        mine, theirs = measured[str(layer)], table[str(layer)]
        for site, key in ((f"resid_mid.{layer}", "mid"), (f"resid_post.{layer}", "post")):
            for which, column in (("position_0_dropped", f"raw_{key}"), ("with_position_0", f"raw_with_pos0_{key}")):
                diff = abs(mine[site][which] - theirs[column])
                worst = max(worst, diff)
                assert diff <= 1e-4, (layer, site, which, mine[site][which], theirs[column])
        for which in ("position_0_dropped", "with_position_0"):
            assert 0.0 < mine["mlp"][which] < 1.0, (layer, which)
    out = TEST_OUT / "measured" / "release2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reader_fvu_vs_table.json").write_text(json.dumps({"largest_difference": worst, "limit": 1e-4}) + "\n")


def test_no_private_paths(settings: Settings) -> None:
    """R8.7 (d)"""
    check_no_private_paths(settings.need_release2(), LAYERS, ("MANIFEST.json",))


def test_no_internal_names(settings: Settings) -> None:
    """R11.3; the staged folder holds the model card as README.md"""
    check_no_internal_names(settings.need_release2(), LAYERS, ("MANIFEST.json", "README.md"))
