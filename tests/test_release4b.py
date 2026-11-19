"""The Qwen3.5-4B set laid out as published (state/L<L>/, mlp/L<L>/, MANIFEST.json).
The pass/fail rules are R10.6, R10.7, R10.8 and R11.1 - R11.3 of tests/EQUIVALENCE_RULES.md (each written before its checks were first run)."""
from __future__ import annotations

import copy
import json

import pytest
import torch
import yaml

from conftest import REPO, TEST_OUT, Settings
from published_checks import check_config_edits, check_manifest_hashes, check_no_internal_names, check_no_private_paths
from statescope.config import MODELS
from statescope.reader import PublicMolt

pytestmark = [pytest.mark.legacy]

MODEL = MODELS["qwen3.5-4b"]
LAYERS = range(MODEL.n_layers)
READER_FVU = REPO / "release" / "qwen35-4b" / "reader_fvu.json"


@pytest.mark.parametrize("layer", LAYERS)
def test_manifest_hashes(layer: int, settings: Settings) -> None:
    """R10.6 (a)"""
    check_manifest_hashes(settings.need_release4b(), layer)


@pytest.mark.parametrize("layer", LAYERS)
def test_published_configs_are_the_originals_with_the_stated_edit(layer: int, settings: Settings) -> None:
    """R11.1, R11.2"""
    check_config_edits(settings.need_release4b(), layer, settings.vocab4b_dir(layer), settings.mlp_copy_4b_dir(layer))


@pytest.mark.parametrize("layer", LAYERS)
def test_mlp_tensors_are_the_tested_exports(layer: int, settings: Settings) -> None:
    """R10.6 (b): the published copy holds the tensors R10.3 read, bit for bit (CPU)"""
    released = PublicMolt(str(settings.need_release4b() / "mlp" / f"L{layer}"))
    tested = PublicMolt(str(settings.molt4b_dir(layer)))
    assert (released.layer, released.eps, released.ranks) == (tested.layer, tested.eps, tested.ranks)
    for name in ("w_eff", "mu_in", "sd_in", "mu_out", "sd_out", "e", "b", "theta"):
        assert torch.equal(getattr(released, name), getattr(tested, name)), name
    for a, b in zip(released.U + released.V, tested.U + tested.V):
        assert torch.equal(a, b)


def test_configs_hold_the_documented_training_facts(settings: Settings) -> None:
    """R10.6 (c), with its correction: the standardizer tables have 24 rows at layers 0-23 and 32 rows at layers 24-31"""
    root = settings.need_release4b()
    reference = None
    for layer in LAYERS:
        cfg = json.loads((root / "state" / f"L{layer}" / "cfg.json").read_text())
        facts = (cfg["tokens"], cfg["steps"], cfg["n_units"], cfg["unit_size"], cfg["k_topk"], cfg["form"], cfg["input"], cfg["rms_eps"],
                 cfg["init_seed"], cfg["d_in"])
        assert facts == (30003200, 7325, 16384, 4, 24, "bsf", "token_rms", 1e-06, 0, MODEL.d_model), layer
        config = yaml.safe_load((root / "mlp" / f"L{layer}" / "config.yaml").read_text())
        args = config["model"]["init_args"]
        assert args["layer"] == layer and config["trainer"]["logger"]["init_args"]["name"] == f"L{layer}"
        rows = 24 if layer < 24 else 32
        standardizers = (args["model"]["init_args"]["input_standardizer"], args["model"]["init_args"]["output_standardizer"])
        assert [s["init_args"]["n_layers"] for s in standardizers] == [rows, rows], layer
        rest = copy.deepcopy(config)
        del rest["model"]["init_args"]["layer"], rest["trainer"]["logger"]["init_args"]["name"]
        for name in ("input_standardizer", "output_standardizer"):
            del rest["model"]["init_args"]["model"]["init_args"][name]["init_args"]["n_layers"]
        reference = reference or rest
        assert rest == reference, f"layer {layer}: the MLP config differs from layer 0 in more than layer, logger name and standardizer rows"
    args = reference["model"]["init_args"]
    assert (args["lambda_sparsity"], reference["trainer"]["max_steps"], reference["data"]["init_args"]["batch_size"], args["learning_rate"],
            args["model"]["init_args"]["N"]) == (3e-05, 58600, 500, 4.0e-05, 320)


def test_reader_fvu_on_all_heldout_tokens(settings: Settings) -> None:
    """R10.7: the numbers of release/qwen35-4b/reader_fvu.json (devtools/measure_release2.py) against the original's check records"""
    measured = json.loads(READER_FVU.read_text())["layers"]
    checks = settings.need_legacy_results() / "mlp_track" / "q4b" / "train3" / "checks"
    worst = 0.0
    for layer in LAYERS:
        mine = measured[str(layer)]
        for which, record_name, count in (("position_0_dropped", "fvu_all_nopos0", "position_0_dropped"), ("with_position_0", "fvu_all", "all")):
            record = json.loads((checks / f"L{layer}.B1.{record_name}.json").read_text())
            assert {split: n[count] for split, n in mine["tokens_per_split"].items()} == record["tokens_per_split"], (layer, which)
            for site in (f"resid_mid.{layer}", f"resid_post.{layer}"):
                diff = abs(mine[site][which] - record["raw"][site])
                worst = max(worst, diff)
                assert diff <= 1e-4, (layer, site, which, mine[site][which], record["raw"][site])
            assert 0.0 < mine["mlp"][which] < 1.0, (layer, which)
    out = TEST_OUT / "measured" / "release4b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reader_fvu_vs_checks.json").write_text(json.dumps({"largest_difference": worst, "limit": 1e-4}) + "\n")


def test_no_private_paths(settings: Settings) -> None:
    """R10.6 (d)"""
    check_no_private_paths(settings.need_release4b(), LAYERS, ("MANIFEST.json", "ALWAYS_ON.json"))


def test_no_internal_names(settings: Settings) -> None:
    """R11.3; the staged folder holds the model card as README.md"""
    check_no_internal_names(settings.need_release4b(), LAYERS, ("MANIFEST.json", "ALWAYS_ON.json", "README.md"))


@pytest.mark.gpu
@pytest.mark.model
def test_reading_a_text_with_the_4b_model(settings: Settings, device: str) -> None:
    """R10.8: the model card's usage lines, once, on the 4B model and the published folder"""
    from statescope.lm import load_language_model
    from statescope.reader import StateVocab, capture, fvu
    layer = 16
    root = settings.need_release4b()
    lm = load_language_model(MODEL, device)
    input_ids = torch.tensor([lm.tokenizer("The committee met on Thursday to review the budget.")["input_ids"]], device=device)
    acts = capture(lm.model, layer, input_ids)
    vocab = StateVocab(str(root / "state" / f"L{layer}"), device)
    molt = PublicMolt(str(root / "mlp" / f"L{layer}"), device)
    measured = {"tokens": int(input_ids.shape[1]), "text": "The committee met on Thursday to review the budget."}
    for kind in ("resid_mid", "resid_post"):
        assert acts[kind].shape == (input_ids.shape[1] - 1, MODEL.d_model)
        measured[f"fvu_state_{kind}"] = fvu(acts[kind], vocab.reconstruct(acts[kind], f"{kind}.{layer}"))
    gates, y_hat = molt.forward(acts["resid_mid"])
    measured["fvu_mlp"] = fvu(acts["mlp_out"], y_hat)
    measured["transforms_on_per_token"] = float((gates > 0).sum() / gates.shape[0])
    out = TEST_OUT / "measured" / "release4b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "usage_lines.json").write_text(json.dumps(measured, indent=1) + "\n")
    for name in ("fvu_state_resid_mid", "fvu_state_resid_post", "fvu_mlp"):
        assert 0.0 <= measured[name] < 1.0, measured
