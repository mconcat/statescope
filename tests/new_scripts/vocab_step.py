"""ONE training step of the new state dictionary from an injected state (rule R4.5): the counterpart of
tests/legacy_scripts/vocab_step_reference.py, with the same outputs.

The dictionary is built and calibrated as statescope.earlier_design.vocab.train.train_vocab does it, every --dead-every-th group is marked dead, and
one step is taken on the first --batch-tokens held-out pile tokens of the calibration folder (x = resid_mid, x + y = resid_post).

Writes <out>/step.safetensors and <out>/step.json.
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import save_file

from statescope.earlier_design.calibration import load_calibration
from statescope.config import MIX_08B, MODELS, DataPaths
from statescope.earlier_design.data.mix import training_stream
from statescope.heldout import load_heldout
from statescope.lm import load_language_model
from statescope.earlier_design.vocab.export import weight_tensors
from statescope.earlier_design.vocab.recipe import VocabRecipe, site_names
from statescope.earlier_design.vocab.train import build_trainer

DEAD_SINCE = 2_000_000          # above the dead window (VocabRecipe.dead_window_tokens)
PARAMETERS = ("D", "log_gamma", "b_dec", "E")      # the original's parameters; log_gamma is the frozen all-zero tensor of the file format


def sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300, help="length of the run the lr schedule is built for (the step taken is step 0)")
    ap.add_argument("--dead-every", type=int, default=97)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--mem-gb", type=float, required=True)
    for name in ("pile-dir", "ultrachat-dir", "view-corpus", "full-corpus"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / total_gb)
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    recipe = VocabRecipe(seed=a.seed)
    if DEAD_SINCE <= recipe.dead_window_tokens:
        raise ValueError("DEAD_SINCE must exceed the recipe's dead window")
    load_calibration(a.calib_dir, device)                 # checks the folder's statistics file, as the trainer's start does
    lm = load_language_model(MODELS[a.model], "cuda:0")
    stream = training_stream(lm.tokenizer, DataPaths(a.pile_dir, a.ultrachat_dir, a.view_corpus, a.full_corpus), MIX_08B)
    trainer, calib_record = build_trainer(lm, a.layer, stream, recipe, schedule_steps=a.steps)
    model = trainer.model
    torch.cuda.empty_cache()
    before = {name: sha256(weight_tensors(model)[name]) for name in PARAMETERS}

    # the injected state and the fixed batch
    dead_groups = torch.arange(0, recipe.groups, a.dead_every, device=device)
    trainer.since[dead_groups] = DEAD_SINCE
    x, y = load_heldout(a.calib_dir, "pile", a.batch_tokens)
    x = x.to(device)
    site_mid, site_post = site_names(a.layer)
    batch = {site_mid: x, site_post: x + y.to(device)}

    record = trainer.step(batch)
    record.update({"lr": trainer.lr, "clip_hits": trainer.clip_hits, "n_always": int(trainer.always_on().sum())})

    weights = weight_tensors(model)
    tensors = {name: weights[name].cpu().contiguous() for name in (*PARAMETERS, "site_mu", "site_scale", "thr")}
    tensors |= {"thr_site": trainer.thr_site.cpu().contiguous(), "since": trainer.since.cpu().contiguous(),
                "use": trainer.use.cpu().contiguous(), "fire": trainer.fire.cpu().contiguous()}
    save_file(tensors, os.path.join(a.out, "step.safetensors"))
    info = {"record": record, "init_hash": model.init_hash, "calib": calib_record, "params_before_sha256": before,
            "n_tokens": int(x.shape[0]), "n_dead_injected": int(dead_groups.numel()), "sites": list(model.sites),
            "gpu": torch.cuda.get_device_name(0)}
    json.dump(info, open(os.path.join(a.out, "step.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
