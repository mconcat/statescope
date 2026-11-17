"""Train the MLP dictionary of one layer with statescope (recipe M1w) and write the run into <out>/run."""
import argparse
import dataclasses
import os

import torch
from safetensors.torch import load_file

from statescope.calibration import load_calibration
from statescope.config import MIX_08B, MODELS, DataPaths
from statescope.data.mix import training_stream
from statescope.lm import Capturer, load_language_model
from statescope.molt.recipe import MoltRecipe
from statescope.molt.train import train_molt

MODEL = "qwen3.5-0.8b"          # this runner is the 0.8B test run: the model and its token mixture (MIX_08B) go together


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--fixmap", required=True, help="safetensors file of statescope.prep.fixedmap (tensor W_skip_std)")
    ap.add_argument("--steps", type=int, required=True, help="training budget in batches")
    ap.add_argument("--log-every", type=int, required=True)
    ap.add_argument("--mem-gb", type=float, required=True, help="cap of this process on the GPU")
    ap.add_argument("--dead-tokens", type=int, default=None, help="override of the recipe's dead window (tests)")
    for name in ("pile-dir", "ultrachat-dir", "view-corpus", "full-corpus"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / total_gb, 0)

    recipe = MoltRecipe()
    if a.dead_tokens is not None:
        recipe = dataclasses.replace(recipe, dead_tokens=a.dead_tokens)
    stats = load_calibration(a.calib_dir, device)
    lm = load_language_model(MODELS[MODEL], "cuda:0")
    stream = training_stream(lm.tokenizer, DataPaths(a.pile_dir, a.ultrachat_dir, a.view_corpus, a.full_corpus), MIX_08B)
    capturer = Capturer(lm, a.layer, MIX_08B.min_pos)
    skip_map = load_file(a.fixmap)["W_skip_std"]
    train_molt(lm, capturer, stream, stats, a.calib_dir, skip_map, os.path.join(a.out, "run"), recipe, tokens=a.steps * recipe.batch,
               log_every=a.log_every, device=device)
    print(f"[train_molt] peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
