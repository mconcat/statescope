"""A short run of the new state-dictionary trainer (writes <out>/run/{export/, log.jsonl, units.jsonl, final.json})."""
import argparse
import os

import torch

from statescope.calibration import load_calibration
from statescope.config import MIX_08B, MODELS, DataPaths
from statescope.data.mix import training_stream
from statescope.lm import load_language_model
from statescope.train.loop import LoopSettings
from statescope.vocab.recipe import VocabRecipe
from statescope.vocab.train import train_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--log-every", type=int, required=True)
    ap.add_argument("--mem-gb", type=float, required=True)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    for name in ("pile-dir", "ultrachat-dir", "view-corpus", "full-corpus"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / total_gb)
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    paths = DataPaths(a.pile_dir, a.ultrachat_dir, a.view_corpus, a.full_corpus)
    stats = load_calibration(a.calib_dir, device)
    lm = load_language_model(MODELS[a.model], "cuda:0")
    stream = training_stream(lm.tokenizer, paths, MIX_08B)
    loop = LoopSettings(tokens=a.steps * 4096, seed=a.seed)
    fvu = train_vocab(lm, a.layer, stream, stats, a.calib_dir, VocabRecipe(seed=a.seed), loop, os.path.join(a.out, "run"), a.log_every)
    print("final held-out FVU", fvu, flush=True)


if __name__ == "__main__":
    main()
