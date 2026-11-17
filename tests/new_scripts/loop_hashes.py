"""The new training loop with no dictionary: per step the sha256 of the batch, per unit the loop's record."""
import argparse
import hashlib
import json
import os

import torch

from statescope.calibration import load_calibration
from statescope.config import MIX_08B, MODELS, DataPaths
from statescope.data.mix import training_stream
from statescope.lm import Capturer, load_language_model
from statescope.train.loop import LoopSettings, training_batches


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--steps", type=int, required=True)
    for name in ("pile-dir", "ultrachat-dir", "view-corpus", "full-corpus"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    paths = DataPaths(a.pile_dir, a.ultrachat_dir, a.view_corpus, a.full_corpus)
    stats = load_calibration(a.calib_dir, device)
    lm = load_language_model(MODELS["qwen3.5-0.8b"], "cuda:0")
    stream = training_stream(lm.tokenizer, paths, MIX_08B)
    capturer = Capturer(lm, a.layer, MIX_08B.min_pos)
    units = []
    with open(os.path.join(a.out, "batches.jsonl"), "w") as f:
        for b in training_batches(capturer, stream, stats, LoopSettings(tokens=a.steps * 4096), device, on_unit=units.append):
            f.write(json.dumps({"step": b.step, "z": digest(b.z), "y_std": digest(b.y_std), "pos0": digest(b.pos0), "tok": digest(b.tok)}) + "\n")
    json.dump(units, open(os.path.join(a.out, "units.json"), "w"), default=str)


if __name__ == "__main__":
    main()
