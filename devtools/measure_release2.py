"""Reconstruction error of the frozen 0.8B set, measured with this repository's reader on the original project's held-out activations
(the calibration folders: four held-out splits per layer with x = raw resid_mid, y = raw MLP output, pos = position in the sequence).

Per layer, on ALL held-out tokens of the four splits (about 257,000), in raw units:
  state dictionary   FVU at resid_mid (x) and resid_post (x + y), position 0 dropped (training dropped it), and with position 0
  MLP dictionary     FVU of the output estimate against y, position 0 dropped, and with position 0 (training included it)
FVU = sum of squared errors / sum over splits of the squared deviations from the split's per-dimension mean.

Writes release/qwen35-08b-v2/reader_fvu.json. The calibration folders are not published; the numbers say what the published files do
on those tokens when read with the published reader.

    python devtools/measure_release2.py --release release_stage2 --calib-root <.../q08c/calib1> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from statescope.heldout import HELDOUT_SPLITS
from statescope.reader import PublicMolt, StateVocab

DEST = Path(__file__).resolve().parents[1] / "release" / "qwen35-08b-v2" / "reader_fvu.json"
BATCH = 8192


def load_split(folder: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """all tokens of one held-out split: x (n, d), y (n, d) in fp32 and pos (n,)"""
    index = json.loads((folder / "index.json").read_text())
    parts = [load_file(str(folder / shard["file"])) for shard in index["shards"]]
    return (torch.cat([p["x"] for p in parts]).float(), torch.cat([p["y"] for p in parts]).float(), torch.cat([p["pos"] for p in parts]).long())


class Fvu:
    """pooled over splits: errors summed, deviations taken from each split's own mean"""

    def __init__(self) -> None:
        self.error, self.total = 0.0, 0.0

    def add(self, target: torch.Tensor, estimate: torch.Tensor) -> None:
        self.error += float(((target - estimate).double() ** 2).sum())
        self.total += float(((target - target.mean(0)).double() ** 2).sum())

    def value(self) -> float:
        return self.error / self.total


@torch.no_grad()
def measure_layer(layer: int, release: Path, calib: Path, device: str) -> dict:
    vocab = StateVocab(str(release / "state" / f"L{layer}"), device)
    molt = PublicMolt(str(release / "mlp" / f"L{layer}"), device)
    sites = (f"resid_mid.{layer}", f"resid_post.{layer}")
    acc = {name: {"position_0_dropped": Fvu(), "with_position_0": Fvu()} for name in (*sites, "mlp")}
    n_tokens = n_position_0 = 0
    for split in HELDOUT_SPLITS:
        x, y, pos = load_split(calib / f"heldout_{split}")           # kept on the CPU; one batch at a time goes to the device
        keep = pos > 0
        n_tokens, n_position_0 = n_tokens + x.shape[0], n_position_0 + int((~keep).sum())
        estimates = {sites[0]: [], sites[1]: [], "mlp": []}
        for s in range(0, x.shape[0], BATCH):
            xb, yb = x[s:s + BATCH].to(device), y[s:s + BATCH].to(device)
            estimates[sites[0]].append(vocab.reconstruct(xb, sites[0]).cpu())
            estimates[sites[1]].append(vocab.reconstruct(xb + yb, sites[1]).cpu())
            estimates["mlp"].append(molt.forward(xb)[1].cpu())
        targets = {sites[0]: x, sites[1]: x + y, "mlp": y}
        for name, target in targets.items():
            estimate = torch.cat(estimates[name])
            acc[name]["with_position_0"].add(target, estimate)
            acc[name]["position_0_dropped"].add(target[keep], estimate[keep])
    return {"tokens": n_tokens, "position_0_tokens": n_position_0,
            **{name: {k: f.value() for k, f in by.items()} for name, by in acc.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--calib-root", required=True, type=Path, help="holds calib_q08_L<L>/heldout_<split>/")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    layers = {}
    for layer in range(24):
        layers[str(layer)] = measure_layer(layer, args.release, args.calib_root / f"calib_q08_L{layer}", args.device)
        row = layers[str(layer)]
        print(layer, {k: round(v["position_0_dropped"], 4) for k, v in row.items() if isinstance(v, dict)}, flush=True)
        torch.cuda.empty_cache()
    out = {"what": "FVU in raw units on all held-out tokens of the original project's four held-out splits per layer, read with statescope.reader",
           "splits": list(HELDOUT_SPLITS), "layers": layers}
    DEST.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
