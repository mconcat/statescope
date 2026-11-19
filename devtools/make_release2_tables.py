"""Markdown tables of the frozen 0.8B set for the model card, from the staged MANIFEST.json and a result of demo/line_count.py.
Also copies the manifest into release/qwen35-08b-v2/ (the copy the repository carries). Every number in the two tables comes from
those two files; nothing is typed by hand.

    python devtools/make_release2_tables.py --release release_stage2 --demo-result <folder with result.json>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "release" / "qwen35-08b-v2"
CELLS = ("count:60", "count:100", "width:100")
ROWS = [("L9 mixer, whole output", "layer_y"), ("L9 mixer, head 10 only", "head_o"),
        ("`resid_post.8`, whole, through the L9 mixer", "l8/full_post8"), ("L8 MLP output, through the L9 mixer", "l8/mlp8_true"),
        ("L8 MLP dictionary, all transforms, through the L9 mixer", "l8/molt8_all"), ("L8 transform 189, through the L9 mixer", "l8/t189"),
        ("state group 11586, through the L9 mixer", "l8/g11586"), ("L9 MLP output", "l9/mlp9_y"),
        ("L9 MLP dictionary, all transforms", "l9/molt9_all"), ("L9 transform 410", "l9/t410"), ("L9 transform 1749", "l9/t1749")]


def layer_table(manifest: dict) -> str:
    lines = ["| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP FVU | MLP transforms on per token | MLP transforms alive | MLP penalty |",
             "|---|---|---|---|---|---|---|"]
    for layer, e in manifest["layers"].items():
        s, m = e["state"], e["mlp"]
        fvu = s["heldout_fvu_own_units"]
        lines.append(f"| {layer} | {fvu[f'resid_mid.{layer}']:.4f} | {fvu[f'resid_post.{layer}']:.4f} | {m['heldout_fvu']:.4f} | "
                     f"{m['heldout_l0']:.1f} | {m['alive']} of {m['transforms']} | {m['penalty']:g} |")
    return "\n".join(lines)


def demo_table(result: dict) -> str:
    lines = ["| Swap | " + " | ".join(CELLS) + " |", "|---|---|---|---|"]
    for name, arm in ROWS:
        cells = []
        for cell in CELLS:
            share, low, high = result["shares"][cell][arm]["even"]
            cells.append(f"{share:.3f} ({low:.3f} to {high:.3f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--demo-result", required=True, type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    result = json.loads((args.demo_result / "result.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.release / "MANIFEST.json", DEST / "MANIFEST.json")
    (DEST / "layers_table.md").write_text(layer_table(manifest) + "\n")
    (DEST / "line_count_table.md").write_text(demo_table(result) + "\n")
    print(f"wrote MANIFEST.json, layers_table.md and line_count_table.md to {DEST}")


if __name__ == "__main__":
    main()
