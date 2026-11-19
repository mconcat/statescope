"""Markdown tables of a set of the second design for its model card, from the staged MANIFEST.json, the set's reader_fvu.json
(devtools/measure_release2.py) and, for the 0.8B set, a result of demo/line_count.py. Also copies the manifest into the set's folder
under release/ (the copy the repository carries). Every number in the tables comes from those files; nothing is typed by hand.

    python devtools/make_release2_tables.py --release release_stage2 --demo-result <folder with result.json>
    python devtools/make_release2_tables.py --release release_stage_4b --dest release/qwen35-4b --always-on release_stage_4b/ALWAYS_ON.json
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


def layer_table(manifest: dict, reader_fvu: dict, always_on: dict | None = None) -> str:
    """state and first MLP column: this repository's reader on all held-out tokens, raw units, position 0 dropped (reader_fvu.json);
    the other MLP columns: recorded by the original project at training time (manifest); with `always_on` (the original project's
    ALWAYS_ON.json), one more column: the number of transforms that fire on at least 90% of its tokens"""
    head = ["Layer", "State FVU, `resid_mid`", "State FVU, `resid_post`", "MLP FVU", "MLP FVU recorded at training", "MLP transforms on per token",
            "MLP transforms alive", "MLP penalty"] + (["MLP transforms on at 90% of tokens or more"] if always_on is not None else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for layer, e in manifest["layers"].items():
        m, r = e["mlp"], reader_fvu["layers"][layer]
        row = (f"| {layer} | {r[f'resid_mid.{layer}']['position_0_dropped']:.4f} | {r[f'resid_post.{layer}']['position_0_dropped']:.4f} | "
               f"{r['mlp']['position_0_dropped']:.4f} | {m['heldout_fvu']:.4f} | {m['heldout_l0']:.1f} | {m['alive']} of {m['transforms']} | {m['penalty']:g} |")
        if always_on is not None:
            row += f" {len(always_on['layers'][layer]['on_90pct'])} |"
        lines.append(row)
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


def link_table(result: dict) -> str:
    lines = ["| Change at the L9 mixer output | " + " | ".join(CELLS) + " |", "|---|---|---|---|"]
    for base, label in (("mix", "whole mixer swap"), ("h10o", "head 10 swap")):
        for part, what in (("full", "all of it"), ("g4885", "only the part inside group 4885's frame"), ("minus_g4885", "the rest")):
            cells = []
            for cell in CELLS:
                six, two, down = (result["shares"][cell][f"link/{base}/{part}/{route}"]["even"][0] for route in ("F2", "T2", "down"))
                cells.append(f"{six:.3f} / {two:.3f} ({down:.3f})")
            lines.append(f"| {label}, {what} | " + " | ".join(cells) + " |")
    cells = []
    for cell in CELLS:
        six, two = (max(abs(result["shares"][cell][f"link/mix/rand{j}/{route}"]["even"][0]) for j in range(5)) for route in ("F2", "T2"))
        cells.append(f"{six:.3f} / {two:.3f}")
    lines.append("| whole mixer swap, only the part inside a random 4-dimensional frame (largest of 5, absolute) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--demo-result", type=Path, default=None, help="a result folder of demo/line_count.py (0.8B set only)")
    ap.add_argument("--dest", type=Path, default=DEST, help="the set's folder under release/ (holds reader_fvu.json)")
    ap.add_argument("--always-on", type=Path, default=None, help="the original project's list of transforms that fire on most tokens (4B set)")
    args = ap.parse_args()
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    shutil.copyfile(args.release / "MANIFEST.json", args.dest / "MANIFEST.json")
    reader_fvu = json.loads((args.dest / "reader_fvu.json").read_text())     # written by devtools/measure_release2.py
    always_on = json.loads(args.always_on.read_text()) if args.always_on is not None else None
    (args.dest / "layers_table.md").write_text(layer_table(manifest, reader_fvu, always_on) + "\n")
    written = ["MANIFEST.json", "layers_table.md"]
    if args.demo_result is not None:
        result = json.loads((args.demo_result / "result.json").read_text())
        (args.dest / "line_count_table.md").write_text(demo_table(result) + "\n")
        (args.dest / "line_count_link_table.md").write_text(link_table(result) + "\n")
        written += ["line_count_table.md", "line_count_link_table.md"]
    print(f"wrote {', '.join(written)} to {args.dest}")


if __name__ == "__main__":
    main()
