"""Write demo/line_count_items.safetensors from a run of the frozen original code (tests/legacy_scripts/demo_reference.py): the 3 x 64
item pairs that the original's build_items(seed 0) makes for the cells count:60, count:100 and width:100.

    <cell>/ids   (64, 256) int32   token ids of the clean text
    <cell>/cor   (64, 256) int32   token ids of the corrupt text
    <cell>/pi    (64, 256) int32   for each clean position, the corrupt position that holds the same word piece, or -1

The texts are excerpts of FineWeb-Edu documents (HuggingFaceFW/fineweb-edu, ODC-By 1.0), re-wrapped to a fixed line width.
tests/test_demo.py::test_items checks the file against the reference run.

    python devtools/make_demo_items.py --reference test_out/legacy/demo_reference-<key>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file

DEST = Path(__file__).resolve().parents[1] / "demo" / "line_count_items.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path, help="output folder of a finished demo_reference.py run")
    args = ap.parse_args()
    if not (args.reference / "DONE").is_file():
        raise FileNotFoundError(f"{args.reference} is not a finished reference run")
    ref = load_file(str(args.reference / "reference.safetensors"))
    items = {key[len("items/"):]: t.contiguous() for key, t in ref.items() if key.startswith("items/")}
    save_file(items, str(DEST), metadata={"source": "FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By 1.0), re-wrapped; Qwen3.5 tokenizer",
                                          "fields": "ids = clean text, cor = corrupt text, pi = clean position -> corrupt position or -1"})
    print(f"wrote {len(items)} tensors to {DEST}")


if __name__ == "__main__":
    main()
