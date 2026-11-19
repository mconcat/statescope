"""Check a downloaded copy of the published set against its MANIFEST.json: every file of state/L<L>/ and mlp/L<L>/ by sha256.

    python demo/verify_release.py --release <folder with MANIFEST.json, state/ and mlp/>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    args = ap.parse_args()
    layers = json.loads((args.release / "MANIFEST.json").read_text())["layers"]
    wrong: list[str] = []
    n = 0
    for layer, entry in layers.items():
        for kind in ("state", "mlp"):
            for name, want in entry[kind]["sha256"].items():
                path = args.release / kind / f"L{layer}" / name
                n += 1
                if sha256(path) != want:
                    wrong.append(str(path))
        print(f"layer {layer}: checked", flush=True)
    if wrong:
        raise SystemExit(f"{len(wrong)} of {n} files do not match the manifest: {wrong}")
    print(f"all {n} files match the manifest")


if __name__ == "__main__":
    main()
