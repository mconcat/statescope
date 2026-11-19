"""Lay out the frozen 0.8B set (second design) in the folder structure it is published in, without copying a tensor.

    <dest>/MANIFEST.json
    <dest>/README.md                                                           the model card (a copy), if --card is given
    <dest>/state/L<L>/{cfg.json, weights.safetensors, usage.safetensors}       state dictionary of layer L (two sites: resid_mid.L, resid_post.L)
    <dest>/mlp/L<L>/{config.yaml, weights.safetensors, postnorm.safetensors}   MLP dictionary of layer L

Every file under state/ and mlp/ is a symbolic link to the original (about 29 GB stay where they are). MANIFEST.json holds, per layer and
file, the sha256 of the bytes behind the link, and the numbers the original project recorded at training time. Both sources are only read.

    python devtools/stage_release2.py --state-root <.../q08c/train1> --mlp-root <.../release_08b_v1> --dest release_stage2 \
        [--card release/qwen35-08b-v2/MODEL_CARD.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

N_LAYERS = 24
STATE_FILES = ("cfg.json", "weights.safetensors", "usage.safetensors")
MLP_FILES = ("config.yaml", "weights.safetensors", "postnorm.safetensors")
ARM = "B1s0"                                  # the original's name of the released state-dictionary run


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def link(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-root", required=True, type=Path, help="holds L<L>/B1/vocab_B1s0/final and L<L>/B1/layer_recipe_done.json")
    ap.add_argument("--mlp-root", required=True, type=Path, help="the frozen folder of the MLP dictionaries: L<L>/ and MANIFEST.json")
    ap.add_argument("--dest", required=True, type=Path, help="folder to create (must not exist)")
    ap.add_argument("--card", type=Path, default=None, help="model card to copy into the folder as README.md")
    args = ap.parse_args()
    if args.dest.exists():
        raise FileExistsError(f"{args.dest} exists; stage into a new folder")
    mlp_manifest = json.loads((args.mlp_root / "MANIFEST.json").read_text())

    layers: dict[str, dict] = {}
    for L in range(N_LAYERS):
        run = args.state_root / f"L{L}" / "B1"
        src_state, src_mlp = run / f"vocab_{ARM}" / "final", args.mlp_root / f"L{L}"
        for name in STATE_FILES:
            link(src_state / name, args.dest / "state" / f"L{L}" / name)
        for name in MLP_FILES:
            link(src_mlp / name, args.dest / "mlp" / f"L{L}" / name)
        cfg = json.loads((src_state / "cfg.json").read_text())
        sites = [f"resid_mid.{L}", f"resid_post.{L}"]
        if cfg["sites"] != sites:
            raise ValueError(f"layer {L}: state dictionary sites {cfg['sites']}")
        recorded = mlp_manifest["layers"][str(L)]
        mlp_sha = {name: sha256(src_mlp / name) for name in MLP_FILES}
        if mlp_sha != recorded["sha256"]:
            raise ValueError(f"layer {L}: the MLP dictionary's files do not match the frozen folder's own manifest")
        layers[str(L)] = {
            "state": {
                "sites": sites, "form": cfg["form"], "input": cfg["input"] if "input" in cfg else cfg["spec"]["input"],
                "groups": cfg["n_units"], "group_size": cfg["unit_size"], "k": cfg["k_topk"], "steps": cfg["steps"],
                "heldout_fvu_own_units": json.loads((run / "layer_recipe_done.json").read_text())["vocab_heldout_fvu"][ARM],
                "sha256": {name: sha256(src_state / name) for name in STATE_FILES},
            },
            "mlp": {
                "penalty": recorded["penalty"], "transforms": recorded["transforms"], "alive": recorded["alive"],
                "heldout_fvu": recorded["heldout_fvu"], "heldout_l0": recorded["heldout_l0"], "heldout_active_rank": recorded["heldout_active_rank"],
                "sha256": mlp_sha, "tensor_bytes_sha256": recorded["data_sha256"],
            },
        }
        print(f"layer {L}: linked and hashed", flush=True)

    manifest = {
        "model": mlp_manifest["model"],
        "frozen": mlp_manifest["frozen"],
        "notes": {
            "state.heldout_fvu_own_units": "the trainer's number: per-token-RMS-normalised, site-standardised units; first 4,096 tokens of each of "
                                           "four held-out splits, position 0 of a sequence included (training dropped position 0)",
            "mlp.heldout_*": "measured by the original project on a held-out shard of the training capture that was deleted afterwards; "
                             "position 0 included (training included it); these numbers cannot be recomputed from the published files",
            "mlp.tensor_bytes_sha256": "sha256 of the tensor bytes only (the safetensors headers' metadata was rewritten once after training)",
        },
        "layers": layers,
    }
    (args.dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.card is not None:
        shutil.copyfile(args.card, args.dest / "README.md")
    print(f"staged {N_LAYERS} layers into {args.dest}")


if __name__ == "__main__":
    main()
