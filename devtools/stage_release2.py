"""Lay out a set of dictionaries (the 0.8B set, or the 4B set with --layers 32) in the folder structure it is published in.

    <dest>/MANIFEST.json
    <dest>/README.md                                                           the model card (a copy), if --card is given
    <dest>/<name>                                                              a copy of every file given with --extra
    <dest>/state/L<L>/{cfg.json, weights.safetensors, usage.safetensors}       state dictionary of layer L (two sites: resid_mid.L, resid_post.L)
    <dest>/mlp/L<L>/{config.yaml, weights.safetensors, postnorm.safetensors}   MLP dictionary of layer L

The safetensors files are symbolic links to the originals: no tensor is copied or rewritten. The two text configs are written new,
without the original project's internal names (rule R11 of tests/EQUIVALENCE_RULES.md):

    cfg.json      `kind` becomes "state_dictionary"; the keys `name`, `trainer`, `arm` and `code_md5` are removed
    config.yaml   the line that names the original project's data module is removed

MANIFEST.json holds, per layer and file, the sha256 of the published bytes, the sha256 of the two configs before the edit, and the
numbers the original project recorded at training time. Both sources are only read.

    python devtools/stage_release2.py --state-root <.../train1> --mlp-root <path-free copy of the MLP dictionaries> --dest <new folder> \
        [--layers 24] [--card release/qwen35-08b-v2/MODEL_CARD.md] [--extra <file> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

STATE_FILES = ("cfg.json", "weights.safetensors", "usage.safetensors")
MLP_FILES = ("config.yaml", "weights.safetensors", "postnorm.safetensors")
ARM = "B1s0"                                  # the original's name of the released state-dictionary run
ORIGINAL_KIND = "sasa_proper_segment"         # the original project's name of its trainer module, the `kind` its cfg.json carries
PUBLISHED_KIND = "state_dictionary"
REMOVED_CFG_KEYS = ("name", "trainer", "arm", "code_md5")
REMOVED_CONFIG_LINE = "  class_path: measure.public_molt.shard_data.ShardDataModule\n"


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


def published_cfg(original: str) -> str:
    """the state dictionary's cfg.json as published: the original text with one value replaced and four keys removed"""
    cfg = json.loads(original)
    if json.dumps(cfg, indent=1) != original:
        raise ValueError("the original cfg.json is not written with indent=1: rewriting it would change more than the stated keys")
    if cfg["kind"] != ORIGINAL_KIND:
        raise ValueError(f"unexpected kind {cfg['kind']!r}")
    cfg["kind"] = PUBLISHED_KIND
    for key in REMOVED_CFG_KEYS:
        del cfg[key]
    return json.dumps(cfg, indent=1)


def published_config(original: str) -> str:
    """the MLP dictionary's config.yaml as published: the original text without the one line that names the data module"""
    if original.count(REMOVED_CONFIG_LINE) != 1:
        raise ValueError("the line that names the data module does not occur exactly once")
    return original.replace(REMOVED_CONFIG_LINE, "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-root", required=True, type=Path, help="holds L<L>/B1/vocab_B1s0/final and L<L>/B1/layer_recipe_done.json")
    ap.add_argument("--mlp-root", required=True, type=Path, help="the path-free copy of the MLP dictionaries: L<L>/ and MANIFEST.json")
    ap.add_argument("--dest", required=True, type=Path, help="folder to create (must not exist)")
    ap.add_argument("--layers", type=int, default=24, help="number of layers of the model (24 for 0.8B, 32 for 4B)")
    ap.add_argument("--card", type=Path, default=None, help="model card to copy into the folder as README.md")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="files to copy into the folder under their own names")
    args = ap.parse_args()
    if args.dest.exists():
        raise FileExistsError(f"{args.dest} exists; stage into a new folder")
    mlp_manifest = json.loads((args.mlp_root / "MANIFEST.json").read_text())

    layers: dict[str, dict] = {}
    for L in range(args.layers):
        run = args.state_root / f"L{L}" / "B1"
        src_state, src_mlp = run / f"vocab_{ARM}" / "final", args.mlp_root / f"L{L}"
        dst_state, dst_mlp = args.dest / "state" / f"L{L}", args.dest / "mlp" / f"L{L}"
        for name in STATE_FILES[1:]:
            link(src_state / name, dst_state / name)
        for name in MLP_FILES[1:]:
            link(src_mlp / name, dst_mlp / name)
        (dst_state / "cfg.json").write_text(published_cfg((src_state / "cfg.json").read_text()))
        (dst_mlp / "config.yaml").write_text(published_config((src_mlp / "config.yaml").read_text()))
        cfg = json.loads((dst_state / "cfg.json").read_text())
        sites = [f"resid_mid.{L}", f"resid_post.{L}"]
        if cfg["sites"] != sites:
            raise ValueError(f"layer {L}: state dictionary sites {cfg['sites']}")
        recorded = mlp_manifest["layers"][str(L)]
        if {name: sha256(src_mlp / name) for name in MLP_FILES} != recorded["sha256"]:
            raise ValueError(f"layer {L}: the MLP dictionary's files do not match the copy's own manifest")
        layers[str(L)] = {
            "state": {
                "sites": sites, "form": cfg["form"], "input": cfg["input"],
                "groups": cfg["n_units"], "group_size": cfg["unit_size"], "k": cfg["k_topk"], "steps": cfg["steps"],
                "heldout_fvu_own_units": json.loads((run / "layer_recipe_done.json").read_text())["vocab_heldout_fvu"][ARM],
                "sha256": {name: sha256(dst_state / name) for name in STATE_FILES},
                "original_sha256": {"cfg.json": sha256(src_state / "cfg.json")},
            },
            "mlp": {
                "penalty": recorded["penalty"], "transforms": recorded["transforms"], "alive": recorded["alive"],
                "heldout_fvu": recorded["heldout_fvu"], "heldout_l0": recorded["heldout_l0"], "heldout_active_rank": recorded["heldout_active_rank"],
                "sha256": {name: sha256(dst_mlp / name) for name in MLP_FILES},
                "original_sha256": {"config.yaml": recorded["sha256"]["config.yaml"]},
                "tensor_bytes_sha256": recorded["data_sha256"],
            },
        }
        print(f"layer {L}: staged and hashed", flush=True)

    manifest = {
        "model": mlp_manifest["model"],
        "frozen": mlp_manifest["frozen"],
        **({"note_frozen": mlp_manifest["note_frozen"]} if "note_frozen" in mlp_manifest else {}),
        "notes": {
            "state.heldout_fvu_own_units": "the trainer's number: per-token-RMS-normalised, site-standardised units; first 4,096 tokens of each of "
                                           "four held-out splits, position 0 of a sequence included (training dropped position 0)",
            "mlp.heldout_*": "measured by the original project on a held-out shard of the training capture that was deleted afterwards; "
                             "position 0 included (training included it); these numbers cannot be recomputed from the published files",
            "mlp.tensor_bytes_sha256": "sha256 of the tensor bytes only (the safetensors headers' metadata was rewritten once after training)",
            "original_sha256": "sha256 of a config file before names of the original, unpublished project were removed for publication: "
                               "in cfg.json the value of `kind` was replaced and four keys were removed (a configuration label, a source "
                               "file path, a run label, and the md5 record of the original project's source files); in config.yaml the "
                               "line naming the training data module was removed. Nothing else differs, and no tensor file was changed",
        },
        "layers": layers,
    }
    (args.dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.card is not None:
        shutil.copyfile(args.card, args.dest / "README.md")
    for extra in args.extra:
        shutil.copyfile(extra, args.dest / extra.name)
    print(f"staged {args.layers} layers into {args.dest}")


if __name__ == "__main__":
    main()
