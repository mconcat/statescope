"""Build the release manifest: per layer the files of both dictionaries (size, sha256) and the measured numbers.

    header  = json.load(open("release/qwen35-08b/header.json"))          # model, revision, free-text descriptions
    numbers = {layer: LayerNumbers(...)}                                  # read from the run folders by the caller
    manifest = build_manifest(header, release_root, numbers)

A layer's JSON files are refused if they contain an absolute home path (they are published).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from statescope.manifest import layer_dir, sha256_file

VOCAB_FILES = ("cfg.json", "weights.safetensors", "usage.safetensors")
MOLT_FILES = ("cfg.json", "input.json", "weights.safetensors", "freq_train.safetensors")
FVU_OUT_OF_RANGE = 0.42          # reported as a note, not a failure
SPLICE_ARMS = ("molt", "mlp_zero", "mlp_mean", "vocab_post", "vocab_mid", "post_mean")


@dataclass
class LayerNumbers:
    """what was measured for one layer (all from the trainers' own records and the checks run on the exports)"""
    vocab_heldout_fvu: dict[str, float]          # site -> FVU
    vocab_groups: int
    vocab_dead_groups: int
    vocab_train_tokens: int
    molt_heldout_fvu: float                      # pooled over the four held-out splits
    molt_n_atoms: int
    molt_train_tokens: int
    molt_silent_fraction: float                  # atoms that never fire on the held-out tokens
    molt_fold_err: float                         # the trainer's export-vs-training-model check
    molt_gate_flips: int | None                  # gate decisions differing between fp32 and fp64 on 4,096 tokens (statescope.foldcheck)
    molt_fold_verdict: str | None                # statescope.foldcheck.fold_error_verdict
    splice: dict | None = None                   # statescope.splice.splice_layer result on 100 documents, if measured
    other_notes: list[str] = field(default_factory=list)


def file_entries(root: str, layer: int, kind: str) -> list[dict]:
    names = VOCAB_FILES if kind == "vocab" else MOLT_FILES
    folder = layer_dir(root, layer, kind)
    entries = []
    for name in names:
        path = os.path.join(folder, name)
        if name.endswith(".json"):
            with open(path) as f:
                if "/home/" in f.read():
                    raise ValueError(f"{path} holds an absolute home path")
        entries.append({"path": f"layers/L{layer:02d}/{kind}/{name}", "bytes": os.path.getsize(path), "sha256": sha256_file(path)})
    return entries


def layer_notes(n: LayerNumbers) -> tuple[list[str], list[str]]:
    """(notes of the layer, notes of its MLP dictionary)"""
    notes = list(n.other_notes)
    if max(n.vocab_heldout_fvu.values()) > FVU_OUT_OF_RANGE:
        notes.append("V0 FVU out of range")
    if n.molt_heldout_fvu > FVU_OUT_OF_RANGE:
        notes.append("MOLT FVU out of range")
    return notes, ([n.molt_fold_verdict] if n.molt_fold_verdict else [])


def layer_entry(root: str, layer: int, n: LayerNumbers) -> dict:
    notes, molt_notes = layer_notes(n)
    entry: dict = {
        "vocab": {"files": file_entries(root, layer, "vocab"), "heldout_fvu": n.vocab_heldout_fvu,
                  f"dead_groups_of_{n.vocab_groups}": n.vocab_dead_groups, "train_tokens": n.vocab_train_tokens},
        "molt": {"files": file_entries(root, layer, "molt"), "heldout_fvu": n.molt_heldout_fvu, "n_atoms": n.molt_n_atoms,
                 "train_tokens": n.molt_train_tokens, "atoms_silent_on_heldout_fraction": n.molt_silent_fraction,
                 "trainer_fold_max_rel_err": n.molt_fold_err, "gate_pairs_differing_fp32_vs_fp64_of_4096_tokens": n.molt_gate_flips,
                 "notes": molt_notes},
        "notes": notes,
    }
    if n.splice is not None:
        s = n.splice
        entry["splice_fineweb_100x256"] = {"ce_clean": s["ce_clean"], **{f"dce_{arm}": s[arm]["dce"] for arm in SPLICE_ARMS},
                                           "kl_molt": s["molt"]["kl"], "kl_vocab_post": s["vocab_post"]["kl"], "kl_vocab_mid": s["vocab_mid"]["kl"]}
    return entry


def build_manifest(header: dict, root: str, numbers: dict[int, LayerNumbers], not_listed: dict[int, str] | None = None) -> dict:
    """header: the fixed top-level fields. numbers: the layers to list. not_listed: layers held back, with the reason."""
    manifest = dict(header)
    manifest["layers"] = {str(layer): layer_entry(root, layer, numbers[layer]) for layer in sorted(numbers)}
    manifest["layers_not_listed"] = {str(layer): reason for layer, reason in (not_listed or {}).items()}
    return manifest
