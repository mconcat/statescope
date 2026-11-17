"""Compare a statescope run of tests/new_scripts/molt_small.py with the legacy run of tests/legacy_scripts/molt_small_reference.py:
every tensor bit for bit, every record equal. Prints the differences and exits 1 if there are any.

    python tests/new_scripts/molt_small_compare.py <legacy dir> <new dir> [small|init]
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file

CFG_FREE_TEXT = {"source", "note"}


def strip(obj):
    return {k: strip(v) for k, v in obj.items() if k not in CFG_FREE_TEXT} if isinstance(obj, dict) else obj


def main() -> None:
    legacy, new, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "small")
    bad = []
    if mode == "init":
        a = json.load(open(os.path.join(legacy, "init.json")))["tensors"]
        b = json.load(open(os.path.join(new, "init.json")))["tensors"]
        bad += [f"init tensor {k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        print(f"{len(a)} legacy / {len(b)} new initial tensors")
    else:
        a, b = load_file(os.path.join(legacy, "state.safetensors")), load_file(os.path.join(new, "state.safetensors"))
        bad += [f"tensor only on one side: {k}" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape:
                bad.append(f"{k}: {a[k].dtype} {tuple(a[k].shape)} vs {b[k].dtype} {tuple(b[k].shape)}")
            elif not torch.equal(a[k], b[k]):
                bad.append(f"{k}: max abs diff {float((a[k].double() - b[k].double()).abs().max()):.3g}")
        ra, rb = json.load(open(os.path.join(legacy, "records.json"))), json.load(open(os.path.join(new, "records.json")))
        for sa, sb in zip(ra["records"], rb["records"]):
            bad += [f"step {sa['step']} {k}: {sa[k]!r} vs {sb.get(k)!r}" for k in sa if sa[k] != sb.get(k)]
        if len(ra["records"]) != len(rb["records"]):
            bad.append("different number of steps")
        bad += [f"final {k}: {v!r} vs {rb['final'].get(k)!r}" for k, v in ra["final"].items() if v != rb["final"].get(k)]
        if strip(ra["cfg"]) != strip(rb["cfg"]):
            bad.append("cfg differs outside source / note")
        if ra["whitener"] != rb["whitener"] or ra["debias_start"] != rb["debias_start"]:
            bad.append("whitener record or debias step differs")
        print(f"{len(a)} legacy / {len(b)} new tensors, {len(ra['records'])} steps; aux loss ran at "
              f"{sum(1 for s in ra['records'] if s['n_aux_selected'] > 0)} steps; dead at the end {ra['final']['n_dead_1M_end']}")
    print("\n".join(bad) if bad else "identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
