"""Reference spliced-in loss from the ORIGINAL measure/mlp_track/splice_loss.py (runs inside the frozen snapshot).

Calls the original `load_ids` and `run_layer` with the frozen standalone reader. The original `main` is NOT used: it writes into the
original repository by a hard-coded path. --rep only makes a second, separately cached run (process-to-process noise floor).

Writes <out>/splice.json = {layer: result}.
"""
import argparse
import importlib.util
import json
import os

import torch

from _hermetic import assert_hermetic, cap_gpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="release root (layers/Lxx/...)")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--rep", default="A")
    a = ap.parse_args()
    cap_gpu()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import measure.mlp_track.splice_loss as SL

    spec = importlib.util.spec_from_file_location("release_reader", os.path.join(os.environ["LEGACY_SNAPSHOT"], "release/qwen35-08b-layer-dicts/reader.py"))
    RD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RD)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda").eval()
    ids = SL.load_ids(tok, a.docs)
    res = {str(L): SL.run_layer(model, RD, ids, L, a.root, "cuda") for L in [int(x) for x in a.layers.split(",")]}
    res["legacy_modules"] = assert_hermetic()
    res["ids_sha256"] = __import__("hashlib").sha256(ids.numpy().tobytes()).hexdigest()
    json.dump(res, open(os.path.join(a.out, "splice.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
