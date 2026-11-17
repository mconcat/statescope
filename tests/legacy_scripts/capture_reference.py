"""Legacy capture vs the new reader's capture(), in one process on one loaded model (runs inside the frozen snapshot).

The computation is part (c) of the original measure/mlp_track/release_parity.py: `measure.sites_v2.run_sites` on the model from
`measure.models.load_model`, against `capture` of the reader given by --reader (a self-contained file, loaded by path).

Writes <out>/info.json with the differences.
"""
import argparse
import importlib.util
import json
import os

import torch

from _hermetic import assert_hermetic, cap_gpu


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reader", required=True, help="path of the reader.py under test")
    ap.add_argument("--model", required=True, help="legacy model name, e.g. qwen3.5-0.8b")
    ap.add_argument("--layers", required=True, help="comma-separated")
    a = ap.parse_args()
    cap_gpu()

    from measure.mlp_track.release_parity import TEXT
    from measure.models import load_model
    from measure.sites_v2 import parse_site, run_sites

    spec = importlib.util.spec_from_file_location("reader_under_test", a.reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    lm = load_model(a.model)
    ids = torch.tensor([lm.tok(TEXT, add_special_tokens=True)["input_ids"]], device=lm.device)
    info = {"model": a.model, "n_ids": int(ids.shape[1]), "gpu": torch.cuda.get_device_name(0),
            "layers_object_is_same": reader.decoder_layers(lm.model) is lm.layers, "layers": {}}
    for L in [int(x) for x in a.layers.split(",")]:
        ref = run_sites(lm, ids, [parse_site(lm, f"resid_mid.{L}"), parse_site(lm, f"delta_mlp.{L}"), parse_site(lm, f"resid_post.{L}")], stop=False)
        ref = {k: (v[0] if isinstance(v, tuple) else v)[0, 1:].float() for k, v in ref.items()}
        got = reader.capture(lm.model, L, ids)
        info["layers"][str(L)] = {
            "n_tokens": int(got["resid_mid"].shape[0]),
            "max_abs_diff_resid_mid": float((got["resid_mid"] - ref[f"resid_mid.{L}"]).abs().max()),
            "max_abs_diff_mlp_out": float((got["mlp_out"] - ref[f"delta_mlp.{L}"]).abs().max()),
            "max_rel_diff_mid_plus_mlp_vs_layer_output": float((got["resid_post"] - ref[f"resid_post.{L}"]).abs().max() / ref[f"resid_post.{L}"].abs().max()),
        }
    info["legacy_modules"] = assert_hermetic()
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
