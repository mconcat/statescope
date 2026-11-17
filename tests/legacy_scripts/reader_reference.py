"""Reference outputs of the ORIGINAL repository readers for one layer (runs inside the frozen snapshot; see tests/legacy_bridge.py).

The computation is the one of the original measure/mlp_track/release_parity.py, parts (a) and (b):
  MLP dictionary    train.molt_adaptive.RawAdaptive(measure.molt_4b_infra.tiered.TieredFolded.from_export(dir))
  state dictionary  measure.molt_vocab_align.lib.Vocab(spec="proper:<dir>")
on the first --n held-out pile tokens of the calibration folder (x = raw resid_mid, y = raw MLP output).

Writes <out>/reference.safetensors and <out>/info.json.
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

from _hermetic import assert_hermetic, cap_gpu


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--molt-dir", default="")
    ap.add_argument("--vocab-dir", default="")
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    dev = cap_gpu()

    from measure.molt_4b_infra import tiered as TRD
    from measure.molt_vocab_align.lib import Vocab
    from train import molt_adaptive as A
    from train import molt_qwen_train as T

    X, Y = T.load_heldout(a.calib_dir, "pile", dev, a.n)
    out = {"x": X.cpu(), "y": Y.cpu()}
    info = {"layer": a.layer, "n_tokens": int(X.shape[0]), "gpu": torch.cuda.get_device_name(0)}
    if a.molt_dir:
        sd = load_file(os.path.join(a.molt_dir, "weights.safetensors"))
        stats = {k: sd[f"calib_{k}"] for k in A.STAT_KEYS}
        stats["norm_eps"] = float(json.load(open(os.path.join(a.molt_dir, "input.json")))["eps"])
        f = A.RawAdaptive(TRD.TieredFolded.from_export(a.molt_dir), stats).to(dev)
        gate, y_hat = f.forward_eval(X)
        rows, atoms = torch.nonzero(gate, as_tuple=True)                     # the gate matrix is (n, atoms), about 60 nonzeros per row
        out |= {"molt_y": y_hat.cpu(), "molt_gate_rows": rows.cpu(), "molt_gate_atoms": atoms.cpu(), "molt_gate_values": gate[rows, atoms].cpu()}
        info["molt_n_atoms"] = int(gate.shape[1])
        del f, sd, gate, y_hat
        torch.cuda.empty_cache()
    if a.vocab_dir:
        V = Vocab(f"L{a.layer}", dev, spec=f"proper:{os.path.abspath(a.vocab_dir)}")
        for kind, x in (("resid_mid", X), ("resid_post", X + Y)):
            site = f"{kind}.{a.layer}"
            groups, codes = V.encode(V.xt(x, site))
            mu, sig = V.site(site)
            recon = mu + sig * (V.recon(groups, codes) + V.b_dec)
            out |= {f"vocab_{kind}_groups": groups.cpu(), f"vocab_{kind}_codes": codes.cpu(), f"vocab_{kind}_recon": recon.cpu()}
        del V
    info["legacy_modules"] = assert_hermetic()
    save_file(out, os.path.join(a.out, "reference.safetensors"))
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
