"""Reference outputs of the second design's readers for one layer (runs inside the frozen snapshot legacy_snapshot2; see tests/legacy_bridge.py).

  state dictionary  measure.mlp_track.vocab_heldout_fvu.Reader (the reader written apart from the trainer), both sites
  MLP dictionary    measure.public_molt.lib.load_public -> the unmodified public Molt.forward_with_pre_activations, input through
                    measure.public_molt.lib.input_transform (as measure/public_molt/eval_heldout.py calls them)
on the first --n tokens of each held-out split of the calibration folder (x = raw resid_mid, y = raw MLP output, position 0 kept).

Decisions (selected groups, gate values) are stored for all tokens, reconstructed tensors for the first --n-tensor tokens of each split.
Writes <out>/reference.safetensors and <out>/info.json.
"""
import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

from _hermetic import assert_hermetic, cap_gpu


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--vocab-dir", required=True)
    ap.add_argument("--molt-dir", required=True, help="holds weights.safetensors and config.yaml")
    ap.add_argument("--post-norm", required=True, help="the layer's RMSNorm coefficients (the config's own path is on the training machine)")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--n-tensor", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=64, help="tokens per public forward (it builds (chunk, transforms, d) tensors)")
    a = ap.parse_args()
    dev = cap_gpu()

    from measure.mlp_track.vocab_heldout_fvu import SPLITS, Reader, load_heldout
    from measure.public_molt.lib import input_transform, load_public

    held = {s: load_heldout(a.calib_dir, s, a.n, 0) for s in SPLITS}
    out: dict[str, torch.Tensor] = {}
    info: dict = {"layer": a.layer, "gpu": torch.cuda.get_device_name(0), "splits": list(SPLITS),
                  "input_sha256": {s: {"x": tensor_sha256(held[s][0]), "y": tensor_sha256(held[s][1])} for s in SPLITS}}

    # ---- state dictionary
    R = Reader(a.vocab_dir, dev)
    info["vocab_fvu_own_units"] = {}
    for j, site in enumerate(R.sites):
        kind = site.split(".")[0]
        sse = sst = 0.0
        for s in SPLITS:
            x, y = held[s]
            x = (x if j == 0 else x + y).to(dev)
            groups, raws, own = [], [], []
            for i in range(0, x.shape[0], 512):
                xs, xh, raw = R.recon(x[i:i + 512], j)
                # the indices Reader.recon selected: its own lines up to the top-k, on its own tensors; the rebuilt reconstruction must be
                # bit-identical to the one recon returned
                z = ((xs - R.b_dec) @ R.E.reshape(R.G * R.b, R.d).T).view(-1, R.G, R.b)
                ix = torch.einsum("ngr,grs,ngs->ng", z, R.Gd, z).clamp_min(0).sqrt().topk(R.k, dim=-1).indices
                zs = torch.gather(z, 1, ix[..., None].expand(-1, -1, R.b))
                if not torch.equal(torch.einsum("nkr,nkrd->nd", zs, R.D[ix]) + R.b_dec, xh):
                    raise RuntimeError("the recomputed group indices do not rebuild Reader.recon's reconstruction")
                groups.append(ix)
                raws.append(raw)
                own.append(xs)
                sse += float(((xs - xh) ** 2).sum())
            own_all = torch.cat(own)
            sst += float(((own_all - own_all.mean(0)) ** 2).sum())
            out[f"vocab_{kind}_{s}_groups"] = torch.cat(groups).to(torch.int16).cpu()       # 16,384 groups fit
            out[f"vocab_{kind}_{s}_raw"] = torch.cat(raws)[:a.n_tensor].cpu()
        info["vocab_fvu_own_units"][site] = sse / sst
    info["vocab_cfg"] = {k: R.cfg.get(k) for k in ("form", "input", "rms_eps", "k_topk")}
    del R
    torch.cuda.empty_cache()

    # ---- MLP dictionary
    m, layer, minfo = load_public(os.path.join(a.molt_dir, "weights.safetensors"), os.path.join(a.molt_dir, "config.yaml"), dev)
    if layer != a.layer:
        raise RuntimeError(f"the config says layer {layer}")
    pre = input_transform({"data": {"init_args": {"post_norm_stats": a.post_norm}}}, dev)
    for s in SPLITS:
        x, _y = held[s]
        rows, cols, vals, ys = [], [], [], []
        for i in range(0, x.shape[0], a.chunk):
            xb = pre(x[i:i + a.chunk].to(dev)).float()
            _pre, gate, _norm, rec = m.forward_with_pre_activations(xb, layer)
            r, c = torch.nonzero(gate, as_tuple=True)
            rows.append(r + i), cols.append(c), vals.append(gate[r, c]), ys.append(rec)
        out[f"molt_{s}_rows"] = torch.cat(rows).to(torch.int32).cpu()
        out[f"molt_{s}_transforms"] = torch.cat(cols).to(torch.int16).cpu()                 # 3,968 transforms fit
        out[f"molt_{s}_values"] = torch.cat(vals).cpu()
        out[f"molt_{s}_y"] = torch.cat(ys)[:a.n_tensor].cpu()
    info["molt_n_transforms"] = int(m.n_features)
    info["molt_class"] = type(m).__module__ + "." + type(m).__name__

    snapshot = os.path.realpath(os.environ["LEGACY_SNAPSHOT"]) + os.sep
    public = [os.path.realpath(p) for p in sys.modules["crosslayer_transcoder"].__path__]
    if not all(p.startswith(snapshot) for p in public):
        raise RuntimeError(f"crosslayer_transcoder was loaded from {public}, not from the frozen snapshot")
    info["public_code"] = [os.path.relpath(p, snapshot) for p in public]
    info["legacy_modules"] = assert_hermetic()
    save_file(out, os.path.join(a.out, "reference.safetensors"))
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
