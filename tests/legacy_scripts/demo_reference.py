"""Reference numbers of the line-count demonstration (runs inside the frozen snapshot legacy_snapshot3; see tests/legacy_bridge.py).

Uses the original's own pieces unmodified: the batch of a registered cell with its captures and patch helpers (demo_line_count.lib.Batch),
the pairing (line_counting.b_gdn.paired), the restart of a clean run at layer 9 (gdn_heads.common.Partial), the share
(line_counting.b_where.share), the MLP dictionaries through the fork's LeanMolt (demo_line_count.lib.load_mlp_molt), the state
dictionary at a site (demo_line_count.lib.Site), the head-10 arm of molt_state.evaluate_state.causal_new. The original's GDN dictionary
is not loaded: demo_line_count.lib.Ctx.__init__ loads it and a cached key-head basis, so the context is filled in here field by field.

Arms (rule R9.2), each a per-item margin after one patch:
  layer_y, head_o                                           the L9 mixer's whole output / head 10's output, swapped
  l8/{full_post8, mlp8_true, molt8_all, t189, g11586}       a change of resid_post.8 sent through the model's own L9 mixer only
  l9/{mlp9_y, molt9_all, t410, t1749}                       a change of the L9 MLP output
Writes <out>/reference.safetensors (margins, items, gates, labels) and <out>/info.json (shares, descriptive numbers).
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

from _hermetic import assert_hermetic, cap_gpu

HEAD = 10
T189, G11586 = 189, 11586
T9 = (410, 1749)
RAND_FRAMES = 5


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp8-dir", required=True, help="L8 MLP dictionary: weights.safetensors, config.yaml, postnorm.safetensors")
    ap.add_argument("--mlp9-dir", required=True)
    ap.add_argument("--vocab8-dir", required=True, help="L8 state dictionary: cfg.json, weights.safetensors")
    ap.add_argument("--vocab9-dir", required=True)
    ap.add_argument("--run", default="a", help="free label: a second value gives a second, separately cached run (run-to-run repeatability)")
    a = ap.parse_args()
    dev = cap_gpu()
    torch.set_num_threads(4)

    from measure.demo_line_count import lib as D
    BW, C, GD = D.BW, D.C, D.GD
    T = D.T

    ctx = object.__new__(D.Ctx)                       # Ctx.__init__ would load the GDN dictionary and the key-head basis
    ctx.eng = C.engine("0.8b")
    ctx.dev = ctx.eng.device
    ctx.m9 = ctx.eng.mixer(9)
    ctx.cols = C.head_cols(ctx.m9, HEAD).float()
    ctx.arm = None
    ctx.M, ctx.pre = {}, {}
    for L, root in ((8, a.mlp8_dir), (9, a.mlp9_dir)):
        ctx.M[L], ctx.pre[L] = D.load_mlp_molt(L, ctx.dev, root=root)
    ctx.V = {8: D.open_vocab(a.vocab8_dir, ctx.dev, "L8"), 9: D.open_vocab(a.vocab9_dir, ctx.dev, "L9")}
    ctx.items, ctx.cl, ctx.seen = {}, {}, set()
    eng, m9 = ctx.eng, ctx.m9
    S8 = D.Site(ctx.V[8], "resid_post.8")
    Q = ctx.V[8].Q[G11586]
    rand = D.rand_frames(eng.d, RAND_FRAMES, 0, ctx.dev)

    out: dict[str, torch.Tensor] = {}
    energy = {"den": 0.0, "g11586": 0.0, **{f"rand{j}": 0.0 for j in range(RAND_FRAMES)}}
    gates = {"l8": [], "l9": []}
    labels = {k: [] for k in ("c", "tk", "nl", "ls", "known")}
    for cell in D.CELLS:
        for b in ctx.batches(cell):
            c, x = b.c, b.x
            sh = (b.B, T, -1)
            N = b.B * T
            # ---- the mixer's whole output; head 10's output (causal_new's reference arms)
            b.run_mix9(b.pair(x["mix9"], c["mix9"]), "layer_y")
            hq_c, hq_x = GD.head_inputs(m9, c["x9"], [HEAD])[HEAD], GD.head_inputs(m9, x["x9"], [HEAD])[HEAD]
            oc, ox = C.recur(hq_c), C.recur(hq_x)
            n_c = C.head_norm_out(m9, oc, hq_c["z"]).float()
            b.run_mix9(c["mix9"] + (C.head_norm_out(m9, b.pair(ox, oc), hq_c["z"]).float() - n_c) @ ctx.cols.T, "head_o")
            del hq_x, oc, ox
            # ---- changes of resid_post.8 through the model's own L9 mixer (s3_mlp8.path, the `true` arm)
            y_true_c = b.true_mix9(c["post8"])

            def through_mixer(dv, name):
                b.run_mix9(c["mix9"] + b.true_mix9(c["post8"] + dv.view(sh)) - y_true_c, name)

            Rc8, Rx8 = b.mlp_molt(8, c["mid8"]), b.mlp_molt(8, x["mid8"])
            d189 = D.atoms_diff(b, Rc8, Rx8, [T189], "write")
            through_mixer(b.pair(x["post8"], c["post8"]) - c["post8"], "l8/full_post8")
            through_mixer(b.pair(x["mlp8"], c["mlp8"]) - c["mlp8"], "l8/mlp8_true")
            through_mixer(b.pair(Rx8.at.reshape(sh), Rc8.at.reshape(sh)) - Rc8.at.reshape(sh), "l8/molt8_all")
            through_mixer(d189, "l8/t189")
            # state group 11586 of resid_post.8 swapped (s4_amend, A1)
            Ac, cc, rc = S8.enc(c["post8"])
            Ax, cx, rx = S8.enc(x["post8"])
            k_, b_ = S8.k, S8.b
            Axp = b.pair(Ax.view(b.B, T, k_), Ac.view(b.B, T, k_)).reshape(N, k_)
            crx = b.pair((cx * rx[:, :, None]).view(b.B, T, k_, b_), (cc * rc[:, :, None]).view(b.B, T, k_, b_)).reshape(N, k_, b_)
            tok, grp, Dr = S8.delta(Ac, cc * rc[:, :, None], Axp, crx)
            mk = torch.zeros(S8.G, dtype=torch.bool, device=ctx.dev)
            mk[G11586] = True
            through_mixer(S8.delta_vec(N, tok, grp, Dr, mk), "l8/g11586")
            del Ax, cx, crx, tok, grp, Dr
            # energy of transform 189's swapped write inside group 11586's frame (s5_amend2, "swap")
            kf = D.keep_mask(b).view(b.B, T, 1).float()
            energy["den"] += float((d189.pow(2) * kf).sum())
            energy["g11586"] += float((D.proj(Q, d189).pow(2) * kf).sum())
            for j, fr in enumerate(rand):
                energy[f"rand{j}"] += float((D.proj(fr, d189).pow(2) * kf).sum())
            # ---- changes of the L9 MLP output (s2_mlp9, pass 1)
            Rc9, Rx9 = b.mlp_molt(9, c["mid9"]), b.mlp_molt(9, x["mid9"])
            b.run_mlp9(c["mlp9"] + b.pair(x["mlp9"], c["mlp9"]) - c["mlp9"], "l9/mlp9_y")
            b.run_mlp9(c["mlp9"] + b.pair(Rx9.at.reshape(sh), Rc9.at.reshape(sh)) - Rc9.at.reshape(sh), "l9/molt9_all")
            for u in T9:
                b.run_mlp9(c["mlp9"] + D.atoms_diff(b, Rc9, Rx9, [u], "write"), f"l9/t{u}")
            # ---- gates on the clean items, token labels
            rr = b.real.to(ctx.dev)
            gates["l8"].append(Rc8.gate[:, [T189]].view(b.B, T, -1)[rr].float().cpu())
            gates["l9"].append(Rc9.gate[:, list(T9)].view(b.B, T, -1)[rr].float().cpu())
            lab = b.labels()
            for k in labels:
                labels[k].append(lab[k][rr.cpu()])
            print(cell, "batch", b.s0, "peak %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
            del Rc8, Rx8, Rc9, Rx9, b
    shares = {}
    for cell in D.CELLS:
        cl = ctx.cl[cell]
        for k in cl.d:
            out[f"margin/{cell}/{k}"] = cl.get(k).float().contiguous()
        shares[cell] = ctx.shares(cell)
        items = ctx.items[cell]
        for f in ("ids", "cor", "pi"):
            out[f"items/{cell}/{f}"] = torch.tensor([it[f] for it in items], dtype=torch.int32)
    out["gate/l8"] = torch.cat(gates["l8"])                                  # (192, T, 1): cells in order, items in order
    out["gate/l9"] = torch.cat(gates["l9"])                                  # (192, T, 2)
    for k, v in labels.items():
        out[f"label/{k}"] = torch.cat(v).to(torch.int32)
    used = assert_hermetic()
    info = {"gpu": torch.cuda.get_device_name(0), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "run": a.run, "cells": list(D.CELLS),
            "shares": shares, "energy_fraction_t189_swap": {k: v / energy["den"] for k, v in energy.items() if k != "den"},
            "item_fields": {cell: {k: [it[k] for it in ctx.items[cell]] for k in ("W", "kind", "count_c", "count_x", "left", "line")} for cell in D.CELLS},
            "research_modules": used, "fla_loaded": "fla" in __import__("sys").modules}
    save_file(out, os.path.join(a.out, "reference.safetensors"))
    with open(os.path.join(a.out, "info.json"), "w") as f:
        json.dump(C.clean(info), f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
