"""CPU-only debugging reference for the MLP dictionary port: the ORIGINAL model classes and step on a small synthetic problem
(runs inside the frozen snapshot; no GPU, no language model).

    --mode small   build train.molt_stream.build_arm's model at a small width, run the original step (train_mlp.Run.step ->
                   fixes.arm_step_fix, with the per-tier auxiliary output patched in as train.molt_stream.main does) on synthetic
                   batches with the bookkeeping of main's train_on, then the per-tier export and the evaluation formulas of
                   main's evaluate_export. Writes data.safetensors (the synthetic inputs, for the new code to read),
                   state.safetensors (initial state, final state, export tensors, evaluation outputs) and records.json.
    --mode init    the full-size initial model of the release recipe (width 1,024): sha256 of every initial tensor -> init.json.

The loop here is a transcription of the parts of molt_stream.main that touch the model; the real main is compared by
tests/test_train_molt.py on the GPU. This script only makes arithmetic differences visible without a GPU run.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, whatever the bridge selected

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from _hermetic import assert_hermetic  # noqa: E402


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synthetic(d: int, B: int, steps: int, n_eval: int) -> dict:
    """inputs of the small problem: raw x with a strong common mean (many atoms never fire: dead atoms), the dictionary input u made
    from it as the trainer does, a smooth target, and made-up calibration statistics"""
    g = torch.Generator().manual_seed(1234)
    r = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    base, mix, teacher = 1.5 * r(d), 0.5 * r(d, d), r(d, d) / d ** 0.5
    out = {"skip": 0.1 * r(d, d), "mu_x": 0.1 * r(d), "sd_x": 0.5 + torch.rand(d, generator=g), "mu_y": 0.1 * r(d),
           "sd_y": 0.5 + torch.rand(d, generator=g), "mu_u": 0.1 * r(d), "sd_u": 0.5 + torch.rand(d, generator=g),
           "norm_w_eff": 0.5 + torch.rand(d, generator=g)}

    def u_of(x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * out["norm_w_eff"] - out["mu_u"]) / out["sd_u"]
    out["u"] = u_of(base + r(steps, B, d) @ mix)
    out["y"] = torch.tanh(out["u"] @ teacher) + 0.1 * r(steps, B, d)
    out["white"] = u_of(base + r(8, B, d) @ mix)
    for split in ("pile", "ultra", "cont", "chat"):
        out[f"x_{split}"] = base + r(n_eval, d) @ mix
        out[f"y_{split}"] = torch.tanh(u_of(out[f"x_{split}"]) @ teacher) * out["sd_y"] + out["mu_y"] + 0.1 * r(n_eval, d)
    return out


def state_of(m, prefix: str) -> dict:
    out = {f"{prefix}e": m.e, f"{prefix}b": m.b, f"{prefix}b_dec": m.b_dec, f"{prefix}W_skip": m.W_skip, f"{prefix}gate_mean": m.m,
           f"{prefix}gate_mean_set": m.m_init, f"{prefix}tier_thr": m.rule_thr, f"{prefix}tier_thr_set": m.rule_thr_init, f"{prefix}tier": m.tier}
    for i, r in enumerate(m.rs):
        out |= {f"{prefix}U_r{r}": m.U_t[i], f"{prefix}V_r{r}": m.V_t[i], f"{prefix}s_r{r}": m.s_t[i], f"{prefix}idx_r{r}": getattr(m, f"idx_{i}")}
    return {k: v.detach().clone().contiguous() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=("small", "init"))
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--dead-tokens", type=int, default=1024)
    ap.add_argument("--reserve", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=1200)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    from gpt2.molt2.metrics import heldout_metrics
    from measure.molt_4b_infra import tiered as TRD
    from measure.molt_dead_atoms import train_dead as TD
    from measure.molt_gate_comparison import gates as GT
    from measure.molt_mlp_init import tiers as TR
    from train import molt_adaptive as A
    from train import molt_adaptive_v2 as AV2
    from train import molt_qwen_models as Q
    from train import molt_qwen_train as T
    from train import molt_stream as MS

    GT.aux_output_signed = TR.aux_output_tier           # as train.molt_stream.main
    if a.mode == "init":
        MS.PRESETS["INIT"] = dict(MS.PRESETS["Q"])     # no whitening: the initial weights do not depend on it
        m, _, cfg = MS.build_arm("M1w", "INIT", 1024, 0, 4, None, None, a.out, MS.MLP_WIDTH["qwen3.5-0.8b"])
        json.dump({"arm": cfg, "tensors": {k: digest(v) for k, v in state_of(m, "").items()}, "legacy_modules": assert_hermetic()},
                  open(os.path.join(a.out, "init.json"), "w"), indent=1)
        return

    d, B, micro = a.d, a.batch, 4
    data = synthetic(d, B, a.steps, a.n_eval)
    save_file({k: v.contiguous() for k, v in data.items()}, os.path.join(a.out, "data.safetensors"))
    stats = {k: data[k] for k in ("mu_x", "sd_x", "mu_y", "sd_y", "mu_u", "sd_u", "norm_w_eff")}
    stats["norm_eps"] = 1e-6
    MS.PRESETS["SMALL"] = dict(MS.PRESETS["Q"], f_white=1)       # what train_fixwhite does to the recipe-Q preset
    m, white_info, arm_cfg = MS.build_arm("M1w", "SMALL", d, 0, micro, stats, lambda: iter(data["white"]), a.out, a.reserve)
    opt = MS.make_opt(m, "adam", 3e-4, host_uv=False)             # the host-memory optimizer wrapper needs CUDA (pinned memory)
    with torch.no_grad():                                          # train.layer_recipe.LayerRecipe.setup, kind Qfix
        m.W_skip.data.copy_(data["skip"])
    m.W_skip.requires_grad_(False)
    tensors = state_of(m, "init/")
    tensors["whitener"] = m.Wg.clone()

    run_steps = total_steps = a.steps
    debias_start = total_steps - int(round(A.DEBIAS_FRAC * total_steps))
    run = MS.StreamRun(0, 0, run_steps, B)
    run.log_thr_every = 0
    run.out_dir_target = os.path.join(a.out, "extras")
    since = torch.zeros(m.n_units, dtype=torch.long)
    seg, seg_tokens = torch.zeros(10, m.n_units, dtype=torch.long), [0] * 10
    tf = Q.u_transform(stats)
    q = {s: T.standardize(stats, data[f"x_{s}"][:512], data[f"y_{s}"][:512]) for s in T.HELDOUT}
    qraw = {s: data[f"y_{s}"][:512] for s in T.HELDOUT}
    qvar = {s: float(((qraw[s] - qraw[s].mean(0)) ** 2).sum(-1).mean()) for s in T.HELDOUT}

    @torch.no_grad()
    def quick_eval():
        out = {}
        for s in T.HELDOUT:
            act, yh = m.forward_eval(tf(q[s][0]))
            yr = yh * stats["sd_y"] + stats["mu_y"]
            out[f"q_fvu_{s}"] = float(((yr - qraw[s]) ** 2).sum(-1).mean() / qvar[s])
            out[f"q_l0_{s}"] = float((act != 0).sum(-1).float().mean())
        return out

    records = []
    for step in range(a.steps):                                    # molt_stream.main's train_on, the lines that touch the model
        if step == debias_start:
            m.freeze_zero_s_()
        zu, ys = data["u"][step], data["y"][step]
        dead_now = TD.dead_mask(since, a.dead_tokens)
        didx = dead_now.nonzero().squeeze(1)
        r, lr, ex = run.step(m, opt, zu, ys, 0.0, 0.0, AV2.PREACT, micro, dead_idx=didx, auxk_k=512, auxk_coef=1.0 / 32, f_aux=False, probe_idx=None)
        run.post(m, ex)
        with torch.no_grad():
            act = r["a"] != 0
            run.update_since(since, act, B)
            sg = min(9, step * 10 // run_steps)
            seg[sg] += act.sum(0)
            seg_tokens[sg] += B
        records.append({"step": step, "lr": lr, "loss": float(r["loss"].detach()), "mse": float(r["mse"]), "l_aux": float(r.get("l_aux", 0.0)),
                        "n_aux_candidates": int(didx.numel()), "n_aux_selected": int(r.get("n_aux_selected", 0)),
                        "l0_train": float(act.sum(-1).float().mean()), "n_dead_1M": int(TD.dead_mask(since, a.dead_tokens).sum()), **quick_eval()})
    for f in run.files():
        f.close()
    tensors |= state_of(m, "final/")
    tensors |= {"final/since": since, "final/seg": seg, "final/seg_tokens": torch.tensor(seg_tokens, dtype=torch.long)}

    # molt_stream.main's evaluate_export, export format "tiers"
    export, cfg = TRD.tiered_tensors(m, stats)
    tensors |= {f"export/{k}": v for k, v in export.items()}
    f = A.RawAdaptive(TRD.TieredFolded.from_model(m, stats), stats)
    fin = {}
    with torch.no_grad():
        Xe, Ye = data["x_pile"], data["y_pile"]
        zc, _ = T.standardize(stats, Xe[:512], Ye[:512])
        y_ref = m.forward_eval(tf(zc))[1] * stats["sd_y"] + stats["mu_y"]
        act_raw, y_raw = f.forward_eval(Xe[:512])
        tensors |= {"eval/y_model": y_ref, "eval/y_export": y_raw, "eval/a_export": act_raw}
        fin["fold_max_rel_err"] = float((y_raw - y_ref).abs().max() / y_ref.abs().max())
        per = {s: heldout_metrics(f, data[f"x_{s}"], data[f"y_{s}"], stats["sd_y"], chunk=500) for s in T.HELDOUT}
        fin["l0"] = T.pooled_units(f, [data[f"x_{s}"] for s in T.HELDOUT], chunk=500)["l0"]
        sst, n_all, s1, s2 = {}, 0, 0.0, 0.0
        for s in T.HELDOUT:
            Y = data[f"y_{s}"].double()
            sst[s] = float(((Y - Y.mean(0)) ** 2).sum())
            n_all, s1, s2 = n_all + Y.shape[0], s1 + Y.sum(0), s2 + (Y * Y).sum(0)
        sst_all = float((s2 - s1 * s1 / n_all).sum())
        fin["fvu_pooled"] = sum(per[s]["fvu"] * sst[s] for s in T.HELDOUT) / sst_all
        for s in T.HELDOUT:
            fin[f"fvu_{s}"] = per[s]["fvu"]
        fin["n_dead_1M_end"] = int(TD.dead_mask(since, a.dead_tokens).sum())
    save_file({k: v.detach().contiguous() for k, v in tensors.items()}, os.path.join(a.out, "state.safetensors"))
    json.dump({"records": records, "final": fin, "cfg": cfg, "whitener": white_info, "arm": arm_cfg, "debias_start": debias_start,
               "legacy_modules": assert_hermetic()}, open(os.path.join(a.out, "records.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
