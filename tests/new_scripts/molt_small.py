"""CPU-only debugging run of statescope's MLP dictionary code on the small synthetic problem of
tests/legacy_scripts/molt_small_reference.py (same inputs, read from that run's data.safetensors; same output names), or the
sha256 of the full-size initial model (--mode init). No GPU, no language model.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python tests/new_scripts/molt_small.py --out <dir> --mode small --data <legacy dir>/data.safetensors

Compare with tests/new_scripts/molt_small_compare.py.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from statescope.calibration import CalibStats  # noqa: E402
from statescope.heldout import HELDOUT_SPLITS  # noqa: E402
from statescope.molt.evaluate import QuickEval, final_eval  # noqa: E402
from statescope.molt.export import export_cfg, export_tensors, write_export  # noqa: E402
from statescope.molt.model import MoltModel, init_model, whitener_from  # noqa: E402
from statescope.molt.recipe import MoltRecipe, RunPlan  # noqa: E402
from statescope.molt.step import Counters, train_step  # noqa: E402
from statescope.reader import MoltOps  # noqa: E402


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_of(m: MoltModel, prefix: str) -> dict:
    out = {f"{prefix}e": m.e, f"{prefix}b": m.b, f"{prefix}b_dec": m.b_dec, f"{prefix}W_skip": m.W_skip, f"{prefix}gate_mean": m.gate_mean,
           f"{prefix}gate_mean_set": m.gate_mean_set, f"{prefix}tier_thr": m.tier_thr, f"{prefix}tier_thr_set": m.tier_thr_set, f"{prefix}tier": m.tier}
    for i, r in enumerate(m.ranks):
        out |= {f"{prefix}U_r{r}": m.U[i], f"{prefix}V_r{r}": m.V[i], f"{prefix}s_r{r}": m.s[i], f"{prefix}idx_r{r}": m.tier_index(i)}
    return {k: v.detach().clone().contiguous() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=("small", "init"))
    ap.add_argument("--data", default="")
    ap.add_argument("--dead-tokens", type=int, default=1024)
    ap.add_argument("--reserve", type=int, default=100)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    if a.mode == "init":
        recipe = MoltRecipe()
        m = init_model(recipe, 1024, 3584, torch.eye(1024), torch.zeros(1024, 1024))
        json.dump({"tensors": {k: digest(v) for k, v in state_of(m, "").items()}}, open(os.path.join(a.out, "init.json"), "w"), indent=1)
        return

    data = load_file(a.data)
    steps, B, d = data["u"].shape
    recipe = dataclasses.replace(MoltRecipe(), batch=B, dead_tokens=a.dead_tokens, quick_eval_tokens=512, eval_chunk=500, fold_check_tokens=512)
    stats = CalibStats(**{k: data[k] for k in ("mu_x", "sd_x", "mu_y", "sd_y", "mu_u", "sd_u", "norm_w_eff")}, norm_eps=1e-6, sha256="",
                       file_metadata={}, folder_meta={})
    whitener, white_info = whitener_from(iter(data["white"]), recipe.whiten_eig_floor)
    m = init_model(recipe, d, a.reserve, whitener, data["skip"])
    opt = torch.optim.Adam(m.parameters(), lr=recipe.lr, betas=recipe.adam_betas, weight_decay=0.0)
    tensors = state_of(m, "init/")
    tensors["whitener"] = m.whitener.clone()

    plan = RunPlan.of(recipe, steps * B)
    counters = Counters.new(m.n_atoms, recipe.freq_segments, "cpu")
    heldout = {s: (data[f"x_{s}"], data[f"y_{s}"]) for s in HELDOUT_SPLITS}
    quick = QuickEval(heldout, stats, recipe.quick_eval_tokens, "cpu")
    records = []
    for step in range(steps):
        if step == plan.debias_start:
            m.freeze_zero_scales_()
        dead = counters.dead_mask(recipe.dead_tokens).nonzero().squeeze(1)
        res = train_step(m, opt, data["u"][step], data["y"][step], dead, recipe)
        counters.update(res.active, step, plan.steps)
        records.append({"step": step, "lr": res.lr, "loss": res.loss, "mse": res.mse, "l_aux": res.aux_loss, "n_aux_candidates": int(dead.numel()),
                        "n_aux_selected": res.n_aux_selected, "l0_train": float(res.active.sum(-1).float().mean()),
                        "n_dead_1M": int(counters.dead_mask(recipe.dead_tokens).sum()), **quick(m)})
    tensors |= state_of(m, "final/")
    tensors |= {"final/since": counters.since, "final/seg": counters.segment_fires, "final/seg_tokens": torch.tensor(counters.segment_tokens, dtype=torch.long)}

    tensors |= {f"export/{k}": v for k, v in export_tensors(m, stats).items()}
    export_dir = os.path.join(a.out, "export")
    write_export(export_dir, m, stats, counters)
    fin = final_eval(m, export_dir, heldout, stats, "cpu")
    fin["n_dead_1M_end"] = int(counters.dead_mask(recipe.dead_tokens).sum())
    with torch.no_grad():
        exported = MoltOps(export_dir, "cpu")
        x, y = heldout["pile"][0][:512], heldout["pile"][1][:512]
        z, _ = stats.standardize(x, y)
        a_export, y_export = exported.forward(x)
        tensors |= {"eval/y_model": m.forward_infer(stats.u_from_z(z))[1] * stats.sd_y + stats.mu_y, "eval/y_export": y_export, "eval/a_export": a_export}
    save_file({k: v.detach().contiguous() for k, v in tensors.items()}, os.path.join(a.out, "state.safetensors"))
    json.dump({"records": records, "final": fin, "cfg": export_cfg(m), "whitener": white_info, "debias_start": plan.debias_start},
              open(os.path.join(a.out, "records.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
