"""Training one layer's MLP dictionary: `train_molt` ties the pieces together.

    whitener  <- the first `whiten_tokens` tokens of the stream (the stream is rewound afterwards)
    model     <- seeded initial weights (CPU), the whitener, the frozen skip map; then moved to the GPU
    loop      <- statescope.earlier_design.train.loop.training_batches; per batch: u = u_from_z(z), one train_step, counters
    at the end: export (release format), evaluation of the export, final.json

Written into `out_dir`:
    whitener.safetensors      W (d, d)
    curve.jsonl               one record per logged step
    export/                   cfg.json, input.json, weights.safetensors, freq_train.safetensors  (statescope.earlier_design.molt.export)
    final.json                held-out FVU per split and pooled, mean active atoms, fold error, dead atoms at the end
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from statescope.earlier_design.calibration import CalibStats
from statescope.earlier_design.data.mix import MixStream
from statescope.heldout import HELDOUT_SPLITS, load_heldout
from statescope.lm import Capturer, LanguageModel
from statescope.earlier_design.molt.evaluate import QuickEval, final_eval
from statescope.earlier_design.molt.export import write_export
from statescope.earlier_design.molt.model import MoltModel, init_model, standardized_mlp_input, whitener_from
from statescope.earlier_design.molt.recipe import MoltRecipe, RunPlan
from statescope.earlier_design.molt.step import Counters, train_step
from statescope.earlier_design.train.loop import LoopSettings, first_raw_inputs, training_batches


def check_calibration(lm: LanguageModel, layer: int, stats: CalibStats) -> None:
    """the calibration folder must have been made for this model and layer: same RMSNorm weight and epsilon, same layer number"""
    if int(stats.folder_meta["layer"]) != layer:
        raise ValueError(f"the calibration folder is layer {stats.folder_meta['layer']}, not {layer}")
    norm = lm.mlp_norm(layer)
    if not torch.equal((1.0 + norm.weight.detach().double()).float().cpu(), stats.norm_w_eff.cpu()) or float(norm.eps) != stats.norm_eps:
        raise ValueError("the calibration's RMSNorm weight / epsilon are not those of this model's layer")


def mlp_neurons(lm: LanguageModel, layer: int) -> int:
    """hidden width of the layer's MLP (the number of rank-1 atoms reserved at the front of the dictionary)"""
    return int(lm.mlp(layer).down_proj.in_features)


def compute_whitener(capturer: Capturer, stream: MixStream, stats_cpu: CalibStats, recipe: MoltRecipe) -> tuple[torch.Tensor, dict]:
    """Whitener of the dictionary's input over the first tokens of the stream, on the CPU. Leaves the stream where it was."""
    chunks = (standardized_mlp_input(stats_cpu, x) for x in first_raw_inputs(capturer, stream, recipe.whiten_tokens, recipe.whiten_chunk))
    return whitener_from(chunks, recipe.whiten_eig_floor)


def train_molt(lm: LanguageModel, capturer: Capturer, stream: MixStream, stats: CalibStats, calib_dir: str, skip_map: torch.Tensor,
               out_dir: str, recipe: MoltRecipe, tokens: int, log_every: int, device: torch.device) -> dict:
    """Train on `tokens` tokens of `stream` (positioned at its start) and write the outputs listed in the module docstring.

    stats      calibration statistics of the layer, on `device`
    calib_dir  the calibration folder (its held-out activations are the evaluation data)
    skip_map   (d, d) the fixed linear part in standardized coordinates (W_skip_std of statescope.earlier_design.prep.fixedmap)
    Returns the content of final.json."""
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("the recipe is defined with full-precision fp32 matmuls: torch.set_float32_matmul_precision('highest')")
    layer = capturer.layer
    check_calibration(lm, layer, stats)
    d = stats.d
    if tuple(skip_map.shape) != (d, d):
        raise ValueError(f"skip map {tuple(skip_map.shape)} for model width {d}")
    plan = RunPlan.of(recipe, tokens)
    os.makedirs(out_dir, exist_ok=True)

    whitener, whitener_record = compute_whitener(capturer, stream, stats.to("cpu"), recipe)
    save_file({"W": whitener.float().contiguous()}, os.path.join(out_dir, "whitener.safetensors"))
    print(f"[molt] whitener from {whitener_record['n_tokens']} tokens: eigenvalues {whitener_record['eig_min']:.4f} .. "
          f"{whitener_record['eig_max']:.2f}, {whitener_record['n_floored']} floored", flush=True)

    model: MoltModel = init_model(recipe, d, mlp_neurons(lm, layer), whitener, skip_map).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr, betas=recipe.adam_betas, weight_decay=0.0)
    counters = Counters.new(model.n_atoms, recipe.freq_segments, device)
    quick = QuickEval({s: load_heldout(calib_dir, s, recipe.quick_eval_tokens) for s in HELDOUT_SPLITS}, stats, recipe.quick_eval_tokens, device)
    print(f"[molt] layer {layer}: {model.n_atoms} atoms in tiers {model.ranks}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"trained parameters, {plan.steps} steps", flush=True)

    settings = LoopSettings(tokens=tokens, batch=recipe.batch, buffer_tokens=recipe.buffer_tokens, seed=recipe.seed)
    with open(os.path.join(out_dir, "curve.jsonl"), "w") as curve:
        for batch in training_batches(capturer, stream, stats, settings, device):
            step = batch.step
            if step == plan.debias_start:
                model.freeze_zero_scales_()
            u = stats.u_from_z(batch.z)
            dead = counters.dead_mask(recipe.dead_tokens).nonzero().squeeze(1)       # dead before this step
            result = train_step(model, optimizer, u, batch.y_std, dead, recipe)
            counters.update(result.active, step, plan.steps)
            if step % log_every == 0:
                if result.loss != result.loss or abs(result.loss) == float("inf"):
                    raise FloatingPointError(f"non-finite loss {result.loss} at step {step}")
                record = {"step": step, "tokens": step * recipe.batch, "lr": result.lr, "loss": result.loss, "mse": result.mse,
                          "l_aux": result.aux_loss, "n_aux_candidates": int(dead.numel()), "n_aux_selected": result.n_aux_selected,
                          "l0_train": float(result.active.sum(-1).float().mean()),
                          "n_dead_1M": int(counters.dead_mask(recipe.dead_tokens).sum()),      # after this step; the key keeps its historical name
                          "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, **quick(model)}
                curve.write(json.dumps(record) + "\n")
                curve.flush()
            del result, u, dead

    # no step follows: free the optimizer's moments before the evaluation
    del optimizer
    torch.cuda.empty_cache()
    export_dir = os.path.join(out_dir, "export")
    write_export(export_dir, model, stats, counters)
    final = final_eval(model, export_dir, {s: load_heldout(calib_dir, s) for s in HELDOUT_SPLITS}, stats, device)
    final["n_dead_1M_end"] = int(counters.dead_mask(recipe.dead_tokens).sum())
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=1)
    print(f"[molt] final: fvu pooled {final['fvu_pooled']:.4f}, l0 {final['l0']:.2f}, fold error {final['fold_max_rel_err']:.2e}, "
          f"dead {final['n_dead_1M_end']}", flush=True)
    return final
