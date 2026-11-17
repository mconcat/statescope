"""Training of the state dictionary of one layer.

`VocabTrainer` holds the model, the optimizer and the per-group counters and takes one step on a batch of raw activations of both
sites. `train_vocab` is the whole run: calibrate the sites, loop over the shared batches, log, export.

One step: for each site, the site's loss on the batch, weighted 1 / S and backpropagated (gradients add up over the sites); gradient
clipping; Adam; learning-rate schedule; balancing; then the counters.
"""
from __future__ import annotations

import functools
import json
import os
import time

import torch

from statescope.calibration import CalibStats
from statescope.data.mix import MixStream, reference_sequences
from statescope.lm import Capturer, LanguageModel
from statescope.train.loop import LoopSettings, training_batches
from statescope.vocab.calibrate import calibrate_sites
from statescope.vocab.evaluate import heldout_fvu, load_heldout_sites
from statescope.vocab.export import export_vocab, vocab_cfg
from statescope.vocab.model import StateDictionary
from statescope.vocab.recipe import VocabRecipe, site_names


class VocabTrainer:
    def __init__(self, model: StateDictionary, schedule_steps: int) -> None:
        """schedule_steps: length of the learning-rate schedule in steps"""
        recipe = model.recipe
        device = model.D.device
        n_sites, G = len(model.sites), recipe.groups
        self.model, self.recipe = model, recipe
        self.optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr, betas=recipe.adam_betas)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: recipe.lr_factor(step, schedule_steps))
        self.since = torch.zeros(G, dtype=torch.long, device=device)                     # tokens since the group was last selected at any site
        self.use = torch.zeros(n_sites, G, dtype=torch.float64, device=device)           # EMA of the group's selection rate per site
        self.fire = torch.zeros(n_sites, G, dtype=torch.long, device=device)             # times selected per site
        self.fire_tokens = 0                                                             # tokens `fire` counts over
        self.thr_site = torch.zeros(n_sites, dtype=torch.float64, device=device)         # per site: EMA of the batch's smallest selected norm
        # The EMA weight is derived as the original did, 1 - (1 - 1 / steps): in floating point that is not exactly 1 / steps.
        self.usage_decay = 1.0 - 1.0 / recipe.usage_ema_steps
        self.steps = 0
        self.clip_hits = 0

    @property
    def lr(self) -> float:
        """the learning rate the next step will use"""
        return self.scheduler.get_last_lr()[0]

    def step(self, X: dict[str, torch.Tensor]) -> dict:
        """X: {site: raw activations (n, d) fp32 on the device}, the same n tokens at every site. Returns the step's numbers."""
        model, recipe = self.model, self.recipe
        dead = self.since > recipe.dead_window_tokens                                     # (G,) bool
        weight = 1.0 / len(model.sites)
        L_rec, L_aux, cuts, masks = [], [], [], []
        for j, site in enumerate(model.sites):
            out = model.losses(model.standardize(X[site], j), dead)
            (weight * out.loss).backward()
            L_rec.append(out.L_rec.detach())          # kept as tensors: reading a number here would make the GPU wait per site
            L_aux.append(out.L_aux.detach())
            cuts.append(out.cut)
            masks.append(out.mask)
            del out
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), recipe.grad_clip)
        if float(grad_norm) > recipe.grad_clip:
            self.clip_hits += 1
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        model.after_step()
        with torch.no_grad():
            model.update_threshold(functools.reduce(torch.minimum, cuts))
            site_cuts = torch.stack([c.double() for c in cuts])
            if self.steps == 0:
                self.thr_site.copy_(site_cuts)
            else:
                self.thr_site.mul_(recipe.threshold_decay).add_(site_cuts, alpha=1 - recipe.threshold_decay)
            n = int(next(iter(X.values())).shape[0])
            self.since.add_(n)
            fired = functools.reduce(torch.logical_or, [m.any(0) for m in masks])         # (G,) selected at some site by some token
            self.since[fired] = 0
            a = 1.0 - self.usage_decay
            for j, mask in enumerate(masks):
                self.use[j].mul_(1 - a).add_(mask.double().mean(0), alpha=a)
                self.fire[j] += mask.sum(0)
            self.fire_tokens += n
        record = {"step": self.steps, **{f"L_rec__{site}": float(L_rec[j]) for j, site in enumerate(model.sites)},
                  "L_aux": weight * sum(float(v) for v in L_aux), "grad_norm": float(grad_norm), "n_dead": int(dead.sum()),
                  "n_fired": int(fired.sum())}
        self.steps += 1
        return record

    def always_on(self) -> torch.Tensor:
        """(G,) bool: groups selected for more than `always_on_rate` of the tokens at some site (bias-corrected EMA)"""
        bias = 1.0 - (1.0 - (1.0 - self.usage_decay)) ** max(self.steps, 1)
        return (self.use / bias).amax(0) > self.recipe.always_on_rate

    def cfg(self, calib_record: dict, batch: int) -> dict:
        return vocab_cfg(self.model, calib_record, self.steps, self.steps * batch, self.clip_hits, self.thr_site)

    def export(self, path: str, cfg: dict) -> None:
        export_vocab(path, self.model, cfg, self.since, self.use, self.fire, self.fire_tokens)


def build_trainer(lm: LanguageModel, layer: int, stream: MixStream, recipe: VocabRecipe, schedule_steps: int) -> tuple[VocabTrainer, dict]:
    """A freshly initialised dictionary with its site statistics set from the stream's calibration sequences, ready to train.
    Returns (trainer, calibration record). The stream is left where it was."""
    state = stream.snapshot()
    reference = reference_sequences(stream)
    stream.restore(state)
    calib_seqs = {name: seqs for name, seqs in reference.items() if name.startswith("calib_")}
    model = StateDictionary(lm.spec.d_model, site_names(layer), recipe, lm.device)
    calibration = calibrate_sites(lm, layer, calib_seqs, recipe.calib_batch_tokens)
    model.site_mu.copy_(calibration.mu)
    model.site_scale.copy_(calibration.scale)
    return VocabTrainer(model, schedule_steps), calibration.record


def train_vocab(lm: LanguageModel, layer: int, stream: MixStream, stats: CalibStats, calib_dir: str, recipe: VocabRecipe,
                loop: LoopSettings, out_dir: str, log_every: int) -> dict[str, float]:
    """Train the state dictionary of `layer` and write

      <out_dir>/export/    cfg.json, weights.safetensors, usage.safetensors (the release format)
      <out_dir>/log.jsonl  one record per logged step
      <out_dir>/units.jsonl  one record per stream unit
      <out_dir>/final.json {"heldout_fvu": {site: FVU}} of the exported dictionary

    stats: the layer's calibration statistics on lm.device (the loop's batches are standardized with them). calib_dir: the calibration
    folder (its held-out splits are evaluated). Returns the final held-out FVU per site.

    The schedule has loop.tokens // loop.batch steps while the loop runs ceil(loop.tokens / loop.batch): with a token budget that is
    not a multiple of the batch, one more step follows the schedule's last one (at the final learning rate, not logged)."""
    if int(stats.folder_meta["layer"]) != layer:
        raise ValueError(f"{calib_dir} is the calibration folder of layer {stats.folder_meta['layer']}, not {layer}")
    device = lm.device
    os.makedirs(out_dir, exist_ok=True)
    site_mid, site_post = site_names(layer)
    schedule_steps = loop.tokens // loop.batch
    held = load_heldout_sites(calib_dir, layer, device, recipe.heldout_tokens)
    trainer, calib_record = build_trainer(lm, layer, stream, recipe, schedule_steps)
    capturer = Capturer(lm, layer, stream.spec.min_pos)
    seconds = 0.0
    with open(os.path.join(out_dir, "log.jsonl"), "w") as log, open(os.path.join(out_dir, "units.jsonl"), "w") as units:
        def on_unit(record: dict) -> None:
            units.write(json.dumps(record, default=str) + "\n")
            units.flush()

        for batch in training_batches(capturer, stream, stats, loop, device, on_unit=on_unit):
            t0 = time.time()
            with torch.no_grad():
                # the dictionary sees raw activations rebuilt from the buffer's bf16 standardized rows, without the first token of each sequence
                x = batch.z * stats.sd_x + stats.mu_x
                y = batch.y_std * stats.sd_y + stats.mu_y
                kept = (~batch.pos0).nonzero(as_tuple=True)[0]
                X = {site_mid: x[kept], site_post: (x + y)[kept]}
            record = trainer.step(X)
            seconds += time.time() - t0
            last = batch.step == schedule_steps - 1
            if batch.step % log_every == 0 or last:
                record.update({"lr": trainer.lr, "clip_hits": trainer.clip_hits, "n_always": int(trainer.always_on().sum()),
                               "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "step_seconds": seconds})
                if batch.step % (log_every * recipe.heldout_every_logs) == 0 or last:
                    record["heldout_fvu"] = heldout_fvu(trainer.model, held)
                log.write(json.dumps(record) + "\n")
                log.flush()
    trainer.export(os.path.join(out_dir, "export"), trainer.cfg(calib_record, loop.batch))
    final = {"heldout_fvu": heldout_fvu(trainer.model, held)}
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=1)
    return final["heldout_fvu"]
