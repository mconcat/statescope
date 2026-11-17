"""ONE training step of the ORIGINAL state dictionary from an injected state (rule R4.5; runs inside the frozen snapshot).

The dead-group auxiliary loss is never reached in a short legacy run (the dead window is a literal 1,000,000 tokens), so it is tested
here by state injection, without changing any legacy code:

  1. the dictionary is built and calibrated exactly as train.layer_recipe does it for the V0 arm (the setup lines of
     train.molt_stream.main, then LayerRecipe.setup's vocabulary part: NVocab, SegmentCapture, sasa_segment.calibrate, setup_tables);
  2. the `since` counter of every --dead-every-th group is set above the dead window;
  3. NVocab.step runs once on a fixed batch: the first --batch-tokens tokens of the held-out pile split of the calibration folder
     (x = raw resid_mid, x + y = resid_post);
  4. the step's record and every parameter and counter after the step are saved.

Writes <out>/step.safetensors (D, E, b_dec, log_gamma, site_mu, site_scale, thr, thr_site, since, use, fire) and <out>/step.json
(the step record, init_hash, the calibration record, sha256 of the parameters before the step).
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import save_file

from _hermetic import assert_hermetic

DEAD_SINCE = 2_000_000          # above the legacy dead window of 1,000,000 tokens (train/layer_recipe_vocab.py, NVocab.step)


def sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--view-corpus", required=True)
    ap.add_argument("--full-corpus", required=True)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--arm", default="V0s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300, help="length of the run the lr schedule is built for (the step taken is step 0)")
    ap.add_argument("--dead-every", type=int, default=97)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--mem-gb", type=float, required=True)
    ap.add_argument("--rep", default="A", help="only distinguishes repeated runs (separate cache entries)")
    a = ap.parse_args()

    import train.layer_recipe as LR
    import train.layer_recipe_vocab as LV
    import train.sasa_segment as SS
    from measure.molt_qwen import cache_acts as C
    from measure.sites_v2 import load_lm
    from train import molt_qwen_train as T
    from train import molt_stream_data as SD

    # ---- train.layer_recipe.main: the arm's spec, thread count, memory cap
    spec_file = os.path.join(os.environ["LEGACY_SNAPSHOT"], "results", "mlp_track", "specs", "v0.json")
    spec = json.load(open(spec_file))[a.arm]
    spec["sites"] = [LR.site_names(a.layer)[s] if s in ("a", "m", "p") else s for s in spec["sites"]]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / total_gb)

    # ---- train.molt_stream.main up to LRX.setup: seed, statistics, language model, token stream
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    stats = T.load_calib(a.calib_dir, dev)
    d = int(stats["mu_x"].shape[0])
    SD.configure_allpos()
    C.LAYER = a.layer
    lm = load_lm(a.model, device="cuda:0", dtype=torch.bfloat16)
    ms = SD.MixStream(lm.tok, a.view_corpus, a.full_corpus, extend=True)
    ms.check_view_prefix()
    ms.excl = SD.prefix_set(SD.reference_seqs(ms, a.full_corpus))

    # ---- LayerRecipe.setup: the vocabulary and its calibration
    state = ms.snapshot()
    refs = SD.reference_seqs(ms, a.full_corpus)
    ms.restore(state)
    calib = {k: refs[k] for k in refs if k.startswith("calib_")}
    V = LV.NVocab(a.arm, spec, d, dev, a.steps, a.out, token_tables=None, log_every=100)
    cap = SS.SegmentCapture(lm, V.sites)
    calib_info = SS.calibrate(cap, V.S, calib)
    del cap
    V.setup_tables()
    torch.cuda.empty_cache()
    S = V.S
    before = {name: sha256(p) for name, p in S.named_parameters()}

    # ---- the injected state and the fixed batch
    dead_groups = torch.arange(0, V.G, a.dead_every, device=dev)
    V.since[dead_groups] = DEAD_SINCE
    x, y = T.load_heldout(a.calib_dir, "pile", dev, a.batch_tokens)
    site_mid, site_post = LR.site_names(a.layer)["m"], LR.site_names(a.layer)["p"]
    batch = {site_mid: x, site_post: x + y}

    # ---- one step, as LayerRecipe.post calls it
    with torch.enable_grad():
        rec = V.step({s: batch[s] for s in V.sites}, None, None)
    rec.update({"lr": V.sched.get_last_lr()[0], "clip_hits": V.clip_hits, "n_always": int(V.always_on().sum())})
    V.log.close()

    tensors = {name: p.detach().float().cpu().contiguous() for name, p in S.named_parameters()}
    tensors |= {"site_mu": S.site_mu.float().cpu().contiguous(), "site_scale": S.site_scale.float().cpu().contiguous(),
                "thr": S.thr.float().cpu().reshape(1), "thr_site": V.thr_site.cpu().contiguous(), "since": V.since.cpu().contiguous(),
                "use": V.use.cpu().contiguous(), "fire": V.fire.cpu().contiguous()}
    save_file(tensors, os.path.join(a.out, "step.safetensors"))
    info = {"record": rec, "init_hash": S.init_hash, "calib": calib_info, "params_before_sha256": before,
            "n_tokens": int(x.shape[0]), "n_dead_injected": int(dead_groups.numel()), "sites": list(V.sites),
            "gpu": torch.cuda.get_device_name(0), "legacy_modules": assert_hermetic()}
    json.dump(info, open(os.path.join(a.out, "step.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
