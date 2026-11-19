"""Calibration folder of one layer: standardization statistics and held-out activations.

Written files (the format the trainers and `statescope.heldout` read):
  calib_stats.safetensors   mu_x, sd_x (raw resid_mid), mu_y, sd_y (raw MLP output), mu_u, sd_u (resid_mid after the layer's RMSNorm),
                            norm_w_eff (= 1 + the RMSNorm weight); metadata: sha256 over mu_x, sd_x, mu_y, sd_y; norm_eps; token counts
  heldout_<split>/          shard_00000.safetensors (x, y, gen, seq, pos, tok, src) + index.json, for pile, ultra, cont, chat
  meta.json                 which rows were used

Statistics are float64 sums over the calibration tokens: 64 Pile rows, 72 UltraChat conversations, 64 + 64 on-policy sequences (MixSpec).
"""
from __future__ import annotations

import hashlib
import json
import os

import torch
from safetensors.torch import save_file

from statescope.config import DataPaths, MixSpec
from statescope.earlier_design.data.sources import UltraChat, corpus_prompt_lengths, corpus_tokens, pile_first_rows
from statescope.lm import Capturer, LanguageModel

SOURCE_CODES = {"pile": 0, "cont": 1, "chat": 2, "ultra": 3}      # the `src` column of the shards
STATS_HASHED = ("mu_x", "sd_x", "mu_y", "sd_y")


def _with_source(captured: dict[str, torch.Tensor], source: str, seq_offset: int = 0) -> dict[str, torch.Tensor]:
    out = dict(captured)
    out["src"] = torch.full((captured["x"].shape[0],), SOURCE_CODES[source], dtype=torch.uint8)
    out["seq"] = captured["seq"] + seq_offset
    return out


def _mean_sd(parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """float64 per-dimension mean and population sd over the rows of several (n, d) tensors"""
    n, s, ss = 0, 0, 0
    for t in parts:
        n += t.shape[0]
        s, ss = s + t.sum(0), ss + (t * t).sum(0)
    mu = s / n
    return mu, (ss / n - mu ** 2).clamp_min(0).sqrt(), n


def calibration_stats(parts: list[dict[str, torch.Tensor]], norm_weight: torch.Tensor, norm_eps: float) -> tuple[dict[str, torch.Tensor], int]:
    """parts: capture dicts of the calibration sequences. norm_weight: the RMSNorm weight of the layer's MLP norm.
    Returns ({mu_x, sd_x, mu_y, sd_y, mu_u, sd_u, norm_w_eff} fp32, number of tokens)."""
    mu_x, sd_x, n = _mean_sd([t["x"].double() for t in parts])
    mu_y, sd_y, _ = _mean_sd([t["y"].double() for t in parts])
    for name, sd in (("sd_x", sd_x), ("sd_y", sd_y)):
        if not torch.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError(f"calibration {name} has non-positive entries")
    w_eff = (1.0 + norm_weight.detach().double()).cpu()
    us = []
    for t in parts:
        x = t["x"].double()
        us.append(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps) * w_eff)
    mu_u, sd_u, _ = _mean_sd(us)
    stats = {"mu_x": mu_x.float(), "sd_x": sd_x.float(), "mu_y": mu_y.float(), "sd_y": sd_y.float(),
             "mu_u": mu_u.float(), "sd_u": sd_u.float(), "norm_w_eff": w_eff.float()}
    return stats, n


def stats_sha256(stats: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in STATS_HASHED:
        h.update(stats[key].float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def x_head_md5(x: torch.Tensor) -> str:
    """md5 of the first 16 MiB of a bf16 activation tensor (recorded per shard, compared when a stream replays a cache)"""
    return hashlib.md5(x.view(torch.int16).numpy().tobytes()[: 1 << 24]).hexdigest()


def _write_heldout(out_dir: str, split: str, tensors: dict[str, torch.Tensor], info: dict) -> None:
    folder = os.path.join(out_dir, f"heldout_{split}")
    os.makedirs(folder, exist_ok=True)
    name = "shard_00000.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(folder, name))
    index = {"split": f"heldout_{split}", "shards": [{"file": name, "tokens": int(tensors["x"].shape[0]), "x_head_md5": x_head_md5(tensors["x"]), **info}]}
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


def calibrate(lm: LanguageModel, layer: int, paths: DataPaths, spec: MixSpec, out_dir: str, batch_tokens: int = 32768) -> dict:
    """Write the calibration folder of `layer` to out_dir (must not hold a meta.json yet). Returns the statistics' metadata."""
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise FileExistsError(f"{out_dir} already holds a calibration folder")
    os.makedirs(out_dir, exist_ok=True)
    corpus = paths.view_corpus
    capturer = Capturer(lm, layer, spec.min_pos, batch_tokens)
    cont, cont_prompt = corpus_tokens(corpus, "cont_train"), corpus_prompt_lengths(corpus, "cont_train")
    chat, chat_prompt = corpus_tokens(corpus, "chat_train"), corpus_prompt_lengths(corpus, "chat_train")
    c0, c1 = spec.onpol_calib_start, spec.onpol_calib_start + spec.calib_onpol_seqs
    for name, seqs in (("cont", cont), ("chat", chat)):
        if len(seqs) < c1:
            raise RuntimeError(f"{name}_train has {len(seqs)} sequences < {c1} (the calibration reserve)")
    ultra_train = UltraChat(os.path.join(paths.ultrachat_dir, spec.ultra_train_file), lm.tokenizer, spec.ultra_max_tokens)
    ultra_test = UltraChat(os.path.join(paths.ultrachat_dir, spec.ultra_test_file), lm.tokenizer, spec.ultra_max_tokens)
    with open(os.path.join(corpus, "chat_train", "progress.json")) as f:
        chat_prompt_cursor = json.load(f)["cursor"]
    if chat_prompt_cursor >= spec.ultra_calib_rows[0]:
        raise RuntimeError("the on-policy chat prompts reached the UltraChat calibration rows")
    with open(os.path.join(corpus, "meta.json"), "rb") as f:
        corpus_meta_md5 = hashlib.md5(f.read()).hexdigest()
    quota = spec.quota()
    meta = {"layer": layer, "model": lm.spec.name, "min_pos": spec.min_pos, "special_ids": [], "keep_special": 1, "shard_tokens": spec.unit_tokens,
            "corpus": os.path.abspath(corpus), "corpus_meta_md5": corpus_meta_md5,
            "n_cont_seqs": len(cont), "n_chat_seqs": len(chat), "onpol_chat_prompt_cursor": chat_prompt_cursor,
            "calib": {"pile_shard": spec.pile_calib_shard, "pile_rows": spec.calib_pile_rows, "ultra_rows": list(spec.ultra_calib_rows), "onpol_seqs": [c0, c1]},
            "heldout_pile": {"shard": spec.pile_heldout_shard, "rows": spec.heldout_pile_rows},
            "heldout_ultra": {"file": spec.ultra_test_file, "convs_from_end": spec.heldout_ultra_convs},
            "ultra": {"file": spec.ultra_train_file, "order": "rows from the end backwards", "max_len": spec.ultra_max_tokens,
                      "template": "tokenizer.apply_chat_template(messages, tokenize=True), native template, no generation prompt"},
            "train_mix": {k: v / spec.unit_tokens for k, v in quota.items()}, "dtype": "bf16 raw", "src_codes": SOURCE_CODES,
            "versions": {"torch": torch.__version__}, "gpu": torch.cuda.get_device_name(lm.device)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # statistics
    parts = [_with_source(capturer.run(pile_first_rows(paths.pile_dir, spec.pile_calib_shard, spec.calib_pile_rows, spec.pile_row_tokens)), "pile"),
             _with_source(capturer.run([ultra_train.get(i) for i in range(*spec.ultra_calib_rows)]), "ultra", spec.ultra_calib_rows[0]),
             _with_source(capturer.run(cont[c0:c1], cont_prompt[c0:c1]), "cont", c0),
             _with_source(capturer.run(chat[c0:c1], chat_prompt[c0:c1]), "chat", c0)]
    norm = lm.mlp_norm(layer)
    stats, n_tokens = calibration_stats(parts, norm.weight, float(norm.eps))
    metadata = {"sha256": stats_sha256(stats), "n_tokens": str(n_tokens), "layer": str(layer),
                "per_src_tokens": json.dumps([int(p["x"].shape[0]) for p in parts]),
                "norm_eps": str(float(norm.eps)), "min_pos": str(spec.min_pos), "keep_special": "1",
                "u_definition": f"x * rsqrt(mean(x^2) + eps) * (1 + w), layer {layer} post_attention_layernorm"}
    save_file(stats, os.path.join(out_dir, "calib_stats.safetensors"), metadata=metadata)
    del parts

    # held-out activations
    pile = _with_source(capturer.run(pile_first_rows(paths.pile_dir, spec.pile_heldout_shard, spec.heldout_pile_rows, spec.pile_row_tokens)), "pile")
    _write_heldout(out_dir, "pile", pile, {"rows": spec.heldout_pile_rows})
    rows = [ultra_test.n - 1 - k for k in range(spec.heldout_ultra_convs)]
    ultra = _with_source(capturer.run([ultra_test.get(i) for i in rows]), "ultra")
    ultra["seq"] = torch.tensor(rows)[ultra["seq"]]                  # the conversation's row in the test file
    _write_heldout(out_dir, "ultra", ultra, {"rows_from_end": spec.heldout_ultra_convs})
    for kind in ("cont", "chat"):
        seqs = corpus_tokens(corpus, f"{kind}_heldout")
        captured = _with_source(capturer.run(seqs, corpus_prompt_lengths(corpus, f"{kind}_heldout")), kind)
        _write_heldout(out_dir, kind, captured, {"seqs": len(seqs)})
    return metadata
