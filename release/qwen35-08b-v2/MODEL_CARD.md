---
license: mit
base_model: Qwen/Qwen3.5-0.8B
---

# Qwen3.5-0.8B per-layer dictionaries (state dictionary and MLP dictionary)

## What this is

This set contains separately trained state and MLP dictionaries for each of the 24 layers of `Qwen/Qwen3.5-0.8B`. The model has width 1024. The base-model revision is `2fc06364715b967f1860aea9cf38778875588b17`. The set was frozen on 2026-11-18.

A mixer is a layer's GatedDeltaNet or attention sublayer. A residual stream site is a location between the mixer and the MLP or after the MLP where a state dictionary reads the residual stream. A state dictionary represents the residual stream at both sites. An MLP dictionary represents a layer's MLP operations, from its post-norm input to its raw output. The set contains no dictionary for the mixers.

The statescope repository contains the reader, demonstration, tests and training code for an earlier design. The weights are separate from the code repository and are not published yet. The earlier design's training code did not train these weights.

## Files

The weights folder uses the following layout. `<L>` is the layer index from 0 to 23 without padding.

```text
MANIFEST.json
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

There are 144 files. State dictionaries occupy about 13 GiB in total. MLP dictionaries occupy about 16 GiB in total. The repository carries a manifest copy at `release/qwen35-08b-v2/MANIFEST.json`. The manifest provides a sha256 for each layer's files and the numbers recorded during training.

The following command checks all 144 files against the manifest:

```sh
python demo/verify_release.py --release <folder>
```

The MLP export stores the trainer's state dict in `weights.safetensors` and its configuration in `config.yaml`. Its RMSNorm coefficients are separate from the state dict. They are stored in `postnorm.safetensors` as tensor `w_eff` and metadata `eps`.

## State dictionary

`statescope.reader.StateVocab` reads one layer's state dictionary. The same dictionary covers `resid_mid.L` and `resid_post.L`. The former is the residual between the mixer and MLP, at the input of the norm before the MLP. The latter is the residual after the MLP, equal to `resid_mid` plus the MLP output.

A group is a set of 4 directions in the state dictionary. A frame is the orthonormal basis of a group's 4-dimensional write subspace. A code gives the token's coordinates in that frame, multiplied by a per-group scale. The dictionary has 16,384 groups in a tied form. The 24 groups with the largest written norm are active at each token.

The input convention is per-token RMS normalization. The dictionary first divides the residual by its own size using `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. It then standardises the result with a mean vector and one scale per site. To recover the raw residual, the reconstruction is multiplied by the token's own `sqrt(mean(x^2) + eps)`. `StateVocab.reconstruct(x, site)` performs both steps. Alternatively, `decode` accepts `rms=V.token_rms(x)`.

Each layer's dictionary was trained on 30,003,200 tokens over 7,325 steps. Position 0 means the first position of every sequence. Training dropped position 0. Measurements with this dictionary should also drop position 0. The reader's `capture` function does so.

The reader specifies the state representation and its input convention as follows:

```text
A site's activation x is standardized, xt = (x - mu_site) / scale_site - b_dec, and written as a sum over the active groups g of
c_g Q_g, where Q_g (b, d) is an orthonormal frame of the group's write subspace and c_g (b,) its code. Active = the k groups with
the largest |c_g| (= the norm of what the group writes).

A dictionary with cfg "input": "token_rms" (the tied form of the second design) reads the residual divided by its own size:
x is replaced by x * rsqrt(mean(x^2) + eps) before the line above. Its reconstructions are then in those units, and the raw
residual is the reconstruction times the token's own sqrt(mean(x^2) + eps): pass `rms=V.token_rms(x)` to `decode`, or call
`reconstruct`, which does both steps.

  sites            ["resid_mid.L", "resid_post.L"]
  G, b, d, k       groups, directions per group, model width, active groups per token  (0.8B release: 16384, 4, 1024, 24)
  Q                (G, b, d) write frames, orthonormal rows
  F                (G, b, d) code functionals: c_g = F_g xt
```

## MLP dictionary

A transform is a linear operation in an MLP dictionary with its own read and write matrices. Its write is what it adds to the MLP output. A gate controls a transform through a learned threshold on its pre-activation. `statescope.reader.PublicMolt` reads the sparse mixture of linear transforms, called MOLT. This dictionary uses a JumpReLU gate with a threshold learned separately for each transform. Its read and write matrices are free. It has no skip connection. Training uses a tanh sparsity penalty.

The form comes from the public `crosslayer-transcoder` code, branch MOLT, commit 376fabe. Training used a fork that changes no mechanism. There are 3,968 transforms per layer in five rank tiers. The tiers contain 128 transforms of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16 and 2,048 of rank 8. They are stored with the largest rank first. The reader's `rank` array gives each transform's rank.

The dictionary takes the layer's post-norm MLP input and produces a reconstruction of the raw MLP output. The reader computes that input from raw `resid_mid` using the layer's own RMSNorm coefficients. The formula is `u = x * rsqrt(mean(x^2) + eps) * w_eff`. The input is then standardised using per-dimension statistics retained by the trainer. The output is returned to raw MLP units with the stored output statistics.

Each layer was trained for 58,600 steps of 500 tokens, or 29.3 million tokens. The learning rate was 4.0e-05. The sparsity penalty coefficient is 7.5e-05 for layers 0 to 22. Layer 23 used 2.5e-05 in a separate run. Unlike the state dictionary, MLP dictionary training included position 0 of every sequence.

The gate is a hard threshold. A transform whose pre-activation lies within float rounding of its threshold can be on in one computation and off in another. The reader gives the following forward-pass formulas:

```text
u = x * rsqrt(mean(x^2) + eps) * w_eff          the layer's own RMSNorm on raw resid_mid x
z = (u - mu_in) / sd_in                         per-dimension statistics kept by the trainer
h_t = e_t . z + b_t ;  g_t = h_t if h_t > theta_t and h_t > 0 else 0                    JumpReLU gate of transform t
y_std = sum over transforms t of  g_t * (z V_t) U_t                                      V_t (d, rank_t), U_t (rank_t, d)
y = y_std * sd_out + mu_out                                                              ~ raw MLP output
```

## Numbers per layer

FVU means fraction of variance unexplained. It is squared reconstruction error divided by the squared deviation from the per-dimension mean. The state columns give the trainer's held-out FVU in the dictionary's own units. These units are per-token-RMS-normalised and site-standardised. The evaluation uses the first 4,096 tokens of each of four held-out splits with position 0 included. The repository's reader reproduces all 48 numbers within 6.9e-7.

The original project measured MLP FVU and transform activity on a held-out shard of the training capture, including position 0. That shard was deleted afterwards. Those held-out numbers cannot be recomputed from the published files. This repository did not re-measure them. “Transforms on per token” is the mean number of transforms whose gate is open at a token. “Transforms alive” counts transforms that fired on at least one token of that shard. The others never fired there. The recorded shard size at layer 9 was 267,271 tokens. At that layer, 2035 of 3968 transforms were alive.

With position 0 included, state FVU in raw residual units is higher than FVU in the dictionary's own units because position 0 has a very large norm. The original project measured raw FVU at layer 6, `resid_mid`, as 0.2524 with position 0 and 0.2184 without it.

Each row below identifies a layer, and the columns give state FVU at both sites, MLP FVU, mean transforms on per token, transforms alive out of the total, and the penalty coefficient. The values come from the manifest with the measurement conventions described above.

| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP FVU | MLP transforms on per token | MLP transforms alive | MLP penalty |
|---|---|---|---|---|---|---|
| 0 | 0.0533 | 0.0568 | 0.0732 | 19.0 | 3955 of 3968 | 7.5e-05 |
| 1 | 0.0917 | 0.1006 | 0.1447 | 11.2 | 3936 of 3968 | 7.5e-05 |
| 2 | 0.1433 | 0.1302 | 0.1358 | 9.9 | 3901 of 3968 | 7.5e-05 |
| 3 | 0.1587 | 0.1529 | 0.1599 | 10.3 | 3838 of 3968 | 7.5e-05 |
| 4 | 0.1791 | 0.1597 | 0.1526 | 10.2 | 3793 of 3968 | 7.5e-05 |
| 5 | 0.1888 | 0.1746 | 0.1736 | 9.9 | 3690 of 3968 | 7.5e-05 |
| 6 | 0.2147 | 0.2084 | 0.1892 | 9.1 | 3167 of 3968 | 7.5e-05 |
| 7 | 0.2393 | 0.2188 | 0.1851 | 9.2 | 2720 of 3968 | 7.5e-05 |
| 8 | 0.2359 | 0.2221 | 0.2066 | 9.1 | 2396 of 3968 | 7.5e-05 |
| 9 | 0.2452 | 0.2331 | 0.2153 | 8.9 | 2035 of 3968 | 7.5e-05 |
| 10 | 0.2540 | 0.2367 | 0.1828 | 8.6 | 1862 of 3968 | 7.5e-05 |
| 11 | 0.2713 | 0.2509 | 0.1763 | 8.3 | 1950 of 3968 | 7.5e-05 |
| 12 | 0.2648 | 0.2453 | 0.1857 | 8.3 | 2290 of 3968 | 7.5e-05 |
| 13 | 0.2594 | 0.2450 | 0.1768 | 8.5 | 2240 of 3968 | 7.5e-05 |
| 14 | 0.2606 | 0.2325 | 0.1216 | 7.5 | 1935 of 3968 | 7.5e-05 |
| 15 | 0.2586 | 0.2284 | 0.1200 | 8.2 | 2880 of 3968 | 7.5e-05 |
| 16 | 0.2314 | 0.2190 | 0.1469 | 8.3 | 3350 of 3968 | 7.5e-05 |
| 17 | 0.2185 | 0.2139 | 0.1555 | 8.4 | 3506 of 3968 | 7.5e-05 |
| 18 | 0.2151 | 0.2059 | 0.1304 | 7.9 | 3462 of 3968 | 7.5e-05 |
| 19 | 0.2294 | 0.2268 | 0.1370 | 8.2 | 3486 of 3968 | 7.5e-05 |
| 20 | 0.2221 | 0.2252 | 0.1491 | 8.3 | 3565 of 3968 | 7.5e-05 |
| 21 | 0.2307 | 0.2379 | 0.1461 | 8.2 | 3407 of 3968 | 7.5e-05 |
| 22 | 0.2455 | 0.2551 | 0.1234 | 7.8 | 2960 of 3968 | 7.5e-05 |
| 23 | 0.2666 | 0.2586 | 0.0206 | 16.1 | 3327 of 3968 | 2.5e-05 |

## Spliced-in loss

Spliced-in loss is the increase in next-token cross-entropy when a dictionary reconstruction replaces the true activation in the forward pass, one layer at a time. The original project measured these increases in nats on text with clean cross-entropy 2.808. This repository did not repeat the measurement.

The recorded MLP result for layer 23 belongs to an earlier dictionary of that layer. It is omitted because it does not describe the released dictionary. On the same text, the earlier set had a smaller increase at every tested layer. That comparison covered all 24 state dictionary layers and MLP dictionary layers 0 to 22. For the state dictionary at layer 9, the recorded increase was +0.243 for this set against +0.161 for the earlier set.

Each table row is a layer, and the columns distinguish replacement at `resid_post` by the state dictionary from replacement by the MLP dictionary. Values are increases in cross-entropy in nats, with the released layer 23 MLP result marked as not measured.

| Layer | State dictionary spliced in at `resid_post` | MLP dictionary spliced in |
|---|---|---|
| 0 | +0.074 | +0.0269 |
| 1 | +0.107 | +0.0386 |
| 2 | +0.117 | +0.0394 |
| 3 | +0.120 | +0.0268 |
| 4 | +0.134 | +0.0223 |
| 5 | +0.159 | +0.0199 |
| 6 | +0.208 | +0.0173 |
| 7 | +0.216 | +0.0156 |
| 8 | +0.240 | +0.0130 |
| 9 | +0.243 | +0.0072 |
| 10 | +0.288 | +0.0086 |
| 11 | +0.274 | +0.0076 |
| 12 | +0.280 | +0.0115 |
| 13 | +0.308 | +0.0101 |
| 14 | +0.337 | +0.0192 |
| 15 | +0.301 | +0.0192 |
| 16 | +0.309 | +0.0197 |
| 17 | +0.317 | +0.0191 |
| 18 | +0.326 | +0.0223 |
| 19 | +0.356 | +0.0221 |
| 20 | +0.376 | +0.0251 |
| 21 | +0.413 | +0.0275 |
| 22 | +0.465 | +0.0261 |
| 23 | +0.866 | not measured for the released dictionary |

## How to read the dictionaries

The reader is the single file `src/statescope/reader.py`. It needs torch, safetensors and PyYAML. It can be copied next to the weights. `StateVocab` provides `encode`, `decode`, `reconstruct` and `group_write`. `PublicMolt` provides `forward` and `transform_write`. The `MoltOps` class in the same file reads the earlier design's MLP dictionaries. It is not used for this set.

`capture` returns `resid_mid`, `mlp_out` and `resid_post` for one layer with position 0 dropped. It needs transformers and flash-linear-attention.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from statescope.reader import StateVocab, PublicMolt, capture

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda").eval()
input_ids = torch.tensor([tokenizer("The committee met on Thursday to review the budget.")["input_ids"]], device="cuda")

V = StateVocab("<release>/state/L9", "cuda")
M = PublicMolt("<release>/mlp/L9", "cuda")
acts = capture(model, 9, input_ids)                      # resid_mid, mlp_out, resid_post; position 0 dropped
groups, codes = V.encode(acts["resid_post"], "resid_post.9")
x_hat = V.reconstruct(acts["resid_post"], "resid_post.9")
gate, y_hat = M.forward(acts["resid_mid"])               # gate (tokens, 3968), y_hat ~ acts["mlp_out"]
write = M.transform_write(acts["resid_mid"], 410)        # what transform 410 adds to the MLP output
```

## Demonstration

`demo/line_count.py` follows a line-counting chain from the L8 MLP through L9 GatedDeltaNet head 10 to the L9 MLP. The model predicts a line break at the last word of a full line. Each item pairs 256-token texts that end on the same words. The clean text's last line is full. The corrupt text moves the last line break or wraps all lines wider, so a line break is not due.

The margin is the last-token logsumexp of newline-token logits minus the logsumexp of logits for tokens starting with a space. A patch replaces a clean-run tensor with a corrupt-run value moved onto the clean token positions. A share is `mean(m_clean - m_patched) / mean(m_clean - m_corrupt)`, where `m` is the margin. A share of 0 means that the patch changes nothing. A share of 1 means that the patch alone turns the clean run into the corrupt one.

A cell is a task setting defined by the wrapping width and the change that makes the corrupt text. The cells are `count:60`, `count:100` and `width:100`. The count cells move the last line break, while the width cell wraps all lines wider. Each contains 64 pairs stored as token ids and position pairing in `demo/line_count_items.safetensors`. These are re-wrapped FineWeb-Edu excerpts.

The original project found the chain. The script takes its choice of transform, group and head as given and re-measures them. It does not search for them. The original project ranked transforms on the odd-numbered items and reads results on the even-numbered ones. Some groups were first chosen after looking at all items; when the original project repeated the selection on odd-numbered items only, the chosen groups, frames and transforms were the same. Head 10 was known from earlier work. The results below use the even-numbered half, with 32 items per cell.

Rows identify swaps, and columns identify the three task cells. Values are shares, with bootstrap intervals from 2.5 % to 97.5 % in parentheses.

| Swap | count:60 | count:100 | width:100 |
|---|---|---|---|
| L9 mixer, whole output | 0.474 (0.421 to 0.524) | 0.590 (0.542 to 0.638) | 0.704 (0.663 to 0.744) |
| L9 mixer, head 10 only | 0.633 (0.585 to 0.675) | 0.678 (0.632 to 0.725) | 0.784 (0.737 to 0.827) |
| `resid_post.8`, whole, through the L9 mixer | 0.343 (0.286 to 0.396) | 0.378 (0.317 to 0.439) | 0.519 (0.471 to 0.562) |
| L8 MLP output, through the L9 mixer | 0.353 (0.295 to 0.410) | 0.323 (0.275 to 0.368) | 0.326 (0.268 to 0.375) |
| L8 MLP dictionary, all transforms, through the L9 mixer | 0.357 (0.301 to 0.414) | 0.319 (0.267 to 0.369) | 0.348 (0.286 to 0.404) |
| L8 transform 189, through the L9 mixer | 0.328 (0.261 to 0.385) | 0.324 (0.265 to 0.375) | 0.254 (0.202 to 0.307) |
| state group 11586, through the L9 mixer | 0.351 (0.295 to 0.399) | 0.312 (0.266 to 0.356) | 0.308 (0.262 to 0.355) |
| L9 MLP output | 0.346 (0.296 to 0.395) | 0.458 (0.404 to 0.510) | 0.508 (0.465 to 0.552) |
| L9 MLP dictionary, all transforms | 0.329 (0.280 to 0.376) | 0.366 (0.296 to 0.434) | 0.482 (0.431 to 0.533) |
| L9 transform 410 | 0.345 (0.281 to 0.401) | 0.013 (-0.039 to 0.064) | 0.026 (0.013 to 0.042) |
| L9 transform 1749 | 0.000 (0.000 to 0.000) | 0.108 (0.070 to 0.146) | 0.524 (0.457 to 0.593) |

In the count cells, swapping transform 189's write or state group 11586 through the L9 mixer moves the answer about as much as swapping the whole L8 MLP output. Swapping head 10's read alone gives a larger share than swapping the whole mixer output. Transform 410 carries the L9 MLP's effect in `count:60`, and transform 1749 carries it in `width:100`.

The next table tests how the L9 mixer reaches the six L9 transforms selected by the original project and transforms 410 and 1749 alone. The whole mixer swap and head 10's swap are each applied in full, only inside the 4-dimensional frame of state group 4885 of `resid_mid.9`, or as the rest.

Rows specify changes at the L9 mixer output, and columns identify the same task cells on even-numbered items. Cells are written `A / B (C)`. A is the share carried through the writes of the six L9 transforms selected by the original project: 410, 1749, 6, 3720, 1394 and 2275. B is the share through transforms 410 and 1749 alone. For A and B, the changed `resid_mid.9` enters the L9 MLP dictionary, and only the change in the selected transforms' writes is patched at the MLP output. C is the share of the same change with everything downstream. The last row has no C.

| Change at the L9 mixer output | count:60 | count:100 | width:100 |
|---|---|---|---|
| whole mixer swap, all of it | 0.320 / 0.268 (0.474) | 0.396 / 0.202 (0.590) | 0.392 / 0.425 (0.704) |
| whole mixer swap, only the part inside group 4885's frame | 0.300 / 0.283 (0.459) | 0.329 / 0.164 (0.594) | 0.374 / 0.414 (0.721) |
| whole mixer swap, the rest | 0.016 / 0.013 (0.156) | 0.032 / 0.052 (0.168) | 0.042 / 0.031 (0.177) |
| head 10 swap, all of it | 0.293 / 0.380 (0.633) | 0.373 / 0.179 (0.678) | 0.398 / 0.467 (0.784) |
| head 10 swap, only the part inside group 4885's frame | 0.314 / 0.374 (0.585) | 0.309 / 0.169 (0.650) | 0.395 / 0.476 (0.860) |
| head 10 swap, the rest | 0.026 / 0.025 (0.153) | 0.041 / 0.066 (0.185) | 0.044 / 0.037 (0.204) |
| whole mixer swap, only the part inside a random 4-dimensional frame (largest of 5, absolute) | 0.003 / 0.004 | 0.003 / 0.005 | 0.005 / 0.007 |

The part inside group 4885's frame moves the transforms about as much as the whole swap. The rest moves them little. For the whole mixer swap, the rest carries 0.016 in `count:60`, 0.032 in `count:100` and 0.042 in `width:100` through the six transforms. Through the two transforms alone, the rest carries 0.013 in `count:60`, 0.052 in `count:100` and 0.031 in `width:100`. A random 4-dimensional frame carries at most 0.005 through the six transforms and 0.007 through the two. With everything downstream, the rest carries 0.156 in `count:60`, 0.168 in `count:100` and 0.177 in `width:100`. The frame is therefore not the only route to the answer. The downstream shares of the two parts do not add up to the whole.

The task uses one model, one dictionary seed and fixed-width texts. It is an easy task with dedicated MLP neurons, so it does not show that dictionary units are better than neurons. Group 4885 is dense rather than sparse. Single-item margins and shares have the rounding limits stated below. The original project registered a pass rule for this link before measurement. The part inside the frame must keep at least 0.75 of the whole share. The rest must keep at most 0.25 of it and at most 0.05. Random frames must carry at most 0.01. The rule applies in both count cells. The original project's verdict was that the rule holds for the route through the six transforms and holds only in part for the route through 410 and 1749 alone, because it recorded a share of 0.051 for the rest in `count:100` through those two transforms.

## Limitations

- The demonstration covers one model, 0.8B, one dictionary seed and texts wrapped to a fixed width.
- This task is easy for the model and has dedicated MLP neurons. The demonstration is not evidence that dictionary units are better than neurons.
- State group 11586 is a sparse first-token-of-a-line group. Not every group is sparse. Some groups encountered by the original project on this task are dense 4-dimensional coordinates active on most tokens. They carry information in their coordinates. Group 4885 is one of these dense groups and is not a sparse feature.
- Single-item margins of dictionary swaps are reproducible only to about 0.1. Shares are reproducible only to about 0.003. Patched tensors are rounded to bf16. A 1e-6 relative difference between implementations flips some rounded elements.
- The trainers for this set are not in the repository. The earlier design's code is under `src/statescope/` in `vocab/`, `molt/`, `prep/`, `data/` and `train/`. The `statescope` command line and `release/qwen35-08b/` also belong to it. That design uses an untied state dictionary and an M1w MLP dictionary with 41,484 atoms. Its code reproduces how an earlier, unpublished set was trained. It did not train these weights. Porting the current trainers is planned.
- The MLP dictionaries' held-out numbers cannot be recomputed because the original evaluation shard was deleted.
- Spliced-in loss was measured by the original project and was not repeated in this repository.
- No full-length training run was repeated for either design.
- Other models, including Qwen3.5-4B, are outside this set.

## License and credits

The code is licensed under MIT. The MLP dictionary's form and trainer come from the public `crosslayer-transcoder` code, which is MIT licensed, Copyright 2025 Georg Lange. The demonstration's texts come from `HuggingFaceFW/fineweb-edu` under ODC-By 1.0.
