# statescope

statescope traces what a language model represents in its hidden states and which operations use that information. We use [Fel et al.’s block-sparse featurizer (BSF)](https://arxiv.org/abs/2606.25234) for residual states and the [public MOLT code](https://github.com/Goreg12345/crosslayer-transcoder/tree/376fabe32ec827e790fcef5b27db1f7aefc188c3) for MLP operations. We release these separately trained dictionaries for **every layer of Qwen3.5-0.8B and Qwen3.5-4B**, with a reader and two studies that test how the states affect predictions. The first builds on [Gurnee et al.’s line-wrapping task](https://arxiv.org/abs/2601.04480).

- **From characters counted to space remaining.** A recurrent head tracks how many characters have been written. A later memory update brings together positions with the same space left across different line widths. Editing that memory shifts the model’s next-token preference for a line break toward earlier positions. [Follow the 0.8B trace](https://statescope.ai/#example)
- **From a prompt edit to later Python code.** We asked for functions such as turning “Hello World” into “HW”. We kept the wording fixed and edited one state direction as the model read a neutral sentence. Across 72 tasks, outputs containing `raise` or `isinstance()` rose from 31.9% to 48.6% when the edit matched the mean shift from an explicit input-validation instruction. [Compare the 4B prompts and generated functions](https://statescope.ai/#prompt-states)

[![Line-count task at width 60. The share of the mean line-break margin gap carried through six layer-9 MLP transforms was 0.320 for the full mixer swap, 0.300 for the part inside a learned four-dimensional state frame, and at most 0.003 in absolute value for five random frames.](docs/l9-frame.png)](https://statescope.ai/#mixer-to-mlp)

In the counting example, we swapped the layer-9 mixer output between paired texts. Keeping only the change inside one learned four-dimensional subspace preserved **94%** of the full swap’s effect through six selected MLP transforms (0.300 / 0.320). Other components still affected the decision through other paths. [Interactive figure](https://statescope.ai/#mixer-to-mlp) · [Measurements for all three conditions](release/qwen35-08b-v2/line_count_link_table.md)

[Read the interactive article](https://statescope.ai/) · [Download weights](#what-this-is) · [Run the demo](#demonstration-line-counting)

## What this is

Each layer has a dictionary for the residual stream and another for the MLP. The reader exposes the coordinates of a token inside a state group, the transforms that fire, and the vectors they write. You can follow a signal from one to the next and test the proposed path against the model’s own heads. More model releases, including Gemma, are planned.

| Base model | Layers | Width | Groups per state dictionary / active groups per token | MLP transforms per layer | Size | Weights |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | 32 | 2,560 | 16,384 / 24 | 9,920 | About 168 GiB | [mconcat/statescope-qwen3.5-4b](https://huggingface.co/mconcat/statescope-qwen3.5-4b) |
| `Qwen/Qwen3.5-0.8B` | 24 | 1,024 | 16,384 / 24 | 3,968 | About 27.4 GiB | [mconcat/statescope-qwen3.5-0.8b](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b) |

In the Qwen3.5 models, a mixer is a layer's GatedDeltaNet or attention sublayer. A residual stream site is a location between the mixer and the MLP or after the MLP where a state dictionary reads the residual stream. A state dictionary represents the residual stream at both of these sites. An MLP dictionary represents the layer's MLP operations, taking the post-norm MLP input and reconstructing the raw MLP output.

A group is a set of 4 directions in a state dictionary. A frame is the orthonormal basis of a group's 4-dimensional write subspace. A code gives a token's coordinates in that frame, multiplied by a per-group scale. A transform is a linear operation in an MLP dictionary with its own read and write matrices. Its write is what it adds to the MLP output. A gate controls a transform through a learned threshold on its pre-activation.

## What the baselines show

There are several ways to find the count signal. In an earlier exploratory comparison on the same 0.8B line-wrapping task, **eight native MLP neurons recovered essentially the full effect of patching the layer-9 MLP output**. Eight selected latents from three conventional TopK transcoders recovered 26% to 68%, depending on the transcoder. Eight transforms from three earlier MOLT configurations recovered 31% to 48%.

<details>
<summary>Layer-9 comparison on 256 test pairs</summary>

We revised the ranking procedure after inspecting initial results. For the reported comparison, we ranked units on pairs from odd-numbered source lines and patched their writes on pairs from even-numbered lines. Each value is the mean change in line-break margin from patching the selected writes, divided by the mean change from patching the whole layer-9 MLP output. The columns hold the number of patched units fixed. Transforms can have higher rank than individual neurons or transcoder latents.

| Component patched | 2 selected units | 8 selected units |
|---|---:|---:|
| Native MLP neurons | 0.42 | 1.00 |
| Earlier MOLT (186 transforms) | 0.41 | 0.41 |
| Earlier MOLT (41,484 transforms) | 0.31 | 0.31 |
| Same 41,484-transform design with per-transform offsets | 0.42 | 0.48 |
| TopK transcoder, 11 active latents | 0.29 | 0.28 |
| TopK transcoder, 192 active latents | 0.04 | 0.26 |
| TopK transcoder, 900 active latents | 0.26 | 0.68 |

These exploratory runs used earlier dictionaries and a separate evaluation set. The figure above uses the released weights and measures the effect along a selected six-transform path. [Baseline measurements by number of patched units](docs/l9-count-baselines.csv)

</details>

The transcoder with 192 active latents also detected count ranges, with several latents responding to overlapping ranges. With the released dictionaries we could ask a further question. How much of the six transforms’ effect survives if we keep only the change along one learned state subspace? The figure above shows that intervention. For the recurrent step, an earlier stateful transcoder trained on 18 GatedDeltaNet layers reproduced about one third of the counting head’s response. We therefore read that step directly from the model’s head. A matched end-to-end comparison with a conventional scalar sparse autoencoder remains open.

## What you can reproduce

The executable trace covers the **early L8–L9 counting pathway on 0.8B**. For the [later width intervention](https://statescope.ai/#evidence) and [4B steering experiment](https://statescope.ai/#prompt-states), we publish [the figure data](https://github.com/mconcat/statescope-site/tree/main/data). The analysis scripts for those two results and the dictionary training code are unpublished. Both studies used the Qwen3.5 family. Generalization to other model families remains open.

- `python demo/verify_release.py --release <folder>` checks a download of either set against its manifest.
- `python demo/line_count.py --release <folder>` remeasures the early counting pathway.
- `pytest -m "not legacy"` runs the 36 tests that need no private data.

## Install

Python 3.12 or newer is required. This command installs the reader with torch, safetensors and PyYAML:

```sh
pip install -e .
```

This command adds transformers, accelerate and flash-linear-attention, which `capture` and the demonstration need:

```sh
pip install -e '.[model]'
```

This command also adds numpy and pytest:

```sh
pip install -e '.[model,dev]'
```

Without flash-linear-attention, transformers silently uses a slow GatedDeltaNet reference implementation. A warning about a missing `causal_conv1d` means that a slower reference implementation is being used with unchanged values.

The comparisons used Python 3.12.3, torch 2.11.0+cu128, transformers 5.20.2, flash-linear-attention 0.6.0 and safetensors 0.8.0. With newer library versions, the demonstration's shares moved by up to about 0.01, as described in [Verification](#verification).

## Reading the dictionaries

This command downloads the 4B weights with the Hugging Face command line tool:

```sh
hf download mconcat/statescope-qwen3.5-4b --local-dir <folder>
```

This command downloads one layer of the 4B set:

```sh
hf download mconcat/statescope-qwen3.5-4b --include "state/L16/*" "mlp/L16/*" MANIFEST.json --local-dir <folder>
```

This command downloads the 0.8B weights:

```sh
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
```

The downloaded `<folder>` is the folder passed to `--release <folder>` in `demo/verify_release.py` and, for the 0.8B set, `demo/line_count.py`.

The separate weights folder has the same layout for both sets. `<L>` is a layer index from 0 to 31 for the 4B set or 0 to 23 for the 0.8B set, without padding.

```text
MANIFEST.json
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

Only the 4B set includes `ALWAYS_ON.json`, which lists frequently firing MLP transforms and their shares of summed write energy on a web text sample, along with counts on a held-out shard of the training mix.

The 4B set contains 192 weight and config files. Its state dictionaries occupy about 40.0 GiB, and its MLP dictionaries occupy about 128.0 GiB. The 0.8B set contains 144 files. Its state dictionaries occupy about 12.0 GiB, and its MLP dictionaries occupy about 15.4 GiB. The manifest records a sha256 for each layer's files and numbers recorded at training time. This command checks a full download, all 192 files of the 4B set or all 144 files of the 0.8B set, against the manifest:

```sh
python demo/verify_release.py --release <folder>
```

The reader is the single file `src/statescope/reader.py`. It can be copied next to the weights. For Qwen3.5, `capture` returns `resid_mid`, `mlp_out` and `resid_post` for the requested layer with position 0 dropped.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from statescope.reader import StateVocab, PublicMolt, capture

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cuda").eval()
input_ids = torch.tensor([tokenizer("The committee met on Thursday to review the budget.")["input_ids"]], device="cuda")

V = StateVocab("<folder>/state/L16", "cuda")
M = PublicMolt("<folder>/mlp/L16", "cuda")
acts = capture(model, 16, input_ids)                     # resid_mid, mlp_out, resid_post; position 0 dropped
groups, codes = V.encode(acts["resid_post"], "resid_post.16")
x_hat = V.reconstruct(acts["resid_post"], "resid_post.16")
gate, y_hat = M.forward(acts["resid_mid"])               # gate (tokens, 9920), y_hat ~ acts["mlp_out"]
write = M.transform_write(acts["resid_mid"], 410)        # what transform 410 adds to the MLP output
```

The 4B model in bf16 plus one layer's two dictionaries needed more than 14 GiB of GPU memory. The 0.8B set is read the same way with `Qwen/Qwen3.5-0.8B` and its own folder, with 3,968 transforms per layer and layers 0 to 23.

<details>
<summary>Dictionary architecture, training settings and reconstruction measurements</summary>

The state dictionary covers `resid_mid.L` and `resid_post.L`. The former is the input of the norm before the MLP. The latter is `resid_mid` plus the MLP output. The dictionary uses the Grassmannian block-sparse featurizer (BSF) of [Fel et al. (2026)](https://arxiv.org/abs/2606.25234). Each of its 16,384 groups reads and writes through the same orthonormal frame, with one learned scale per group. At each token, the 24 groups with the largest written norm are active. The input is divided by its own size using `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. It is then standardised with a mean vector and one scale per site. The raw residual reconstruction multiplies the reconstruction in those units by the token's own `sqrt(mean(x^2) + eps)`. `StateVocab.reconstruct(x, site)` handles both steps. Each layer's dictionary was trained on 30,003,200 tokens over 7,325 steps.

The MLP dictionary uses the sparse mixture of linear transforms, or MOLT, from the public `crosslayer-transcoder` code. It has 9,920 transforms in the 4B set and 3,968 in the 0.8B set, in five rank tiers. The 4B set has 320 transforms of rank 128, 640 of rank 64, 1,280 of rank 32, 2,560 of rank 16 and 5,120 of rank 8. The 0.8B set has 128 transforms of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16 and 2,048 of rank 8. It uses a JumpReLU gate. Each transform has free read and write matrices. There is no skip connection. Training uses a tanh sparsity penalty. `PublicMolt` applies the layer's own RMSNorm to raw `resid_mid` using coefficients in `postnorm.safetensors`. Training used 58,600 steps of 500 tokens, or 29.3 million tokens, per layer. The learning rate was 4.0e-05. The 4B set used a sparsity penalty coefficient of 3e-05 at all 32 layers. In the 0.8B set, the sparsity penalty coefficient was 7.5e-05 for layers 0 to 22. Layer 23 of the 0.8B set used 2.5e-05 in a separate run. A pre-activation within float rounding of its hard threshold can be on in one computation and off in another.

Position 0 means the first position of each sequence. It was dropped during state dictionary training and included during MLP dictionary training. The raw FVU columns in the model cards drop position 0 for both dictionaries. Measurements with the state dictionary drop position 0. The `capture` function already does this.

FVU means fraction of variance unexplained. It is squared reconstruction error divided by squared deviation from the per-dimension mean of each held-out split. The model cards' state FVU and MLP FVU columns were re-measured with the reader in raw residual and raw MLP output units. They use all tokens of four held-out splits, with position 0 dropped. The splits for the 4B set contain 257,102 tokens per layer, of which 181 are at position 0. The splits for the 0.8B set contain 257,122 tokens per layer, of which 179 are at position 0. Each model card separately gives the MLP FVU recorded at training on different tokens. The held-out shards behind the MLP numbers recorded at training and the held-out activations behind the reconstruction numbers are not published.

This repository measured raw FVU with and without position 0 for the 0.8B set. At layer 6, `resid_mid`, state FVU is 0.2524 with position 0 and 0.2184 without it. At layer 0, it is 0.0778 with position 0 and 0.0777 without it. For the MLP dictionary, the two differ by at most 0.014 at any layer.

The state trainer's held-out FVU was recorded at training in `MANIFEST.json` as `heldout_fvu_own_units`. It uses the dictionary's per-token-RMS-normalised, site-standardised units. It includes position 0 on the first 4,096 tokens of each of four held-out splits. For the 0.8B set, the reader reproduces all 48 numbers within 6.9e-7. At early layers of the 0.8B set, this number is lower than the number on all tokens because the first 4,096 tokens are an easier subset, not because of position 0. At layer 0, `resid_mid`, it is 0.0533 on the subset against 0.0693 on all tokens in the same units, or 0.0777 in raw units. At layer 9, it is 0.2452 on the subset against 0.2438 on all tokens in the same units.

</details>

## Qwen3.5-4B set

The 4B set contains a state dictionary and an MLP dictionary for each of the 32 layers of `Qwen/Qwen3.5-4B`. The dictionaries were trained in 2026-11. Spliced-in loss was measured at layer 16. Its `ALWAYS_ON.json` lists frequently firing MLP transforms, their shares of summed write energy and held-out counts. The set's [model card](https://huggingface.co/mconcat/statescope-qwen3.5-4b) describes its reconstruction measurements and transform usage.

## Demonstration: line counting

The demonstration uses the 0.8B set on texts wrapped to a fixed width. The task comes from Gurnee et al. (2026), "When Models Manipulate Manifolds: The Geometry of a Counting Task" (see [Related work](#related-work)). At the last word of a line, the model must predict a line break, so it must know how many characters the line already holds. Each item pairs 256-token texts ending on the same words. In every cell, the clean text's last line is full, at most 6 characters short of the width, so a line break is due.

The margin is the last-token logsumexp of newline-token logits minus the logsumexp of logits for tokens starting with a space.

```text
m = logsumexp(logits of newline tokens) - logsumexp(logits of tokens that start with a space)
```

A patch replaces a tensor in the clean run with a value built from the corrupt run and moved onto the clean run's token positions. We measure the effect of the patch by comparing mean margins across the paired texts in each condition.

```text
share = mean(m_clean - m_patched) / mean(m_clean - m_corrupt)
```

A share of 0 means the patch leaves the mean margin unchanged. At 1, the patched run has the same mean margin as the corrupt run. Changes to individual items can cancel out in the mean. A share can be negative or exceed 1, and adding shares from overlapping paths can count the same effect twice.

A cell is a task setting defined by the wrapping width and how the corrupt text is made. The three cells are `count:60`, `count:100` and `width:100`. In the count cells, the corrupt text's last line starts at a later word than the clean text's, chosen so that the line holds between 35% and 65% of the width at the last word, and the preceding lines are wrapped backward from that word at the same width, moving their line breaks too. In the width cell, the corrupt text's last line starts at the same word as the clean text's, and the preceding lines are wrapped backward at 1.75 times the width. Each cell contains 64 pairs in `demo/line_count_items.safetensors`, which stores token ids and position pairing. The texts are re-wrapped excerpts from FineWeb-Edu documents.

We found the following chain. The script uses our chosen transform, group and head and re-measures their effects. It does not search for them.

1. Transform 189 of the L8 MLP dictionary fires on the first token of a line. Its write lies mostly inside the frame of state group 11586 of `resid_post.8`, the first-token-of-a-line group.
2. The model's L9 mixer reads that write. Swapping only transform 189's write or only group 11586 moves the answer about as much as swapping the whole L8 MLP output in the count cells. GatedDeltaNet head 10 carries the count inside the mixer.
3. Transforms 410 and 1749 of the L9 MLP dictionary read different parts of the count range and influence the line-break margin. Transform 410 is on while the line is short, and transform 1749 is on once it is long. What the L9 transforms read of the mixer's write lies mostly inside the 4-dimensional frame of state group 4885 of `resid_mid.9`. That part alone moves them about as much as the whole write. The rest moves them little but still reaches the answer through other routes. This is measured for the six selected L9 transforms (410, 1749, 6, 3720, 1394, 2275) together and for 410 and 1749 alone.

This command remeasures the early L8–L9 line-count pathway. It ran in under two minutes on one RTX 5090, including model loading.

```sh
python demo/line_count.py --release <folder with state/ and mlp/> --device cuda:0 --out <folder>
```

<details>
<summary>Patch results, unit selection, controls and pass criteria</summary>

Each row below identifies a swap, and each column identifies a task cell. Each value is a share on even-numbered items, with the bootstrap interval from 2.5 % to 97.5 % in parentheses.

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

The table uses 32 items per cell. Transforms were ranked on the odd-numbered items, and results are read on the even-numbered ones. Some groups were first chosen after looking at all items; repeating the selection on odd-numbered items only gave the same groups, frames and transforms. Head 10 was known from earlier work. In the count cells, transform 189 or group 11586 carries about as much as the whole L8 MLP output through the L9 mixer. Head 10's read alone gives a larger share than the whole mixer output. Transform 410 carries the L9 MLP's effect in `count:60`, and transform 1749 carries it in `width:100`.

Transform 189 fires on 1.000 of line-first tokens, 0.001 of line-break tokens and 0.010 of other tokens. Of the energy in its swapped write, 0.769 lies inside group 11586's frame. The largest value among five random frames is 0.016. The chance value for a 4-dimensional frame in 1,024 dimensions is 0.004.

Transform 410 fires on 0.84 to 0.89 of tokens when a line holds 20 to 49 characters. It fires on 0.25 at 60 to 69 characters and 0.03 at 70 to 79. Transform 1749 fires on 0.00 below 70 characters, 0.19 at 80 to 89 and 0.60 at 90 to 99.

The next table's rows specify changes at the L9 mixer output, and its columns are the same task cells on even-numbered items. The whole mixer swap and head 10's swap are each applied in full, only inside the 4-dimensional frame of state group 4885 of `resid_mid.9`, or as the rest. Cells are written `A / B (C)`. A is the share carried through the writes of the six selected L9 transforms: 410, 1749, 6, 3720, 1394 and 2275. B is the share through transforms 410 and 1749 alone. For A and B, the changed `resid_mid.9` enters the L9 MLP dictionary, and only the change in the selected transforms' writes is patched at the MLP output. C is the share of the same change with everything downstream. The last row has no C.

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

The same computation is also visible in individual MLP neurons.

Further findings:

- Group 4885 is a dense group with 4-dimensional coordinates that are active on most tokens and carry information in those coordinates.
- Single-item margins of dictionary swaps are reproducible to about 0.1. Shares are reproducible to about 0.003. A patched tensor is rounded to bf16. A 1e-6 relative difference between implementations flips some rounded elements.
- We registered a pass rule for the mixer-to-MLP link. The rule requires at least 0.75 of the whole share inside the frame, at most 0.25 of it and at most 0.05 in the rest, and at most 0.01 in random frames, in both count cells. Our verdict was that the rule holds for the route through the six transforms and holds only in part for the route through 410 and 1749 alone, because we recorded a share of 0.051 for the rest in `count:100` through those two transforms.

</details>

## Repository layout

- `src/statescope/reader.py` contains the reader.
- `src/statescope/config.py` defines the two models, and `src/statescope/lm.py` loads the language model.
- `src/statescope/heldout.py` and `src/statescope/metrics.py` are used by the measuring tools and tests.
- `demo/line_count.py` runs the demonstration, and `demo/verify_release.py` checks the weights. `demo/line_count_items.safetensors` holds the demonstration items.
- `release/qwen35-08b-v2/` holds a manifest copy, the model card and generated tables.
- `tests/` holds the tests and `tests/EQUIVALENCE_RULES.md`, which records comparison rules.
- `devtools/` holds tools that stage and measure the published set, and a tool that compares two runs of the demonstration.

An earlier, unpublished design's code and tests are on the branch `earlier-design`; it did not train these weights.

## Tests

Install the `dev` extra for pytest. This command runs the 36 tests that need no private data:

```sh
pytest -m "not legacy"
```

The other tests carry the marker `legacy` and need the frozen reference code, its results and the staged weight folders named in `tests/local.env.example`. This command runs everything:

```sh
pytest
```

A missing required path fails its test. Nothing is skipped silently.

The pass/fail rules were written down before the comparisons were run and are in `tests/EQUIVALENCE_RULES.md`.

## Verification

<details>
<summary>Reader agreement, weight checks and reproduction across library versions</summary>

**Readers**

- For the 0.8B set, the state reader was compared with the reference reader at 24 layers on 4 splits of 4,096 tokens at 2 sites. Of 786,432 tokens, 4 selected a different group set; raw reconstructions elsewhere agreed within 7.1e-7 relative.
- For the 0.8B set, the MLP reader was compared with the unmodified public forward pass at 24 layers. There was 1 differing gate decision, at layer 3; outputs on other tokens agreed within 1.9e-6 relative.
- For the 4B set, the state reader was compared with the reference code at all 32 layers on 4 splits of 4,096 held-out tokens at both sites. There were 36 differing group sets; raw reconstructions elsewhere agreed within 1.7e-6 relative.
- For the 4B set, the MLP reader was compared with the reference code at all 32 layers on 4 splits of 4,096 held-out tokens. There were 5 differing gate decisions; outputs elsewhere agreed within 2.0e-6 relative.

**Published files**

- All 144 files in the 0.8B set match the manifest. MLP tensors at layers 0 to 22 are bit-identical to the training exports; layer 23 passed its comparison with the public forward pass.
- All 192 files in the 4B set match the manifest, and MLP tensors at all 32 layers are bit-identical to the training exports.
- On all 257,122 held-out tokens per layer, the 0.8B state raw FVU agrees with the records within 6.5e-8 at all 24 layers and both sites, against a bound of 1e-4.
- On all 257,102 held-out tokens per layer, the 4B state raw FVU agrees with the records within 1.7e-6 at all 32 layers and both sites, against a bound of 1e-4.

**Demonstration and install**

- Across 351 share comparisons with the reference run and recorded results, the largest difference was 0.0030, against a tolerance of 0.005.
- A fresh clone, a new environment and a fresh download reproduce the demonstration byte for byte with torch 2.11.0+cu128 and transformers 5.20.2.
- With newer library versions, torch 2.15.0+cu132 and transformers 5.23.0, shares differ by at most 0.011.
- The 4B usage lines were run once on the 4B model.

</details>

## Related work

- The state dictionary applies the Grassmannian BSF to the residual stream before and after each MLP, with normalization specific to each site. Thomas Fel et al. (2026), [“Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds.”](https://arxiv.org/abs/2606.25234)
- A related method, SASA, learns subspaces with group-level sparsity and uses a nuclear-norm penalty to regularize their effective rank. Seyed Arshan Dalili and Mehrdad Mahdavi (2026), [“Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability.”](https://arxiv.org/abs/2606.06333)
- The form of the MLP dictionary, a sparse mixture of linear transforms (MOLT), follows this work. Jack Lindsey, Brian Chen, Adam Pearce, Sasha Hydrie, Thomas Conerly; edited by Jeff Wu. ["Sparse mixtures of linear transforms."](https://transformer-circuits.pub/2025/bulk-update/index.html) Transformer Circuits Thread, 2025.
- This is closely related contemporaneous work on decomposing MLP layers. James Oldfield, Shawn Im, Sharon Li, Mihalis A. Nicolaou, Ioannis Patras, Grigorios G Chrysos. ["Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders."](https://arxiv.org/abs/2505.21364) arXiv:2505.21364, 2025 (NeurIPS 2025).
- The demonstration's line-count task comes from this paper on Claude 3.5 Haiku; here the task is followed in Qwen3.5-0.8B, where one head of a GatedDeltaNet layer carries the count. Wes Gurnee, Emmanuel Ameisen, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson. ["When Models Manipulate Manifolds: The Geometry of a Counting Task."](https://arxiv.org/abs/2601.04480) arXiv:2601.04480, 2026.
- The ABA / ABB rule task used with the 4B set comes from this work. Yukang Yang, Declan Campbell, Kaixuan Huang, Mengdi Wang, Jonathan Cohen, Taylor Webb. ["Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models."](https://arxiv.org/abs/2502.20332) arXiv:2502.20332, 2025 (ICML 2025).
- This is related later work on the ABA / ABB rule task. Melody Zixuan Li, Taylor Webb. ["What drives the emergence of symbolic mechanisms in large language models?"](https://arxiv.org/abs/2611.13270) arXiv:2611.13270, 2026.

## License and credits

The dictionaries were trained on `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and on `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.

The code is licensed under MIT in `LICENSE`. The dictionary weights in the Hugging Face repository are released under MIT, as stated by the model card's `license: mit`. The base model `Qwen/Qwen3.5-0.8B` is licensed under the Apache License, Version 2.0. The base model is not part of this set and is not redistributed here.

The file `mlp/L<L>/postnorm.safetensors` holds numbers copied from the base model. Its tensor `w_eff` has 1,024 numbers equal to 1 + the weight of that layer's `post_attention_layernorm`, the RMSNorm before the MLP, and its metadata holds the norm's `eps` (1e-06). Qwen3.5 stores this weight as an offset from 1. No other file of the set holds base-model parameters. All other tensors were trained on activations of the base model.

The base model `Qwen/Qwen3.5-4B` is also licensed under the Apache License, Version 2.0, and the 4B dictionary weights are released under MIT.

The MLP dictionary's form and trainer come from the public `crosslayer-transcoder` code, branch MOLT, commit 376fabe. That code is MIT licensed, Copyright 2025 Georg Lange. The training fork changes no mechanism. The demonstration texts come from `HuggingFaceFW/fineweb-edu`, licensed under ODC-By 1.0.
