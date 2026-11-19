# statescope

statescope is a reader in one Python file and a set of dictionary weights for `Qwen/Qwen3.5-0.8B`, with one state dictionary for two residual stream sites and one MLP dictionary of 3,968 gated low-rank transforms at each of the 24 layers. A second set for `Qwen/Qwen3.5-4B` uses the same reader and is described in [Qwen3.5-4B set](#qwen35-4b-set).

**You can reproduce**

- `python demo/verify_release.py --release <folder>` checks the 144 downloaded files against the manifest.
- `python demo/line_count.py --release <folder>` re-runs the traced line-count chain for the model's line-break decision, from one layer-8 MLP transform through one state group, the model's layer-9 mixer and a second state group to two layer-9 MLP transforms.
- `pytest -m "not legacy"` runs the 36 tests that need no private data.

**You cannot reproduce from the published files**

- Training cannot be reproduced because the trainers for this set are not in the repository.
- The per-layer reconstruction numbers and the spliced-in loss cannot be reproduced because the held-out activations and texts behind them are not published.

**Limits**

- The set covers one model, 0.8B, and one dictionary seed.
- There are no dictionaries for the mixers, which are the GatedDeltaNet and attention sublayers.
- The demonstration covers one easy task with dedicated MLP neurons and is not evidence that dictionary units are better than neurons.
- Every comparison ran on one machine with one RTX 5090.

## What this is

statescope contains a reader for per-layer dictionaries of `Qwen/Qwen3.5-0.8B`. The base model has 24 layers and width 1024. Its revision is `2fc06364715b967f1860aea9cf38778875588b17`. The set was frozen on 2026-11-18. Each layer has a state dictionary and an MLP dictionary that were trained separately.

A mixer is a layer's GatedDeltaNet or attention sublayer. A residual stream site is a location between the mixer and the MLP or after the MLP where a state dictionary reads the residual stream. A state dictionary represents the residual stream at both of these sites. An MLP dictionary represents the layer's MLP operations, taking the post-norm MLP input and reconstructing the raw MLP output. This set contains no dictionaries for the mixers.

A group is a set of 4 directions in a state dictionary. A frame is the orthonormal basis of a group's 4-dimensional write subspace. A code gives a token's coordinates in that frame, multiplied by a per-group scale. A transform is a linear operation in an MLP dictionary with its own read and write matrices. Its write is what it adds to the MLP output. A gate controls a transform through a learned threshold on its pre-activation.

The repository contains the reader, a demonstration and tests. The weights are in [mconcat/statescope-qwen3.5-0.8b](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b).

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

The comparisons used Python 3.12.3, torch 2.11.0+cu128, transformers 5.20.2, flash-linear-attention 0.6.0 and safetensors 0.8.0. With newer library versions, the demonstration's shares moved by up to about 0.01, as described in [What the tests showed for this set](#what-the-tests-showed-for-this-set).

## Reading the dictionaries

This command downloads the weights with the Hugging Face command line tool:

```sh
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
```

The downloaded `<folder>` is the folder passed to `--release <folder>` in `demo/verify_release.py` and `demo/line_count.py`.

The separate weights folder has this layout. `<L>` is a layer index from 0 to 23 without padding.

```text
MANIFEST.json
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

The set contains 144 files. State dictionaries occupy about 12.0 GiB, and MLP dictionaries occupy about 15.4 GiB. The manifest records a sha256 for each layer's files and numbers recorded at training time. This command checks all 144 files against the manifest:

```sh
python demo/verify_release.py --release <folder>
```

The reader is the single file `src/statescope/reader.py`. It can be copied next to the weights. `capture` returns `resid_mid`, `mlp_out` and `resid_post` for the requested layer with position 0 dropped.

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

The state dictionary covers `resid_mid.L` and `resid_post.L`. The former is the input of the norm before the MLP. The latter is `resid_mid` plus the MLP output. The dictionary's tied form has 16,384 groups. At each token, the 24 groups with the largest written norm are active. The input is divided by its own size using `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. It is then standardised with a mean vector and one scale per site. The raw residual reconstruction multiplies the reconstruction in those units by the token's own `sqrt(mean(x^2) + eps)`. `StateVocab.reconstruct(x, site)` handles both steps. Each layer's dictionary was trained on 30,003,200 tokens over 7,325 steps.

The MLP dictionary uses the sparse mixture of linear transforms, or MOLT, from the public `crosslayer-transcoder` code. It has 3,968 transforms in five rank tiers. There are 128 transforms of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16 and 2,048 of rank 8. It uses a JumpReLU gate. Each transform has free read and write matrices. There is no skip connection. Training uses a tanh sparsity penalty. `PublicMolt` applies the layer's own RMSNorm to raw `resid_mid` using coefficients in `postnorm.safetensors`. Training used 58,600 steps of 500 tokens, or 29.3 million tokens, per layer. The learning rate was 4.0e-05. The sparsity penalty coefficient was 7.5e-05 for layers 0 to 22. Layer 23 used 2.5e-05 in a separate run. A pre-activation within float rounding of its hard threshold can be on in one computation and off in another.

Position 0 means the first position of each sequence. The original project dropped it during state dictionary training and included it during MLP dictionary training. The raw FVU columns in the model card drop position 0 for both dictionaries. Drop position 0 when measuring with the state dictionary. The `capture` function already does this.

FVU means fraction of variance unexplained. It is squared reconstruction error divided by squared deviation from the per-dimension mean of each held-out split. The model card's state FVU and MLP FVU columns are this repository's measurements with the published reader in raw residual and raw MLP output units. They use all tokens of the original project's four held-out splits, with position 0 dropped. The splits contain 257,122 tokens per layer, of which 179 are at position 0. The model card separately gives the MLP FVU recorded at training by the original project on different tokens.

This repository measured raw FVU with and without position 0. At layer 6, `resid_mid`, state FVU is 0.2524 with position 0 and 0.2184 without it. At layer 0, it is 0.0778 with position 0 and 0.0777 without it. For the MLP dictionary, the two differ by at most 0.014 at any layer.

The original project also recorded the state trainer's own held-out FVU in `MANIFEST.json` as `heldout_fvu_own_units`. It uses the dictionary's per-token-RMS-normalised, site-standardised units. It includes position 0 on the first 4,096 tokens of each of four held-out splits. This repository's reader reproduces all 48 numbers within 6.9e-7. At early layers, this number is lower than the number on all tokens because the first 4,096 tokens are an easier subset, not because of position 0. At layer 0, `resid_mid`, it is 0.0533 on the subset against 0.0693 on all tokens in the same units, or 0.0777 in raw units. At layer 9, it is 0.2452 on the subset against 0.2438 on all tokens in the same units.

Before publication, the state configs were edited to set `kind` to `state_dictionary` and remove the original project's configuration label, source file path, run label and source-file md5 record. The line naming the original project's data module was removed from the MLP configs, and no safetensors file was changed. The model card's [Config files](release/qwen35-08b-v2/MODEL_CARD.md#config-files) section describes these edits.

## Demonstration: line counting

The task uses texts wrapped to a fixed width. At the last word of a line, the model must predict a line break, so it must know how many characters the line already holds. Each item pairs 256-token texts ending on the same words. In every cell, the clean text's last line is full, at most 6 characters short of the width, so a line break is due.

The margin is the last-token logsumexp of newline-token logits minus the logsumexp of logits for tokens starting with a space.

```text
m = logsumexp(logits of newline tokens) - logsumexp(logits of tokens that start with a space)
```

A patch replaces a tensor in the clean run with a value built from the corrupt run and moved onto the clean run's token positions. A share is the part of the clean/corrupt margin difference carried by the patched tensor.

```text
share = mean(m_clean - m_patched) / mean(m_clean - m_corrupt)
```

A share of 0 means that the patch changes nothing. A share of 1 means that the patch alone turns the clean run into the corrupt one. A cell is a task setting defined by the wrapping width and how the corrupt text is made. The three cells are `count:60`, `count:100` and `width:100`. In the count cells, the corrupt text's last line starts at a later word than the clean text's, chosen so that the line holds between 35% and 65% of the width at the last word, and the preceding lines are wrapped backward from that word at the same width, moving their line breaks too. In the width cell, the corrupt text's last line starts at the same word as the clean text's, and the preceding lines are wrapped backward at 1.75 times the width. Each cell contains 64 pairs in `demo/line_count_items.safetensors`, which stores token ids and position pairing. The texts are re-wrapped excerpts from FineWeb-Edu documents.

The original project found the following chain. The script uses its chosen transform, group and head and re-measures their effects. It does not search for them.

1. Transform 189 of the L8 MLP dictionary fires on the first token of a line. Its write lies mostly inside the frame of state group 11586 of `resid_post.8`, the first-token-of-a-line group.
2. The model's L9 mixer reads that write. Swapping only transform 189's write or only group 11586 moves the answer about as much as swapping the whole L8 MLP output in the count cells. GatedDeltaNet head 10 carries the count inside the mixer.
3. Transforms 410 and 1749 of the L9 MLP dictionary turn the count into the line-break decision. Transform 410 is on while the line is short, and transform 1749 is on once it is long. What the L9 transforms read of the mixer's write lies mostly inside the 4-dimensional frame of state group 4885 of `resid_mid.9`. That part alone moves them about as much as the whole write. The rest moves them little but still reaches the answer through other routes. This is measured for the six L9 transforms selected by the original project (410, 1749, 6, 3720, 1394, 2275) together and for 410 and 1749 alone.

This command runs the demonstration. It ran in under two minutes on one RTX 5090, including model loading.

```sh
python demo/line_count.py --release <folder with state/ and mlp/> --device cuda:0 --out <folder>
```

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

The table uses 32 items per cell. The original project ranked transforms on the odd-numbered items and reads results on the even-numbered ones. Some groups were first chosen after looking at all items; when the original project repeated the selection on odd-numbered items only, the chosen groups, frames and transforms were the same. Head 10 was known from earlier work. In the count cells, transform 189 or group 11586 carries about as much as the whole L8 MLP output through the L9 mixer. Head 10's read alone gives a larger share than the whole mixer output. Transform 410 carries the L9 MLP's effect in `count:60`, and transform 1749 carries it in `width:100`.

Transform 189 fires on 1.000 of line-first tokens, 0.001 of line-break tokens and 0.010 of other tokens. Of the energy in its swapped write, 0.769 lies inside group 11586's frame. The largest value among five random frames is 0.016. The chance value for a 4-dimensional frame in 1,024 dimensions is 0.004.

Transform 410 fires on 0.84 to 0.89 of tokens when a line holds 20 to 49 characters. It fires on 0.25 at 60 to 69 characters and 0.03 at 70 to 79. Transform 1749 fires on 0.00 below 70 characters, 0.19 at 80 to 89 and 0.60 at 90 to 99.

The next table's rows specify changes at the L9 mixer output, and its columns are the same task cells on even-numbered items. The whole mixer swap and head 10's swap are each applied in full, only inside the 4-dimensional frame of state group 4885 of `resid_mid.9`, or as the rest. Cells are written `A / B (C)`. A is the share carried through the writes of the six L9 transforms selected by the original project: 410, 1749, 6, 3720, 1394 and 2275. B is the share through transforms 410 and 1749 alone. For A and B, the changed `resid_mid.9` enters the L9 MLP dictionary, and only the change in the selected transforms' writes is patched at the MLP output. C is the share of the same change with everything downstream. The last row has no C.

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

The demonstration has these caveats:

- It covers one model, 0.8B, and one dictionary seed, with texts wrapped to a fixed width.
- This is an easy task for which the model has dedicated MLP neurons. The demonstration is not evidence that dictionary units are better than neurons.
- State group 11586 is sparse, but not every group is sparse. Some groups encountered by the original project on this task are dense 4-dimensional coordinates active on most tokens. They carry information in their coordinates. Group 4885 is one of these dense groups, not a sparse feature.
- Single-item margins of dictionary swaps are reproducible only to about 0.1. Shares are reproducible only to about 0.003. A patched tensor is rounded to bf16. A 1e-6 relative difference between implementations flips some rounded elements.
- The original project registered a pass rule for the mixer-to-MLP link before measuring it. The part inside the frame must keep at least 0.75 of the whole share. The rest must keep at most 0.25 of it and at most 0.05. Random frames must carry at most 0.01. The rule applies in both count cells. The original project's verdict was that the rule holds for the route through the six transforms and holds only in part for the route through 410 and 1749 alone, because it recorded a share of 0.051 for the rest in `count:100` through those two transforms.

## Qwen3.5-4B set

A second set for `Qwen/Qwen3.5-4B`, with 32 layers and width 2,560, is published at [mconcat/statescope-qwen3.5-4b](https://huggingface.co/mconcat/statescope-qwen3.5-4b) and occupies about 168 GiB. It uses the same forms, reader and folder layout, with `ALWAYS_ON.json` added and 9,920 MLP transforms per layer. It is a copy made on 2026-11-20 of what the original project trained with one seed, rather than a frozen configuration like the 0.8B set. There is no demonstration for it; `demo/line_count.py` is for 0.8B only.

Read its model card's limits before use: reconstruction is worse than for 0.8B, with state FVU up to 0.42 against about 0.25 to 0.27 in the middle layers of 0.8B, and the original project measured spliced-in loss at one layer only. Many MLP transforms never fired on the original project's held-out shard, up to 60% at the last layer, while at most layers a few transforms fire on almost every token and carry a large share of the write energy; `ALWAYS_ON.json` lists them.

This repository tested the set under rule R10, written before the runs, on 2026-11-20 with one RTX 5090 per run:

- Readers were compared with the original project's code at all 32 layers on 4 splits of 4,096 held-out tokens. There were 36 differing state group sets and 5 differing MLP gate decisions, within the rule's limits. Raw state reconstructions elsewhere agreed within 1.7e-6 relative and MLP outputs within 2.0e-6 relative. The trainer's 64 held-out numbers were reproduced within 8.8e-6, against a limit of 1e-5.
- Reconstruction error was measured on all 257,102 held-out tokens per layer. State raw FVU agreed with the original project's records within 1.7e-6 at all 32 layers and both sites, against a limit of 1e-4.
- All 192 files matched the manifest, and MLP tensors were bit-identical to the training exports at all 32 layers. The configs were the originals with the stated edit. No machine path or removed name was found.
- The usage lines were run once on the 4B model. Its `capture` was not compared with the original project's capture. The model in bf16 plus one layer's two dictionaries needed more than 14 GiB of GPU memory.

## Repository layout

- `src/statescope/reader.py` contains the reader.
- `src/statescope/config.py` defines the two models, and `src/statescope/lm.py` loads the language model.
- `src/statescope/heldout.py` and `src/statescope/metrics.py` are used by the measuring tools and tests.
- `demo/line_count.py` runs the demonstration, and `demo/verify_release.py` checks the weights. `demo/line_count_items.safetensors` holds the demonstration items.
- `release/qwen35-08b-v2/` holds a manifest copy, the model card and generated tables.
- `tests/` holds the tests and `tests/EQUIVALENCE_RULES.md`, which records comparison rules.
- `devtools/` holds tools that stage and measure the published set, and a tool that compares two runs of the demonstration.

## Tests

Install the `dev` extra for pytest. This command runs the 36 tests that need no private data:

```sh
pytest -m "not legacy"
```

The other tests carry the marker `legacy` and need the frozen original code, the original project's results and the staged weight folders named in `tests/local.env.example`. This command runs everything:

```sh
pytest
```

A missing required path fails its test. Nothing is skipped silently.

## What the tests showed for this set

- All comparisons ran on one RTX 5090 machine with Qwen3.5-0.8B. Rules R8 and R9 were written and committed before the runs they judge.
- The whole 0.8B suite was run again on 2026-11-20 after the earlier design moved off the main branch and the published configs were edited. All 167 tests passed: the 36 that need no private data and 131 that do. In comparisons with the original project's code, that code read the original files and this repository's reader read the published copy with the edited configs, as required by rule R11.4, written before the run.
- The state reader was compared with a reader the original project wrote separately from its trainer. This covered 24 layers and 4 splits of 4,096 tokens at 2 sites. Of 786,432 tokens, 4 selected a different group set. Each had its 24th and 25th group norms within 7.3e-7. Raw reconstructions elsewhere agreed within 7.1e-7 relative. All 48 of the trainer's held-out FVU numbers in its own units were reproduced within 6.9e-7. These use the first 4,096 tokens of each split with position 0 included.
- The MLP reader was compared with the unmodified public forward pass at 24 layers. There was 1 differing gate decision, at layer 3. Outputs on other tokens agreed within 1.9e-6 relative.
- The sum of firing transforms' writes plus the output mean matched the forward pass within 1.5e-6 relative. The largest difference was 1.476e-6 at layer 22.
- This repository measured reconstruction error on all held-out tokens, 257,122 per layer. The state dictionary's raw FVU agreed with the original project's table within 6.5e-8 at all 24 layers and both sites. The limit was 1e-4. Repeating the measurement with the reduced reader on the published folder gave identical values.
- All 144 release files matched the manifest. MLP tensors for layers 0 to 22 were bit-identical to the training exports used by the reader tests. Layer 23 passed its own comparison with the public forward pass. No file header or config held a machine path.
- Rule R11 was written and committed before the configs were rewritten. At all 24 layers, the parsed published `cfg.json` equalled the original with only the stated edit, and the published `config.yaml` equalled the original project's path-free copy byte for byte without the one removed line. The state dictionaries' tensor files were the original files. None of the removed names occurred in any config, any safetensors header, including tensor names and metadata, the manifest or the model card.
- Every demonstration share agreed within 0.0030 with the original harness's reference run and the original project's recorded numbers. There were 39 swaps x 3 cells x 3 halves = 351 comparisons. The original registration's tolerance was 0.005. The item file matched the original items exactly. With the edited configs and the reduced reader, the demonstration wrote a result byte-identical to the run tested before the changes, as checked with `devtools/compare_demo_runs.py`, so these comparison numbers are unchanged.
- The capture comparison was run again with the reduced reader. `resid_mid` and the MLP output were bit-identical to the original project's capture at layers 0 and 19.
- The weights repository was uploaded again with the edited configs, and its earlier history was removed, leaving one commit. The 96 safetensors files on the server had the sha256 recorded in the manifest. Both repositories were made public on 2026-11-20.
- On 2026-11-20, after the re-upload, the code was cloned from GitHub into an empty folder and installed into a new environment, and the weights were downloaded into an empty folder: 147 files in 16 min 46 s. Installing with `pip install -e '.[model,dev]'` gave torch 2.15.0+cu132, transformers 5.23.0, flash-linear-attention 0.6.0 and safetensors 0.8.0. The 36 tests that need no private data passed. The check also found that loading the model with `device_map` needs `accelerate`, which the `model` extra did not install. It was added to that extra; the tested environment already had it.
- `demo/verify_release.py` reported that all 144 downloaded files matched the manifest. The downloaded `README.md` and `MANIFEST.json` were byte-identical to the repository's card and the staged manifest. With the cloned code and downloaded weights, `demo/line_count.py` in the tested versions, torch 2.11.0+cu128 and transformers 5.20.2, wrote a result byte-identical to the tested run.
- In the new environment with torch 2.15.0+cu132 and transformers 5.23.0, the same demonstration ran in 1 min 49 s and was not byte-identical. The model's unpatched single-item margins differed by at most 0.146, and the 351 shares differed from the tested run by at most 0.011. The largest share difference was on a swap measuring the rest outside group 4885's frame: 0.009 in the tested run and 0.020 in the new environment. The documented numbers are exact for the tested versions; newer versions moved shares by up to about 0.01. This check still used only one machine and one GPU model, RTX 5090.

## What is not shown

- The trainers for this set are absent. Porting them is planned.
- The original project recorded MLP FVU, transforms on per token and transforms alive at training on a held-out shard of each layer's training capture. Position 0 was included. The shards were subsequently deleted, including the shard for layer 23's separate run. These training-time numbers cannot be recomputed from the published files. Transforms on per token is the mean number of open gates. Transforms alive counts those that fired at least once on that shard. The others never fired there. At layer 9, 2035 of 3968 were alive on 267,271 tokens. Layer 23's shard also held 267,271 tokens. The held-out activations behind this repository's raw state and MLP FVU measurements are not published, so users cannot recompute those exact numbers either.
- Spliced-in loss is the increase in next-token cross-entropy when a dictionary reconstruction replaces the true activation in the forward pass, one layer and one dictionary at a time. The original project measured it in nats on text with clean cross-entropy 2.808. It used FineWeb-Edu rows outside the training mixture, taking the first 100 documents of at least 256 tokens and the first 256 tokens of each. Position 0 was not replaced. Loss was averaged over positions 1 to 254, or 25,400 positions. The dictionaries are the released ones. The original project separately measured the released layer 23 MLP dictionary on the same kind of web text, 100 documents of 256 tokens, and recorded +0.0502. In the original project's comparison on the same text, the earlier set had smaller increases at all 24 state layers and MLP layers 0 to 22. For the state dictionary at layer 9, it recorded +0.243 here against +0.161 for the earlier set. At layer 23, the released MLP dictionary had the smaller increase, +0.0502 against +0.0840 for the earlier set.
- This repository did not repeat the original project's training-time MLP held-out measurements or the spliced-in loss measurements.
- No full-length training run was repeated for either design.
- The demonstration and the 0.8B numbers do not carry over to the separate Qwen3.5-4B set.

## Earlier design

The earlier design used an untied state dictionary and an M1w MLP dictionary with 41,484 atoms. Its code and tests are on the branch `earlier-design`. It did not train these weights.

## License and credits

The code is licensed under MIT in `LICENSE`. The dictionary weights in the Hugging Face repository are released under MIT, as stated by the model card's `license: mit`. The base model `Qwen/Qwen3.5-0.8B` is licensed under the Apache License, Version 2.0. The base model is not part of this set and is not redistributed here.

The file `mlp/L<L>/postnorm.safetensors` holds numbers copied from the base model. Its tensor `w_eff` has 1,024 numbers equal to 1 + the weight of that layer's `post_attention_layernorm`, the RMSNorm before the MLP, and its metadata holds the norm's `eps` (1e-06). Qwen3.5 stores this weight as an offset from 1. No other file of the set holds base-model parameters. All other tensors were trained on activations of the base model.

The MLP dictionary's form and trainer come from the public `crosslayer-transcoder` code, branch MOLT, commit 376fabe. That code is MIT licensed, Copyright 2025 Georg Lange. The training fork changes no mechanism. The demonstration texts come from `HuggingFaceFW/fineweb-edu`, licensed under ODC-By 1.0.
