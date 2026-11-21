---
license: mit
base_model: Qwen/Qwen3.5-4B
---

# Qwen3.5-4B per-layer dictionaries

This statescope set for Qwen3.5-4B has two dictionaries per layer and a Python reader. Sets for other models are planned. The state dictionary reconstructs the residual stream, the vector layers pass along, using a few active groups, each a set of four learned directions selected together. Those directions form its frame, and a token's four coordinates in the frame are its code. The MLP (multilayer perceptron) dictionary reconstructs the layer's MLP output as a sum of transforms, low-rank linear operations on the input. Each transform has a gate that opens when a reading along one input direction exceeds a learned threshold. The gate scales the transform's write.

Our in-depth study of these dictionaries follows [line counting in the Qwen3.5-0.8B set](https://statescope.ai/#example).

### What you can do with it

- You can inspect active groups on a token (a piece of text) and reconstruct its activation (internal vector).
- You can inspect which transforms fire and what each adds to the MLP output.
- You can change a group's code (its coordinates) or a transform's gate and decode back to vectors.

### What these units have been used for

**Input validation in Python.** We projected the mean state difference between paired validation and neutral prompts into group 2879's frame at `resid_post.10`, just after the layer-10 MLP. We kept the neutral prompt text fixed and added the vector to the state at each of the 17 tokens of its neutral sentence. One addition at the vector's measured size is the mean dose. We scored the generated code and ran the functions. Brackets give 95% intervals over tasks.

| Measure | No edit | Mean dose | Twice the dose |
|---|---:|---:|---:|
| Code contains `raise` or `isinstance` | 31.9% [21.8, 42.6] | 48.6% [37.5, 59.3] | 72.7% [63.9, 81.0] |
| Wrong-type inputs rejected | 5.3% | 12.4% | 51.0% |
| Wrong-value inputs rejected | 56.9% | 84.6% | 94.0% |
| Functions passing every valid case | 94.4% | 90.7% | 88.9% |

With the instruction written into the prompt, 67.1% of functions passed every valid case. On two other neutral sentences, the same edit increased the fraction containing `raise` or `isinstance` by +4.2 and +6.5 percentage points at the mean dose and by +28.7 and +14.8 points at twice the dose, respectively.

**ABA and ABB rules.** The third word repeats the first in ABA and the second in ABB. Two directions in group 12487 track whether a word repeats and which word it repeats. Layer-15 attention head 14 reads both. Moving the state it reads toward the other rule reproduced 0.24 to 0.39 of the unedited mean difference between rules inside four frames at the answer position. Along the full rule direction the fractions were 0.20 and 0.26 for the two directions of the edit.

**Digit comparison.** Group 9719 at `resid_mid.7`, just before the layer-7 MLP, separates smaller, equal and larger digits in the same slot on successive lines. Transform 357 reads whether the current digit is smaller and 472 reads whether it is larger. The parts of their read directions inside the group's frame have cosine -0.90. The full directions have cosine +0.06.

**A randomly drawn group.** In group 15417 at `resid_post.27`, we moved one coordinate of “vital” to match “fatal”, “deadly”, “lethal” or “mortal” in six sentence templates. We compared the summed probabilities of two sets of eight nouns. The log ratio shifted toward the set associated with death words in all 24 pairs, by +1.41 nats on average. In “It was a W”, all 41 adjectives had negative second coordinates. “Vital”, “fatal” and “pivotal” were the most negative.

**Python rule completion.** We asked for a function that fit a few input-output cases. Moving the state inside group 9167's frame at `resid_mid.27` just before the function body led the model to reason in comments. Across 24 held-out rule families the correct-rule rate rose from 0.516 to 0.773. Forcing the first body token to `#` gave a similar rate in a separate comparison.

**JavaScript comparison.** At layer 7, the MLP reconstruction carried 0.89 of the MLP's own contribution to a hand-traced JavaScript comparison.

### Try it

In the [code checkout](https://github.com/mconcat/statescope), these commands install dependencies and download one layer: about 1.25 gibibytes (GiB) for state and 4.0 GiB for MLP files. `<folder>` is the destination.

```sh
pip install -e '.[model]'
hf download mconcat/statescope-qwen3.5-4b --include "state/L16/*" --include "mlp/L16/*" --include MANIFEST.json --local-dir <folder>
```

The full download is about 168 GiB (180 gigabytes, GB).

```sh
hf download mconcat/statescope-qwen3.5-4b --local-dir <folder>
```

This example reconstructs a layer's activations:

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
x_hat = V.reconstruct(acts["resid_post"], "resid_post.16")
gate, y_hat = M.forward(acts["resid_mid"])               # gate (tokens, 9920), y_hat ~ acts["mlp_out"]
```

The model in bfloat16 (bf16), together with one layer's two dictionaries, needed more than 14 GiB of graphics processing unit (GPU) memory when we tested these lines.

The file check compares a full download against the manifest's file hashes:

```sh
python demo/verify_release.py --release <folder>
```

### What is in the box

- The 192 weight and config files cover 32 layers; the manifest, transform firing record and model card accompany them.
- State files occupy about 40.0 GiB (43.0 GB) and MLP files about 128.0 GiB (137.5 GB).
- Each state dictionary activates 24 of 16,384 groups per token.
- Each MLP dictionary has 9,920 transforms.
- We read each mixer, the GatedDeltaNet or attention sublayer that combines information across tokens, directly through the model's heads and channels. The dictionaries cover the residual stream and the MLPs. On the 0.8B model, a stateful transcoder trained on 18 GatedDeltaNet layers reproduced 33 to 47 percent of the counting head's response to a swap of the layer-8 MLP output and 68 to 74 percent of its response to a swap of the whole residual. The counting head carried the count about as well as the whole mixer.

A smaller set for [Qwen3.5-0.8B](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b) is published separately and comes with a runnable demonstration.

## For agents reading this repo

### Files

The [weights](https://huggingface.co/mconcat/statescope-qwen3.5-4b) use unpadded layer indices `<L>` from 0 to 31:

```text
MANIFEST.json
ALWAYS_ON.json
README.md
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

- `MANIFEST.json`: file hashes and numbers recorded at training; `frozen` holds the date the files were assembled, 2026-11-20.
- `ALWAYS_ON.json`: firing lists and write-energy shares from 65,408 FineWeb-Edu web-text tokens; separate counts from 65,536 held-out training-mix tokens.
- `README.md`: model card.
- `state/L<L>/cfg.json`: form, sites, sizes and training options.
- `state/L<L>/weights.safetensors`: encoder, decoder, bias, site means and scales.
- `state/L<L>/usage.safetensors`: about 0.5 megabytes (MB) of training counters: selections `fire`, window token totals `n_tokens`, running selection rates `use`, tokens since selection `since`.
- `mlp/L<L>/config.yaml`: MLP configuration.
- `mlp/L<L>/weights.safetensors`: gates, read/write matrices and input/output statistics.
- `mlp/L<L>/postnorm.safetensors`: pre-MLP root mean square normalization (RMSNorm) coefficients `w_eff` and metadata `eps`.

### Load and call

In the [code checkout](https://github.com/mconcat/statescope), which holds the reader and its tests (training code will follow):

```sh
pip install -e '.[model,dev]'
```

The [Try it](#try-it) example loads the model and dictionaries. These calls are in `statescope.reader`; `V` and `M` are its dictionary instances. `Tensor = torch.Tensor`, `Device = str | torch.device`; shapes use tokens `n`, width `d = 2560`, groups `G = 16384`, directions `b = 4`, active groups `k = 24`, transforms `T = 9920`.

- `StateVocab(path: str, device: Device = "cpu")`: state folder.
- `V.token_rms(x: Tensor) -> Tensor`: raw `(n, d)` → root mean square (RMS) `(n, 1)`.
- `V.standardize(x: Tensor, site: str) -> Tensor`: raw `(n, d)` → standardized `(n, d)`.
- `V.encode(x: Tensor, site: str, chunk: int = 1024) -> tuple[Tensor, Tensor]`: raw `(n, d)` → `torch.long` group indices `(n, k)`, codes `(n, k, b)`.
- `V.decode(groups: Tensor, codes: Tensor, site: str, rms: Tensor, chunk: int = 1024) -> Tensor`: indices, codes, RMS `(n, 1)` → raw `(n, d)`.
- `V.reconstruct(x: Tensor, site: str) -> Tensor`: raw `(n, d)` → raw `(n, d)`.
- `V.group_write(group: int, code: Tensor, site: str, rms: Tensor) -> Tensor`: code `(b,)` → raw contribution `scale_site * (code @ Q[group]) * rms`; `(d,)` before RMS scaling, excluding mean/bias.
- `PublicMolt(path: str, device: Device = "cpu", post_norm: str | None = None, chunk: int = 512)`: MLP folder; default norm `postnorm.safetensors`.
- `M.standardize_input(x: Tensor) -> Tensor`: raw `(n, d)` → standardized `(n, d)`.
- `M.gate(z: Tensor) -> Tensor`: standardized `(n, d)` → gates `(n, T)`.
- `M.forward(x: Tensor, g: Tensor | None = None) -> tuple[Tensor, Tensor]`: raw `(n, d)` → gates `(n, T)`, raw output `(n, d)`; accepts edited gates.
- `M.transform_write(x: Tensor, transform: int) -> Tensor`: raw `(n, d)` → contribution `(n, d)`, zero when off.
- `capture(model: torch.nn.Module, layer: int, input_ids: Tensor, attention_mask: Tensor | None = None) -> dict[str, Tensor]`: IDs/mask `(batch, sequence)` → `resid_mid`, `mlp_out`, `resid_post`, each 32-bit floating-point `(tokens, d)`.
- `decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList`: `model.model.layers`.
- `write_post_norm(model: torch.nn.Module, layer: int, path: str) -> None`: writes layer RMSNorm coefficients.
- `fvu(x: Tensor, x_hat: Tensor) -> float`: `(n, d)` arrays → fraction of variance unexplained (FVU).

### Conventions

- `resid_mid.L` is between the mixer (GatedDeltaNet or attention sublayer) and MLP, at `post_attention_layernorm` input; `resid_post.L = resid_mid + mlp_out`. One state dictionary covers both sites.
- The state dictionary reads the residual stream before any norm module, divided by its own size: `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. This is the model's RMSNorm without its learned weight; site standardization follows. The MLP dictionary reads the same quantity multiplied by that weight (`w_eff`), which is the model's own post-norm MLP input. `reconstruct` returns raw units; manual decoding takes `rms=V.token_rms(x)`.
- `capture` drops position 0 and masked padding. State training drops position 0; MLP training includes it. This training convention is inferred from shared code.
- FVU is squared error divided by squared deviation from the per-dimension mean. State/MLP FVU uses raw residual/output units; `heldout_fvu_own_units` uses RMS-normalized, site-standardized state units. Transforms on per token is a mean; alive is a count; penalty is a sparsity coefficient; spliced-in loss is in nats (natural-log cross-entropy).
- Transforms are stored largest rank first: 320 × rank 128, 640 × 64, 1,280 × 32, 2,560 × 16, 5,120 × 8. Rank is the intermediate dimension; `M.rank` has shape `(T,)`.
- `PublicMolt` takes raw `resid_mid` and applies the layer's norm from `postnorm.safetensors`, then input standardization; output statistics restore raw MLP units.
- Standardizer tables hold 24 rows at layers 0–23 and 32 at layers 24–31; only the layer's own row is filled and read.
- A few transforms fire on almost every token and carry a large share of summed write energy; `ALWAYS_ON.json` lists them.

### Verify

For a full download, from the code checkout:

```sh
python demo/verify_release.py --release <folder>
pytest -m "not legacy"
```

The hash check prints `layer <L>: checked`, then `all 192 files match the manifest`. The pytest command selects tests that run without private data.

## Appendix: details of the measurements

### Where the numbers come from

We trained the dictionaries and made the first measurements with research code that is not published. Some numbers were recorded then ("recorded at training"); others were measured again with the published reader on the published files ("re-measured").

### State dictionary

Group 12112 at `resid_post.13` is one state-dictionary unit: four directions that represent part of a token's residual vector after layer 13. From the normalized input, it computes four numbers, its code: the token's coordinates along those directions multiplied by a scale for the group. The group is "active" when its contribution is among the 24 largest by length on that token. If active, it weights its four directions by those numbers to contribute a vector to the reconstruction. We add the selected contributions and restore the token's original units to reconstruct its residual vector. In our rule experiment, shifting the residual within this group's four directions by the mean difference between rules changed ABA (repeat the first item) to ABB (repeat the second) in 0.45 of prompts.

The dictionary normalizes each token by its RMS with `eps = 1e-6`, then applies a site mean and scale. The normalized input is the model's RMSNorm without its learned weight; the MLP dictionary's input, below, includes that weight.

The formula shows how codes reconstruct a token's residual, with write frames `Q`, code maps `F`, and decoder bias `b_dec`:

```text
x_rms = x * rsqrt(mean(x^2) + eps)
xt = (x_rms - mu_site) / scale_site - b_dec
c_g = F_g xt
active = the k groups with the largest |c_g|
x_hat = (mu_site + scale_site * (sum over active g of c_g Q_g + b_dec)) * rms(x)

sites       ["resid_mid.L", "resid_post.L"]
G, b, d, k  16384, 4, 2560, 24
Q           (G, b, d), with orthonormal rows
F           (G, b, d)
groups      (n, k)
codes       (n, k, b)
```

**Settings**

| Setting | Value |
|---|---|
| Groups per layer | 16,384 |
| Directions per group | 4; an orthonormal frame (perpendicular unit directions) |
| Model width | 2,560 |
| Form | Tied, block-sparse; shared read/write frames |
| Selection | Top-k by written norm; 24 active groups per token |
| Active dimensions | 96, or 3.75% of model width |
| Training per layer | 30,003,200 tokens over 7,325 steps |
| Seed | 0 |
| Position 0 in training | Dropped; inferred from shared code |
| File sizes | About 1.25 GiB per layer; about 40.0 GiB (43.0 GB) for all state files |

The default of 24 active groups came from comparing 24, 40 and 60 at layers 7, 16, 19 and 23 on one JavaScript ternary expression, which chooses between two values according to a condition.

### MLP dictionary

One MLP-dictionary unit is a transform that reads one token's MLP input and writes a contribution to the reconstructed MLP output; transform 357 at layer 7 ranked first in our hand-traced JavaScript comparison. Its rank of 64 means it maps the normalized and standardized input to 64 intermediate numbers, then maps those numbers to an output vector. It also computes a gate score from the input and "fires" only when that score exceeds both its learned threshold and zero. When it fires, the score scales the vector it writes; when it is off, it writes zero. We sum the transforms' writes and restore output units to reconstruct the MLP output. We recorded a mean of 31.6 firing transforms per token at layer 7 on the held-out training mix. The form follows "Sparse mixtures of linear transforms" (Lindsey et al., 2025); the Mixture of Decoders of Oldfield et al. (2025) is closely related work from the same period.

The reader maps raw `resid_mid` through the layer's RMSNorm, using `w_eff = 1 + weight` of `post_attention_layernorm` and `eps = 1e-06`. In the formula, `z` is standardized input, `h_t` is gate pre-activation, `theta_t` its learned threshold, and `V_t`/`U_t` the read/write matrices.

The formula shows how each gate scales a transform's write and how we reconstruct the MLP output:

```text
u = x * rsqrt(mean(x^2) + eps) * w_eff
z = (u - mu_in) / sd_in
h_t = e_t . z + b_t
g_t = h_t if h_t > theta_t and h_t > 0 else 0
y_std = sum over transforms t of g_t * (z V_t) U_t
y = y_std * sd_out + mu_out

V_t has shape (d, rank_t); U_t has shape (rank_t, d).
```

**Settings**

| Setting | Value |
|---|---|
| Form | Sparse mixture of linear transforms (MOLT) |
| Transforms per layer | 9,920 |
| Rank tiers, largest first | 320 × rank 128; 640 × 64; 1,280 × 32; 2,560 × 16; 5,120 × 8 |
| Read and write matrices | Free |
| Gate | JumpReLU; pre-activation passes through when it exceeds both its learned threshold and zero |
| Skip connection | No linear skip part |
| Sparsity penalty | Hyperbolic tangent (`tanh`); coefficient 3e-05 at all 32 layers |
| Training | 58,600 steps of 500 tokens; 29.3 million tokens |
| Learning rate | 4.0e-05 |
| Position 0 in training | Included; inferred from shared code |
| File sizes | About 4.0 GiB per layer; about 128.0 GiB (137.5 GB) for all MLP files |

Layer 16 comes from a separate run of the same recipe; its config matches the others apart from layer index, logger name and standardizer row count. We chose the penalty at layer 16 alone: spliced-in loss was +0.0068 nat at 3e-05, versus +0.0095 nat at 7.5e-05.

Rounding near thresholds can change gate decisions and group selections.

`ALWAYS_ON.json` lists transforms firing on at least 90% of tokens and those firing on 50 to 90%, with each transform's share of summed write energy. Its web-text sample has 128 FineWeb-Edu documents of 512 tokens, giving 65,408 after position 0 is dropped; `training_mix_heldout_counts` covers 65,536 training-mix tokens including position 0. Counts differ at 17 layers.

Counted with our research code after training: examples of counts firing on at least 90% of tokens, for those two texts.

| Layer | FineWeb-Edu web text sample | Held-out training-mix shard |
|---|---|---|
| 7 | 18 | 18 |
| 13 | 11 | 6 |
| 19 | 10 | 5 |
| 30 | 3 | 0 |

On web text, these transforms carry up to 0.69 of summed write energy, at layer 11. On the training mix, they carry up to 0.68, at layer 7, and at least 0.30 at 21 of the 32 layers. We trace them as one dense background block; individual transforms need not correspond to single circuit operations.

A transform is counted as dead when it never fires on the held-out training shard. The dead fraction was 0% at layer 0, 35 to 37% at layers 11 to 13, 40 to 43% at layers 14 to 16, and 60% at layer 31, where 3,974 of 9,920 were alive. These counts describe that shard.

### Numbers per layer

Raw FVU pools squared errors over four held-out calibration splits and divides by the sum of squared deviations from each split's per-dimension mean. The splits contain 257,102 tokens per layer, including 181 at position 0, leaving 256,921 after dropping it.

- **Layer**: model layer index, 0 to 31.
- **State FVU, `resid_mid`**: re-measured in raw residual units between mixer and MLP, on 256,921 calibration tokens per layer, position 0 dropped.
- **State FVU, `resid_post`**: re-measured in raw residual units after the MLP, on the same 256,921 tokens.
- **MLP FVU**: re-measured in raw MLP output units on the same 256,921 tokens.
- **MLP FVU recorded at training**: raw MLP reconstruction FVU on 65,536 held-out training-capture tokens, position 0 included.
- **MLP transforms on per token**: recorded at training; mean firing count on those 65,536 tokens.
- **MLP transforms alive**: recorded at training; transforms firing at least once out of 9,920 on those 65,536 tokens.
- **MLP penalty**: recorded at training; sparsity coefficient from configs, a configuration value rather than a token measurement.
- **MLP transforms on at 90% of tokens or more (web text / training mix)**: counted with our research code after training, on 65,408 FineWeb-Edu tokens with position 0 dropped / on 65,536 held-out training-mix tokens with position 0 included.

The two MLP FVU columns use different texts. The training-mix shard was deleted, and the calibration activations are unpublished, so the exact values cannot be recomputed from the published files.

| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP FVU | MLP FVU recorded at training | MLP transforms on per token | MLP transforms alive | MLP penalty | MLP transforms on at 90% of tokens or more (web text / training mix) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0988 | 0.1112 | 0.0701 | 0.0849 | 30.4 | 9894 of 9920 | 3e-05 | 0 / 0 |
| 1 | 0.1548 | 0.1737 | 0.1540 | 0.1724 | 21.5 | 9894 of 9920 | 3e-05 | 0 / 0 |
| 2 | 0.2081 | 0.2013 | 0.2002 | 0.2234 | 15.7 | 9880 of 9920 | 3e-05 | 0 / 0 |
| 3 | 0.2208 | 0.2272 | 0.2294 | 0.2478 | 19.7 | 9778 of 9920 | 3e-05 | 1 / 1 |
| 4 | 0.2463 | 0.2620 | 0.2713 | 0.2890 | 21.4 | 9676 of 9920 | 3e-05 | 2 / 2 |
| 5 | 0.2904 | 0.2819 | 0.2388 | 0.2595 | 25.1 | 9493 of 9920 | 3e-05 | 7 / 7 |
| 6 | 0.3186 | 0.2913 | 0.2027 | 0.2097 | 30.2 | 9090 of 9920 | 3e-05 | 15 / 14 |
| 7 | 0.3331 | 0.3146 | 0.2167 | 0.2284 | 31.6 | 8573 of 9920 | 3e-05 | 18 / 18 |
| 8 | 0.3266 | 0.3162 | 0.2440 | 0.2593 | 31.4 | 8343 of 9920 | 3e-05 | 17 / 17 |
| 9 | 0.3373 | 0.3338 | 0.2843 | 0.2970 | 27.7 | 7918 of 9920 | 3e-05 | 12 / 11 |
| 10 | 0.3495 | 0.3364 | 0.2528 | 0.2635 | 29.8 | 7227 of 9920 | 3e-05 | 14 / 13 |
| 11 | 0.3710 | 0.3496 | 0.2660 | 0.2696 | 28.9 | 6492 of 9920 | 3e-05 | 14 / 12 |
| 12 | 0.3650 | 0.3491 | 0.2770 | 0.2773 | 28.0 | 6478 of 9920 | 3e-05 | 12 / 10 |
| 13 | 0.3657 | 0.3530 | 0.2898 | 0.2932 | 27.0 | 6280 of 9920 | 3e-05 | 11 / 6 |
| 14 | 0.3674 | 0.3556 | 0.2711 | 0.2832 | 26.0 | 5919 of 9920 | 3e-05 | 9 / 7 |
| 15 | 0.3833 | 0.3663 | 0.2579 | 0.2629 | 24.8 | 5643 of 9920 | 3e-05 | 10 / 7 |
| 16 | 0.3787 | 0.3623 | 0.2451 | 0.2520 | 23.7 | 5923 of 9920 | 3e-05 | 6 / 6 |
| 17 | 0.3712 | 0.3491 | 0.2295 | 0.2393 | 23.5 | 6112 of 9920 | 3e-05 | 7 / 4 |
| 18 | 0.3639 | 0.3371 | 0.1790 | 0.1835 | 21.6 | 6185 of 9920 | 3e-05 | 4 / 3 |
| 19 | 0.3599 | 0.3265 | 0.1724 | 0.1763 | 23.4 | 6823 of 9920 | 3e-05 | 10 / 5 |
| 20 | 0.3183 | 0.3070 | 0.2063 | 0.2017 | 22.4 | 8104 of 9920 | 3e-05 | 7 / 6 |
| 21 | 0.3063 | 0.3066 | 0.2200 | 0.2170 | 21.4 | 8493 of 9920 | 3e-05 | 6 / 4 |
| 22 | 0.3123 | 0.2938 | 0.1610 | 0.1607 | 21.8 | 8599 of 9920 | 3e-05 | 8 / 7 |
| 23 | 0.3131 | 0.3116 | 0.1836 | 0.1823 | 24.7 | 8443 of 9920 | 3e-05 | 10 / 10 |
| 24 | 0.3319 | 0.3417 | 0.2077 | 0.2201 | 24.4 | 8630 of 9920 | 3e-05 | 10 / 10 |
| 25 | 0.3345 | 0.3344 | 0.2066 | 0.2350 | 26.5 | 8634 of 9920 | 3e-05 | 13 / 13 |
| 26 | 0.3520 | 0.3951 | 0.1438 | 0.1616 | 25.1 | 8488 of 9920 | 3e-05 | 13 / 13 |
| 27 | 0.3591 | 0.3533 | 0.1631 | 0.1864 | 23.2 | 8330 of 9920 | 3e-05 | 10 / 9 |
| 28 | 0.3585 | 0.3537 | 0.1839 | 0.2008 | 22.8 | 8121 of 9920 | 3e-05 | 8 / 8 |
| 29 | 0.3535 | 0.3525 | 0.1698 | 0.1811 | 21.6 | 7838 of 9920 | 3e-05 | 8 / 5 |
| 30 | 0.3155 | 0.3232 | 0.1601 | 0.1654 | 18.1 | 6808 of 9920 | 3e-05 | 3 / 0 |
| 31 | 0.4038 | 0.4202 | 0.0425 | 0.0364 | 14.5 | 3974 of 9920 | 3e-05 | 5 / 5 |

Middle-layer state FVU is 0.31 to 0.42 here and 0.25 to 0.27 for the smaller set; the proposed explanation in terms of the smaller active fraction, 3.75% against 9.4% of model width, is an inference.

### Spliced-in loss

Spliced-in loss is the increase in next-token cross-entropy when a reconstruction replaces the true activation, one layer and dictionary at a time, in nats. We measured it at layer 16 only.

On held-out text the model itself answered, comprising 53 rows and 123,745 positions, the state dictionary at `resid_post.16` gave +0.303. The MLP dictionary gave +0.0068; replacing the MLP output with zero gave +0.0307, so reconstruction recovered 0.78 of that gap. On FineWeb-Edu text with 25,400 positions, MLP loss was +0.0058 versus +0.0194 for zero output, recovering 0.70.

Recorded at training: state FVU in raw residual units at `resid_post.16`, with spliced-in loss in nats.

| Active groups per token | Raw state FVU at `resid_post.16` | Spliced-in loss |
|---|---|---|
| 24 | 0.362 | +0.303 |
| 40 | 0.317 | +0.171 |
| 60 | 0.281 | +0.110 |

The 40- and 60-group dictionaries are outside this set. Reading the 24-group dictionary with 60 groups gave +0.172 on the odd-numbered 26 rows; the dictionary trained with 40 groups gave +0.170 on those same rows.

### What these units have been used for, in full

We used templated prompts in these experiments. Their scripts are unpublished.

**Input validation in Python.** On 16 tasks we measured the mean residual difference between paired prompts. One asked for argument validation and an error for invalid inputs. The other asked for a description of the arguments and readable code. At `resid_post.10`, group 2879's frame held the largest projection, 0.45 of the full difference in norm. We added that projected vector to the state at each of the 17 tokens of the neutral sentence. One addition at its measured size is the mean dose. The prompt text stayed the same, and twelve unedited tokens followed the edit through the opening Python code fence.

On 72 evaluation tasks we generated one greedy and two sampled answers per task, giving 216 functions per condition. We ran them on 678 valid inputs, 297 wrong-type inputs and 87 wrong-value inputs. The wrong-value cases came from the 38 tasks with a stated value restriction. We counted an invalid input as rejected when a `raise` statement in the function fired. Rejection rates are means over tasks. Brackets give 95% bootstrap intervals over tasks.

| Condition | Contains `raise` or `isinstance` | Wrong-type inputs rejected | Wrong-value inputs rejected | Passes every valid case |
|---|---:|---:|---:|---:|
| No edit | 31.9% [21.8, 42.6] | 5.3% | 56.9% | 94.4% |
| Mean dose | 48.6% [37.5, 59.3] | 12.4% | 84.6% | 90.7% |
| Twice the dose | 72.7% [63.9, 81.0] | 51.0% | 94.0% | 88.9% |
| Validation instruction in the prompt | 100.0% | 94.5% | 98.2% | 67.1% |

The first three rows use the neutral wording from which we measured the direction. On two other neutral sentences the same vector increased the rate of `raise` or `isinstance` by +4.2 and +6.5 percentage points at the mean dose. At twice the dose the increases were +28.7 [19.4, 38.9] and +14.8 [8.3, 21.8] points, respectively. Most lost valid cases were inputs such as an empty list or string that the function rejected where the specification was silent.

**ABA and ABB rules.** In the task from Yang et al. (2025), the third item repeats the first in ABA and the second in ABB. At that item, one direction in group 12487's plane tracks repetition and another distinguishes which word repeats. Layer-15 attention head 14 reads both and writes at the answer position. We shifted the mixer's input within group 12487's frame. Moving toward the other rule reproduced 0.24 to 0.39 of the unedited mean difference between rules inside four frames at the answer position. Along the full rule direction the fractions were 0.20 and 0.26 for the two edit directions. An equal move in one randomly drawn frame gave 0.03 to 0.12 inside the four frames.

Across histories with different numbers of ABA, ABB and ABC lines (three distinct words), the whole residual at `resid_mid.15` lay near a plane at the answer position. Head 14's write accounted for 0.53 of the variation between histories. The whole mixer, including that head, accounted for 0.80 and the incoming residual for 0.20.

We also ranked groups using the rule labels in prompts with two examples of comma-separated words. At layers 11 to 14, one frame carried 0.18 to 0.42 of the effect of swapping the whole residual difference at the positions where the rule is abstracted. Four groups carried 0.50 to 0.56 and sixteen carried 0.67 to 0.78. One direction chosen with supervision carried 0.78 to 0.89 and a random 64-dimensional subspace carried 0.01 to 0.02. Shifting inside group 12112's frame at `resid_post.13` by the mean difference between rules flipped ABA to ABB in 0.45 of prompts, or 0.98 at twice the shift. The reverse flip reached 0.03, or 0.67 at twice the shift. These measurements used one plain-text format. The model did not solve the task in its chat template.

**Digit comparison.** We compared digits in the same slot on successive lines of dates or weights. At `resid_mid.7`, one axis in group 9719's frame separates smaller from larger and another separates equal digits. Transform 357 reads less and transform 472 reads greater. Projected into this frame, their read directions have cosine -0.90. Their full read directions have cosine +0.06. The frame holds 0.15 of 357's squared read length and 0.25 of 472's. It supplies 0.23 of 357's gate-input variance on lists and 0.21 in JavaScript, compared with 0.67 and 0.51 for 472. Group 5915 contributes more to 357 (0.51 on lists and 0.49 in JavaScript).

**A randomly drawn group.** Among groups drawn at random, group 15417 at `resid_post.27` separated “vital” from death words. In each of six sentence templates we moved “vital” inside the frame to match one coordinate of “fatal”, “deadly”, “lethal” or “mortal”. We summed next-token probabilities over each of two sets of eight nouns. The log ratio of those sums shifted toward the death-word set in all 24 pairs, by +1.41 nats on average, covering 0.24 of the original gap. “Mistake” and “accident” are members of that set. Equally sized random moves gave changes from -0.28 to +0.07 nats.

In the template “It was a W”, all 41 adjectives had negative values on the second coordinate of the learned frame. “Vital”, “fatal” and “pivotal” had the most negative values.

**Python rule completion.** We chose group 9167 by comparing prompts with and without the instruction “Reply with code only.” Shifting the state within its frame at `resid_mid.27` just before the function body led the model to reason in comments. At a fixed dose, the correct-rule rate rose from 0.516 to 0.773 on 24 held-out rule families (1,536 generations, one greedy and one sampled per prompt). Of 768 greedy answers, 23 became wrong and 232 became correct. In 96 separate generations a random group changed the rate by -0.01. Forcing the first body token to `#` gave a similar rate, 0.65 against 0.72 with the unit, with the difference inside the interval.

**JavaScript comparison.** In a hand-traced ternary expression, swapping the layer-7 MLP output gave a patch share of 0.083, the fraction of the paired prompts' mean answer-margin gap reproduced by that swap. The MLP reconstruction carried 0.89 of the MLP's contribution (interval 0.78 to 1.00). Transform 357, rank 64, ranked first. Two MLP neurons carried 0.80 of the same MLP contribution.

### How the published files were checked

We ran these checks on one machine with one RTX 5090.

- **State reader:** against our training-code reader, across 32 layers × 4 splits × 2 sites × 4,096 tokens = 1,048,576 tokens, position 0 included. Group sets differed on 36 tokens, at most 2 per split/site; other raw reconstructions agreed within 1.7e-6 relative. All 64 `heldout_fvu_own_units` values agreed within 8.8e-6.
- **MLP reader:** against the unmodified public forward pass, over 524,288 tokens × 9,920 transforms. Gates differed 5 times, at layers 6, 10, 14, 16 and 21; other outputs agreed within 2.0e-6 relative.
- **Individual writes:** firing transforms plus output mean matched the forward pass within 1.02e-6 relative; group writes plus mean and bias matched decoding within 8.4e-7 relative.
- **Files:** all 192 weight/config hashes matched `MANIFEST.json`; state tensor files were unchanged from training exports, and all reader-held MLP tensors were bit-identical to their training-export counterparts. All 32 layers' norm files matched coefficients rebuilt from the base model bit for bit.
- **All-token measurements:** state FVU at both sites and all 32 layers matched the record within 1.7e-6, with position 0 both included and dropped; split token counts matched. MLP FVU was 0.0425 to 0.2898 with position 0 dropped; including it changed FVU by at most 0.0067. The MLP training record uses different tokens.
- **Model usage:** the Try it lines ran at layer 16; the sentence had 10 tokens, leaving 9 and shapes `(9, 2560)`. The bf16 model plus both dictionaries needed more than 14 GiB GPU memory and ran under a 26 GiB cap. A larger check used eight FineWeb-Edu texts of 256 tokens, leaving 2,040 after dropping position 0.

Re-measured: raw FVU and mean transforms on per token for the eight-text check (2,040 tokens).

| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP output FVU | Transforms on per token |
|---|---|---|---|---|
| 0 | 0.1057 | 0.1214 | 0.0829 | 31.5 |
| 16 | 0.4327 | 0.4131 | 0.2788 | 22.7 |
| 31 | 0.4649 | 0.4236 | 0.0431 | 14.4 |

### Config files

In `cfg.json`, the format tag `kind = state_dictionary` and fields `form = bsf`, `gate = topk`, `input = token_rms` specify the tied block-sparse form and RMS input; `lam`, `lam_dr`, `lam_pc`, `lam_nest` and `n_dense` are all 0, corresponding to no penalty term or dense block. From `config.yaml`, the reader uses `model` for class paths, sizes and ranks, and `data.init_args.post_norm_stats`. The manifest's `original_sha256` holds each config's hash before a small edit made before publication.

### Licence and credits

The code and dictionary weights are MIT licensed. The separately loaded base model, `Qwen/Qwen3.5-4B`, is Apache-2.0 licensed at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

`postnorm.safetensors` holds the base model's 2,560 RMSNorm coefficients and epsilon under Apache-2.0; all other tensors were trained on its activations. The MLP dictionary's form and trainer come from the MIT-licensed `crosslayer-transcoder` code (branch MOLT, commit 376fabe), Copyright 2025 Georg Lange.

The MLP dictionary's form was proposed by Jack Lindsey, Brian Chen, Adam Pearce, Sasha Hydrie and Thomas Conerly (edited by Jeff Wu) in "Sparse mixtures of linear transforms", Transformer Circuits Thread, 2025, https://transformer-circuits.pub/2025/bulk-update/index.html. Closely related work: James Oldfield, Shawn Im, Sharon Li, Mihalis A. Nicolaou, Ioannis Patras and Grigorios G Chrysos, "Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders", arXiv:2505.21364 (27 May 2025), https://arxiv.org/abs/2505.21364.

The ABA / ABB task was studied by Yukang Yang, Declan Campbell, Kaixuan Huang, Mengdi Wang, Jonathan Cohen and Taylor Webb in "Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models", arXiv:2502.20332 (27 February 2025), https://arxiv.org/abs/2502.20332. Related later work: Melody Zixuan Li and Taylor Webb, "What drives the emergence of symbolic mechanisms in large language models?", arXiv:2611.13270 (9 November 2026), https://arxiv.org/abs/2611.13270.
