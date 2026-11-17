---
license: mit
base_model: Qwen/Qwen3.5-0.8B
---

# Qwen3.5-0.8B per-layer dictionaries

## What this is

This release contains per-layer dictionaries of the language model `Qwen/Qwen3.5-0.8B`. A **state dictionary**, read by `StateVocab`, reconstructs residual-stream activations from selected **groups**, which are sets of write directions with vectors of coefficients called codes. A **multilayer perceptron (MLP) dictionary**, read by `MoltOps`, approximates the MLP output with gated low-rank linear maps called **atoms**. The state and MLP dictionaries were trained separately at each layer.

## What is in the repository

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Model revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Layers | 24 |
| Model width | 1024 |

Each layer uses the folders `layers/Lxx/vocab` and `layers/Lxx/molt`, where `xx` is the layer index written with two digits. The following files are used by the reader, with an additional evaluation record for the MLP dictionary.

| Folder | Files | Reported size per layer |
|---|---|---|
| `layers/Lxx/vocab` | `cfg.json`, `weights.safetensors` | 513 megabytes (MB) |
| `layers/Lxx/molt` | `cfg.json`, `weights.safetensors`, `input.json`; `budget.json` records evaluation and export errors | About 1.2 gigabytes (GB) |

The release notes describe a standalone `reader.py`, a reconstruction example named `example.py`, and `manifest.json`. The statescope repository supplies the reader and the reconstruction example used below. The reader is self-contained and imports no other code from the statescope repository.

**Fraction of variance unexplained (FVU)** is the squared reconstruction error divided by the squared deviation from the per-dimension mean, pooled over tokens and dimensions. The manifest records file sizes, `sha256` checksums, held-out FVU, fractions of dead groups and quiet atoms, and folding errors from exporting the trained dictionaries. Held-out FVU uses tokens from the training mixture, with state-dictionary values reported separately for each activation site and MLP-dictionary values pooled over the four held-out sets. Only layers with both dictionaries exported and no failure flag from the collector enter the manifest.

## Quick start

Install `torch` and `safetensors` to read the dictionaries. Running the model and `capture` also requires `transformers`. Install `flash-linear-attention` for the model run; without it, transformers uses a slow reference implementation of the model's GatedDeltaNet component. If `causal_conv1d` is missing, transformers warns that its reference implementation is slower but produces the same values.

For a layer indexed by `L`, **`resid_mid`** is the residual-stream activation received by `layers[L].post_attention_layernorm`, before the MLP sub-block. **`resid_post`** is the layer-output activation represented by `resid_mid` plus the MLP output. The reader constructs this sum explicitly.

With the model loaded through `transformers.AutoModelForCausalLM` and tokenized text in `input_ids`, run the reader's example from the directory containing `layers`. The code uses the reader from the statescope repository.

```python
from statescope.reader import StateVocab, MoltOps, capture

V = StateVocab("layers/L19/vocab", "cuda")
M = MoltOps("layers/L19/molt", "cuda")
acts = capture(model, 19, input_ids)
groups, codes = V.encode(acts["resid_post"], "resid_post.19")
x_hat = V.decode(groups, codes, "resid_post.19")
a, y_hat = M.forward(acts["resid_mid"])
```

`capture` returns `resid_mid`, `mlp_out`, and `resid_post` as tensors with a token axis and a model-width axis. It drops position 0 from every sequence and also drops padded positions when an attention mask is supplied. Here, `x_hat` reconstructs the layer-output activation, `a` contains atom gate values, and `y_hat` estimates the raw MLP output. The reconstruction example in the statescope repository loads the model and tokenizer, measures English prose, Python code, and Korean prose, and prints the strongest groups and atoms at the last token.

## The two dictionaries

For the state dictionary, let $x$ be a raw activation at a chosen site. The site has a mean $\mu_{\mathrm{site}}$ and scale $\mathrm{scale}_{\mathrm{site}}$, and $b_{\mathrm{dec}}$ is the decoder bias. Standardization and the code for group $g$ are

$$
\widetilde{x} = \frac{x-\mu_{\mathrm{site}}}{\mathrm{scale}_{\mathrm{site}}}-b_{\mathrm{dec}},
\qquad c_g = F_g\widetilde{x}.
$$

The rows of $Q_g$ form an orthonormal frame for the group's write subspace, and $F_g$ maps the standardized activation to its code. The written contribution is $c_g Q_g$, whose norm equals the code norm. The active set $A$ contains the $k$ groups with the largest code norms. Decoding gives

$$
\widehat{x} = \mu_{\mathrm{site}} + \mathrm{scale}_{\mathrm{site}}
\left(b_{\mathrm{dec}} + \sum_{g\in A}c_g Q_g\right).
$$

`group_write` returns a group's contribution in raw activation units, excluding the mean and bias. Codes can be edited, or groups dropped, before calling `decode`.

For the MLP dictionary, $x$ is raw `resid_mid`. First apply the layer's own root mean square normalization, called RMSNorm, and then standardize its output:

$$
u = x\odot\operatorname{rsqrt}\left(\operatorname{mean}(x^2)+\epsilon\right)\odot w_{\mathrm{eff}},
\qquad z = \frac{u-\mu_u}{\mathrm{sd}_u}.
$$

Here, $\odot$ denotes elementwise multiplication, $\operatorname{rsqrt}$ is the reciprocal square root, and the mean is taken across the activation's dimensions. The normalization uses $\epsilon$ and effective weights $w_{\mathrm{eff}}$; $\mu_u$ and $\mathrm{sd}_u$ are the input standardization statistics. Atom $t$ has a gate read direction $e_t$, bias $b_t$, score divisor $\mathrm{score\_div}_t$, and threshold $\mathrm{thr}_t$:

$$
h_t = e_t\cdot z-b_t,
\qquad
a_t = \begin{cases}
\operatorname{relu}(h_t), & h_t/\mathrm{score\_div}_t>\mathrm{thr}_t,\\
0, & \text{otherwise}.
\end{cases}
$$

The rectified linear function $\operatorname{relu}(h)$ returns $\max(h,0)$. The atom contributions combine with a fixed linear term and decoder bias to estimate the MLP output:

$$
y_{\mathrm{std}} = zW_{\mathrm{skip}}+b_{\mathrm{dec}}
+\sum_t a_t\left[\left(zV_t^\mathsf{T}\right)\odot s_t\right]U_t,
\qquad
y = y_{\mathrm{std}}\odot\mathrm{sd}_y+\mu_y.
$$

For each atom, $V_t$ and $U_t$ are its read and write matrices, with one row per rank dimension, and $s_t$ supplies their scales. The output standardization statistics are $\mu_y$ and $\mathrm{sd}_y$. `atom_write` returns one atom's contribution to the raw MLP output, with no contribution where its gate is off. Passing edited gate values to `forward` changes the atom contributions.

For either dictionary, FVU is computed as

$$
\mathrm{FVU}(x,\widehat{x}) =
\frac{\sum_{i,j}(x_{ij}-\widehat{x}_{ij})^2}
{\sum_{i,j}(x_{ij}-\overline{x}_j)^2}.
$$

Here, $i$ indexes tokens, $j$ indexes activation dimensions, and $\overline{x}_j$ is the mean over the evaluated tokens in dimension $j$.

## How they were trained

The state dictionary was trained at both `resid_mid` and `resid_post`. The MLP dictionary uses recipe `M1w`, with the atoms of recipe `Q`, a fixed mean-gate linear map, and whitened gate inputs.

| Setting | State dictionary | MLP dictionary |
|---|---|---|
| Number of groups or atoms | 16,384 groups | 41,484 atoms |
| Directions or ranks | 4 directions per group | Ranks 1-16, stored in tiers 1, 2, 4, 8, 16 |
| Selection or rank budget | 24 groups per token, selected by written norm | Rank budget 131,072 |
| Training tokens per layer | 50M | 29.3M |
| Learning rate | 3e-4 | 3e-4 |
| Seed | 0 | 0 |

The training texts mix Pile, UltraChat, and text generated by the base model itself. The mixture is primarily English and code.

| Text source | Mixture share |
|---|---|
| Pile | 1/3 |
| UltraChat | 1/3 |
| Model-generated continuations | 1/6 |
| Model-generated chat answers | 1/6 |

The following script-type shares were measured on samples from these sources, weighted by the mixture shares. ASCII means the American Standard Code for Information Interchange character set.

| Token character category | Measured share |
|---|---|
| ASCII only | 98.7% |
| Hangul | .002% |
| Han characters | .08% |
| Kana | .006% |
| Cyrillic | .03% |
| Other non-ASCII | .5% |

The Pile and UltraChat samples contained 0 Hangul tokens. These are character-script measurements, not language measurements, so the share of other languages written in Latin script is unknown.

## Measured reconstruction and spliced-in loss

The example was measured on layers 19 and 9 using the standalone reader. Each row below describes one short text, and the token count excludes position 0. The state columns give FVU at `resid_mid` and `resid_post`, while the MLP column gives FVU for the MLP output. Lower FVU means less unexplained variance under the formula above. The final column is the mean number of active atoms per token. These short-text results are separate from the held-out training-mixture measurements.

| Layer | Text | Tokens | State FVU at resid_mid | State FVU at resid_post | MLP FVU | Mean active atoms |
|---|---|---|---|---|---|---|
| 19 | English prose | 67 | .211 | .207 | .138 | 56.9 |
| 19 | Python code | 89 | .302 | .294 | .153 | 45.3 |
| 19 | Korean prose | 64 | .524 | .520 | .385 | 108.1 |
| 9 | English prose | 67 | .263 | .250 | .292 | 57.2 |
| 9 | Python code | 89 | .318 | .306 | .281 | 61.6 |
| 9 | Korean prose | 64 | .454 | .437 | .366 | 74.6 |

**Spliced-in loss** is the next-token cross-entropy loss after replacing an activation at a single layer with a dictionary reconstruction or a baseline value. The measurements used 100 fineweb-edu documents with 256 tokens each, outside the training mixture. The unmodified model's loss was 2.808 nats, where a nat is a unit of loss using natural logarithms. The replacement columns below report the increase over that unmodified loss. The MLP replacements act on the MLP output; the state replacements act on `resid_post` or `resid_mid`, as indicated. A **recovered fraction** is the fraction of a baseline's loss increase avoided by using the dictionary reconstruction. It equals one minus the dictionary's loss increase divided by the baseline's loss increase. **The recovered MLP column uses the zeroed MLP output as its baseline, while the recovered state column uses the layer output replaced by its mean.** The recovered state column refers to the `resid_post` reconstruction. The two recovered columns therefore must not be compared directly.

| Layer | MLP output → MLP dictionary | MLP output → 0 | MLP output → mean | resid_post → state dictionary | resid_mid → state dictionary | resid_post → mean | Recovered MLP | Recovered state |
|---|---|---|---|---|---|---|---|---|
| 0 | .017 | .184 | .171 | .056 | .046 | 8.71 | .91 | .99 |
| 1 | .025 | .120 | .108 | .078 | .074 | 7.79 | .79 | .99 |
| 2 | .021 | .222 | .153 | .075 | .097 | 7.88 | .91 | .99 |
| 3 | .012 | .129 | .117 | .080 | .091 | 7.62 | .91 | .99 |
| 4 | .011 | .099 | .091 | .083 | .094 | 6.64 | .89 | .99 |
| 5 | .013 | .089 | .081 | .102 | .103 | 6.50 | .86 | .98 |
| 6 | .012 | .178 | .064 | .128 | .132 | 6.34 | .93 | .98 |
| 7 | .009 | .058 | .055 | .137 | .140 | 6.75 | .84 | .98 |
| 8 | .009 | .040 | .031 | .155 | .155 | 6.12 | .77 | .97 |
| 9 | .006 | .030 | .025 | .161 | .165 | 6.11 | .79 | .97 |
| 10 | .006 | .040 | .031 | .192 | .191 | 6.04 | .84 | .97 |
| 11 | .007 | .034 | .033 | .189 | .187 | 6.05 | .81 | .97 |
| 12 | .009 | .043 | .042 | .202 | .205 | 6.10 | .79 | .97 |
| 13 | .009 | .050 | .045 | .229 | .227 | 5.66 | .82 | .96 |
| 14 | .013 | .137 | .132 | .244 | .266 | 5.69 | .90 | .96 |
| 15 | .012 | .137 | .130 | .219 | .237 | 5.69 | .91 | .96 |
| 16 | .015 | .130 | .118 | .227 | .224 | 5.62 | .88 | .96 |
| 17 | .014 | .119 | .103 | .232 | .227 | 5.41 | .88 | .96 |
| 18 | .016 | .164 | .135 | .240 | .241 | 5.25 | .90 | .95 |
| 19 | .017 | .158 | .119 | .265 | .249 | 4.98 | .89 | .95 |
| 20 | .019 | .118 | .110 | .280 | .260 | 5.04 | .84 | .94 |
| 21 | .021 | .125 | .108 | .320 | .305 | 5.28 | .83 | .94 |
| 22 | .019 | .196 | .126 | .352 | .333 | 6.07 | .90 | .94 |
| 23 | .084 | .785 | .876 | .648 | .354 | 7.72 | .89 | .92 |

Reader parity was also measured on 4,096 held-out Pile tokens. The MLP dictionaries at layers 19 and 0 had identical gates and outputs under the standalone reader and the statescope repository reader. At layer 19, the state dictionary selected identical group sets at both sites. Its largest code difference was 2.9e-6, and its relative reconstruction difference was at most 1.2e-7.

## Cautions when using them

**Non-English text.** Both dictionaries reconstructed the measured Korean prose substantially worse than the English prose. Only one Korean passage was measured, containing 64 tokens. We infer that the near absence of Hangul in the training mixture contributes to this result. Worse reconstruction on Chinese and Japanese text is a prediction; their FVU has not been measured. We also infer that the higher FVU on short texts is partly due to estimating the mean from only 60–90 tokens.

**Hard gate threshold and flipped pairs.** An atom near its hard gate threshold can switch on or off as batch shape or numerical precision changes. Checks covered 4,096 tokens and 41,484 atoms. The measured number of flipped token–atom pairs was 0–2. The affected token's output changed by .1–.5% of the largest output. Such flips can exceed the export folding-error threshold of 1e-4. For layers above that threshold, the recorded folding error was assessed by whether it matched the contribution of the flipped pair within 1%. The training-time model was no longer available, so the remaining error with that pair excluded was not measured directly.

**The MLP dictionary is not the published MOLT recipe.** The MLP dictionary is a variant of the sparse mixture of linear transforms described by Anthropic. A comparison with that description and with public implementations found six differences in recipe `M1w`. It has no sparsity penalty and instead keeps a fixed number of active atoms per rank tier in each batch. Its gate is dimensionless. The read, write and gate directions are constrained to unit norm. The ranks are 1-16 instead of a 512-32 pyramid. The skip path is the frozen mean-gate linear map of the layer's own MLP. Only the gate reads a whitened input. No public implementation with all six properties was found. The comparison was made by reading the published text and code; it did not measure how each difference changes the results.

**Position 0.** The first token of every sequence was excluded from training and evaluation and must also be excluded when measuring reconstruction. `capture` already performs this exclusion. Its captured `resid_mid` and MLP output matched the statescope repository's capture code exactly. Their explicit sum differed from the actual layer output by .2–.3% of the largest activation value. We infer that this difference comes from rounding in the model's `torch.bfloat16` arithmetic.

**Single seed.** This release uses seed 0. A second state-dictionary seed already exists for layer 9. Coverage across all layers is not included.

**Earlier training batches.** The state dictionaries for layers 9 and 18 were trained in earlier batches and copied into this release. Their specification filenames were `p1a.json` and `v0.json`, respectively. For layer 18, the earlier specification has the same `v0.json` basename as the main recipe but comes from a different location. The registration record says the configurations differ only in activation sites and learned thresholds. The code map did not recheck their original launch commands. The layer 9 MLP dictionary also used a different calibration cache and fixed map from the common launch sequence. The layer 9 state dictionary was also trained with an earlier version of two of the trainer's source files than the other 47 runs, according to the source hashes each run recorded.

## What is not included yet

- A second training seed across all layers is not included.
- A learning-rate sweep for the MLP dictionaries is not included.
- FVU measurements on non-English text beyond the Korean example are not included.
- Organized code for reproducing dictionary training is not yet included in the release described by the sources.

## License

This release is licensed under the MIT license.
