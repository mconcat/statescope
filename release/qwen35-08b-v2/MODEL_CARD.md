---
license: mit
base_model: Qwen/Qwen3.5-0.8B
---

# Qwen3.5-0.8B per-layer dictionaries

These files provide two dictionaries for every layer of Qwen3.5-0.8B and a Python reader. The state dictionary writes the residual stream (the vector layers pass along) as a few active groups of directions. The MLP (multilayer perceptron) dictionary writes the layer's MLP as a sum of small gated linear operations called transforms; gates control when and how strongly they act.

### What you can do with it

- You can see active groups on a token (a piece of text) and reconstruct its activation (internal vector).
- You can see which transforms fire and what each adds to the MLP output.
- You can change a group's code (its coordinates) or a transform's gate and decode back to vectors.
- You can follow the demonstrated computation across layers through named units.

### An example: how the model decides to break a line

With text wrapped at a fixed width, the model must place line breaks by tracking how long the current line is.

1. Layer-8 transform 189 fires on the first token of a line.
2. State group 11586 holds most of that signal.
3. Head 10 of the layer-9 mixer carries the count. A mixer is a layer's GatedDeltaNet or attention sublayer, the part that combines information across tokens.
4. Layer-9 transforms read it mostly through state group 4885.
5. Transform 410 is on while the line is short; transform 1749 turns on when it is long. They turn the count into the line-break decision.

The test swaps in values from a second version of the same text, wrapped at 60 characters so its last line is about half full and no line break is due. Swapping only the first transform's contribution through the mixer reproduces 0.328 of the average change in the model's preference for newline over space. Swapping the whole layer-8 MLP output the same way reproduces 0.353 of that change. In this setting, one transform out of 3,968 moves the answer about as much as the whole MLP output of its layer. These demonstration numbers are exact for the tested library versions (torch 2.11.0+cu128, transformers 5.20.2); with newer versions, the reported fractions moved by up to about 0.01.

We found this chain; `demo/line_count.py` re-measures it from published files in under two minutes.

### Try it

These commands install dependencies in the [code checkout](https://github.com/mconcat/statescope) and download [weights](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b); `<folder>` is the destination.

```sh
pip install -e '.[model]'
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
```

This example reconstructs a layer's activations:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from statescope.reader import StateVocab, PublicMolt, capture
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda").eval()
input_ids = torch.tensor([tokenizer("The committee met on Thursday to review the budget.")["input_ids"]], device="cuda")
V = StateVocab("<folder>/state/L9", "cuda")
M = PublicMolt("<folder>/mlp/L9", "cuda")
acts = capture(model, 9, input_ids)                      # resid_mid, mlp_out, resid_post; position 0 dropped
x_hat = V.reconstruct(acts["resid_post"], "resid_post.9")
gate, y_hat = M.forward(acts["resid_mid"])               # gate (tokens, 3968), y_hat ~ acts["mlp_out"]
```

File and demonstration checks:

```sh
python demo/verify_release.py --release <folder>
python demo/line_count.py --release <folder>
```

### What is in the box

- The 146 files occupy about 27.4 gibibytes (GiB).
- The 144 weight and config files cover 24 layers; the other files are the manifest and model card.
- Each state dictionary activates 24 of 16,384 groups per token.
- Each MLP dictionary has 3,968 transforms.

A larger set for Qwen3.5-4B is published separately: [Qwen3.5-4B](https://huggingface.co/mconcat/statescope-qwen3.5-4b).

### Scope

This set covers one model and one dictionary training seed (the number controlling randomness); we chose the line-break chain's units, and the demonstration re-measures them. Line counting is an easy task for which the model has dedicated MLP neurons (individual MLP units). The per-layer numbers recorded at training and the spliced-in loss rest on unpublished data.

## For agents reading this repo

### Files

`<L>` is an unpadded layer index, 0 to 23.

```text
MANIFEST.json
README.md
.gitattributes
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

- `MANIFEST.json`: file hashes and numbers recorded at training.
- `README.md`: model card.
- `.gitattributes`: added by Hugging Face; a download holds 147 files, the 146 listed plus this file.
- `state/L<L>/cfg.json`: form, sites, sizes and training options.
- `state/L<L>/weights.safetensors`: encoder, decoder, bias, site means and scales.
- `state/L<L>/usage.safetensors`: training counters `fire`, window totals `n_tokens`, running selection rates `use`, and tokens since last selection `since`.
- `mlp/L<L>/config.yaml`: MLP configuration.
- `mlp/L<L>/weights.safetensors`: gates, read/write matrices and input/output statistics.
- `mlp/L<L>/postnorm.safetensors`: pre-MLP root mean square normalization (RMSNorm) coefficients `w_eff` and `eps`.

### Load and call

Python 3.12 or newer; installation in the code checkout:

```sh
pip install -e '.[model]'
```

Calls in `statescope.reader`, with `V: StateVocab`, `M: PublicMolt`, `Tensor = torch.Tensor`, `Device = str | torch.device`; shapes use tokens `n`, width `d=1024`, groups `G=16384`, directions `b=4`, active groups `k=24`, transforms `T=3968`.

- `StateVocab(path: str, device: Device = "cpu")`: loads `state/L<L>`.
- `V.token_rms(x: Tensor) -> Tensor`: raw `(n,d)` → root mean square (RMS) `(n,1)`.
- `V.standardize(x: Tensor, site: str) -> Tensor`: raw `(n,d)` → standardized `(n,d)`.
- `V.encode(x: Tensor, site: str, chunk: int = 1024) -> tuple[Tensor, Tensor]`: raw `(n,d)` → long groups `(n,k)`, codes `(n,k,b)`.
- `V.decode(groups: Tensor, codes: Tensor, site: str, rms: Tensor, chunk: int = 1024) -> Tensor`: `(n,k)`, `(n,k,b)`, RMS `(n,1)` → raw `(n,d)`.
- `V.reconstruct(x: Tensor, site: str) -> Tensor`: raw `(n,d)` → raw `(n,d)`.
- `V.group_write(group: int, code: Tensor, site: str, rms: Tensor) -> Tensor`: code `(b,)` → `scale_site * (code @ Q[group]) * rms`, with `(d,)` before RMS multiplication.
- `PublicMolt(path: str, device: Device = "cpu", post_norm: str | None = None, chunk: int = 512)`: loads `mlp/L<L>`.
- `M.standardize_input(x: Tensor) -> Tensor`: raw `(n,d)` → standardized `(n,d)`.
- `M.gate(z: Tensor) -> Tensor`: standardized `(n,d)` → gates `(n,T)`.
- `M.forward(x: Tensor, g: Tensor | None = None) -> tuple[Tensor, Tensor]`: raw `(n,d)` → gates `(n,T)`, raw output `(n,d)`; accepts edited gates.
- `M.transform_write(x: Tensor, transform: int) -> Tensor`: raw `(n,d)` → contribution `(n,d)`.
- `capture(model: torch.nn.Module, layer: int, input_ids: Tensor, attention_mask: Tensor | None = None) -> dict[str, Tensor]`: `(batch,sequence)` → `resid_mid`, `mlp_out`, `resid_post`, each float32 `(n,d)`.
- `decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList`: `model.model.layers`.
- `write_post_norm(model: torch.nn.Module, layer: int, path: str) -> None`: writes RMSNorm coefficients.
- `fvu(x: Tensor, x_hat: Tensor) -> float`: `(n,d)` → fraction of variance unexplained (FVU).

The Python example in [Try it](#try-it) shows loading and reconstruction.

### Conventions

- `resid_mid.L` is between mixer and MLP, before `post_attention_layernorm`; `resid_post.L = resid_mid + mlp_out`. One state dictionary covers both sites.
- The state dictionary reads the residual stream before any norm module, divided by its own size: `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. This is the model's RMSNorm without its learned weight; site standardization follows. The MLP dictionary reads the same quantity multiplied by that weight (`w_eff`), which is the model's own post-norm MLP input. `reconstruct` returns raw units; manual decoding takes `rms=V.token_rms(x)`.
- `capture` drops position 0 and masked padding. State training excluded position 0; MLP training included it.
- FVU divides squared reconstruction error by squared deviation from the per-dimension mean. Raw state/MLP FVU uses residual/MLP-output units; `heldout_fvu_own_units` uses RMS-normalized, site-standardized units. Transforms on per token is a mean count; alive is a count; penalty is a sparsity coefficient. Spliced-in loss is in nats. Demonstration firing rates and frame energies are fractions.
- Transforms are stored largest rank first: 128 of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16, 2,048 of rank 8; `M.rank` has shape `(T,)`.
- `PublicMolt` takes raw `resid_mid`, applies the layer's norm from `postnorm.safetensors`, then standardizes. The standardizer tables hold one row per model layer; only the layer's own row is filled.
- A demonstration share is a ratio of mean margin changes: `mean(m_clean - m_patched) / mean(m_clean - m_corrupt)`. Margin is the last-token logsumexp (logarithm of a sum of exponentials) of newline-token logits (model scores) minus that of logits for tokens starting with a space. Bootstrap intervals, estimated by resampling, span 2.5 % to 97.5 %.

### Verify

```sh
python demo/verify_release.py --release <folder>
python demo/line_count.py --release <folder>
```

The file check prints `layer <L>: checked`, then `all 144 files match the manifest` on success. It compares Secure Hash Algorithm 256-bit (SHA-256) hashes using the Python standard library. The demonstration prints shares and firing rates; it reports measurements without a pass/fail verdict. Its documented numbers are exact for torch 2.11.0+cu128 and transformers 5.20.2; with newer versions, shares moved by up to about 0.01. With the `dev` extra installed, `pytest -m "not legacy"` runs 36 tests without private data.

## Appendix: details of the measurements

### Where the numbers come from

We trained the dictionaries and made the first measurements with research code that is not published. Some numbers were recorded then ("recorded at training"); others were measured again with the published reader on the published files ("re-measured").

### State dictionary

A group contains 4 directions; its frame is an orthonormal basis of its write subspace, and its code gives coordinates in that frame multiplied by a per-group scale. Each tied dictionary has 16,384 groups at model width 1024 and activates the 24 groups with the largest written norm at each token. It covers `resid_mid.L` and `resid_post.L` with separate site means and scales.

For raw residual `x`, site mean `mu_site`, site scale `scale_site` and decoder bias `b_dec`:

```text
rms = sqrt(mean(x^2) + eps)                      eps = 1e-6
xt = (x / rms - mu_site) / scale_site - b_dec
c_g = F_g xt                                    F, Q: (G, b, d); c_g: (b,)
active = the k groups with largest norm(c_g)      G=16384, b=4, d=1024, k=24
x_hat = (mu_site + scale_site * (sum_active c_g Q_g + b_dec)) * rms
```

`Q_g` has orthonormal rows. The normalized input `x / rms` is the model's RMSNorm without its learned weight; the MLP dictionary's input, below, includes that weight. We trained each layer's dictionary on 30,003,200 tokens over 7,325 steps, excluding position 0. The state files occupy about 12.0 GiB.

### MLP dictionary

A transform has read and write matrices; its write is its contribution to the MLP output. The dictionary is a sparse mixture of linear transforms, called MOLT. Each transform's JumpReLU gate retains a positive pre-activation above its learned threshold and otherwise returns zero. The read and write matrices are free, with no skip connection, and training uses a tanh sparsity penalty.

Each layer has 3,968 transforms in the five rank tiers listed under Conventions. The MLP files occupy about 15.4 GiB. The reader applies RMSNorm to raw `resid_mid`, standardizes each dimension, and restores raw output units:

```text
u = x * rsqrt(mean(x^2) + eps) * w_eff
z = (u - mu_in) / sd_in
h_t = e_t . z + b_t
g_t = h_t if h_t > theta_t and h_t > 0 else 0
y_std = sum over transforms t of g_t * (z V_t) U_t
y = y_std * sd_out + mu_out
```

`V_t` is `(d, rank_t)` and `U_t` is `(rank_t, d)`; `w_eff = 1 + weight` of `post_attention_layernorm`, with `eps = 1e-06`. Hard gates can differ near a floating-point threshold.

We trained each layer for 58,600 steps of 500 tokens, or 29.3 million tokens, with learning rate 4.0e-05 and position 0 included. The sparsity penalty is 7.5e-05 at layers 0 to 22. Layer 23 used 2.5e-05 in a separate run.

### Numbers per layer

FVU is squared reconstruction error divided by squared deviation from the per-dimension mean of each held-out split. The four held-out splits contain 257,122 tokens per layer, including 179 at position 0. The training shards held 267,271 tokens at layers 9 and 23; those shards were deleted, and the held-out activations are unpublished.

- **Layer:** layer index.
- **State FVU, `resid_mid`:** re-measured in raw residual units on all held-out-split tokens except position 0.
- **State FVU, `resid_post`:** re-measured in raw residual units on all held-out-split tokens except position 0.
- **MLP FVU:** re-measured in raw MLP output units on all held-out-split tokens except position 0.
- **MLP FVU recorded at training:** recorded at training on each layer's held-out training shard, including position 0.
- **MLP transforms on per token:** recorded at training; mean count of open gates on that shard, including position 0.
- **MLP transforms alive:** recorded at training; count firing on at least one token of that shard, including position 0, out of the total.
- **MLP penalty:** recorded at training; sparsity coefficient in the manifest.

| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP FVU | MLP FVU recorded at training | MLP transforms on per token | MLP transforms alive | MLP penalty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0777 | 0.0858 | 0.0879 | 0.0732 | 19.0 | 3955 of 3968 | 7.5e-05 |
| 1 | 0.1153 | 0.1317 | 0.1627 | 0.1447 | 11.2 | 3936 of 3968 | 7.5e-05 |
| 2 | 0.1617 | 0.1478 | 0.1481 | 0.1358 | 9.9 | 3901 of 3968 | 7.5e-05 |
| 3 | 0.1710 | 0.1666 | 0.1734 | 0.1599 | 10.3 | 3838 of 3968 | 7.5e-05 |
| 4 | 0.1899 | 0.1703 | 0.1633 | 0.1526 | 10.2 | 3793 of 3968 | 7.5e-05 |
| 5 | 0.1956 | 0.1836 | 0.1812 | 0.1736 | 9.9 | 3690 of 3968 | 7.5e-05 |
| 6 | 0.2184 | 0.2132 | 0.1961 | 0.1892 | 9.1 | 3167 of 3968 | 7.5e-05 |
| 7 | 0.2403 | 0.2216 | 0.1908 | 0.1851 | 9.2 | 2720 of 3968 | 7.5e-05 |
| 8 | 0.2390 | 0.2270 | 0.2134 | 0.2066 | 9.1 | 2396 of 3968 | 7.5e-05 |
| 9 | 0.2491 | 0.2380 | 0.2243 | 0.2153 | 8.9 | 2035 of 3968 | 7.5e-05 |
| 10 | 0.2576 | 0.2416 | 0.1975 | 0.1828 | 8.6 | 1862 of 3968 | 7.5e-05 |
| 11 | 0.2713 | 0.2521 | 0.1863 | 0.1763 | 8.3 | 1950 of 3968 | 7.5e-05 |
| 12 | 0.2634 | 0.2472 | 0.1973 | 0.1857 | 8.3 | 2290 of 3968 | 7.5e-05 |
| 13 | 0.2590 | 0.2472 | 0.1883 | 0.1768 | 8.5 | 2240 of 3968 | 7.5e-05 |
| 14 | 0.2625 | 0.2395 | 0.1406 | 0.1216 | 7.5 | 1935 of 3968 | 7.5e-05 |
| 15 | 0.2623 | 0.2319 | 0.1289 | 0.1200 | 8.2 | 2880 of 3968 | 7.5e-05 |
| 16 | 0.2293 | 0.2196 | 0.1557 | 0.1469 | 8.3 | 3350 of 3968 | 7.5e-05 |
| 17 | 0.2167 | 0.2131 | 0.1650 | 0.1555 | 8.4 | 3506 of 3968 | 7.5e-05 |
| 18 | 0.2132 | 0.2053 | 0.1394 | 0.1304 | 7.9 | 3462 of 3968 | 7.5e-05 |
| 19 | 0.2259 | 0.2227 | 0.1455 | 0.1370 | 8.2 | 3486 of 3968 | 7.5e-05 |
| 20 | 0.2166 | 0.2216 | 0.1618 | 0.1491 | 8.3 | 3565 of 3968 | 7.5e-05 |
| 21 | 0.2249 | 0.2327 | 0.1602 | 0.1461 | 8.2 | 3407 of 3968 | 7.5e-05 |
| 22 | 0.2402 | 0.2518 | 0.1385 | 0.1234 | 7.8 | 2960 of 3968 | 7.5e-05 |
| 23 | 0.2614 | 0.2404 | 0.0246 | 0.0206 | 16.1 | 3327 of 3968 | 2.5e-05 |

The two MLP FVU columns use different tokens. Including position 0 changes state `resid_mid` FVU at layer 6 from 0.2184 to 0.2524, and at layer 0 from 0.0777 to 0.0778. MLP FVU differs by at most 0.014 at any layer.

**`heldout_fvu_own_units` in `MANIFEST.json`:** recorded at training; the state trainer's FVU in per-token-RMS-normalized, site-standardized units on the first 4,096 tokens of each split, including position 0. All 48 values agree with the reader within 6.9e-7. Early-layer values are lower because that subset is easier. At layer 0, `resid_mid`, FVU is 0.0533 on the subset versus 0.0693 on all tokens in the same units, or 0.0777 in raw units. At layer 9, it is 0.2452 on the subset versus 0.2438 on all tokens in the same units.

### Spliced-in loss

Spliced-in loss is the increase in next-token cross-entropy when a dictionary reconstruction replaces an activation, one layer and dictionary at a time. We used FineWeb-Edu rows outside the training mixture: the first 100 documents of at least 256 tokens and the first 256 tokens of each. Position 0 stayed intact; loss was averaged over positions 1 to 254, or 25,400 positions. Clean cross-entropy was 2.808. The texts are unpublished. Layer 23's MLP result comes from a separate measurement on the same kind of web text, 100 documents of 256 tokens.

Recorded at training: increases in cross-entropy in nats. Rows identify layers; columns distinguish state replacement at `resid_post` from MLP replacement.

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
| 23 | +0.866 | +0.0502 |

Our earlier, unpublished set had smaller increases at all 24 state layers and MLP layers 0 to 22 on the same text. At state layer 9, this set gives +0.243 versus +0.161 for the earlier set. At MLP layer 23, this set gives +0.0502 versus +0.0840 for the earlier set.

### Demonstration

`demo/line_count.py` follows the layer-8 (L8) MLP through layer-9 (L9) GatedDeltaNet head 10 to the L9 MLP as the model predicts a line break at the last word of a full line. Each item pairs 256-token FineWeb-Edu excerpts ending on the same words. The clean text's last line is full; the corrupt text is re-wrapped so a line break is not due.

A cell is a task setting defined by wrapping width and the change producing the corrupt text: `count:60`, `count:100` or `width:100`. In the count cells, the corrupt text's last line starts at a later word, chosen so it holds between 35% and 65% of the width at the last word; preceding lines are wrapped backward from that word at the same width, moving their breaks too. In the width cell, the last line starts at the same word as in the clean text, and preceding lines are wrapped backward at 1.75 times the width. Each cell contains 64 pairs stored as token ids and position pairing in `demo/line_count_items.safetensors`.

The margin is the last-token logsumexp (logarithm of a sum of exponentials) of newline-token logits (model scores) minus that of logits for tokens starting with a space. A patch replaces a clean-run tensor with a corrupt-run value moved onto clean token positions. A share is `mean(m_clean - m_patched) / mean(m_clean - m_corrupt)`, where `m` is the margin: 0 means no change; 1 means the patch reproduces the mean clean-to-corrupt margin change.

We chose the units: transforms were ranked on odd-numbered items and results read on even-numbered ones. Some groups were first chosen on all items; repeating selection on odd-numbered items returned the same groups, frames and transforms. Head 10 was known from earlier work. The script re-measures these units.

Transform 189 writes mostly inside state group 11586's frame at `resid_post.8`. Head 10 carries the count; L9 transforms 410 and 1749 read it mostly through group 4885's frame at `resid_mid.9`.

Re-measured firing and frame-energy fractions on even-numbered items, 32 per cell:

Transform 189 fires on 1.000 of line-first tokens, 0.001 of line-break tokens and 0.010 of other tokens on clean count-cell texts. Group 11586's frame holds 0.769 of its swapped-write energy; the largest of five random frames holds 0.016. Chance for a 4-dimensional frame in 1,024 dimensions is 0.004.

Transform 410 fires on 0.84 to 0.89 of tokens at line lengths of 20 to 49 characters, 0.25 at 60 to 69, and 0.03 at 70 to 79. Transform 1749 fires on 0.00 below 70 characters, 0.19 at 80 to 89, and 0.60 at 90 to 99.

Re-measured shares on the even-numbered half, 32 items per cell. Rows identify swaps; columns identify task cells. Parentheses hold bootstrap intervals from 2.5 % to 97.5 %.

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

In the count cells, transform 189 or group 11586 through the L9 mixer moves the answer about as much as the whole L8 MLP output. Head 10's read alone gives a larger share than the whole mixer output. Transform 410 carries the L9 MLP's effect in `count:60`; transform 1749 carries it in `width:100`.

The next table applies mixer-output or head-10 swaps in full, inside group 4885's 4-dimensional frame, or as the rest. Cells are `A / B (C)`: A is the share through the six L9 transforms we selected (410, 1749, 6, 3720, 1394, 2275); B is through 410 and 1749 alone. For A and B, changed `resid_mid.9` enters the L9 MLP dictionary, and only the selected transforms' changed writes are patched at the MLP output. C includes everything downstream; the last row omits C.

Re-measured shares on even-numbered items, 32 per cell. Rows identify mixer-output changes; columns identify the same task cells.

| Change at the L9 mixer output | count:60 | count:100 | width:100 |
|---|---|---|---|
| whole mixer swap, all of it | 0.320 / 0.268 (0.474) | 0.396 / 0.202 (0.590) | 0.392 / 0.425 (0.704) |
| whole mixer swap, only the part inside group 4885's frame | 0.300 / 0.283 (0.459) | 0.329 / 0.164 (0.594) | 0.374 / 0.414 (0.721) |
| whole mixer swap, the rest | 0.016 / 0.013 (0.156) | 0.032 / 0.052 (0.168) | 0.042 / 0.031 (0.177) |
| head 10 swap, all of it | 0.293 / 0.380 (0.633) | 0.373 / 0.179 (0.678) | 0.398 / 0.467 (0.784) |
| head 10 swap, only the part inside group 4885's frame | 0.314 / 0.374 (0.585) | 0.309 / 0.169 (0.650) | 0.395 / 0.476 (0.860) |
| head 10 swap, the rest | 0.026 / 0.025 (0.153) | 0.041 / 0.066 (0.185) | 0.044 / 0.037 (0.204) |
| whole mixer swap, only the part inside a random 4-dimensional frame (largest of 5, absolute) | 0.003 / 0.004 | 0.003 / 0.005 | 0.005 / 0.007 |

The part inside group 4885's frame moves the transforms about as much as the whole swap. For the whole mixer swap, the rest carries 0.016, 0.032 and 0.042 through the six transforms in `count:60`, `count:100` and `width:100`, respectively; through the two transforms, it carries 0.013, 0.052 and 0.031. A random 4-dimensional frame carries at most 0.005 through six transforms and 0.007 through two. With everything downstream, the rest carries 0.156, 0.168 and 0.177 in those cells. Other routes therefore reach the answer, and the parts' downstream shares do not add up to the whole.

We registered the pass rule before measurement: in both count cells, the frame retains at least 0.75 of the whole share; the rest retains at most 0.25 of it and at most 0.05; random frames carry at most 0.01. The rule held through the six transforms and held only in part through 410 and 1749, where the rest in `count:100` was 0.051 in that comparison. Group 4885 is a dense 4-dimensional coordinate group active on most tokens; its coordinates carry information. Line counting is an easy task with dedicated MLP neurons.

### How the published files were checked

We ran these checks on one machine with one RTX 5090, including an empty-folder download and installation into a new environment on 2026-11-20.

- Against our research-code state reader, across 24 layers, 4 splits of 4,096 tokens and 2 sites, 4 of 786,432 tokens selected different groups; their 24th and 25th group norms differed by at most 7.3e-7. Other raw reconstructions agreed within 7.1e-7 relative. All 48 own-unit FVU values agreed within 6.9e-7.
- Against the unmodified public MLP forward pass at 24 layers, 1 gate differed, at layer 3; other-token outputs agreed within 1.9e-6 relative. The sum of firing transforms' writes plus the output mean agreed within 1.5e-6 relative.
- All-token state FVU agreed with the record within 6.5e-8 at all 24 layers and both sites, with and without position 0, within the 1e-4 limit. Repeating all-token reconstruction measurements with the published reader and folder gave identical values. Captured `resid_mid` and MLP output were bit-identical to the research capture at layers 0 and 19.
- All 144 weight/config files matched the manifest hashes, including the 96 safetensors files on the server. State tensor files are the training exports; MLP tensors at layers 0 to 22 were bit-identical to the training exports, and layer 23 passed its own forward-pass comparison.
- The demonstration items matched the reference items exactly. All 351 shares (39 swaps × 3 cells × 3 halves) agreed within 0.0030 with the reference run and recorded numbers, within tolerance 0.005. Single-item margins agreed to about 0.1 and shares to about 0.003: rounding patched tensors to bfloat16 can change elements after a 1e-6 relative implementation difference.
- In the empty-folder check, all 144 hashes matched; downloaded `README.md` and `MANIFEST.json` were byte-identical to the card and staged manifest. With torch 2.11.0+cu128 and transformers 5.20.2, the downloaded demonstration result was byte-identical to the tested run. With torch 2.15.0+cu132 and transformers 5.23.0, it ran in 1 min 49 s; the largest unpatched single-item margin difference was 0.146 and the largest share difference was 0.011 (0.009 versus 0.020 for a swap outside group 4885's frame). The 36 tests needing no private data passed in the new environment.

### Config files

In `cfg.json`, `kind: state_dictionary` is the format tag; `form: bsf`, `gate: topk` and `input: token_rms` specify the tied block-sparse form, top-k group selection and per-token RMS input, while `lam`, `lam_dr`, `lam_pc`, `lam_nest` and `n_dense` are all 0 (no penalty or dense block). The reader takes class paths, sizes and ranks from `config.yaml`'s `model` section and reads `data.init_args.post_norm_stats`. The manifest's `original_sha256` is the hash of each config before a small edit made before publication.

### Limits

- The set covers one model and one dictionary seed.
- We chose the demonstration units beforehand; the script re-measures them.
- Line counting is an easy task with dedicated MLP neurons.
- Training-shard metrics come from deleted data; held-out activations and spliced-in-loss texts are unpublished.
- Demonstration numbers are exact for the tested library versions.
- The trainers are not published yet.

### Licence and credits

The code and dictionary weights use MIT. The separately loaded `Qwen/Qwen3.5-0.8B` base model (revision `2fc06364715b967f1860aea9cf38778875588b17`) uses Apache 2.0; `capture` takes a model loaded by the caller. Its copied norm values in `postnorm.safetensors` also come from the Apache-2.0-licensed model: `w_eff` has 1,024 values equal to 1 plus the layer's norm weight, with `eps = 1e-06`. Other tensors were trained on residual activations or MLP inputs and outputs.

The MLP form and trainer come from MIT-licensed `crosslayer-transcoder` (branch MOLT, commit 376fabe), Copyright 2025 Georg Lange. The demonstration texts come from `HuggingFaceFW/fineweb-edu` under the Open Data Commons Attribution License (ODC-By 1.0).
