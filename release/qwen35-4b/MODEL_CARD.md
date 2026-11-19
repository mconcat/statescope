---
license: mit
base_model: Qwen/Qwen3.5-4B
---

# Qwen3.5-4B per-layer dictionaries

This larger companion of the [0.8B set](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b) provides two dictionaries per model layer, read by the same Python reader. The state dictionary writes the residual stream (the vector layers pass along) as a few active groups of directions. The MLP (multilayer perceptron) dictionary writes the layer's MLP as a sum of small gated linear operations called transforms; gates control when and how strongly they act.

### What you can do with it

- See active groups on a token (a piece of text) and reconstruct its activation (internal vector).
- See which transforms fire and what each adds to the MLP output.
- Change a group's code (its coordinates) or a transform's gate and decode back to vectors.

### What these units have been used for

The original project ran these experiments with this set's units; they cannot be re-run from the published files and code.

- The model had to infer an unknown rule from a few input-output cases and complete a Python function; the test covered 24 rule families it had not been tuned on. The original project moved the residual stream inside a single state group's frame (the four directions of a group), at just the token before the function body. It used group 9167 of `resid_mid.27`, and the model then reasoned in comments before writing the rule. The rate of correct rules rose from 0.516 to 0.773. The effect is not distinguishable from simply forcing the first token of the function body to be the comment sign `#`.
- In an ABA rule the third item repeats the first; in an ABB rule it repeats the second. At layers 11 to 14, the sixteen highest-ranked state groups carried 0.67 to 0.78 of the effect of swapping the whole residual difference. Moving the residual stream inside the frame of group 12112 at `resid_post.13` changed which rule the model followed. It flipped from ABA to ABB in 0.45 of prompts; the opposite direction was much harder, at 0.03. The project ranked groups with the rule labels and used one text format; inside the chat template, the model did not solve the task.
- At layer 7, the MLP dictionary's reconstruction carries 0.89 of the MLP's contribution to a hand-traced JavaScript comparison. That MLP is itself a small part of the circuit: swapping its whole output moves 0.083 of the answer. Two MLP neurons carry 0.80 of the same contribution, so neurons describe this more compactly.

These experiments used one dictionary seed (the number controlling randomness), prompts built from templates, and units chosen by a contrast between two kinds of prompts; none shows these units are interpretable in general.

### Try it

In the [code checkout](https://github.com/mconcat/statescope), install dependencies and download one layer: about 1.25 gibibytes (GiB) for state and 4.0 GiB for MLP files.

```sh
pip install -e '.[model]'
hf download mconcat/statescope-qwen3.5-4b --include "state/L16/*" "mlp/L16/*" MANIFEST.json --local-dir <folder>
```

The full download is about 168 GiB (180 gigabytes, GB).

```sh
hf download mconcat/statescope-qwen3.5-4b --local-dir <folder>
```

Replace `<folder>` with the destination. Reconstruct a layer's activations:

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

The 4B model in bf16 plus one layer's two dictionaries needed more than 14 GiB of GPU memory in the run that tested these lines.

Check a full download against the manifest (the file-hash record):

```sh
python demo/verify_release.py --release <folder>
```

There is no demonstration script for this set.

### What is in the box

- The 192 weight and config files cover 32 layers; the manifest, transform firing record and model card accompany them.
- State files occupy about 40.0 GiB and MLP files about 128.0 GiB.
- Each state dictionary activates 24 of 16,384 groups per token.
- Each MLP dictionary has 9,920 transforms.

### What this does not show

Reconstruction is worse than in the smaller companion. Loss after replacing an activation with its reconstruction was measured at one layer only. Many transforms never fire. A few fire almost always and carry a large share of what is written. This is a dated copy without an owner decision to freeze it. It covers one seed, with no mixer dictionaries (for sublayers combining information across tokens) or trainers.

See [For agents reading this repo](#for-agents-reading-this-repo) for exact calls and conventions, and [Appendix: details of the measurements](#appendix-details-of-the-measurements) for the full evidence.

## For agents reading this repo

### What is here and what is not

- Get weights from [mconcat/statescope-qwen3.5-4b](https://huggingface.co/mconcat/statescope-qwen3.5-4b). Get the reader, tests and staging/measuring tools from [mconcat/statescope](https://github.com/mconcat/statescope). There is no demonstration script for this set.
- Use this set for the 32 layers of `Qwen/Qwen3.5-4B`: one state dictionary for two residual sites and one MLP dictionary per layer. Do not look for mixer dictionaries. A mixer is a layer's GatedDeltaNet or attention sublayer.
- Use the main-branch reader for the published form only. The earlier design and its reader are on branch `earlier-design`; that code did not train these weights. The published set's trainers are absent; porting is planned.
- Load the base model separately under its own licence. It is not redistributed as part of this set.

### File layout

Use unpadded layer indices from 0 to 31 for `<L>`.

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

- Read `MANIFEST.json` for each layer's file hashes, training records and `original_sha256` for the configs before their edits. Its `frozen` key is the copy date, 2026-11-20; `note_frozen` explains that meaning.
- Read `ALWAYS_ON.json`, made by the original project, for per-layer transforms firing on at least 90% of tokens and those firing on 50 to 90%, with each transform's share of summed write energy. These lists use a FineWeb-Edu web text sample: 128 documents of 512 tokens, position 0 dropped, 65,408 tokens. The file also gives counts from the held-out training-mix shard, 65,536 tokens, under `training_mix_heldout_counts`. Always identify which text a count is from.
- Read `state/L<L>/cfg.json` for the state dictionary's form, sites, sizes and training options.
- Load `state/L<L>/weights.safetensors` for the trained decoder and encoder tensors, decoder bias, site means and site scales.
- Keep `state/L<L>/usage.safetensors` with the state files. It holds training counters: per-site group selections in the last counting window (`fire`), per-site token totals for that window (`n_tokens`), running selection rates (`use`) and tokens since each group's last selection (`since`). The reader does not load it.
- Read `mlp/L<L>/config.yaml` for the MLP configuration; the reader uses its `model` section and `data.init_args.post_norm_stats`.
- Load `mlp/L<L>/weights.safetensors` for the trainer's unchanged state-dict tensors, including gates, read and write matrices and input/output statistics. The unused `last_active` tensor holds one integer per transform.
- Load `mlp/L<L>/postnorm.safetensors` for tensor `w_eff` and metadata `eps`, the base model's root mean square normalization (RMSNorm) coefficients before the MLP.

The weights repository contains 192 weight and config files, plus `MANIFEST.json`, `ALWAYS_ON.json` and `README.md` (the model card).

### How to load and call

Install `pip install -e '.[model]'` in the code checkout for model capture; use `pip install -e '.[model,dev]'` to add pytest and numpy. The standalone reader needs torch, safetensors and PyYAML. The `model` extra adds transformers, accelerate and flash-linear-attention. There is no `train` extra or installed `statescope` command. The model loader is `load_language_model` in `src/statescope/lm.py`; there is no `load_model` function. The reader does not load the model: `capture` takes a model that the caller has loaded.

Use `StateVocab`, `PublicMolt` and `capture` from `statescope.reader`. Tensor shapes below use `n` for tokens, `d = 2560` for model width, `G = 16384` for groups, `b = 4` for directions per group, `k = 24` for active groups and `T = 9920` for transforms. `Tensor` means `torch.Tensor`; `Device` is `str | torch.device`.

- Construct `StateVocab(path: str, device: Device = "cpu")` from `state/L<L>`. Use site strings `"resid_mid.L"` or `"resid_post.L"`, replacing `L` with the layer index.
- Call `V.token_rms(x: Tensor) -> Tensor` on raw `x (n, d)` to obtain each token's root mean square (RMS), `(n, 1)`, with the stored epsilon added inside the square root.
- Call `V.standardize(x: Tensor, site: str) -> Tensor` on raw `x (n, d)` to obtain dictionary coordinates `xt (n, d)` after token normalization and site standardization.
- Call `V.encode(x: Tensor, site: str, chunk: int = 1024)` on raw `x (n, d)`. It returns group indices `(n, k)` with dtype `torch.long` and codes `(n, k, b)`.
- Call `V.decode(groups: Tensor, codes: Tensor, site: str, rms: Tensor, chunk: int = 1024) -> Tensor` with groups `(n, k)`, codes `(n, k, b)` and `rms=V.token_rms(x) (n, 1)` from the encoded tokens. It returns raw activations `(n, d)`. Edit codes or drop groups before decoding to steer.
- Call `V.reconstruct(x: Tensor, site: str) -> Tensor` on raw `x (n, d)` to encode and decode with its RMS. The result is `(n, d)` in raw units.
- Call `V.group_write(group: int, code: Tensor, site: str, rms: Tensor) -> Tensor` with a single code `(b,)` and the token's RMS. Its contribution is `scale_site * (code @ Q[group]) * rms`, with no mean or bias. `Q[group]` has shape `(b, d)`, so `code @ Q[group]` has shape `(d,)` before multiplication by `rms`.
- Construct `PublicMolt(path: str, device: Device = "cpu", post_norm: str | None = None, chunk: int = 512)` from `mlp/L<L>`. The default norm file is `postnorm.safetensors` in that folder. `chunk` controls tokens per decoding step and memory use, not results.
- Call `M.standardize_input(x: Tensor) -> Tensor` on raw `resid_mid (n, d)` to apply the layer's own RMS normalization (RMSNorm) and the trainer's per-dimension standardization, returning `z (n, d)`.
- Call `M.gate(z: Tensor) -> Tensor` on standardized `z (n, d)` to obtain gates `(n, T)`, zero where transforms are off.
- Call `M.forward(x: Tensor, g: Tensor | None = None)` on raw `resid_mid (n, d)`. It returns gates `(n, T)` and an estimate `(n, d)` in raw MLP output units. Pass edited gates of the same shape as `g` to steer.
- Call `M.transform_write(x: Tensor, transform: int) -> Tensor` on raw `resid_mid (n, d)` to obtain that transform's contribution `(n, d)` in raw MLP output units, zero where it is off.
- Call `capture(model: torch.nn.Module, layer: int, input_ids: Tensor, attention_mask: Tensor | None = None)` for one forward pass. Token identifiers and the optional mask have shape `(batch, sequence)`. It returns a dictionary with `resid_mid`, `mlp_out` and `resid_post`, each a 32-bit floating-point tensor `(tokens, d)`. Position 0 of every sequence and masked padding are dropped.
- Call `decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList` to get `model.model.layers` for a text-only Qwen3.5 causal language model loaded with `transformers.AutoModelForCausalLM`.
- Call `write_post_norm(model: torch.nn.Module, layer: int, path: str) -> None` to write that layer's pre-MLP RMSNorm coefficients in the form read by `PublicMolt`.
- Call `fvu(x: Tensor, x_hat: Tensor) -> float` on activation and reconstruction arrays `(n, d)` to compute fraction of variance unexplained (FVU), pooled over tokens and dimensions around the per-dimension mean of `x`.

### Conventions that prevent silent errors

- Use `resid_mid.L` for the residual stream site between the mixer and MLP: the input to `post_attention_layernorm`. Use `resid_post.L` for the site after the MLP: `resid_mid + mlp_out`. One state dictionary covers both sites; pass the matching site string.
- Normalize the state input per token as `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`, then use the site's mean vector and scale. `reconstruct` restores raw units. If decoding manually, supply `rms=V.token_rms(x)`; codes alone do not retain the token's raw magnitude.
- Drop position 0, the first token of each sequence, when measuring anything involving the state dictionary. The original project's training convention drops it for state dictionaries and includes it for MLP dictionaries. Applying that convention to this set is an inference from the shared training code, not a separate check of the 4B training. `capture` drops it.
- Compute raw state fraction of variance unexplained (FVU) in residual units and raw MLP FVU in MLP output units. FVU is a ratio of squared reconstruction error to squared deviation from the per-dimension mean of each held-out split. Keep `heldout_fvu_own_units` separate: it uses per-token-RMS-normalized, site-standardized state units.
- Read transforms on per token as a mean count, transforms alive as a count out of the total, and MLP penalty as the sparsity coefficient. Training-time MLP numbers include position 0 and use the deleted training-mix shard; this repository's reconstruction numbers use other tokens.
- Report spliced-in loss in nats, the units of cross-entropy calculated with natural logarithms. It is the increase in next-token cross-entropy after replacing one layer's activation with a dictionary reconstruction. The original project measured it only at layer 16.
- Keep transform indices in stored order: largest rank first. Tiers have 320 transforms of rank 128, 640 of rank 64, 1,280 of rank 32, 2,560 of rank 16 and 5,120 of rank 8. Read `M.rank (T,)` for each transform's rank. Hard gates can disagree near a floating-point threshold.
- Pass raw `resid_mid` to `PublicMolt` methods that take `x`. The dictionary was trained on the post-norm MLP input. The reader computes it as `u = x * rsqrt(mean(x^2) + eps) * w_eff`, then standardizes by the trainer's per-dimension statistics. `postnorm.safetensors` is separate from the trainer's state dict; `w_eff = 1 + weight` of `post_attention_layernorm`, and metadata `eps` is `1e-06`. Output statistics restore raw MLP units.
- The standardizer tables have 24 rows at layers 0 to 23 and 32 rows at layers 24 to 31. Only the layer's own row is filled; the reader reads that row. Do not substitute another row or assume every table has the same row count.
- Treat the transforms firing on at least 90% of tokens as one dense background block when tracing. Do not equate one transform with one operation. Always say whether an always-on count comes from the FineWeb-Edu sample or the held-out training-mix shard; counts differ at 17 layers.

### How to verify

Run these commands from the code checkout. Replace `<folder>` with the weights destination.

```sh
hf download mconcat/statescope-qwen3.5-4b --local-dir <folder>
python demo/verify_release.py --release <folder>
pytest -m "not legacy"
```

- Expect the file check to print `layer <L>: checked` and finish with `all 192 files match the manifest`. It compares Secure Hash Algorithm 256-bit (SHA-256) hashes only and uses only the Python standard library, so it can run before installation. A mismatched hash makes it fail. Use a full download for this check.
- Install the `dev` extra for pytest. `pytest -m "not legacy"` selects tests that need no private data. The other tests have the `legacy` marker and require preserved original code, the original project's results and the staged weight folders specified in `tests/local.env.example`. Run `pytest` for everything. A missing required path fails its test; nothing is skipped silently. There is no `release` marker.
- The appendix records this set's comparisons under rules R10 and R11. These file and reader checks do not establish the quality or interpretation of the units. There is no demonstration command for this set.

### Where the published numbers come from

- Treat this repository's state and MLP FVU columns as measurements with its reader on held-out calibration activations; the MLP column uses different tokens from the original training record. Treat the reader comparisons, file checks and text-capture checks as this repository's measurements on the published files, on one machine with one RTX 5090 on 2026-11-20.
- Treat state `heldout_fvu_own_units`, training-time MLP FVU, transforms on per token, transforms alive, spliced-in loss, always-on counts and the experiments in the first part as the original project's records. The training-time MLP columns use 65,536 held-out training-capture tokens with position 0 included; the MLP penalty is the sparsity coefficient from the configs.
- Exact reconstruction numbers cannot be recomputed without the unpublished held-out activations; the training-time MLP shard was deleted. The first part's experiments cannot be re-run from the published files and code because their scripts are not published.

- Read “Layer” as the model layer index.
- Read “State FVU, `resid_mid`”, “State FVU, `resid_post`” and “MLP FVU” as this repository's reader measurements in raw units on 256,921 held-out calibration tokens per layer, with position 0 dropped.
- Read “MLP FVU recorded at training”, “MLP transforms on per token” and “MLP transforms alive” as the original project's records on 65,536 held-out training-capture tokens, with position 0 included; that shard was deleted.
- Read “MLP penalty” as the sparsity coefficient in the original project's MLP configs, checked by this repository.
- Read the last column as the original project's counts of transforms firing on at least 90% of tokens: FineWeb-Edu web text (65,408 tokens, position 0 dropped) / held-out training mix (65,536 tokens, position 0 included).

### Do not claim

- Do not describe this set as finalized by an owner decision. It is a copy dated 2026-11-20; the manifest's `frozen` key records that date only.
- Do not claim that spliced-in loss was measured per layer. It was measured at layer 16 only, with no measurement at the other 31 layers.
- Do not claim that the units are interpretable in general, that the smaller companion's experiments read, edited or steered any unit of this set, or that the first part's experiments can be reproduced from the published files and code.
- Do not claim that 24 active groups are optimal. The default came from one circuit comparison; spliced-in loss favours more groups.
- Do not claim that this set reconstructs as well as the smaller companion, or that its smaller active fraction has been measured to cause the difference. That explanation is an inference.
- Do not claim that every transform fires, that all transforms are sparse, or that a transform is one operation. Many never fired on the held-out training shard; those that fire almost always carry a large share of write energy at most layers and must be treated as one dense background block.
- Do not claim that the sparsity penalty was compared across all layers. The original project chose it at layer 16 only.
- Do not claim that the original project's four checks per layer or this repository's tests establish quality. They check that files are built and read as defined.
- Do not claim that state training included position 0, that state measurements should keep it, or that the 4B training convention was separately checked. The convention is inferred from shared training code, and `capture` drops position 0.
- Do not generalize beyond one seed and one model. The original project saw large variation between seeds for single-group numbers on the smaller companion.
- Do not claim that training-time MLP FVU, transforms on per token or alive counts can be reproduced from the published files. Their shard was deleted. Do not expect this repository's MLP FVU on other tokens to equal the training record, or claim the activations behind the reconstruction columns are public.
- Do not claim that the published code trains these weights, reproduces a full training run, provides a demonstration for this set, or contains mixer dictionaries.
- Do not claim that `capture` on this model was compared with the original project's capture. This repository ran the card's usage once and the larger text check described below.
- Do not claim exact agreement for every group selection or gate decision. Rounding near thresholds can change selections.
- Do not claim verification on another machine or another graphics processing unit (GPU) model. The reported runs used one machine and one RTX 5090 per run.

### Appendix pointers

Use “State dictionary” and “MLP dictionary” for definitions and formulas; “Config files” for accepted formats and edits; “Numbers per layer” and “Spliced-in loss” for metrics; “What the original project did with the units” for experiments and caveats; “What this repository measured and tested on the published files” for agreement limits; “The code repository and the earlier design”, “Limitations” and “License and credits” for scope and provenance.

## Appendix: details of the measurements

### What this is

This set contains separately trained state and MLP (multilayer perceptron) dictionaries for each of the 32 layers of `Qwen/Qwen3.5-4B`. The model has width 2,560. The base-model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, as recorded in the original project's training machines' model cache.

The owner did not make a decision to freeze this set. It is a copy made on 2026-11-20 of the dictionaries the original project trained for this model. The `frozen` key in `MANIFEST.json` holds the date of the copy; the adjacent `note_frozen` says so. This is the larger companion of [mconcat/statescope-qwen3.5-0.8b](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b), with the same dictionary forms and reader.

A mixer is a layer's GatedDeltaNet or attention sublayer. A residual stream site is a location between the mixer and the MLP or after the MLP where a state dictionary reads the residual stream. A state dictionary represents the residual stream at both sites. An MLP dictionary represents a layer's MLP operations, from its post-norm input to its raw output. The set contains no dictionary for the mixers.

The original project trained this set with one training seed. At a few layers it also trained variants with 40 or 60 active groups per token, a dictionary twice the size, and 50 million training tokens. Those variants are not in this set.

The code is public at [mconcat/statescope](https://github.com/mconcat/statescope), and the weights are public at [mconcat/statescope-qwen3.5-4b](https://huggingface.co/mconcat/statescope-qwen3.5-4b). The weights repository's `README.md` is this model card. The base model is loaded separately.

### Files and download

The weights folder uses the following layout. `<L>` is the layer index from 0 to 31 without padding.

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

There are 192 weight and config files, plus the three files at the top. This repository measured sizes on the staged folder. State dictionaries occupy about 40.0 GiB (43.0 GB), and MLP dictionaries about 128.0 GiB (137.5 GB). The total is about 168 GiB (180 GB). One layer is about 1.25 GiB of state dictionary and 4.0 GiB of MLP dictionary. GiB means gibibytes; GB means gigabytes.

Download one layer with the Hugging Face command line tool:

```sh
hf download mconcat/statescope-qwen3.5-4b --include "state/L16/*" "mlp/L16/*" MANIFEST.json --local-dir <folder>
```

Download the full set and check every weight and config file against the manifest:

```sh
hf download mconcat/statescope-qwen3.5-4b --local-dir <folder>
python demo/verify_release.py --release <folder>
```

The file check is in the code repository. It compares SHA-256 hashes only, uses only the Python standard library and can run before installation. It prints `layer <L>: checked` and, for a matching full download, `all 192 files match the manifest`.

`MANIFEST.json` provides each layer's file hashes and training records. `ALWAYS_ON.json` holds the original project's transform firing lists, write-energy shares and separate training-mix counts described under “MLP dictionary”. `cfg.json` gives the state form, sites, sizes and training options. The state weights hold trained decoder and encoder tensors, decoder bias, site means and site scales. The MLP export stores the trainer's state dict in `weights.safetensors` and its configuration in `config.yaml`. Its RMSNorm coefficients are separate from that state dict, in `postnorm.safetensors` as tensor `w_eff` and metadata `eps`.

`state/L<L>/usage.safetensors` holds the trainer's group usage counters at the end of training, about 0.5 MB (megabytes). `fire` has shape `(groups, 2 sites)` and counts training tokens selecting each group at each site in the last counting window. `n_tokens` has shape `(2,)` and gives that window's token totals. `use` has shape `(groups, 2 sites)` and gives each group's running average selection rate at each site. `since` has shape `(groups,)` and counts training tokens since each group was last selected. The reader does not load this file.

Other unused content is retained so the files stay the trained ones: `last_active` inside the MLP weights holds one integer per transform and is not part of the model, and the standardizer tables retain their unfilled rows.

### The code repository and the earlier design

The main branch of [mconcat/statescope](https://github.com/mconcat/statescope) contains the reader at `src/statescope/reader.py`, its neighboring modules, the file check at `demo/verify_release.py`, staging and measuring tools in `devtools/`, and tests with their rules file. `config.py` defines the models, `lm.py` loads the language model, and `heldout.py` and `metrics.py` serve the measuring tools and tests. The smaller companion's line-count demonstration is not a demonstration of this set; no demonstration script exists for this set.

The main-branch reader handles only the published form: the tied state dictionary with per-token RMS input and the MLP dictionary in the public penalty form. The published set's trainers are not in the repository; porting them is planned. The earlier design and its reader are on branch [`earlier-design`](https://github.com/mconcat/statescope/tree/earlier-design). That code did not train these weights.

Install the standalone reader and its torch, safetensors and PyYAML dependencies with `pip install -e .`. Install `pip install -e '.[model]'` for transformers, accelerate and flash-linear-attention, or `pip install -e '.[model,dev]'` to add pytest and numpy. There is no `train` extra or installed `statescope` command. The reader is self-contained and can be copied next to the weights. The function that loads the language model is `load_language_model` in `src/statescope/lm.py`; there is no function named `load_model`. The reader does not load the model: `capture` takes a model that the caller has loaded.

### Config files

Before publication, the config files were edited at every layer to remove internal names of the original project. This rewrite was decided on 2026-11-20, after rule R11 had been written.

- In `state/L<L>/cfg.json`, `kind` is now `state_dictionary`; it previously named the original project's trainer module. Four keys were removed: a configuration label, a source file path, a run label and the Message Digest 5 (MD5) record of eight source files of the original project. That record held file hashes. Every other key and value is unchanged.
- In `mlp/L<L>/config.yaml`, the one line naming the original project's data module was removed. Every other byte is unchanged. The `data` section now contains only `init_args`. The file is no longer a complete Lightning command line configuration. It could not be run before this edit either, because the data module is not published.
- No safetensors file changed during these config edits. Tensor bytes and headers are the ones used by the tests. Before this rewrite, the original project had already removed training-machine paths from the MLP configuration and the header metadata of the MLP weights, without changing tensor bytes.

`MANIFEST.json` lists the SHA-256 hashes of the published files. Under `original_sha256`, it also lists the hash of each of the two configs before the edit.

The reader accepts the state dictionary only when `kind` is `state_dictionary`, `gate` is `topk`, `form` is `bsf` and `input` is `token_rms`. These fields identify the tied block-sparse dictionary, its selection of the groups with the largest written norms and its per-token RMS input convention described below. `lam`, `lam_dr`, `lam_pc`, `lam_nest` and `n_dense` are options of the original trainer. All are 0: the state dictionaries used no penalty term and no dense block.

The MLP configuration retains the training-run sections `seed_everything`, `trainer`, `model`, `data` and `ckpt_path`. Class paths starting with `crosslayer_transcoder.` name classes of the public `crosslayer-transcoder` code and its training fork. The reader reads the `model` section for class paths, sizes and ranks, and `data.init_args.post_norm_stats`. It ignores the rest.

In the header metadata of `mlp/L<L>/weights.safetensors`, `source` is `training.ckpt`, the trainer's checkpoint file name. `keys` says that tensor names are the trainer's unchanged state-dict names.

### State dictionary

`statescope.reader.StateVocab` reads one layer's state dictionary. The same dictionary covers `resid_mid.L` and `resid_post.L`. The former is the residual between the mixer and MLP, at the input of the norm before the MLP. The latter is the residual after the MLP, equal to `resid_mid` plus the MLP output.

A group is a set of 4 directions in the state dictionary. A frame is the orthonormal basis of a group's 4-dimensional write subspace. A code gives the token's coordinates in that frame, multiplied by a per-group scale. The dictionary has 16,384 groups in a tied form, with groups sharing their read and write frames. It is block-sparse: only the 24 groups with the largest written norm are active at each token.

The active groups use 96 of 2,560 dimensions, or 3.75% of the width. For the smaller companion, they use 96 of 1,024 dimensions, or 9.4%. The original project used the same dictionary form and size for both sets.

The input convention is per-token root mean square (RMS) normalization. The dictionary first divides the residual by its own size using `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. It then standardises the result with a mean vector and one scale per site. To recover the raw residual, the reconstruction is multiplied by the token's own `sqrt(mean(x^2) + eps)`. `StateVocab.reconstruct(x, site)` performs both steps. Alternatively, `decode` accepts `rms=V.token_rms(x)`.

This repository checked that every layer's configuration records training on 30,003,200 tokens over 7,325 steps. Position 0 means the first position of every sequence. The original project's state-training convention drops position 0. Applying that convention to the 4B training follows from the shared training code and was not checked separately for this model. Drop position 0 when measuring with a state dictionary; `capture` does so.

The original project chose 24 active groups by comparing 24, 40 and 60 groups per token on one hand-traced circuit, a JavaScript ternary expression, at layers 7, 16, 19 and 23. A ternary expression chooses between two values according to a condition. A rule registered before that comparison returned a verdict of 24. This is a default from one comparison, not an optimum; spliced-in loss favours 40 and 60 groups, as reported below.

The reader specifies the state representation and its input convention as follows. Here `n` counts tokens, `d` is model width, `G` counts groups, `b` counts directions per group and `k` counts active groups. `Q` holds orthonormal write frames, and `F` maps the standardized input to group codes.

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

### MLP dictionary

A transform is a linear operation in an MLP dictionary with its own read and write matrices. Its write is what it adds to the MLP output. A gate controls a transform through a learned threshold on its pre-activation. `statescope.reader.PublicMolt` reads the sparse mixture of linear transforms, called MOLT. This dictionary uses a JumpReLU gate with a threshold learned separately for each transform. Its read and write matrices are free. It has no linear skip part. Training uses a hyperbolic tangent (`tanh`) sparsity penalty.

The original project used the public `crosslayer-transcoder` form, branch MOLT, commit 376fabe. Training used a fork that changes no mechanism. There are 9,920 transforms per layer in five rank tiers. Rank is the dimension of a transform's intermediate linear representation. The tiers contain 320 transforms of rank 128, 640 of rank 64, 1,280 of rank 32, 2,560 of rank 16 and 5,120 of rank 8. They are stored with the largest rank first. The reader's `rank` array gives each transform's rank.

The dictionary takes the layer's post-norm MLP input and produces a reconstruction of the raw MLP output. The reader computes that input from raw `resid_mid` using the layer's own RMSNorm coefficients. The formula is `u = x * rsqrt(mean(x^2) + eps) * w_eff`. The input is then standardised using per-dimension statistics retained by the trainer. The output is returned to raw MLP units with the stored output statistics.

This repository checked that every layer's configuration records 58,600 steps of 500 tokens, or 29.3 million tokens, a learning rate of 4.0e-05 and a sparsity penalty coefficient of 3e-05. That coefficient is the same at all 32 layers. The original project's MLP-training convention includes position 0. As for state training, applying this convention to 4B follows from the shared code and was not separately checked.

The original project's layer 16 dictionary comes from a separate, earlier run of the same recipe. This repository checked that its config equals the others apart from layer index, logger name and standardizer row count. The two per-dimension standardizer tables inside `weights.safetensors` have 24 rows at layers 0 to 23 and 32 rows at layers 24 to 31. Only the layer's own row is filled and read. The unfilled rows remain in the files.

The original project chose the penalty 3e-05 at one layer, 16. Its spliced-in loss was +0.0068 nat, lower than +0.0095 nat with 7.5e-05. No other layer compared penalties.

The gate is a hard threshold. A transform whose pre-activation lies within floating-point rounding distance of its threshold can be on in one computation and off in another. The reader gives the following forward-pass formulas. `z` is standardized input, `h_t` is a transform's gate pre-activation, `theta_t` is its learned threshold, `g_t` is its gate value, and `V_t` and `U_t` are its read and write matrices.

```text
u = x * rsqrt(mean(x^2) + eps) * w_eff          the layer's own RMSNorm on raw resid_mid x
z = (u - mu_in) / sd_in                       per-dimension statistics kept by the trainer
h_t = e_t . z + b_t
g_t = h_t if h_t > theta_t and h_t > 0 else 0
y_std = sum over transforms t of g_t * (z V_t) U_t
y = y_std * sd_out + mu_out                  the raw MLP output estimate

V_t has shape (d, rank_t); U_t has shape (rank_t, d).
```

#### Transforms that fire almost always

The original project made `ALWAYS_ON.json`. For each layer it lists transforms firing on at least 90% of tokens and transforms firing on 50 to 90%, with each transform's share of summed write energy. The lists were counted on a FineWeb-Edu web text sample, not the training mix: 128 documents of 512 tokens, position 0 dropped, giving 65,408 tokens. Under `training_mix_heldout_counts`, the same file gives counts on the held-out shard of the training mix, with 65,536 tokens and position 0 included.

The counts differ at 17 layers. The following examples report counts of transforms firing on at least 90% of tokens; each column names its text.

| Layer | FineWeb-Edu web text sample | Held-out training-mix shard |
|---|---|---|
| 7 | 18 | 18 |
| 13 | 11 | 6 |
| 19 | 10 | 5 |
| 30 | 3 | 0 |

On the web text sample, transforms firing on at least 90% of tokens carry up to 0.69 of summed write energy, at layer 11. On the training-mix shard, they carry up to 0.68, at layer 7; at 21 of the 32 layers they carry at least 0.30. When tracing, these transforms have to be treated as one dense background block. A transform of this set must not be read as “one transform = one operation”. Always say which text an always-on count is from.

#### Transforms that never fired

The original project counted a transform as dead if it never fired on the held-out training shard. The dead fraction was 0% at layer 0, 35 to 37% at layers 11 to 13, 40 to 43% at layers 14 to 16, and 60% at layer 31. At layer 31, 3,974 of 9,920 transforms were alive. These counts describe that shard.

### How to read the dictionaries

The example in “Try it” loads the base model separately and reads layer 16. `capture` returns `resid_mid`, `mlp_out` and `resid_post` for one layer as 32-bit floating-point tensors of shape `(tokens, 2560)`. It drops position 0 and, when given an attention mask, padded positions. `resid_mid` is the input of `post_attention_layernorm`, `mlp_out` is the raw MLP output, and `resid_post` is their sum.

`StateVocab` provides `encode`, `decode`, `reconstruct` and `group_write`. Decoding requires the original tokens' RMS values to restore raw magnitude. `group_write` gives one group's contribution without the mean or bias. `PublicMolt` provides `forward` and `transform_write`; pass raw `resid_mid` to methods taking `x`. A transform's write is zero where it is off. Edited codes or gates can be passed to the decoding calls. These calls do not establish what a unit means.

#### Exact calls and shapes

Every class and method of the reader, with argument and return shapes, is listed once, under [How to load and call](#how-to-load-and-call) in the part for agents above.

### Numbers per layer

This repository measured raw state and MLP reconstruction FVU with its reader on the published files, using all four held-out splits of each layer's calibration folder from the original project: 257,102 tokens per layer, including 181 at position 0, or 256,921 with position 0 dropped. FVU is the sum of squared errors divided by the sum over splits of squared deviations from each split's per-dimension mean.

- **Layer**: the model layer index, from 0 to 31.
- **State FVU, `resid_mid`**: this repository's reconstruction FVU in raw residual units at the site between the mixer and MLP, on the 256,921 calibration tokens per layer with position 0 dropped.
- **State FVU, `resid_post`**: this repository's reconstruction FVU in raw residual units at the site after the MLP, on the same calibration tokens with position 0 dropped.
- **MLP FVU**: this repository's reconstruction FVU in raw MLP output units, on the same calibration tokens with position 0 dropped.
- **MLP FVU recorded at training**: the original project's MLP reconstruction FVU on the held-out shard of the training capture, with 65,536 tokens and position 0 included.
- **MLP transforms on per token**: the mean number of firing transforms per token, recorded by the original project on the same 65,536-token training-capture shard with position 0 included.
- **MLP transforms alive**: the number of transforms that fired at least once, out of 9,920, recorded by the original project on the same 65,536-token training-capture shard with position 0 included.
- **MLP penalty**: the sparsity coefficient in the original project's MLP configs, checked by this repository; this is a configuration value, not a token measurement.
- **MLP transforms on at 90% of tokens or more (web text / training mix)**: the original project's counts of transforms firing on at least 90% of tokens. The first count is from the FineWeb-Edu web text sample in `ALWAYS_ON.json`: 128 documents of 512 tokens, position 0 dropped, 65,408 tokens. The second is from the held-out training-mix shard: 65,536 tokens, position 0 included, recorded under `training_mix_heldout_counts` in that file.

The training-time MLP numbers use other tokens than this repository's MLP FVU column. Their shard was deleted, so the recorded FVU, transforms on per token and alive counts cannot be recomputed from the published files, and this repository's MLP FVU is not expected to equal the training record. The held-out calibration activations are not published either, so users cannot recompute this repository's exact FVU values.

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

### Spliced-in loss

Spliced-in loss is the increase in next-token cross-entropy when a dictionary reconstruction replaces the true activation in the forward pass, one layer and one dictionary at a time. The original project measured it in nats at layer 16 only. The other 31 layers were not measured.

For the state dictionary at `resid_post.16`, the original project recorded +0.303 on held-out text the model itself answered, comprising 53 rows and 123,745 positions. On the same text, the MLP dictionary gave +0.0068. Replacing the MLP output by zero gave +0.0307, so the dictionary recovered 0.78 of that gap.

On FineWeb-Edu web text, with 25,400 positions, the layer 16 MLP dictionary gave +0.0058. Replacing the MLP output by zero gave +0.0194, with 0.70 recovered.

The original project also compared active-group counts at layer 16. FVU means fraction of variance unexplained: squared reconstruction error divided by squared deviation from the per-dimension mean, pooled over tokens and dimensions. The following state FVU values are in raw residual units at `resid_post`. The loss column gives increases in nats.

| Active groups per token | Raw state FVU at `resid_post.16` | Spliced-in loss |
|---|---|---|
| 24 | 0.362 | +0.303 |
| 40 | 0.317 | +0.171 |
| 60 | 0.281 | +0.110 |

The dictionaries trained with 40 and 60 active groups are not in this set. Part of the gain needs no retraining: reading the 24-group dictionary with 60 groups gave +0.172 on the odd-numbered 26 rows. The dictionary trained with 40 groups gave +0.170 on those same rows. These results do not make 24 an optimum.

### What the original project did with the units

The following experiments belong to the original, unpublished project and used the dictionaries of this set. They were checked against its result files on 2026-11-20. They cannot be re-run from the published files and code because no script for them is published. They used one dictionary seed, prompts built from templates, and units chosen by contrasting two kinds of prompts rather than selected without labels. Nothing here shows that the units of this set are interpretable in general.

#### One state group changes how the model answers

The original project asked the model to complete a Python function whose body began with a few input-output cases of an unknown rule. The model had to write the rule. It moved the residual stream inside the 4-dimensional frame of state group 9167 of `resid_mid.27`, at layer 27. The shift was applied at one token: the token just before the function body.

The model first wrote its reasoning as code comments and then wrote the rule. On 24 rule families it had not been tuned on, comprising 1,536 generations from 768 prompts each generated twice (once greedily, once sampled), the rate of correct rules rose from 0.516 to 0.773. The shift strength was fixed in an earlier experiment and was not re-tuned. For the 768 greedy generations only, 23 answers that were correct without the shift became wrong; 232 wrong answers became correct. In the earlier experiment, with 96 generations, a random group in place of this one changed the result by -0.01.

The effect was not distinguishable from simply forcing the first token of the function body to be the comment sign `#`. Correct-rule rates were 0.72 with the unit and 0.65 with the forced token; the difference lay inside the interval. The original project chose the unit by contrasting prompts with and without an instruction to reason in comments first.

#### A few groups carry an abstract variable of an in-context rule

The original project tested in-context pattern rules of the forms ABA and ABB: the third item repeats the first or the second. It used plain text in one format, with words separated by commas and two examples.

At positions where the rule is abstracted, at layers 11 to 14, moving the residual stream inside one state group's frame carried 0.18 to 0.42 of the effect of swapping the whole residual difference. Four groups carried 0.50 to 0.56, and sixteen groups carried 0.67 to 0.78. For comparison, one direction found with supervision carried 0.78 to 0.89; a random 64-dimensional subspace carried 0.01 to 0.02.

Moving the residual stream inside the frame of state group 12112 of `resid_post.13` by the mean difference between the two rules flipped the answer from ABA to ABB in 0.45 of prompts, and in 0.98 at twice the shift. The opposite direction, ABB to ABA, was much harder: 0.03 at the original shift and 0.67 at twice the shift.

The original project ranked the units using the rule labels. This used one text format only. Inside the chat template, the model does not solve the task.

#### The MLP dictionary carries most of the MLP's part of a hand-traced circuit

The original project traced a JavaScript comparison, a ternary expression, by hand. At layer 7, the MLP dictionary's reconstruction carried 0.89 of the MLP's contribution to that circuit, with an interval of 0.78 to 1.00. Transform 357, of rank 64, ranked first. The layer-7 MLP is itself a small part of this circuit: swapping its whole output moved 0.083 of the answer, and the 0.89 is a share of that part. Two MLP neurons carried 0.80 of the same contribution, so neurons described it more compactly.

### How to verify

The commands, what a pass looks like and what the other tests need are listed under [How to verify](#how-to-verify) in the part for agents above. The file check requires a full download. There is no demonstration script for this set.

### What this repository measured and tested on the published files

Rules R10 and R11 in `tests/EQUIVALENCE_RULES.md` were written and committed before the first run. All runs used one machine, with one RTX 5090 per run, on 2026-11-20. “Reference” means the original project's code in a fixed copy, reading the original files. This repository's reader read the published copy.

#### Reader comparisons

The comparisons covered all 32 layers, using the first 4,096 tokens of each of four held-out splits. Position 0 was kept, as in the trainers' own reported numbers.

- The state reader was compared with the reader the original project wrote separately from its trainer. The comparison covered 32 layers × 4 splits × 2 sites × 4,096 tokens = 1,048,576 tokens. Of these, 36 selected a different group set, at most 2 per split and site; the limit was 5 per 4,096 tokens. Raw reconstructions on the other tokens agreed within 1.7e-6 relative, against a limit of 1e-4. All 64 of the trainer's held-out FVU numbers in its own units were reproduced within 8.8e-6, against a limit of 1e-5. These own units are per-token-RMS-normalized and site-standardized state units, recorded as `heldout_fvu_own_units`.
- The MLP reader was compared with the unmodified public forward pass over 524,288 tokens × 9,920 transforms. There were 5 differing gate decisions, one each at layers 6, 10, 14, 16 and 21; the limit was 5 per split. Outputs on the other tokens agreed within 2.0e-6 relative, against a limit of 1e-4.
- The sum of firing transforms' writes plus the output mean agreed with the forward pass within 1.02e-6 relative. The largest difference was at layer 13; the limit was 1.5e-6. The sum of a token's group writes plus mean and bias agreed with decoding within 8.4e-7 relative, against a limit of 1e-6.
- The `postnorm.safetensors` files at all 32 layers hold the language model's RMSNorm coefficients bit for bit. This repository rebuilt them from `Qwen/Qwen3.5-4B` with `write_post_norm` for the comparison.

#### Checks of the published folder

- All 192 weight and config files have the SHA-256 hashes listed in `MANIFEST.json`.
- At all 32 layers, the parsed published `cfg.json` equals the original with the stated edit and nothing else. The published `config.yaml` equals the original project's copy with machine paths removed, byte for byte except for the one removed line. The state dictionaries' tensor files are the original files.
- At all 32 layers, every tensor the reader holds for the published MLP dictionary is bit-identical to its counterpart in the training export read by the reference of the reader comparisons above.
- All 32 state configs hold the same training facts: 30,003,200 tokens, 7,325 steps, 16,384 groups of 4 directions, 24 active groups, form `bsf`, input `token_rms`, epsilon 1e-6, seed 0 and width 2,560.
- All 32 MLP configs are equal apart from layer index, logger name and standardizer row count: 24 rows at layers 0 to 23 and 32 rows at layers 24 to 31. They record penalty 3e-05, 58,600 steps, batch size 500 and learning rate 4e-05. The configuration check also records “base count N = 320 (320, 640, 1,280, 2,560 and 5,120 transforms in the five rank tiers)”; the stored tier counts are listed under “MLP dictionary”.
- No config, safetensors header metadata, manifest or `ALWAYS_ON.json` holds a machine path, lab host name or Internet Protocol version 4 (IPv4) address. None of the removed internal names occurs in configs, header metadata or tensor names, the manifest, or `ALWAYS_ON.json`. The card is checked the same way before upload.

#### Reconstruction error on all held-out tokens

This repository measured reconstruction error with its reader on the published files, using all tokens of the four held-out splits in each layer's calibration folder from the original project. There were 257,102 tokens per layer, including 181 at position 0, leaving 256,921 with position 0 dropped. FVU was measured in raw units as the sum of squared errors divided by the sum over splits of squared deviations from each split's per-dimension mean.

- At all 32 layers and both state sites, FVU agreed with the original project's record within 1.7e-6, both with position 0 dropped and with it included. The limit was 1e-4, set before the measurement, and the token counts per split matched the record. With position 0 dropped, state FVU ranged from 0.0988 at layer 0 to 0.4038 at layer 31 for `resid_mid`, and from 0.1112 at layer 0 to 0.4202 at layer 31 for `resid_post`.
- MLP FVU on the same tokens, with position 0 dropped, ranged from 0.0425 to 0.2898. No original-project record exists for comparison on these tokens: its training-time MLP FVU was recorded on other tokens, shown in “MLP FVU recorded at training”. Including position 0 changed MLP FVU by at most 0.0067.

The held-out activations are not published, so users cannot recompute these exact reconstruction measurements.

#### Reading text with the model

This repository ran `capture` on `Qwen/Qwen3.5-4B` and then read the two dictionaries from the published folder. The card's usage lines were run once as written, at layer 16, on “The committee met on Thursday to review the budget.” The sentence had 10 tokens, with 9 left after dropping position 0. The lines ran, and the activation and reconstruction shapes were `(9, 2560)`. FVU was 0.598 at `resid_mid`, 0.567 at `resid_post` and 0.464 for the MLP output, with 21.0 transforms on per token. Nine tokens are too few for these values to mean much; they show that the lines run.

A larger check used eight web texts of 256 tokens: the clean FineWeb-Edu texts of the smaller companion's demonstration items named `count:100`. There were 2,040 tokens after position 0 was dropped. The table gives this repository's raw reconstruction FVU at each site and the mean number of transforms on per token on those texts.

| Layer | State FVU, `resid_mid` | State FVU, `resid_post` | MLP output FVU | Transforms on per token |
|---|---|---|---|---|
| 0 | 0.1057 | 0.1214 | 0.0829 | 31.5 |
| 16 | 0.4327 | 0.4131 | 0.2788 | 22.7 |
| 31 | 0.4649 | 0.4236 | 0.0431 | 14.4 |

These values are close to, and a little above, those on the held-out tokens used for the per-layer table, which are other text wrapped to a fixed width. The Qwen3.5-4B tokenizer gave the same token identifiers as the Qwen3.5-0.8B tokenizer on these texts. This repository did not compare `capture` on 4B with the original project's capture; that comparison exists only for the smaller companion.

The model in bfloat16 (bf16), together with one layer's two dictionaries, needed more than 14 GiB of GPU memory in this run. It ran under a 26 GiB cap.

### Limitations

1. Reconstruction is worse than in the smaller companion. Middle-layer state FVU is 0.31 to 0.42 for this set, against 0.25 to 0.27 for the 0.8B set. The proposed cause, the smaller active fraction of 3.75% against 9.4% of the width, is an inference, not a measurement.
2. The original project measured spliced-in loss at layer 16 only. The other 31 layers were not measured.
3. The 24 active groups per token are a default from one circuit comparison. Spliced-in loss favours more groups, as the measurements with 40 and 60 groups show.
4. Many MLP transforms never fired on the held-out training shard. The dead fraction was 0% at layer 0, 35 to 37% at layers 11 to 13, 40 to 43% at layers 14 to 16, and 60% at layer 31, where 3,974 of 9,920 were alive.
5. Transforms that fire almost always carry a large share of write energy at most layers. For transforms firing on at least 90% of tokens, the maximum was 0.69 on the web text sample, at layer 11, and 0.68 on the training-mix shard, at layer 7. On the training-mix shard the share was at least 0.30 at 21 of 32 layers. Treat these transforms as one dense background block when tracing, and always identify the text behind an always-on count.
6. The original project chose the sparsity penalty at one layer only, layer 16. No other layer compared penalties.
7. No unit of this set was read, edited or steered by the experiments behind the smaller companion's documents. Nothing here supports a claim that this set's groups or transforms are interpretable in general. The original project used this set to trace one circuit, the JavaScript ternary, and in steering experiments whose scripts are not published with this set. Those experiments used one seed, template-built prompts and units selected by contrasts between prompt types; they cannot be re-run from the published files and code.
8. The original project's four checks per layer checked shape, reproduction of the trainer's number, all held-out tokens, and position 0 dropped. Those checks and this repository's tests show that the files are built and read as defined. They are not a statement about quality.
9. State dictionaries were trained without position 0, while MLP dictionaries were trained with it. This account of the 4B training follows from shared training code and was not checked separately for 4B; it is an inference. Drop position 0 when measuring with a state dictionary. The reader's `capture` drops it.
10. This set covers one seed and one model. The original project saw large variation between seeds in numbers resting on a single group in the 0.8B set.
11. The original project's training-time MLP FVU, transforms on per token and alive counts came from a held-out shard of the training capture, with position 0 included and 65,536 tokens. That shard was deleted, so those numbers cannot be recomputed from the published files. This repository's MLP FVU column uses other tokens, from the calibration folders' held-out splits, and is not expected to equal the training record.
12. The held-out activations behind this repository's FVU columns are not published. Users cannot recompute those exact measurements from the published files.
13. No demonstration script exists for this set, and its trainers are not in the repository.
14. This repository did not compare `capture` on 4B with the original project's capture. That comparison exists for the smaller companion only. The card's usage was run once, and the separate larger text check is reported under “Reading text with the model”.

This set is a dated copy without an owner decision to freeze it. It contains no mixer dictionaries. All reported comparison runs used one machine with one RTX 5090 per run.

### License and credits

The code is licensed under MIT in `LICENSE`. The dictionary weights in the Hugging Face repository are released under MIT, as stated by `license: mit` in this card's header. The header names `Qwen/Qwen3.5-4B` as the base model. That model is licensed under the Apache License, Version 2.0: its repository's `LICENSE` and model card state that licence at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The base model is not part of this set and is not redistributed here. `load_language_model` in `src/statescope/lm.py` loads it separately under its own licence. The reader does not load the model; `capture` takes a model that the caller has loaded.

The file `mlp/L<L>/postnorm.safetensors` holds numbers copied from the base model. Its tensor `w_eff` has 2,560 numbers equal to 1 + the weight of that layer's `post_attention_layernorm`, the RMSNorm before the MLP, and its metadata holds the norm's `eps` (1e-06). Qwen3.5 stores this weight as an offset from 1. No other file of the set holds base-model parameters. These norm values come from the Apache-2.0-licensed base model. All other tensors were trained on activations of the base model: state dictionaries on the residual stream, and MLP dictionaries on MLP inputs and outputs.

The MLP dictionary's form and trainer come from the public `crosslayer-transcoder` code, which is MIT licensed, Copyright 2025 Georg Lange.
