---
license: mit
base_model: Qwen/Qwen3.5-0.8B
---

# Qwen3.5-0.8B per-layer dictionaries

These files provide two dictionaries for every layer of Qwen3.5-0.8B and a Python reader. The state dictionary writes the residual stream (the vector layers pass along) as a few active groups of directions. The MLP (multilayer perceptron) dictionary writes the layer's MLP as a sum of small gated linear operations called transforms; gates control when and how strongly they act.

### What you can do with it

- See active groups on a token (a piece of text) and reconstruct its activation (internal vector).
- See which transforms fire and what each adds to the MLP output.
- Change a group's code (its coordinates) or a transform's gate and decode back to vectors.
- Follow the demonstrated computation across layers through named units.

### An example: how the model decides to break a line

With text wrapped at a fixed width, the model must place line breaks by tracking how long the current line is.

1. Layer-8 transform 189 fires on the first token of a line.
2. State group 11586 holds most of that signal.
3. Head 10 of the layer-9 mixer carries the count. A mixer is a layer's GatedDeltaNet or attention sublayer, the part that combines information across tokens.
4. Layer-9 transforms read it mostly through state group 4885.
5. Transform 410 is on while the line is short; transform 1749 turns on when it is long. They turn the count into the line-break decision.

The test swaps in values from text wrapped at 60 characters with its last line break moved. Swapping only the first transform's contribution through the mixer reproduces 0.328 of the average change in the model's preference for newline over space. Swapping the whole layer-8 MLP output the same way reproduces 0.353 of that change. In this setting, one transform out of 3,968 moves the answer about as much as the whole MLP output of its layer.

The original project found this chain with its own tools; `demo/line_count.py` re-measures it from published files in under two minutes on one RTX 5090 graphics card.

### Try it

In the [code checkout](https://github.com/mconcat/statescope), install dependencies and download [weights](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b); replace `<folder>` with the destination.

```sh
pip install -e '.[model]'
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
```

Reconstruct a layer's activations:

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

Check files and run the demonstration:

```sh
python demo/verify_release.py --release <folder>
python demo/line_count.py --release <folder>
```

### What is in the box

- The 146 files occupy about 27.4 gibibytes (GiB).
- The 144 weight and config files cover 24 layers; the other files are the manifest and model card.
- Each state dictionary activates 24 of 16,384 groups per token.
- Each MLP dictionary has 3,968 transforms.

### What this does not show

This set covers one model and one dictionary training seed (the number controlling randomness). It has no mixer dictionaries. This task does not establish an advantage over neurons (individual MLP units). Unpublished trainers and evaluation data prevent reproducing training and the exact reported measurements. All comparisons used one machine with one RTX 5090.

See [For agents reading this repo](#for-agents-reading-this-repo) for exact calls and conventions, and [Appendix: details of the measurements](#appendix-details-of-the-measurements) for the full evidence.

## For agents reading this repo

### What is here and what is not

- Get weights from [mconcat/statescope-qwen3.5-0.8b](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b). Get the reader, demonstration, tests and staging/measuring tools from [mconcat/statescope](https://github.com/mconcat/statescope).
- Use this set for the 24 layers of `Qwen/Qwen3.5-0.8B`: one state dictionary for two residual sites and one MLP dictionary per layer. Do not look for mixer dictionaries. A mixer is a layer's GatedDeltaNet or attention sublayer.
- Use the main-branch reader for the published form only. The earlier design and its reader are on branch `earlier-design`; that code did not train these weights. The published set's trainers are absent; porting is planned.
- Load the base model separately under its own licence. It is not redistributed as part of this set.

### File layout

Use unpadded layer indices from 0 to 23 for `<L>`.

```text
MANIFEST.json
state/L<L>/cfg.json
state/L<L>/weights.safetensors
state/L<L>/usage.safetensors
mlp/L<L>/config.yaml
mlp/L<L>/weights.safetensors
mlp/L<L>/postnorm.safetensors
```

- Read `MANIFEST.json` for each layer's file hashes, training records and `original_sha256` for the configs before their edits.
- Read `state/L<L>/cfg.json` for the state dictionary's form, sites, sizes and training options.
- Load `state/L<L>/weights.safetensors` for the trained decoder and encoder tensors, decoder bias, site means and site scales.
- Keep `state/L<L>/usage.safetensors` with the state files. It holds training counters: per-site group selections in the last counting window (`fire`), per-site token totals for that window (`n_tokens`), running selection rates (`use`) and tokens since each group's last selection (`since`). The reader does not load it.
- Read `mlp/L<L>/config.yaml` for the MLP configuration; the reader uses its `model` section and `data.init_args.post_norm_stats`.
- Load `mlp/L<L>/weights.safetensors` for the trainer's unchanged state-dict tensors, including gates, read and write matrices and input/output statistics.
- Load `mlp/L<L>/postnorm.safetensors` for tensor `w_eff` and metadata `eps`, the base model's root mean square normalization (RMSNorm) coefficients before the MLP.

The weights repository contains 146 files: 144 weight and config files, `MANIFEST.json` and `README.md` (the model card).

### How to load and call

Use Python 3.12 or newer. Install `pip install -e '.[model]'` in the code checkout for model capture and the demonstration; use `pip install -e '.[model,dev]'` to add pytest and numpy. The standalone reader needs torch, safetensors and PyYAML. The `model` extra adds transformers and flash-linear-attention. There is no `train` extra or installed `statescope` command.

Use `StateVocab`, `PublicMolt` and `capture` from `statescope.reader`. Tensor shapes below use `n` for tokens, `d = 1024` for model width, `G = 16384` for groups, `b = 4` for directions per group, `k = 24` for active groups and `T = 3968` for transforms. `Tensor` means `torch.Tensor`; `Device` is `str | torch.device`.

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
- Drop position 0, the first token of each sequence, when measuring anything involving the state dictionary. The original project dropped it during state training and included it during MLP training. `capture` drops it. Both raw FVU columns for state sites and the raw MLP FVU column drop it; the original training-time metrics include it.
- Compute raw state FVU in residual units and raw MLP FVU in MLP output units. FVU is a ratio of squared reconstruction error to squared deviation from the per-dimension mean of each held-out split. Keep `heldout_fvu_own_units` separate: it uses per-token-RMS-normalized, site-standardized state units.
- Read transforms on per token as a mean count, transforms alive as a count out of the total, and MLP penalty as the sparsity coefficient.
- Report spliced-in loss in nats. It is the increase in next-token cross-entropy after replacing one layer's activation with a dictionary reconstruction.
- Read demonstration firing rates and frame energies as fractions. Read a share as a ratio of mean margin changes, not an accuracy: `mean(m_clean - m_patched) / mean(m_clean - m_corrupt)`. The margin is the last-token logsumexp of newline-token logits minus that of tokens starting with a space. Table intervals are bootstrap intervals from 2.5 % to 97.5 %.
- Keep transform indices in stored order: largest rank first. Tiers have 128 transforms of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16 and 2,048 of rank 8. Read `M.rank (T,)` for each transform's rank. Hard gates can disagree near a floating-point threshold.
- Pass raw `resid_mid` to `PublicMolt` methods that take `x`. The dictionary was trained on the post-norm MLP input. The reader computes it as `u = x * rsqrt(mean(x^2) + eps) * w_eff`, then standardizes by the trainer's per-dimension statistics. `postnorm.safetensors` is separate from the trainer's state dict; `w_eff = 1 + weight` of `post_attention_layernorm`, and metadata `eps` is `1e-06`. Output statistics restore raw MLP units.

### How to verify

Run these commands from the code checkout. Replace `<folder>` with the weights destination.

```sh
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
python demo/verify_release.py --release <folder>
python demo/line_count.py --release <folder>
pytest -m "not legacy"
```

- Expect the file check to print `layer <L>: checked` and finish with `all 144 files match the manifest`. It compares Secure Hash Algorithm 256-bit (SHA-256) hashes only and uses only the Python standard library, so it can run before installation. A mismatched hash makes it fail.
- Expect the demonstration to print shares and firing rates. Add `--device cuda:0 --out <result_folder>` to select the device and write `result.json` (shares and firing rates) and `result.safetensors` (per-item margins and gates). It prints measurements rather than a pass/fail verdict. The recorded comparison passed with all 351 shares within 0.0030 of the reference run and original records; the registered tolerance was 0.005.
- Install the `dev` extra for pytest. `pytest -m "not legacy"` selects 36 of 333 tests and needs no private data. The other tests have the `legacy` marker and require frozen original code, the original project's results and the staged weight folders specified in `tests/local.env.example`. Run `pytest` for everything. A missing required path fails its test; nothing is skipped silently. There is no `release` marker.

### Where the published numbers come from

- Treat raw state and MLP FVU as this repository's measurements on all tokens of the original project's four held-out splits, dropping position 0; the splits have 257,122 tokens per layer, including 179 at position 0. Their unpublished activations prevent exact recomputation.
- Treat `heldout_fvu_own_units` as the original project's state-training record on the first 4,096 tokens of each split, including position 0; this reader reproduced all 48 values within 6.9e-7, but those activations are unpublished.
- Treat training-time MLP FVU, transforms on per token and transforms alive as the original project's records on held-out training shards with position 0 included; those shards were deleted, so these numbers cannot be recomputed.
- Read layer training sizes, rates and penalties from the manifest and configs; this repository checked the training facts in all 24 state and all 24 MLP configs.
- Treat spliced-in loss as the original project's measurements on FineWeb-Edu text outside the training mixture, with clean cross-entropy 2.808 and loss over positions 1 to 254; this repository did not repeat them, and the texts are unpublished.
- Treat demonstration shares, firing fractions and frame-energy fractions as this repository's re-measurement of the original project's chosen chain on the published items; the tables use the even-numbered half, with 32 items per cell. The firing fractions use clean count-cell texts.
- Treat reader agreement, file checks and demonstration agreement as this repository's recorded comparisons on one machine with one RTX 5090; see the appendix for each comparison and its tolerance.

### Do not claim

- Do not claim that the published code trains these weights, reproduces a full training run, or makes unpublished evaluation data available.
- Do not claim that the demonstration discovers its units, that dictionary units outperform neurons, or that the result generalizes beyond this model, seed and fixed-width task.
- Do not call every state group sparse. Group 4885 is a dense coordinate group active on most tokens.
- Do not claim that group 4885 is the only route to the answer, that downstream shares add, or that the original pass rule fully held through transforms 410 and 1749 alone.
- Do not claim exact agreement for every gate or single-item margin. Rounding near thresholds and conversion to bfloat16 affect results.
- Do not claim that this set improved spliced-in loss over the earlier set at every layer.
- Do not claim verification on another machine, another graphics processing unit (GPU) model or a fresh installation in a new environment.
- Do not report results for another model from this set. The test suite also covers a 4B set with 32 layers; no results for it are reported here.

### Appendix pointers

Use “State dictionary” and “MLP dictionary” for definitions and formulas; “Config files” for accepted formats and edits; “Numbers per layer” and “Spliced-in loss” for metrics; “Demonstration” for interventions and tables; “What the tests showed for this set” for agreement limits; “The code repository and the earlier design”, “Limitations” and “License and credits” for scope and provenance.

## Appendix: details of the measurements

### What this is

This set contains separately trained state and MLP (multilayer perceptron) dictionaries for each of the 24 layers of `Qwen/Qwen3.5-0.8B`. The model has width 1024. The base-model revision is `2fc06364715b967f1860aea9cf38778875588b17`. The set was frozen on 2026-11-18.

A mixer is a layer's GatedDeltaNet or attention sublayer. A residual stream site is a location between the mixer and the MLP or after the MLP where a state dictionary reads the residual stream. A state dictionary represents the residual stream at both sites. An MLP dictionary represents a layer's MLP operations, from its post-norm input to its raw output. The set contains no dictionary for the mixers.

The code is public at [mconcat/statescope](https://github.com/mconcat/statescope), and the weights are public at [mconcat/statescope-qwen3.5-0.8b](https://huggingface.co/mconcat/statescope-qwen3.5-0.8b). The weights repository's `README.md` is the model card, with the same text as `release/qwen35-08b-v2/MODEL_CARD.md` in the code repository. The base model is loaded separately.

### Files and download

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

There are 144 weight and config files. The weights repository holds 146 files including `MANIFEST.json` and `README.md`, with a measured total size of about 27.4 gibibytes (GiB). State dictionaries occupy about 12.0 GiB in total. MLP dictionaries occupy about 15.4 GiB in total. The repository carries a manifest copy at `release/qwen35-08b-v2/MANIFEST.json`. The manifest provides a Secure Hash Algorithm 256-bit (SHA-256) hash for each layer's files and the numbers recorded during training.

The following command checks all 144 files against the manifest:

```sh
python demo/verify_release.py --release <folder>
```

The MLP export stores the trainer's state dict in `weights.safetensors` and its configuration in `config.yaml`. Its root mean square normalization (RMSNorm) coefficients are separate from the state dict. They are stored in `postnorm.safetensors` as tensor `w_eff` and metadata `eps`.

- Read `MANIFEST.json` for each layer's file hashes, training records and `original_sha256` for the configs before their edits.
- Read `state/L<L>/cfg.json` for the state dictionary's form, sites, sizes and training options.
- Load `state/L<L>/weights.safetensors` for the trained decoder and encoder tensors, decoder bias, site means and site scales.
- `state/L<L>/usage.safetensors` holds training counters: `fire` (groups, 2 sites) counts training tokens selecting each group at each site in the last counting window, `n_tokens` (2,) gives that window's token totals, `use` (groups, 2 sites) gives each group's running average selection rate at each site, and `since` (groups,) counts training tokens since each group was last selected. The reader does not load it.
- Read `mlp/L<L>/config.yaml` for the MLP configuration; the reader uses its `model` section and `data.init_args.post_norm_stats`.
- Load `mlp/L<L>/weights.safetensors` for the trainer's unchanged state-dict tensors, including gates, read and write matrices and input/output statistics.
- Load `mlp/L<L>/postnorm.safetensors` for tensor `w_eff` and metadata `eps`, the base model's root mean square normalization (RMSNorm) coefficients before the MLP.

Download with the Hugging Face command line tool:

```sh
hf download mconcat/statescope-qwen3.5-0.8b --local-dir <folder>
```

Use that same `<folder>` as `--release <folder>` for the file check and demonstration.

### The code repository and the earlier design

The main branch of [mconcat/statescope](https://github.com/mconcat/statescope) contains the reader (`src/statescope/reader.py`), its neighboring modules, the demonstration and items (`demo/line_count.py` and `demo/line_count_items.safetensors`), the file check (`demo/verify_release.py`), staging and measuring tools (`devtools/`), and the tests and their rules file. `config.py` defines the two models, `lm.py` loads the language model, and `heldout.py` and `metrics.py` serve the measuring tools and tests. The main-branch reader handles only the published form: the tied state dictionary with per-token RMS input and the MLP dictionary in the public penalty form. The published set's trainers are not in the repository; porting them is planned.

The earlier design is kept on branch [`earlier-design`](https://github.com/mconcat/statescope/tree/earlier-design) of the same repository. It used an untied state dictionary and an MLP dictionary with 41,484 atoms, called M1w by the original project. That branch keeps the token stream, preparation, both trainers, manifest and gate check, the `statescope` command line, the spliced-in-loss module, its release folder and example, its reader classes (including `MoltOps` for its MLP files), and the tests for rules R1 to R7. Its code reproduces how an earlier, unpublished set was trained; it did not train these weights. Read its files with the reader on that branch.

`tests/EQUIVALENCE_RULES.md` retains the text of R1 to R7 because R8 to R11 cite their limits. Of those earlier rules, only the capture comparison R1.5 still has a test on the main branch. The package no longer installs a `statescope` command and has no `train` extra. Its extras are `model` (transformers, accelerate and flash-linear-attention) and `dev` (pytest and numpy). The earlier-set comparison under “Spliced-in loss” concerns the original project's dictionaries.

### Config files

Before publication, the config files were edited at every layer to remove internal names of the original project.

- In `state/L<L>/cfg.json`, `kind` is now `state_dictionary`; it previously named the original project's trainer module. Four keys were removed: a configuration label, a source file path, a run label and the md5 record of eight source files of the original project. That record held file hashes. Every other key and value is unchanged.
- In `mlp/L<L>/config.yaml`, the one line naming the original project's data module was removed. Every other byte is unchanged. The `data` section now contains only `init_args`. The file is no longer a complete Lightning command line configuration. It could not be run before this edit either, because the data module is not published.
- No safetensors file changed during these edits. Tensor bytes and headers are the ones used by the tests.

`MANIFEST.json` lists the SHA-256 hashes of the published files. Under `original_sha256`, it also lists the hash of each of the two configs before the edit. Rule R11 was written before this config rewrite.

The reader accepts the state dictionary only when `kind` is `state_dictionary`, `gate` is `topk`, `form` is `bsf` and `input` is `token_rms`. These fields identify the tied block-sparse dictionary, its top-k group selection and its per-token RMS input convention described below. `lam`, `lam_dr`, `lam_pc`, `lam_nest` and `n_dense` are options of the original trainer. All are 0: the state dictionaries used no penalty term and no dense block.

The MLP configuration retains the training-run sections `seed_everything`, `trainer`, `model`, `data` and `ckpt_path`. Class paths starting with `crosslayer_transcoder.` name classes of the public `crosslayer-transcoder` code and its training fork. The reader reads the `model` section for class paths, sizes and ranks, and `data.init_args.post_norm_stats`. It ignores the rest.

In the header metadata of `mlp/L<L>/weights.safetensors`, `source` is `training.ckpt`, the trainer's checkpoint file name. `keys` says that tensor names are the trainer's unchanged state-dict names.

### State dictionary

`statescope.reader.StateVocab` reads one layer's state dictionary. The same dictionary covers `resid_mid.L` and `resid_post.L`. The former is the residual between the mixer and MLP, at the input of the norm before the MLP. The latter is the residual after the MLP, equal to `resid_mid` plus the MLP output.

A group is a set of 4 directions in the state dictionary. A frame is the orthonormal basis of a group's 4-dimensional write subspace. A code gives the token's coordinates in that frame, multiplied by a per-group scale. The dictionary has 16,384 groups in a tied form. The 24 groups with the largest written norm are active at each token.

The input convention is per-token root mean square (RMS) normalization. The dictionary first divides the residual by its own size using `x * rsqrt(mean(x^2) + eps)`, with `eps = 1e-6`. It then standardises the result with a mean vector and one scale per site. To recover the raw residual, the reconstruction is multiplied by the token's own `sqrt(mean(x^2) + eps)`. `StateVocab.reconstruct(x, site)` performs both steps. Alternatively, `decode` accepts `rms=V.token_rms(x)`.

Each layer's dictionary was trained on 30,003,200 tokens over 7,325 steps. Position 0 means the first position of every sequence. The original project dropped position 0 during training. The raw fraction of variance unexplained (FVU) columns below also drop position 0. The reader's `capture` function does so.

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

### MLP dictionary

A transform is a linear operation in an MLP dictionary with its own read and write matrices. Its write is what it adds to the MLP output. A gate controls a transform through a learned threshold on its pre-activation. `statescope.reader.PublicMolt` reads the sparse mixture of linear transforms, called MOLT. This dictionary uses a JumpReLU gate with a threshold learned separately for each transform. Its read and write matrices are free. It has no skip connection. Training uses a tanh sparsity penalty.

The form comes from the public `crosslayer-transcoder` code, branch MOLT, commit 376fabe. Training used a fork that changes no mechanism. There are 3,968 transforms per layer in five rank tiers. The tiers contain 128 transforms of rank 128, 256 of rank 64, 512 of rank 32, 1,024 of rank 16 and 2,048 of rank 8. They are stored with the largest rank first. The reader's `rank` array gives each transform's rank.

The dictionary takes the layer's post-norm MLP input and produces a reconstruction of the raw MLP output. The reader computes that input from raw `resid_mid` using the layer's own RMSNorm coefficients. The formula is `u = x * rsqrt(mean(x^2) + eps) * w_eff`. The input is then standardised using per-dimension statistics retained by the trainer. The output is returned to raw MLP units with the stored output statistics.

Each layer was trained for 58,600 steps of 500 tokens, or 29.3 million tokens. The learning rate was 4.0e-05. The sparsity penalty coefficient is 7.5e-05 for layers 0 to 22. Layer 23 used 2.5e-05 in a separate run. The original project included position 0 of every sequence during MLP dictionary training, unlike state dictionary training.

The gate is a hard threshold. A transform whose pre-activation lies within float rounding of its threshold can be on in one computation and off in another. The reader gives the following forward-pass formulas:

```text
u = x * rsqrt(mean(x^2) + eps) * w_eff          the layer's own RMSNorm on raw resid_mid x
z = (u - mu_in) / sd_in                         per-dimension statistics kept by the trainer
h_t = e_t . z + b_t ;  g_t = h_t if h_t > theta_t and h_t > 0 else 0                    JumpReLU gate of transform t
y_std = sum over transforms t of  g_t * (z V_t) U_t                                      V_t (d, rank_t), U_t (rank_t, d)
y = y_std * sd_out + mu_out                                                              ~ raw MLP output
```

### How to read the dictionaries

Use Python 3.12 or newer. Install the reader with torch, safetensors and PyYAML:

```sh
pip install -e .
```

Install transformers, accelerate and flash-linear-attention for `capture` and the demonstration:

```sh
pip install -e '.[model]'
```

Add pytest and numpy for development:

```sh
pip install -e '.[model,dev]'
```

There is no `train` extra or installed `statescope` command. Without flash-linear-attention, transformers silently uses a slow GatedDeltaNet reference implementation. A warning about missing `causal_conv1d` means a slower reference implementation is being used with unchanged values.

The reader is the single file `src/statescope/reader.py`. It needs torch, safetensors and PyYAML. It can be copied next to the weights. `StateVocab` provides `encode`, `decode`, `reconstruct` and `group_write`. `PublicMolt` provides `forward` and `transform_write`. The earlier design's reader classes are on branch `earlier-design`.

`capture` returns `resid_mid`, `mlp_out` and `resid_post` for one layer with position 0 dropped. It returns 32-bit floating-point tensors and drops padded positions when given an attention mask. It needs transformers and flash-linear-attention.

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
groups, codes = V.encode(acts["resid_post"], "resid_post.9")
x_hat = V.reconstruct(acts["resid_post"], "resid_post.9")
gate, y_hat = M.forward(acts["resid_mid"])               # gate (tokens, 3968), y_hat ~ acts["mlp_out"]
write = M.transform_write(acts["resid_mid"], 410)        # what transform 410 adds to the MLP output
```

#### Exact calls and shapes

Every class and method of the reader, with argument and return shapes, is listed once, under [How to load and call](#how-to-load-and-call) in the part for agents above.

### Numbers per layer

FVU means fraction of variance unexplained. It is squared reconstruction error divided by the squared deviation from the per-dimension mean of each held-out split.

The “Layer” column identifies the layer. “State FVU, `resid_mid`” and “State FVU, `resid_post`” give reconstruction error in raw residual units at the two state sites. “MLP FVU” gives reconstruction error in raw MLP output units. This repository measured these three columns with its reader on all tokens of the original project's four held-out splits, with position 0 dropped. The splits contain 257,122 tokens per layer, of which 179 are at position 0. These columns show what the published files do under the published reader with position 0 dropped. The held-out activations are not published, so users cannot recompute these exact numbers. The state values agree with the original project's table, made with its separately written reader, within 6.5e-8 at every layer and both sites, both with position 0 dropped and with it included.

The original project recorded “MLP FVU recorded at training”, “MLP transforms on per token” and “MLP transforms alive” on a held-out shard of each layer's training capture, with position 0 included. “MLP FVU recorded at training” gives the FVU on that shard. “MLP transforms on per token” is the mean number of transforms whose gate is open at a token. “MLP transforms alive” counts transforms that fired on at least one token of that shard, out of the total. The others never fired there. The shards held 267,271 tokens at layers 9 and 23. At layer 9, 2035 of 3968 transforms were alive. The shards were deleted afterwards, including the shard for layer 23's separate run. These three columns cannot be recomputed from the published files. This repository did not repeat those measurements. The two MLP FVU columns differ because they are measured on different tokens.

“MLP penalty” is the sparsity penalty coefficient from the manifest. It is 7.5e-05 for layers 0 to 22 and 2.5e-05 for layer 23, which used a separate run. The training facts stated for each layer were checked in all 24 state configs and all 24 MLP configs. The MLP configs differ only in layer index, logger name and penalty.

This repository measured raw FVU with and without position 0. Position 0 matters for the state dictionary at some layers. At layer 6, `resid_mid`, raw FVU is 0.2524 with position 0 and 0.2184 without it. At layer 0, it is 0.0778 with position 0 and 0.0777 without it. For the MLP dictionary, the two differ by at most 0.014 at any layer.

The original project also recorded the state trainer's own held-out FVU in `MANIFEST.json` as `heldout_fvu_own_units`. It uses the dictionary's own units, which are per-token-RMS-normalised and site-standardised. The evaluation uses the first 4,096 tokens of each split with position 0 included. The repository's reader reproduces all 48 numbers within 6.9e-7. At early layers, this number is lower than the number on all tokens because the first 4,096 tokens are an easier subset, not because of position 0. At layer 0, `resid_mid`, it is 0.0533 on the subset against 0.0693 on all tokens in the same units, or 0.0777 in raw units. At layer 9, it is 0.2452 on the subset against 0.2438 on all tokens in the same units.

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

### Spliced-in loss

Spliced-in loss is the increase in next-token cross-entropy when a dictionary reconstruction replaces the true activation in the forward pass, one layer and one dictionary at a time. The original project measured these increases in nats on text with clean cross-entropy 2.808. This repository did not repeat the measurement.

The original project used FineWeb-Edu rows that were not in the training mixture. It took the first 100 documents of at least 256 tokens and the first 256 tokens of each. Position 0 was not replaced. The loss was averaged over positions 1 to 254, or 25,400 positions.

The dictionaries are the released ones. The MLP result for layer 23 comes from a separate measurement by the original project on the same kind of web text, 100 documents of 256 tokens. It recorded +0.0502 for the released layer 23 dictionary.

In the original project's comparison with the earlier set of dictionaries on the same text, the earlier set had a smaller increase for the state dictionary at all 24 layers and for the MLP dictionary at layers 0 to 22. For the state dictionary at layer 9, the recorded increase was +0.243 for this set against +0.161 for the earlier set. At layer 23, the released MLP dictionary had the smaller increase, +0.0502 against +0.0840 for the earlier set.

Each table row is a layer, and the columns distinguish replacement at `resid_post` by the state dictionary from replacement by the MLP dictionary. Values are increases in cross-entropy in nats.

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

### Demonstration

`demo/line_count.py` follows a line-counting chain from the layer-8 (L8) MLP through layer-9 (L9) GatedDeltaNet head 10 to the L9 MLP. The model predicts a line break at the last word of a full line. Each item pairs 256-token texts that end on the same words. The clean text's last line is full. The corrupt text moves the last line break or wraps all lines wider, so a line break is not due.

The margin is the last-token logsumexp of newline-token logits minus the logsumexp of logits for tokens starting with a space. A patch replaces a clean-run tensor with a corrupt-run value moved onto the clean token positions. A share is `mean(m_clean - m_patched) / mean(m_clean - m_corrupt)`, where `m` is the margin. A share of 0 means that the patch changes nothing. A share of 1 means that the patch alone turns the clean run into the corrupt one.

A cell is a task setting defined by the wrapping width and the change that makes the corrupt text. The cells are `count:60`, `count:100` and `width:100`. The count cells move the last line break, while the width cell wraps all lines wider. Each contains 64 pairs stored as token ids and position pairing in `demo/line_count_items.safetensors`. These are re-wrapped FineWeb-Edu excerpts.

The original project found the chain. The script takes its choice of transform, group and head as given and re-measures them. It does not search for them. The original project ranked transforms on the odd-numbered items and reads results on the even-numbered ones. Some groups were first chosen after looking at all items; when the original project repeated the selection on odd-numbered items only, the chosen groups, frames and transforms were the same. Head 10 was known from earlier work. The results below use the even-numbered half, with 32 items per cell.

The chain has these steps:

1. Transform 189 of the layer-8 MLP dictionary fires on the first token of a line. Its write lies mostly inside the frame of state group 11586 of `resid_post.8`, the first-token-of-a-line group.
2. The model's layer-9 mixer reads that write. Swapping only transform 189's write or only group 11586 moves the answer about as much as swapping the whole layer-8 MLP output in the count cells. GatedDeltaNet head 10 carries the count inside the mixer.
3. Transforms 410 and 1749 of the layer-9 MLP dictionary turn the count into the line-break decision. Transform 410 is on while the line is short; transform 1749 is on once it is long. What those transforms read of the mixer's write lies mostly inside the frame of state group 4885 of `resid_mid.9`. The rest still reaches the answer through other routes.

Transform 189 fires on 1.000 of line-first tokens, 0.001 of line-break tokens and 0.010 of other tokens on the clean texts of the count cells. Of the energy in its swapped write, 0.769 lies inside group 11586's frame. The largest value among five random frames is 0.016. The chance value for a 4-dimensional frame in 1,024 dimensions is 0.004.

Transform 410 fires on 0.84 to 0.89 of tokens when a line holds 20 to 49 characters. It fires on 0.25 at 60 to 69 characters and 0.03 at 70 to 79. Transform 1749 fires on 0.00 below 70 characters, 0.19 at 80 to 89 and 0.60 at 90 to 99.

Run the demonstration with:

```sh
python demo/line_count.py --release <folder> --device cuda:0 --out <result_folder>
```

It ran in under two minutes on one RTX 5090, including model loading. It needs transformers with Qwen3.5 support and, on a graphics processing unit (GPU), flash-linear-attention; otherwise the GatedDeltaNet layers use a slow reference kernel. The script prints shares and firing rates. `--out` writes `result.json` with shares and firing rates, and `result.safetensors` with per-item margins and gates. `--model` accepts the Hugging Face identifier or a local folder of Qwen3.5-0.8B; `--items` defaults to the bundled item file; `--device` defaults to `cuda:0`.

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

### How to verify

The commands, what a pass looks like and what the other tests need are listed once, under [How to verify](#how-to-verify) in the part for agents above.

### What the tests showed for this set

- All comparisons ran on one RTX 5090 machine with Qwen3.5-0.8B. Rules R8 and R9 were written and committed before the runs they judge.
- The state reader was compared with a reader the original project wrote separately from its trainer. This covered 24 layers and 4 splits of 4,096 tokens at 2 sites. Of 786,432 tokens, 4 selected a different group set. Each had its 24th and 25th group norms within 7.3e-7. Raw reconstructions elsewhere agreed within 7.1e-7 relative. All 48 of the trainer's held-out FVU numbers in its own units were reproduced within 6.9e-7. These use the first 4,096 tokens of each split with position 0 included.
- The MLP reader was compared with the unmodified public forward pass at 24 layers. There was 1 differing gate decision, at layer 3. Outputs on other tokens agreed within 1.9e-6 relative.
- The sum of firing transforms' writes plus the output mean matched the forward pass within 1.5e-6 relative. The largest difference was 1.476e-6 at layer 22.
- This repository measured reconstruction error on all held-out tokens, 257,122 per layer. The state dictionary's raw FVU agreed with the original project's table within 6.5e-8 at all 24 layers and both sites. The limit was 1e-4.
- All 144 release files matched the manifest. MLP tensors for layers 0 to 22 were bit-identical to the training exports used by the reader tests. Layer 23 passed its own comparison with the public forward pass. No file header or config held a machine path.
- Every demonstration share agreed within 0.0030 with the original harness's reference run and the original project's recorded numbers. There were 39 swaps x 3 cells x 3 halves = 351 comparisons. The original registration's tolerance was 0.005. The item file matched the original items exactly.
- On 2026-11-19, the weights were downloaded from Hugging Face into an empty folder and the code was cloned from GitHub into an empty folder. `demo/verify_release.py` reported that all 144 files matched the manifest. `demo/line_count.py` on the downloaded weights wrote a result identical, byte for byte, to the tested run on the same machine, as checked with `devtools/compare_demo_runs.py`. This check used one machine with one RTX 5090. A second machine, a different GPU model and a fresh `pip install -e .` into a new environment were not checked.

These are the recorded comparisons for this set. The later config edits left every safetensors file's tensor bytes and header unchanged. The test suite now covers the 0.8B set with 24 layers and a 4B set with 32 layers; no 4B results are reported here.

### Limitations

- The demonstration covers one model, 0.8B, one dictionary seed and texts wrapped to a fixed width.
- This task is easy for the model and has dedicated MLP neurons. The demonstration is not evidence that dictionary units are better than neurons.
- State group 11586 is a sparse first-token-of-a-line group. Not every group is sparse. Some groups encountered by the original project on this task are dense 4-dimensional coordinates active on most tokens. They carry information in their coordinates. Group 4885 is one of these dense groups and is not a sparse feature.
- Single-item margins of dictionary swaps are reproducible only to about 0.1. Shares are reproducible only to about 0.003. Patched tensors are rounded to bfloat16 (bf16). A 1e-6 relative difference between implementations flips some rounded elements.
- The trainers for this set are not in the repository. Porting them is planned. The earlier design's training code is on branch `earlier-design`; it did not train these weights and reproduces an unpublished earlier set.
- The MLP dictionaries' numbers recorded at training by the original project cannot be recomputed because the evaluation shards were deleted, including the shard for layer 23's separate run. The held-out activations behind this repository's raw state and MLP FVU columns are not published, so users cannot recompute those exact numbers either.
- Spliced-in loss was measured by the original project and was not repeated in this repository.
- No full-length training run was repeated for either design.
- Other models, including Qwen3.5-4B, are outside this set.
- There are no dictionaries for the mixers. The demonstration uses the original project's selected units; it does not search for them.
- The held-out texts behind the spliced-in-loss measurements are not published, so the exact measurements cannot be recomputed from the published files.
- All comparisons used one machine with one RTX 5090. A second machine, a different GPU model and a fresh installation into a new environment were not checked.

### License and credits

The code is licensed under MIT in `LICENSE`. The dictionary weights in the Hugging Face repository are released under MIT, as stated by `license: mit` in this card's header. The base model `Qwen/Qwen3.5-0.8B` is licensed under the Apache License, Version 2.0. Its repository holds the Apache 2.0 text in `LICENSE`, and its model card states `license: apache-2.0`. The base model is not part of this set and is not redistributed here. `load_language_model` in `src/statescope/lm.py` and the demonstration download or load it separately under its own licence. The reader does not load the model; `capture` takes a model that the caller has loaded.

The file `mlp/L<L>/postnorm.safetensors` holds numbers copied from the base model. Its tensor `w_eff` has 1,024 numbers equal to 1 + the weight of that layer's `post_attention_layernorm`, the RMSNorm before the MLP, and its metadata holds the norm's `eps` (1e-06). Qwen3.5 stores this weight as an offset from 1. No other file of the set holds base-model parameters. These norm values come from the Apache-2.0-licensed base model. All other tensors were trained on activations of the base model: state dictionaries on the residual stream, and MLP dictionaries on MLP inputs and outputs.

The MLP dictionary's form and trainer come from the public `crosslayer-transcoder` code, which is MIT licensed, Copyright 2025 Georg Lange. The demonstration's texts come from `HuggingFaceFW/fineweb-edu` under the Open Data Commons Attribution License (ODC-By 1.0).
