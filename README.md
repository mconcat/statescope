# What this is

This repository provides per-layer dictionaries for Qwen3.5-0.8B. The model has 24 layers and width 1024. A **state dictionary** represents the residual stream at two sites. An **MLP dictionary** represents the layer's MLP operations. The two dictionaries are trained separately.

The repository contains the reader, checks and training code. Separately published checkpoints total 40.1 GiB. The [model card](release/qwen35-08b/MODEL_CARD.md) contains all measured numbers. The MLP recipe, M1w, is a variant of Anthropic's sparse mixture of linear transforms; the card lists its differences.

This is a cleaned-up rewrite of research code. The rule of each comparison with the original code was written in [tests/EQUIVALENCE_RULES.md](tests/EQUIVALENCE_RULES.md) before the comparison was run. One rule, R4.6, was written after its comparison; the file says so and gives the reason.

## Install

Use Python 3.12 or newer. Install the reader with torch and safetensors:

```sh
pip install -e .
```

The `model` extra adds transformers and flash-linear-attention for running the language model. Without flash-linear-attention, transformers silently uses a slow GatedDeltaNet reference implementation. The warning about missing causal_conv1d indicates a slower reference implementation with unchanged values.

The `train` extra adds numpy and pyarrow; `dev` adds pytest and coverage. Install these extras with:

```sh
pip install -e '.[model,train,dev]'
```

Comparisons used Python 3.12.3 and torch 2.11.0+cu128. They used transformers 5.20.2 and flash-linear-attention 0.6.0. The safetensors version was 0.8.0.

## Quick start with the reader

Replace `<release folder>` with the separately published checkpoints' folder:

```sh
statescope verify --manifest release/qwen35-08b/manifest.json --root "<release folder>"
python examples/reconstruct_texts.py --layer 19 --root "<release folder>"
```

Verification checks every file's size and sha256. The example loads the model and layer dictionaries, printing reconstruction quality for three short texts.

The reader snippet assumes `model` and `input_ids` are available. **resid_mid** is the captured residual-stream site supplied to the MLP dictionary. **resid_post** is the captured residual-stream site encoded and decoded below. Capture drops position 0.

```python
from statescope.reader import StateVocab, MoltOps, capture

V = StateVocab("layers/L19/vocab", "cuda")
M = MoltOps("layers/L19/molt", "cuda")
acts = capture(model, 19, input_ids)  # Also includes "mlp_out".
groups, codes = V.encode(acts["resid_post"], "resid_post.19")
x_hat = V.decode(groups, codes, "resid_post.19")
a, y_hat = M.forward(acts["resid_mid"])
```

## Repository layout

**Held-out FVU** means fraction of variance unexplained on held-out tokens. Within `src/statescope/`:

- The self-contained `reader.py` provides `StateVocab`, `MoltOps`, `capture` and `fvu`, needs only torch and safetensors, and can be copied beside checkpoints.
- `manifest.py` and `release.py` manage manifest sizes, sha256, measured numbers, verification and regeneration.
- `config.py` centralizes settings in `ModelSpec`, `DataPaths` and `MixSpec`.
- `data/` supplies training tokens in units of 262,144 tokens.
- `lm.py`, `calibration.py`, `heldout.py` and `prep/` handle model loading, layer capture, calibration, held-out activations and the fixed mean-gate map.
- `train/loop.py` shares streaming, capture, a shuffle buffer and batching; `vocab/` and `molt/` contain the separate trainers.
- `metrics.py`, `foldcheck.py` and `splice.py` measure trainer-defined held-out FVU, hard-gate sensitivity to float rounding, and next-token loss with dictionaries spliced into the forward pass.
- `cli.py` provides the `statescope` commands.

At the root, `release/qwen35-08b/` holds `manifest.json`, `header.json` and the model card. `tests/` holds checks; `devtools/` holds original-code freezing and tracing tools.

## Training pipeline for one layer

Training needs these data sources:

- Pre-tokenized Pile shards come from `chanind/pile-uncopyrighted-qwen-3.5-1024-abbrv-2B`.
- UltraChat 200k provides parquet files.
- The model supplies continuations and chat answers; their generation code is not included yet.

Supply `--pile-dir`, `--ultrachat-dir`, `--view-corpus` and `--full-corpus`.

1. Run `statescope calibrate` for calibration statistics and held-out activations.
2. Run `statescope fixedmap` for the fixed mean-gate map, needed only by the MLP dictionary.
3. Run `statescope train-vocab` and `statescope train-molt` independently. Release budgets were 50M and 29.3M tokens, respectively.
4. Run `statescope foldcheck` and `statescope splice` on the exports.

## Tests

Install the `dev` extra, then run:

```sh
pytest -m "not release and not legacy"
```

These need no private data. They cover reader formulas on tiny synthetic dictionaries, unit tests and repository rules forbidding original-code imports, swallowed exceptions and machine-specific strings.

For other tests, supply frozen original code, checkpoints and settings from [tests/local.env.example](tests/local.env.example), then run:

```sh
pytest
```

A missing required path fails its test; nothing is skipped silently.

### What the comparisons showed

Every part was compared with the frozen original. Comparisons used Qwen3.5-0.8B, except for the reader pilot. Measurements ran on one RTX 5090 machine.

- All 168 release files match manifest sizes and sha256. Reader outputs are bit-identical to the released reader at layers 0 and 19. Against original readers on 4,096 held-out tokens, there are 0 differing gate decisions and 0 tokens with different group sets. Reader-only FVU matches the manifest within 1e-5 at layers 0, 9, 19 and 23.
- A Qwen3.5-4B pilot export at layer 16, width 2,560, loads with 0 differing gate decisions against the original readers.
- Capture is bit-identical. Spliced losses at layers 9 and 19 match original and recorded values exactly.
- The first 200 token-stream units, or 52.4M tokens, match token for token. Calibration and held-out activations match the release's training files bit for bit; the fixed map matches its recorded md5. All 300 short-run batches match original sha256 values.
- State training at layer 19 for 300 steps produces bit-identical tensors, usage counters and per-step numbers. Held-out FVU agrees within 1.2e-8; original evaluation digits themselves vary at that level.
- MLP training at layer 19 for 300 steps matches all exported tensors, the whitening matrix, firing statistics, and per-step and final numbers bit for bit. The release arguments and a one-batch dead window each cover 96 quantities.
- Neither trainer reaches dead-unit auxiliary loss in 300 steps with release arguments. The short MLP dead window and a state optimizer step with injected dead groups cover it with bit-identical results.
- Both original trainers are bit-deterministic across two runs each, permitting bit-identity comparisons. Gate-threshold results and the published manifest are regenerated.

### What the comparisons do NOT show

- Full-length training over tens of millions of tokens was not repeated.
- Trainer comparisons covered only layer 19 and seed 0. Budgets not divisible by the batch were not compared.
- The layer 9 state dictionary used earlier versions of two source files absent from the frozen copy. Only reader-level tests cover it.
- The Qwen3.5-4B configuration is outside this stage, despite the reader pilot.

## License

This repository is licensed under [MIT](LICENSE).
