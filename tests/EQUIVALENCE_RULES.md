# Equivalence rules

Every pass/fail rule for a comparison between the refactored code and the frozen original is written here **before** the comparison is
first run, and this file is committed before the test that uses the rule. A tolerance is never widened after a result has been seen: a
failing test means the cause is looked for. If a rule turns out to be wrong in itself, the old rule stays in the file with the date and the
reason next to its replacement.

"Legacy" = the frozen copy of the original research code (`legacy_snapshot/`, not committed, hashes in its `SHA256SUMS`).
"Release" = the 24 exported layers of Qwen3.5-0.8B listed in `manifest.json`.

## R0. Why some comparisons are not bit-for-bit

The MLP dictionary's gate is a hard threshold (`a = relu(h)` if `h / score_div > thr`, else 0). A (token, atom) pair whose `h` lies
within float rounding of its threshold can come out on in one computation and off in another whenever the two differ in operation order,
batch shape, precision or GPU model. The original measured 0-2 such pairs among 4,096 tokens x 41,484 atoms, each changing that one
token's output by 0.1-0.5 % of the largest output (`notes/mlp_track/fold_check.md` in the snapshot). The state dictionary has the same
kind of edge at its top-k boundary (the 24th and 25th group by written norm).

So wherever two computations of a gate are not the same sequence of floating-point operations, the rule is the original's own
(`fold_check.md`, "the check has to change"), applied per 4,096 tokens:

- **(a)** gate decisions that differ: at most 5 (token, atom) pairs;
- **(b)** over the tokens with no differing pair, max |y1 - y2| / max |y1| <= 1e-4.

For the state dictionary, read "(token) whose selected group set differs" for "pair", with the same limits.

## R1. Reader (written 2026-11-17, before any reader test was run)

Data: the first 4,096 held-out pile tokens of the layer's calibration folder (`x` = raw resid_mid, `y` = raw MLP output), layers 0 and 19,
fp32 on one GPU. This is the data of the original's `release_parity.py`.

**R1.1 Files.** Every file of every layer in `manifest.json` has the listed size and sha256. Exact.

**R1.2 New reader vs the frozen standalone reader** (`release/qwen35-08b-layer-dicts/reader.py` in the snapshot). The new reader keeps the
same floating-point operations in the same order, so on the same device and inputs every returned tensor must be **bit-identical**
(`torch.equal`): gate values, MLP output estimate, selected groups, codes, reconstructions, `atom_write`, `group_write`, `capture`.

**R1.3 New reader vs the legacy repository readers** (`RawAdaptive(TieredFolded.from_export)` for the MLP dictionary, `molt_vocab_align.lib.Vocab`
for the state dictionary). Pass/fail is rule R0 (a)+(b). In addition the numbers the original measured on processor (RTX 5090) are asserted,
because the same card model is used here; if these stricter assertions fail while R0 holds, that is a rounding finding to explain, not to waive.

| quantity | measured by the original | asserted |
|---|---|---|
| MLP dictionary: gate decisions differing (L0, L19) | 0 | == 0 |
| MLP dictionary: max abs gate value diff, max rel output diff | 0.0 | == 0.0 |
| MLP dictionary: FVU on these tokens | L19 .11542812, L0 .05644568 | abs diff <= 1e-7 from the recorded value |
| MLP dictionary: sum of all atom writes + skip + bias vs forward (first 64 tokens) | <= 3.3e-7 | <= 1.5e-6 |
| state dictionary: tokens with a different group set (L19, both sites) | 0 | == 0 |
| state dictionary: max abs code diff | 2.9e-6 | <= 1.2e-5 |
| state dictionary: max rel reconstruction diff | <= 1.24e-7 | <= 5e-7 |
| state dictionary: FVU on these tokens | mid .17954032, post .17697692 | abs diff <= 1e-7 |
| state dictionary: sum of group writes vs decode (token 0) | <= 2.4e-7 | <= 1e-6 |

The float limits are 4x the measured value, rounded; the factor was chosen here, before running.

**R1.4 Manifest FVU from the reader alone.** The reader must reproduce `heldout_fvu` of the manifest for the sample layers 0, 9, 19, 23 with
abs diff <= 1e-5, using the trainers' definitions:

- MLP dictionary: for each held-out split s in (pile, ultra, cont, chat), all tokens: `fvu_s = SSE_s / SST_s` with SST around the split's own
  per-dimension mean; pooled = sum_s fvu_s * SST_s / SST_all with SST_all around the grand mean (float64 accumulation).
- state dictionary, per site: the first 4,096 tokens of each of the four splits (position 0 included, as the trainer did); x = resid_mid,
  x + y = resid_post; sum over splits of SSE / sum over splits of SST (each split around its own mean).

The trainer evaluated the state dictionary with its training-form model, the reader uses the exported form; whether these agree to 1e-5 has
not been measured before. 1e-5 is the size expected from fp32 rounding alone [prediction]. A larger gap means the two forms select or decode
differently and has to be explained.

**R1.5 Capture.** `capture()` on the Hugging Face model vs the legacy capture (`measure.sites_v2.run_sites`) on the same token ids:
resid_mid and MLP output bit-identical (measured 0.0). resid_mid + mlp_out vs the layer's own output: max abs diff / max abs <= 5e-3
(measured 2.0e-3 and 3.0e-3: bf16 rounding of the residual add inside the model; 5e-3 is about one bf16 step, 2^-8 = 3.9e-3, plus margin).

**R1.6 Width 2,560 (Qwen3.5-4B).** The new reader loads the existing 4B L16 test exports and satisfies R1.3's structure against the legacy
readers on the first 4,096 held-out pile tokens of that run's calibration folder: rule R0 (a)+(b), plus the two "sum of writes" identities
at the R1.3 limits. No recorded 4B numbers exist for the reader, so nothing stricter is asserted.

**R1.5, clarification (2026-11-17, before the capture test was first run).** As in the original measurement, both captures run in one
process on the same loaded model object: the legacy reference process loads the new `reader.py` by file path (the file is self-contained)
and calls its `capture` on the model that `measure.models.load_model` returned. Bit-identity across two separately loaded models is not
claimed: the GatedDeltaNet kernels are tuned per process.

**R1.7 Example run (2026-11-17, before first run).** The example (model loaded the documented way with `AutoModelForCausalLM`, three built-in
texts) must reproduce the table of the snapshot's `notes/mlp_track/release.md` for layers 19 and 9: token counts exact; each FVU within
0.002 of the printed value; mean active atoms within 0.2. The table is printed to three decimals (one decimal for the atoms), so 0.0005
(0.05) is print rounding; the rest is allowance for the bf16 forward on another machine, whose size has not been measured [prediction: it
is below 0.001].

## R5. Spliced-in loss (written 2026-11-17, before the new splice code was first run)

Data and arms as the original (`measure/mlp_track/splice_loss.py`): 100 fineweb-edu documents x 256 tokens from row 131,500 on, layers 9 and 19,
bf16 model, batch of 2 documents. Quantities: `ce_clean`, and per arm `dce`, `kl`, `fvu_batchwise`.

**R5.1 Noise floor.** The legacy code is run twice on this machine in two separate processes (A and B). |A - B| per quantity is the
process-to-process variation (kernel tuning of the bf16 forward); it is recorded, whatever it is.

**R5.2 New vs legacy.** For every quantity q: |new_q - A_q| <= max(3 x |A_q - B_q|, 1e-6). `n_positions` exact. The identity-splice KL of the
first batch <= 1e-4 (the original's own self-check).

**R5.3 New vs the recorded numbers** (measured on processor, 2026-11-16; `results/mlp_track/splice_loss/splice.json` in the snapshot): `ce_clean`
and every arm's `dce` and `kl` within 0.002 nat. The public table prints three decimals; the allowance is for another machine's bf16 forward,
whose size has not been measured [prediction: below 0.001].

## R2. Token stream (written 2026-11-17, before the new stream code was first run)

CPU only, no model. The legacy `train.molt_stream_data.MixStream` is set up exactly as `train.molt_stream.main` does for the 0.8B runs
(`extend=True`, `check_view_prefix()`, `excl = prefix_set(reference_seqs(...))`) and asked for its first 200 units (52.4M tokens: more than
the longest 0.8B run, 50M, so the stream passes the end of the original cache at unit 114 and the on-policy pools' reserved rows).

Everything is compared **exactly**: per unit the shard index, the source parts in order, every sequence token for token (sha256 of its
int64 bytes and its length), the `info` record, and the stream state before the unit; plus the set of excluded 64-token prefixes and the
calibration / held-out reference sequences. One more exact check: a stream restored from the state before unit 120 yields the same units
120-124 (the trainer's whitening pass relies on snapshot / restore).
