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
