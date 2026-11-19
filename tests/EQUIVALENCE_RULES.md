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
for the state dictionary). Pass/fail is rule R0 (a)+(b). In addition the numbers the original measured on another machine (also an RTX 5090) are asserted,
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

**R5.3 New vs the recorded numbers** (measured on another machine, 2026-11-16; `results/mlp_track/splice_loss/splice.json` in the snapshot): `ce_clean`
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

## R3. Preparation (written 2026-11-17, before the new preparation code was first run)

Layer 19 of Qwen3.5-0.8B. The references are the files the released dictionaries were trained from, already on disk in the original
results: the calibration folder (`calib_stats.safetensors`, `meta.json`, four `heldout_*` folders) and the fixed mean-gate map
(`fixedmap/L19.safetensors`). Their hashes are recorded in the release run's configuration (`calib.sha256`, `fixmap_md5`).

**R3.1** The new calibration writes the same tensors: every tensor of `calib_stats.safetensors` and of every held-out shard equal bit for bit
(same keys, dtypes, shapes), and the statistics' sha256 (over mu_x, sd_x, mu_y, sd_y) equal to the recorded one.
**R3.2** The new fixed map writes the same tensors bit for bit, and the file's md5 equals the recorded `fixmap_md5` if the container bytes
also agree (tensor equality is the rule; the md5 is reported).
**R3.3 If a tensor is not bit-identical**, the legacy preparation is rerun on this machine before anything else is concluded:
legacy-rerun == disk means the new code is wrong (fix it); legacy-rerun != disk means the files depend on the machine that made them,
and then the rule is new == legacy-rerun bit for bit, with the size of the machine difference reported.

## R4. Short training (written 2026-11-17, before any two training runs were compared)

Layer 19 of Qwen3.5-0.8B, seed 0, 300 steps of 4,096 tokens, the release launcher's arguments (`tests/train_args.py`), one GPU, one process
at a time. Two configurations per dictionary:

- **base**: the release arguments with the short token budget. A traced legacy run showed that no atom is dead within 300 steps
  (`n_aux_candidates` 0 at every logged step), so this configuration never runs the dead-atom auxiliary loss.
- **dead**: the same with `--dead-tokens 40960` (10 steps instead of 1M tokens), so that the auxiliary loss runs. The reference is valid
  only if the legacy run logs `n_aux_selected > 0` at some step; otherwise the configuration is changed (before any new-code run) until it does.

Compared: every tensor of the exported `weights.safetensors`; `whitener.safetensors`; `freq_train.safetensors` / `usage.safetensors`;
`input.json`; `cfg.json`; and per logged step the trainer's numbers (loss, mse, auxiliary loss, counts of dead / selected atoms or groups,
active counts, quick held-out FVU). Wall-clock, memory, host, path and code-hash fields are provenance and are not compared; the list of
such `cfg.json` keys is fixed in the test before the first comparison and every other key must be equal.

**R4.1 Noise floor.** The legacy code is run twice (A, B) per configuration. 
**R4.2 If A and B are bit-identical** in everything compared: the new code must be bit-identical to A in everything compared. Stage 1 then
changes structure, names and configuration only, never arithmetic or the order of random draws.
**R4.3 If A and B differ**: per tensor, max |new - A| <= 3 x max |A - B| (and the same per logged number), which by itself is a weak test
after 300 steps of amplification. It is therefore paired with checks that do not amplify and must hold exactly or at rounding level:
(i) `whitener.safetensors` and the initial model (state before the first step) bit-identical; (ii) the first logged step's numbers
(step 0, before any update) within 1e-6 relative; (iii) one optimizer step from an identical saved state (a legacy checkpoint) within
1e-6 relative per tensor. The cause of the A/B difference is looked for and written down.
**R4.4 Final evaluation.** The numbers the trainer writes at the end (held-out FVU per split and pooled, fold error, dead counts) follow R4.2 / R4.3.

**R4.0 The shared loop, without any dictionary (2026-11-17, before first run).** A legacy run records, for every one of the 300 steps, the
sha256 of the batch it is about to train on (standardized x, standardized y, the first-position flag and the token ids, as they reach
`LayerRecipe.pre`); the recording only reads the tensors. The new loop must produce the same 300 x 4 hashes **exactly**, and the same
per-unit records as the legacy `mix_*.jsonl` (token counts per source, rows used, and the two float statistics of the standardized head)
**exactly**. Both runs use the same GPU.

**R4.5 The state dictionary's dead-group auxiliary loss (2026-11-17, before any vocabulary comparison).** A traced legacy run showed that
no group is dead within 300 steps (`n_dead` 0, `L_aux` 0.0 at every logged step), and the legacy dead window (1,000,000 tokens) is a
literal in the code, not an argument, so no legacy configuration reaches the auxiliary loss in a short run. The base 300-step comparison
therefore does not test it. It is tested by state injection instead, without changing any legacy code: a legacy process builds the
dictionary exactly as the trainer does (same seed, same calibration), sets the `since` counter of a fixed set of groups (every 97th group)
above the window, runs ONE training step on a fixed batch, and saves the losses and every parameter after the step; the new code does the
same from the same state. Rule: the legacy step must report `n_dead > 0` and `L_aux != 0` (otherwise the test is void); legacy run twice
gives the noise floor; if the two legacy results are bit-identical the new result must be bit-identical, else R4.3's 3x rule applies per
tensor with the difference reported.

## R6. Gate-threshold check of an export (written 2026-11-17, before the new check was first run)

The original `measure/mlp_track/fold_check.py` recomputes an exported MLP dictionary's gate in fp32 and fp64 on the first 4,096 held-out
pile tokens and looks for the (token, atom) pair whose single flip explains the trainer's recorded fold error. Its results for the release
are in the snapshot (`results/mlp_track/fold_check/fold_check.json`). The new check (`statescope.foldcheck`) must reproduce, for layers
15, 22, 23 (recorded fold error above 1e-4) and 19 (below):
`gate_decisions_differing_fp32_vs_fp64` exactly; `output_rel_err_from_fp32_vs_fp64_gate_alone` within 1e-6 relative; and for layers
15, 22, 23 the best-matching near-threshold pair: same (token, atom), `ratio_to_recorded` within 1e-6.
The original estimated the rounding step (99 % quantile of |h32 - h64|) from a random sample whose draws depend on the order of layers in one
invocation; the new check samples with its own seeded generator, so that quantile is asserted only within 10 % of the recorded value.

## R7. Manifest (written 2026-11-17, before the new manifest builder was first run)

The manifest regenerated by `statescope.release` from the release files and the original result files must equal the manifest that was
published next to the checkpoints (`release/qwen35-08b-layer-dicts/manifest.json` in the snapshot) **exactly**, key for key and digit for digit,
except the one free-text header field `splice`, which named a file of the original repository and now names the module of this one.

**R4, change of the `dead` configuration (2026-11-17, made from legacy runs only, before any new-code run of this variant under the test).**
The legacy run with `--dead-tokens 40960` did not reach the auxiliary loss either (`n_aux_candidates` 0 at steps 0, 100, 200; about 41,300
of the 41,484 atoms fire in every batch, so none stays silent for ten batches). As the rule above prescribes, the configuration was changed:
`--dead-tokens 4096` (an atom that was silent in the previous batch counts as dead). The legacy run with it logs
`n_aux_candidates = n_aux_selected` = 205 at step 100 and 329 at step 200, auxiliary loss 0.0315 and 0.0317, and its loss differs from the
base run from step 100 on, so the auxiliary gradient moves the weights. Everything else in R4 is unchanged.

**R4.6 Amendment for the state dictionary's held-out FVU numbers (2026-11-17, written AFTER the first vocabulary comparison; the reason is
a measured property of the legacy code, and the old rule text above is left as it was).**

What was found: the two legacy vocabulary runs are bit-identical in every exported tensor, every usage counter and every per-step training
number, but not in the six held-out FVU numbers (step 0, step 299, final; two sites), which differ by 1.9e-9 to 7.7e-9. Four legacy runs
with identical weights (A, B and two earlier runs) give four different sets of digits, and one legacy process reports two different values
for the same weights. Cause, measured: the legacy evaluation sums each token's 24 group writes with `index_add_` on repeated rows on the GPU,
whose order of additions is not fixed; five repeats on identical inputs differ by up to 3.8e-6 in the reconstruction (values up to 13.8),
while the same sum in a fixed order is bit-identical. Training does not use that operation.

Why the old rule cannot be applied to these numbers: R4.3 takes |A - B| of one pair as each number's noise floor. For a number that jumps
between a few discrete values about 1.9e-9 apart, one pair gives 0 for some numbers by chance (it did for step 299, resid_post), and
then the rule demands bit-identity with a value the legacy code itself does not reproduce.

Replacement, for the held-out FVU numbers of the state dictionary only: |new - A| <= 1e-7 absolute, and likewise |A - B| <= 1e-7 (so a
legacy pair outside the bound would fail the test too). 1e-7 is about 13 times the largest legacy-to-legacy difference seen (7.7e-9) and
is 1.3e-7 of the smallest value concerned (0.39); it was chosen from the legacy spread, and the new code's differences (at most 1.2e-8)
were already known when it was written. These numbers are excluded when deciding whether the two legacy runs are bit-identical; for
everything else R4.2 applies unchanged (bit-identity). The three exact checks R4.3 asks for exist: the initial state (init hash, parameter
hashes, calibration record) and one optimizer step from an identical state are bit-identical in the R4.5 test, and the step-0 training
numbers are bit-identical in this one.

The new evaluation keeps the same operation as the legacy one (so it has the same run-to-run variation). Replacing it by a fixed-order sum
would make it repeatable but would change the reported digits; that is noted as a possible later change, not made here.

## R8. Second 0.8B design: readers (written 2026-11-18, before any test of these readers was run)

The second design keeps the per-layer layout (one state dictionary and one MLP dictionary per layer) with other forms:

- state dictionary: the tied form (`cfg.form == "bsf"`: orthonormal frames `D_g`, encoder `E_g = gamma_g D_g`) reading the residual
  divided by its own RMS per token (`cfg.input == "token_rms"`; site statistics stored as `site_mu_rms` / `site_scale_rms`);
- MLP dictionary: the public penalty form (crosslayer-transcoder, branch MOLT, commit 376fabe): JumpReLU gate with a learned threshold per
  transform, free `U` / `V`, no skip; exported as the trainer's state dict (`weights.safetensors`, `config.yaml`).

"Reference" = the frozen copy `legacy_snapshot2/` (made by `devtools/freeze_design2.py`; not committed; hashes in its `SHA256SUMS`). It
holds the training bundle that was shipped to the training machines (public code, its fork, capture, loader) and the state dictionary's
launcher and separately written reader from the original repository. The first snapshot and every rule above are untouched; the first
design's reader path must stay as it is, which the existing reader tests (R1.2: bit-identical to the released standalone reader) show.

Data, all 24 layers: the first 4,096 tokens of each of the four held-out splits of the layer's calibration folder of the new corpus
(`x` = raw resid_mid, `y` = raw MLP output; position 0 is kept, as in the trainers' own held-out numbers), fp32 on one GPU. Decisions
(selected groups, gate values) are compared on all 4 x 4,096 tokens; reconstructed tensors on the first 1,024 tokens of each split (the
reference tensors of 24 layers would otherwise take about 12 GB of the 20 GB allowed for test outputs).

**R8.1 State reader vs the reference reader** (`measure/mlp_track/vocab_heldout_fvu.Reader`). The new reader does not repeat the
reference's operations (it keeps the first design's route: QR of the frames, selection by the norm of what a group writes), so pass/fail
is R0, per split and site: (a) at most 5 tokens whose selected group set differs; (b) over the other tokens with a reference tensor,
max |raw reconstruction difference| / max |reference raw reconstruction| <= 1e-4. The reference's group indices are recomputed in the
reference script from the reference reader's own tensors; the script asserts that rebuilding the reconstruction from them is bit-identical
to what `Reader.recon` returns, so they are the indices it used.

**R8.2 Trainer's held-out number from the reader alone.** For each layer and site, the FVU in the dictionary's own units over the four
splits (SSE / sum over splits of the squared deviations from the split's own mean) computed with the new reader differs from
`vocab_heldout_fvu` in the run's `layer_recipe_done.json` by at most 1e-5 absolute (the limit of R1.4).

**R8.3 MLP reader vs the public forward** (`measure/public_molt/lib.load_public` -> the unmodified public `Molt.forward_with_pre_activations`,
input through the frozen `input_transform`). The new reader sums over transforms in another order (gate applied to the rank-r latents,
as the fork's `LeanMolt`), so pass/fail is R0, per split: (a) at most 5 (token, transform) pairs whose gate is on in one and off in the
other; (b) over the tokens with no differing pair and a reference tensor, max |y1 - y2| / max |y1| <= 1e-4. Prediction, reported but not a
pass/fail condition: 0 differing pairs and gate values equal to within 1e-5 relative, since the gate's input is computed by the same
operations.

*Result of the first full run (2026-11-18, added after it; the rule above is unchanged).* The prediction missed by one pair: among
24 layers x 16,384 tokens x 3,968 transforms, one gate differs (layer 3, pile, token 4027, transform 3816: the new reader's
pre-activation is 0.68840134 against a threshold of 0.68840033, 1.5e-6 above it relatively, and the reference had it off). Gate values
elsewhere differ by up to 1.5e-6 relatively, so the two computations of the gate input are not bit-identical (the reference runs 64
tokens per call, the new reader all tokens of a split), and this pair lies inside that difference. R8.1 likewise: four tokens of 786,432
have another group set, each with its 24th and 25th group norm within 7.3e-7 of each other relatively.

**R8.4 Post-norm coefficients.** The export does not hold the layer's RMSNorm coefficients that the input transform needs (the config
names a file by an absolute path on the training machine). The tests rebuild them from the language model as the frozen capture script
does (`w_eff = (1 + weight.double()).float()`, `eps` of `layers[L].post_attention_layernorm`). Where a copy rebuilt by the original
project exists next to the export (`postnorm_rebuilt/postnorm.safetensors`), the tensor must be bit-identical and `eps` equal.

**R8.5 Sum of writes.** For the first 64 tokens of the pile split: the sum of `transform_write` over the firing transforms plus the
output mean equals `forward` within 1.5e-6 relative to the largest output (the limit of R1.3); the sum of `group_write` over a token's
selected groups plus mean and bias equals `decode` within 1e-6 relative (token 0).

Not covered by R8: the recorded held-out numbers of the MLP dictionaries (`heldout.json`) were measured on a captured shard that was
deleted after training, so they cannot be recomputed here; the FVU of the new reader on the calibration folders' held-out tokens is
reported as a new measurement, not compared with a record. Trainers, token stream, spliced-in loss and manifest of the second design are
not part of R8.

**R8.6 Re-exported format (written 2026-11-18, before this check was run).** The original project changed its export after R8 was
written: the export folder now holds `postnorm.safetensors` and the config names it by a relative path. Two such exports exist (layer 9,
trained on the old corpus; not among the 24 layers): one with post-norm input, one with pre-norm input.
(a) `PublicMolt(<post-norm export>)` with no `post_norm` argument reads the folder's own file; its `w_eff` must be bit-identical to the
one rebuilt from the language model and `eps` equal, and gate values and outputs on the first 1,024 held-out pile tokens of the new
corpus's layer-9 calibration folder must be bit-identical to those of the same reader given the rebuilt file explicitly.
(b) `PublicMolt(<pre-norm export>)` must raise `ValueError` (this reader handles post-norm input only), not read the dictionary with the
wrong input.
This checks the format only; the layer-9 dictionary of the old corpus is not compared with a reference forward.

**R8.7 The frozen release folder (written 2026-11-19, before these checks were run).** The original project froze the 0.8B set on
2026-11-18: the MLP dictionaries were copied into one folder per layer (`weights.safetensors`, `postnorm.safetensors`, `config.yaml`
without training-time paths, `MANIFEST.json`), layers 0-22 from the exports R8 read and layer 23 from another run (lower penalty); on
2026-11-19 it rewrote the safetensors headers' metadata (which held absolute paths) and left the tensor bytes alone.
(a) Every file's sha256 equals the manifest's.
(b) Layers 0-22: every tensor that `PublicMolt(<release folder>)` holds is bit-identical to the one `PublicMolt(<export R8 read>,
post_norm=<rebuilt file>)` holds, and `eps` is equal. With (b), R8.3-R8.5 carry over to these layers without a new reference run.
(c) Layer 23: R8.3, R8.4 and R8.5 as written, with the release folder as the export and its own `postnorm.safetensors`.
(d) No safetensors header and no config of the release folder, and no `cfg.json` or header of the 24 state dictionaries, contains
an absolute path under a home, mount, root or workspace folder, a host name of the lab or an IPv4 address (the patterns are in
`tests/test_release2.py`; this sentence named the path prefixes literally when it was written, which this repository's own rule
against committed machine paths rejects, so it was reworded on the same day without changing what is checked).
The state dictionaries of the frozen set are the 24 folders R8.1-R8.2 read; nothing new is checked for them here.

**R8.8 Reconstruction error on all held-out tokens, and what the configs say (written 2026-11-19, before the measurement of all
layers was run).** The trainer's held-out number of the state dictionary (R8.2) uses the first 4,096 tokens of each split. At early
layers that subset is easier than the whole split: with this reader, layer 0 `resid_mid` gives 0.0533 on the subset and 0.0693 on all
tokens (own units), layer 9 gives 0.2452 and 0.2438; position 0 changes neither by more than 0.001. (These two layers were looked at
before this rule was written, to find the cause of the mismatch with the original project's table.) The model card therefore
reports numbers on all held-out tokens, measured by `devtools/measure_release2.py` with this reader.
(a) State dictionary, raw units, position 0 dropped, all tokens of the four held-out splits: for every layer and both sites the new
number is within 1e-4 of the original project's `table.json` (`raw_mid`, `raw_post`), and the same with position 0 included
(`raw_with_pos0_mid`, `raw_with_pos0_post`). The original's reader is the one of R8.1.
(b) MLP dictionary on the same tokens: reported, with and without position 0; there is no record to compare with (the original's
held-out shard is gone), so nothing is asserted except that the number is finite and below 1.
(c) Configs: all 24 state `cfg.json` hold the same training facts (tokens 30,003,200, steps 7,325, 16,384 groups of 4, k 24, form
`bsf`, input `token_rms`, eps 1e-6, seed 0); all 24 MLP `config.yaml` are equal apart from the layer index, the logger name and the
penalty, which is 7.5e-05 at layers 0-22 and 2.5e-05 at layer 23.

## R9. Line-count demonstration (written 2026-11-19, before the reference script or the port was run)

`demo/line_count.py` repeats, on the frozen 0.8B set and with this package's readers, part of a chain the original project traced
(`notes/demo_line_count_L8_L9.md` in snapshot 3): L8 MLP transform 189 -> state group 11586 of `resid_post.8` -> the model's own L9
mixer (GDN head 10) -> L9 MLP transforms 410 and 1749. Only the links that do not pass through the original's GDN dictionary (which is
not released) are repeated. The reference is `tests/legacy_scripts/demo_reference.py`, run inside snapshot 3 on the same device: the
original's own batch, patch, pairing and share functions, with its GDN dictionary left unloaded.

Definitions, all the original's: items = `build_items(seed 0)` of the cells count:60, count:100, width:100, 64 pairs each, 256 tokens;
margin m = logsumexp(newline tokens) - logsumexp(space tokens) at the last token; a patch replaces a tensor of the clean run by a value
built from the corrupt run moved onto the clean positions (`paired`), the patched tensor is rounded to bf16 as the original does;
share = mean(m_c - m_p) / mean(m_c - m_x); halves = even / odd item index; batches of 32 with the short batch padded by its first item.

**R9.1 Items.** The item file in the repository equals the reference's items exactly (token ids of both texts, pairing, all 192 pairs).

**R9.2 Shares.** For every arm, cell and half (all, even, odd): |share(new) - share(reference)| <= 0.005. This is the tolerance the
original registration set for reproducing a recorded share (`REGISTRATION.md`, gates G1 and G2: "±.005"). Arms:
unpatched margins (m_c, m_x enter every share); the L9 mixer's whole output swapped; head 10's output swapped (the original's exact
fp32 recurrence, gated norm with the clean gate, head 10's output columns); through the model's own L9 mixer, with the residual's
direct path held at clean: all of `resid_post.8` swapped, the true MLP 8 output swapped, all L8 transforms' writes swapped, transform
189's write swapped, state group 11586's contribution swapped; at the L9 MLP output: the true output swapped, all transforms' writes
swapped, transform 410's write, transform 1749's write. Arms that the original project adds for the group-4885 link on the model's own
mixer (its registration `true_mixer_REGISTRATION.md`, measured after this rule was written) come under this same rule when ported.

**R9.3 Recorded numbers.** Where the note or its result files record a share of one of these arms (three decimals), the new share is
within 0.005 of the record as well. Records measured on another day's process are not expected to be bit-reproducible.

**R9.4 Gates and descriptive numbers.** Gate values of transforms 189 (L8), 410 and 1749 (L9) on the clean items: rule R0 per 4,096
tokens. The firing rates (line-first tokens / newline tokens / other tokens, position 0 and tokens before the first newline dropped)
and the fraction of transform 189's swapped-write energy inside group 11586's 4-dimensional frame: |new - reference| <= 0.005, and
within 0.005 of the record where one exists.

**R9.5 Reported, not pass/fail.** The largest per-item margin difference between the new script and the reference, per arm; the same
between two runs of the reference (run-to-run repeatability of the original code); peak GPU memory. Prediction: unpatched margins
are bit-identical, every share difference is below 0.001.

*Result of the first run (2026-11-19, added after it; the rules above are unchanged).* R9.1-R9.4 passed on the first run. Both
predictions of R9.5 missed. (1) Unpatched margins are not bit-identical: they differ by up to 2e-5 (the reference sums the logits in
vocabulary chunks). (2) The largest share difference is 0.0030, not below 0.001. The arms that swap a tensor of the model itself agree
to 2e-5 per item; the arms that swap something a dictionary computed (a transform's write, a group's contribution) differ by up to
0.097 per item. Cause, checked directly: the two implementations of a dictionary differ at about 1e-6 relatively, and the patched
tensor is then rounded to bf16; multiplying such a change by 1 + 1e-6 x noise before the rounding flips about 1,600 of 8.4 million bf16
elements of one batch and moves single items' margins by 0.04-0.09 (three draws each at 1e-7, 1e-6, 1e-5; the same input twice gives
0). Two runs of the reference are bit-identical in every margin. So single-item margins of dictionary arms are reproducible only to
about 0.1, and shares over 32 or 64 items to about 0.003.

*Link arms (2026-11-19, added after their first run).* The original project registered and measured the group-4885 link on the
model's own mixer the same day (`true_mixer_REGISTRATION.md`, `true_mixer_link.json` in its results; code
`measure/demo_line_count_tm/tm_link.py`, written after snapshot 3 was frozen). Seventeen arms were ported under R9.2 and R9.3 as
announced: the whole mixer swap and head 10's swap, each whole / inside group 4885's frame / the rest, each with everything downstream
and through the writes of transforms 410 and 1749 only, and the whole mixer swap inside five random frames. The reference script
repeats the original's definitions with snapshot 3's own helpers; the record is the original's result file, read from the original
results folder, not from a snapshot. First run: all 153 share comparisons within 0.0030 of the reference and within 0.0022 of the
record.
After the original project reported its verdict (the registered rule holds for the route through its six L9 transforms and holds
only in part for the route through 410 and 1749 alone), the six-transform route was added for the same arms: 28 link arms in all.

Not covered by R9: everything in the note that passes through the GDN dictionary's registers; the selection of the transforms and
groups (the demonstration takes the original's choices as given and says so); other cells, seeds or models.

## Note on module paths (2026-11-19, written before the modules were moved; no rule and no tolerance is changed)

The earlier design's modules are gathered under `statescope.earlier_design`: the folders `vocab/`, `molt/`, `prep/`, `data/`, `train/`
and the modules `calibration`, `manifest`, `release`, `foldcheck` keep their names and contents and get the longer path. Where a rule
above names a module, read it by its new path: `statescope.foldcheck` (R6) is `statescope.earlier_design.foldcheck`, and
`statescope.release` (R7) is `statescope.earlier_design.release`. `statescope/splice.py` keeps its path, because the published
manifest's `splice` field (R7) names that file. The lines above are left as they were written.

## R10. Qwen3.5-4B set (written 2026-11-20, before any test or measurement of these files was run)

The original project trained the second design's two dictionaries for every layer of `Qwen/Qwen3.5-4B` (32 layers, width 2,560): the
state dictionary in the same form and size (tied, per-token RMS input, 16,384 groups of 4, k 24) and the MLP dictionary in the same
public form with 9,920 transforms per layer (320 x the five rank tiers) and penalty 3e-05. Before this rule was written, only the
configs and the original project's record files were read (to learn the layout and the training facts of R10.6 c); no reader of this
repository was run on a 4B file of this set. (R1.6 ran the first design's reader on older 4B test exports of one layer.)

R10 is R8 with other folders, and every limit is the one R8 states. "Reference" is the same frozen copy (`legacy_snapshot2/`) and the
same reference script. Data, all 32 layers: the first 4,096 tokens of each of the four held-out splits of the layer's 4B calibration
folder, position 0 kept; decisions on all 4 x 4,096 tokens, reconstructed tensors on the first 1,024 tokens of each split.

**R10.1 State reader vs the reference reader.** R8.1 (a) and (b) as written, per layer, split and site.

**R10.2 Trainer's held-out number from the reader alone.** R8.2 as written (1e-5 absolute against `vocab_heldout_fvu` of the run's
`layer_recipe_done.json`).

**R10.3 MLP reader vs the public forward.** R8.3 (a) and (b) as written, per layer and split. Prediction, reported and not a pass/fail
condition: at most a few differing gate decisions over all layers (0.8B had one among 24 layers; this set has 2.5 times the transforms
and more layers).

**R10.4 Post-norm coefficients.** For every layer, `w_eff` of the `postnorm.safetensors` in the layer's export folder is bit-identical
to the one this repository's `write_post_norm` builds from the language model, and `eps` is equal.

**R10.5 Sum of writes.** R8.5 as written (1.5e-6 and 1e-6 relative).

**R10.6 The folder that is published.** The original project is asked for a copy of the MLP dictionaries without training-time paths
(config lines and safetensors header metadata rewritten, tensor bytes untouched) with a manifest, as for 0.8B (R8.7). Layer 16's MLP
dictionary comes from a separate run of the same recipe.
(a) Every file's sha256 equals the manifest's.
(b) All 32 layers: every tensor that `PublicMolt(<copy>)` holds is bit-identical to the one `PublicMolt(<export R10.3 read>)` holds,
and `eps` is equal. With (b), R10.3-R10.5 carry over to the copy without a new reference run.
(c) Configs: all 32 state `cfg.json` hold the same training facts (tokens 30,003,200, steps 7,325, 16,384 groups of 4, k 24, form
`bsf`, input `token_rms`, eps 1e-6, seed 0, width 2,560); all 32 MLP `config.yaml` of the copy are equal apart from the layer index
and the logger name, with penalty 3e-05, 58,600 steps, batch 500, learning rate 4e-05 and 320 transforms per rank tier.
(d) R8.7 (d) as written, for every header and config of the 32 + 32 folders.
*Correction of (c), 2026-11-20, before any run:* after the rule was committed the original project reported that the two
standardizer tables of the MLP dictionaries have 24 rows at layers 0-23 and 32 rows at layers 24-31 (`n_layers` of
`input_standardizer` and `output_standardizer` in the config; only the layer's own row is read). So in (c) the MLP configs are equal
apart from the layer index, the logger name and these two `n_layers` values, which are 24 at layers 0-23 and 32 at layers 24-31.

**R10.7 Reconstruction error on all held-out tokens.** Measured with this reader on all tokens of the four held-out splits, raw units.
(a) State dictionary: for every layer and both sites, within 1e-4 of `raw` in the original project's
`checks/L<L>.B1.fvu_all_nopos0.json` (position 0 dropped) and of `raw` in `checks/L<L>.B1.fvu_all.json` (position 0 included), and
the token counts per split equal the record's.
(b) MLP dictionary on the same tokens: reported, with and without position 0. Nothing is asserted except that the number is finite
and below 1: the original project's recorded number (`heldout.json`) is on a held-out shard of the training capture, other tokens
than the calibration folder's.

Not covered by R10: the capture path on the 4B model (R1.5 compared `capture` with the original's capture on 0.8B only), spliced-in
loss, trainers, and any demonstration (the line-count demonstration is on 0.8B).
