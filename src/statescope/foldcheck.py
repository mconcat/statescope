"""How sensitive is an exported MLP dictionary's hard gate to float rounding?

The gate of atom t is on when h_t / score_div_t > thr_t (and h_t > 0). A (token, atom) pair whose h lies within rounding of the threshold
can come out differently in fp32 and fp64, or between two orders of the same operations; one such flip changes that token's output by
the whole write of the atom. The trainer records a "fold error" (largest difference between the training-form model and the exported form
on 4,096 held-out tokens, relative to the largest output); when it exceeds 1e-4 this check finds whether a single flipped pair has exactly
that size.
"""
from __future__ import annotations

import torch

from statescope.reader import MoltOps

NEAR_THRESHOLD_H = 2e-3     # pairs whose h is within this of its threshold are examined
ROWS_PER_CHUNK = 256
SAMPLES_PER_CHUNK = 200_000


@torch.no_grad()
def gate_flip_check(molt: MoltOps, x: torch.Tensor, recorded_fold_err: float | None = None, seed: int = 0) -> dict:
    """x (n, d) raw resid_mid. Returns counts of gate decisions that differ between fp32 and fp64, the output change they cause, the
    rounding step of h, and (if the trainer's recorded fold error is given) the near-threshold pair whose single flip matches it best."""
    device = molt.e.device
    z = molt.standardize_input(x)
    a32, y32 = molt.forward(x)
    sd_y, mu_y = molt.sd_y.double(), molt.mu_y.double()
    y_raw = y32.double()                                                    # forward() already returns raw units
    y_max = float(y_raw.abs().max())
    e64, b64 = molt.e.double(), molt.b.double()
    score_div64, thr64 = molt.score_div.double(), molt.thr.double()
    thr_h = thr64 * score_div64                                             # the threshold in h units
    h32_full = z @ molt.e.T - molt.b
    a64 = torch.zeros_like(a32)
    n_flip, n_flip_shape = 0, 0
    samples, near = [], []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, z.shape[0], ROWS_PER_CHUNK):
        zc = z[start:start + ROWS_PER_CHUNK]
        h64 = zc.double() @ e64.T - b64
        h32 = h32_full[start:start + ROWS_PER_CHUNK]
        h32_chunk = zc @ molt.e.T - molt.b                                  # the same product with another matrix shape
        on64 = (h64 / score_div64 > thr64) & (h64 > 0)
        on32 = (h32 / molt.score_div > molt.thr) & (h32 > 0)
        on32_chunk = (h32_chunk / molt.score_div > molt.thr) & (h32_chunk > 0)
        n_flip += int((on64 != on32).sum())
        n_flip_shape += int((on32_chunk != on32).sum())
        a64[start:start + ROWS_PER_CHUNK] = (torch.relu(h64) * on64).float()
        dh = (h32.double() - h64).abs().reshape(-1)
        samples.append(dh[torch.randint(0, dh.numel(), (SAMPLES_PER_CHUNK,), generator=generator).to(device)])
        margin = (h64 - thr_h).abs()
        idx = torch.nonzero((margin < NEAR_THRESHOLD_H) & (h64 > 0))
        if len(idx):
            near.append(torch.cat([(idx[:, :1] + start).double(), idx[:, 1:].double(), margin[idx[:, 0], idx[:, 1]][:, None], h64[idx[:, 0], idx[:, 1]][:, None]], 1))
    sampled = torch.cat(samples)[:2_000_000]
    q99 = float(torch.quantile(sampled, 0.99))
    # the output change caused by the fp32-vs-fp64 gate decisions alone (same weights, same z)
    y64 = torch.cat([molt._decode(zc, ac) for zc, ac in zip(z.split(molt.chunk), a64.split(molt.chunk))]).double() * sd_y + mu_y
    result: dict = {
        "n_tokens": int(z.shape[0]), "n_atoms": molt.n_atoms, "ymax_raw": y_max, "l0": float((a32 > 0).float().sum(1).mean()),
        "rounding_step_h_q99": q99, "thr_h_median": float(thr_h.median()),
        "gate_decisions_differing_fp32_vs_fp64": n_flip, "gate_decisions_differing_between_two_fp32_gemm_shapes": n_flip_shape,
        "output_rel_err_from_fp32_vs_fp64_gate_alone": float((y64 - y_raw).abs().max() / y_max),
    }
    pairs = torch.cat(near) if near else torch.zeros(0, 4, dtype=torch.double, device=device)   # columns: token, atom, |h - threshold|, h
    result["pairs_within_100x_rounding_step"] = int((pairs[:, 2] < 100 * q99).sum())
    if recorded_fold_err is None or len(pairs) == 0:
        return result
    # what flipping each near pair alone would change: |a(h) * (the atom's map applied to z) * sd_y|_max / ymax
    tokens, atoms = pairs[:, 0].long(), pairs[:, 1].long()
    single_flip = torch.zeros(len(pairs), dtype=torch.double, device=device)
    for r in molt.ranks:
        mask = molt.rank[atoms] == r
        if not mask.any():
            continue
        p = molt.pos[atoms[mask]]
        latent = torch.einsum("kd,krd->kr", z[tokens[mask]].double(), molt.Vs[r].double()[p]) * pairs[mask, 3][:, None]
        contribution = torch.einsum("kr,krd->kd", latent, molt.U[r].double()[p]) * sd_y
        single_flip[mask] = contribution.abs().amax(1) / y_max
    ratio = single_flip / recorded_fold_err
    candidates = torch.nonzero(pairs[:, 2] < 100 * q99).reshape(-1)
    result["recorded_fold_max_rel_err"] = recorded_fold_err
    if len(candidates):
        j = candidates[(ratio[candidates] - 1).abs().argmin()]
        result["best_match_within_100x_q99"] = {"token": int(tokens[j]), "atom": int(atoms[j]), "rank": int(molt.rank[atoms[j]]),
                                                "dist_in_units_of_q99_dh": float(pairs[j, 2] / q99), "single_flip_rel": float(single_flip[j]),
                                                "ratio_to_recorded": float(ratio[j])}
    return result


def fold_error_verdict(fold_err: float, check: dict | None) -> str | None:
    """None if the recorded fold error is within 1e-4; otherwise whether one flipped pair explains it (ratio within 1 %, at most 5 flips)."""
    if fold_err <= 1e-4:
        return None
    if check is None:
        return "fold: not checked yet"
    best = check.get("best_match_within_100x_q99")
    explained = best is not None and abs(best["ratio_to_recorded"] - 1) < 0.01 and check["gate_decisions_differing_fp32_vs_fp64"] <= 5
    return "fold: one flipped pair" if explained else "FOLD NOT EXPLAINED"
