"""Every hyperparameter of the MLP dictionary recipe "M1w", in one place. The defaults are the values of the released dictionaries.

The recipe in words. The MLP of one layer is approximated, in standardized coordinates, by a frozen linear map (the "skip map", computed
beforehand by statescope.earlier_design.prep.fixedmap) plus a sum of gated low-rank maps ("atoms"). Atoms come in rank tiers 1, 2, 4, 8, 16. An atom's
gate reads the whitened input; per training batch a fixed number of (token, atom) entries is switched on in each tier (the largest gate
inputs of the tier over the whole batch), and at inference each tier uses a threshold learned as a running mean of the batch cut-off.
Atoms that have not fired for `dead_tokens` tokens get an auxiliary loss that asks them to explain the current residual.

There are no switches for variants that were never run (no un-whitened gate, no trainable skip map).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoltRecipe:
    name: str = "M1w"                         # written to cfg.json

    # ---- atoms and their ranks
    rank_per_width: int = 128                 # total rank over all atoms = rank_per_width * d  (131,072 at d = 1,024)
    # share of the atoms in each tier above rank 1; rank 1 takes whatever total rank is left, so the total rank is exact
    tier_shares: tuple[tuple[int, float], ...] = ((16, 0.03), (8, 0.10), (4, 0.22), (2, 0.35))
    mean_rank: float = 3.16                   # number of atoms = round(total rank / mean_rank)
    tier_seed: int = 12345                    # seed of the permutation that assigns tiers to atoms
    seed: int = 0                             # initial weights and the shuffle buffer's permutations
    init_scale: float = 1.0 / math.sqrt(3.0)  # an atom of rank r starts with every scale s = init_scale / sqrt(r)  ("tau" in cfg.json)

    # ---- gate: entries switched on per token in each tier while training (60 atoms, rank sum 192 per token)
    k_by_rank: dict[int, int] = field(default_factory=lambda: {1: 18, 2: 21, 4: 13, 8: 6, 16: 2})
    budget: int = 192                         # written to cfg.json only; under the per-tier rule the rank sum follows from k_by_rank
    threshold_decay: float = 0.99             # running mean of each tier's batch cut-off (the inference threshold)
    gate_mean_decay: float = 0.99             # running mean of each atom's firing gate value (the gate is divided by it)

    # ---- whitening of the gate input
    whiten_tokens: int = 1 << 20              # the first tokens of the stream (one shuffle buffer)
    whiten_chunk: int = 4096                  # tokens per covariance update
    whiten_eig_floor: float = 1e-3            # eigenvalues below this fraction of their mean are raised to it

    # ---- optimization
    lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    batch: int = 4096
    micro_batches: int = 4                    # the batch is forwarded and backpropagated in this many equal pieces
    buffer_tokens: int = 1 << 20              # shuffle buffer
    preact_coef: float = 3e-6                 # weight of sum_t relu(-h_t) in the loss
    debias_frac: float = 0.05                 # for the last this-fraction of the steps, scales that are exactly zero stay zero

    # ---- dead atoms
    dead_tokens: int = 1_000_000              # an atom is dead after this many tokens without firing
    aux_k: int = 512                          # dead atoms per token in the auxiliary output (the largest gate inputs)
    aux_coef: float = 1.0 / 32

    # ---- evaluation and export
    quick_eval_tokens: int = 4096             # per held-out split, at every logged step
    eval_chunk: int = 2048                    # tokens per forward of the final evaluation
    fold_check_tokens: int = 4096
    freq_segments: int = 10                   # firing counts are kept per tenth of the run (freq_train.safetensors)

    # ---- fields of the checkpoint format that this recipe does not use (an earlier gate had a per-atom threshold theta). They are
    # written to cfg.json / weights.safetensors unchanged so that existing readers of the format keep working.
    unused_theta0: float = 0.03
    unused_bandwidth: float = 1.0
    unused_ste_to_input: bool = True

    @property
    def ranks(self) -> list[int]:
        return sorted([1] + [r for r, _ in self.tier_shares])

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def k_total(self) -> int:
        """atoms switched on per token while training"""
        return sum(self.k_by_rank.values())

    def total_rank(self, d: int) -> int:
        return self.rank_per_width * d

    def tier_counts(self, d: int) -> dict[int, int]:
        """atoms per rank at model width d  (d = 1,024: 12,450 / 14,517 / 9,125 / 4,148 / 1,244 for ranks 1 / 2 / 4 / 8 / 16)"""
        total = self.total_rank(d)
        n = round(total / self.mean_rank)
        counts = {r: round(share * n) for r, share in self.tier_shares}
        counts[1] = total - sum(r * c for r, c in counts.items())
        return dict(sorted(counts.items()))

    def n_atoms(self, d: int) -> int:
        return sum(self.tier_counts(d).values())

    def init_seed(self, n_atoms: int) -> int:
        """Seed of the generator that draws the initial directions. The formula (and the word "adaptive" in it) is the original
        trainer's: the released weights started from exactly this stream."""
        return (self.seed * 1_000_003 + zlib.crc32(f"adaptive:{n_atoms * 1000 + self.max_rank}".encode())) % (2 ** 63)


@dataclass(frozen=True)
class RunPlan:
    """Step counts of one run of `tokens` training tokens."""
    steps: int                # optimizer steps: ceil(tokens / batch)
    debias_start: int         # the step at which zero scales are frozen

    @staticmethod
    def of(recipe: MoltRecipe, tokens: int) -> "RunPlan":
        whole_steps = tokens // recipe.batch
        return RunPlan(steps=int(math.ceil(tokens / recipe.batch)), debias_start=whole_steps - int(round(recipe.debias_frac * whole_steps)))
