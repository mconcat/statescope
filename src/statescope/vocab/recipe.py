"""Every hyperparameter of the state dictionary in one place. The defaults are the released Qwen3.5-0.8B dictionaries (V0, seed 0).

Shapes used in comments across this package: n tokens, d model width, G groups, b directions per group, k active groups per token,
S sites (2).
"""
from __future__ import annotations

from dataclasses import dataclass

SITE_KINDS = ("resid_mid", "resid_post")     # the MLP sub-block's input, and the layer's output


def site_names(layer: int) -> list[str]:
    """the two sites of a layer, in the order the dictionary stores their statistics: ["resid_mid.L", "resid_post.L"]"""
    return [f"{kind}.{layer}" for kind in SITE_KINDS]


@dataclass(frozen=True)
class VocabRecipe:
    # ---- size
    groups: int = 16384                  # G
    group_size: int = 4                  # b
    active_groups: int = 24              # k: per token, the k groups with the largest written norm are kept
    seed: int = 0                        # seed of the initial frames (a CPU generator of its own)

    # ---- optimizer
    lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    lr_end_factor: float = 0.1           # the learning rate is constant, then falls linearly to lr * lr_end_factor ...
    lr_decay_fraction: float = 0.2       # ... over this last fraction of the steps
    grad_clip: float = 1.0               # maximum gradient norm over all parameters
    encode_in_bf16: bool = True          # the dense encode (tokens x G*b matmul) runs under bf16 autocast; everything else is fp32

    # ---- balancing of encoder and decoder (keeps E_g^T D_g, makes E_g E_g^T = D_g D_g^T)
    balance_every: int = 10              # steps between two balancing passes

    # ---- dead groups and their auxiliary loss
    dead_window_tokens: int = 1_000_000  # a group not selected at either site for more than this many tokens is dead
    aux_width_divisor: int = 2           # the auxiliary loss decodes from up to (d // aux_width_divisor) // b dead groups per token
    aux_coefficient: float = 1.0

    # ---- running statistics kept next to the model
    threshold_decay: float = 0.99        # EMA of the batch's smallest selected norm (stored as `thr`; not used by per-token top-k)
    usage_ema_steps: int = 500           # time constant of the per-group selection-rate EMA, in steps
    always_on_rate: float = 0.5          # a group whose (bias-corrected) selection rate exceeds this at some site counts as always on

    # ---- calibration and held-out evaluation
    calib_batch_tokens: int = 32768      # tokens per forward pass (after padding) when the site statistics are computed
    heldout_tokens: int = 4096           # tokens per held-out split used for the FVU during and after training
    heldout_every_logs: int = 4          # the held-out FVU is computed at every this-many-th log step (and at the last one)

    # ---- chunk sizes. They bound memory; the values are the original ones, kept so that every reduction sees the same operands.
    norm_chunk: int = 1024               # tokens per chunk when the written norms are computed
    balance_chunk: int = 2048            # groups per batched eigendecomposition
    basis_chunk: int = 2048              # groups per batched QR
    heldout_chunk: int = 1024            # tokens per reconstruction call in the held-out evaluation

    def aux_groups(self, d: int) -> int:
        """how many dead groups per token the auxiliary loss decodes from"""
        return (d // self.aux_width_divisor) // self.group_size

    def lr_factor(self, step: int, schedule_steps: int) -> float:
        """multiplier of `lr` at `step` of a schedule of `schedule_steps` steps"""
        decay = int(round(self.lr_decay_fraction * schedule_steps))
        start = schedule_steps - decay
        if step < start or decay <= 0:
            return 1.0
        return 1.0 + (self.lr_end_factor - 1.0) * min((step - start) / max(decay, 1), 1.0)
