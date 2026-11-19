"""Limits of rule R0 of EQUIVALENCE_RULES.md for a comparison that contains a hard decision (a selected group set, a gate)."""
from __future__ import annotations

MAX_DIFFERING = 5                  # decisions that may differ per 4,096 tokens
MAX_REL_DIFF_ELSEWHERE = 1e-4      # largest relative difference of the outputs on the tokens whose decisions agree
