"""Comparison of a new short training run with two legacy runs A and B (rules R4.1 - R4.4 of EQUIVALENCE_RULES.md).

A run is described by `RunFiles`: the exported tensors, the cfg / input JSON, the per-step numbers and the final numbers. The legacy
layout and the new layout differ; each test builds the three RunFiles and calls `compare`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class RunFiles:
    tensors: dict[str, Path]                  # label -> safetensors file (every tensor in it is compared)
    jsons: dict[str, dict]                    # label -> JSON object, compared for equality after dropping provenance keys
    steps: list[dict]                         # per logged step: {"step": n, <number name>: value, ...}
    final: dict[str, float]                   # numbers written at the end
    provenance: dict[str, set[str]] = field(default_factory=dict)    # label -> keys of that JSON that are not compared
    # Numbers that the LEGACY code itself does not reproduce from run to run (rule R4.6): name fragment -> absolute tolerance.
    # A step or final number whose name contains a fragment is compared with that tolerance and does not count against bit-identity.
    unrepeatable: dict[str, float] = field(default_factory=dict)


def _strip(obj, drop: set[str]):
    if isinstance(obj, dict):
        return {k: _strip(v, drop) for k, v in obj.items() if k not in drop}
    return obj


def _tensor_diffs(path_a: Path, path_b: Path) -> tuple[list[str], dict[str, float]]:
    """(structural problems, {key: max abs difference}) between two safetensors files"""
    a, b = load_file(str(path_a)), load_file(str(path_b))
    problems = [f"key only in one file: {k}" for k in sorted(set(a) ^ set(b))]
    diffs = {}
    for k in sorted(set(a) & set(b)):
        if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape:
            problems.append(f"{k}: {a[k].dtype} {tuple(a[k].shape)} vs {b[k].dtype} {tuple(b[k].shape)}")
        elif torch.equal(a[k], b[k]):
            diffs[k] = 0.0
        else:
            diffs[k] = max(float((a[k].double() - b[k].double()).abs().max()), 5e-324)     # > 0: not bit-identical
    return problems, diffs


def _number_diffs(a: dict, b: dict) -> dict[str, float]:
    return {k: (0.0 if a[k] == b[k] else max(abs(a[k] - b[k]), 5e-324)) for k in a}


def _tolerance(name: str, unrepeatable: dict[str, float]) -> float | None:
    for fragment, tol in unrepeatable.items():
        if fragment in name:
            return tol
    return None


def noise_floor(a: RunFiles, b: RunFiles) -> dict:
    """R4.1: every difference between the two legacy runs. `identical` ignores the numbers listed in a.unrepeatable (R4.6)."""
    floor: dict = {"tensors": {}, "steps": [], "final": _number_diffs(a.final, b.final), "problems": []}
    for label in a.tensors:
        problems, diffs = _tensor_diffs(a.tensors[label], b.tensors[label])
        floor["problems"] += [f"legacy A vs B, {label}: {p}" for p in problems]
        floor["tensors"][label] = diffs
    if [s["step"] for s in a.steps] != [s["step"] for s in b.steps]:
        floor["problems"].append("legacy A and B logged different steps")
    floor["steps"] = [_number_diffs(sa, sb) for sa, sb in zip(a.steps, b.steps)]
    for label in a.jsons:
        if _strip(a.jsons[label], a.provenance.get(label, set())) != _strip(b.jsons[label], b.provenance.get(label, set())):
            floor["problems"].append(f"legacy A vs B: {label} differs")
    repeatable_numbers = [v for d in floor["steps"] + [floor["final"]] for k, v in d.items() if _tolerance(k, a.unrepeatable) is None]
    floor["identical"] = (not floor["problems"] and all(v == 0.0 for d in floor["tensors"].values() for v in d.values())
                          and all(v == 0.0 for v in repeatable_numbers))
    return floor


def compare(new: RunFiles, a: RunFiles, b: RunFiles) -> tuple[list[str], dict]:
    """Returns (failures, report). No failures = the new run satisfies R4.2 (if A and B are bit-identical) or R4.3's tolerance part;
    numbers listed in a.unrepeatable follow R4.6 (an absolute tolerance, applied to |new - A| and to |A - B|)."""
    floor = noise_floor(a, b)
    failures = list(floor["problems"])
    exact = floor["identical"]
    report = {"legacy_runs_bit_identical": exact, "max_new_vs_a": {}, "noise": {}, "unrepeatable_in_legacy": sorted(a.unrepeatable)}

    def check(name: str, diff: float, noise: float) -> None:
        report["max_new_vs_a"][name], report["noise"][name] = diff, noise
        tol = _tolerance(name, a.unrepeatable)
        if tol is not None:
            if diff > tol or noise > tol:
                failures.append(f"{name}: |new - A| = {diff:.3g}, |A - B| = {noise:.3g}, allowed {tol:.3g} (R4.6)")
        elif (diff != 0.0) if exact else (diff > 3 * noise):
            failures.append(f"{name}: |new - A| = {diff:.3g}, |A - B| = {noise:.3g}")

    for label in a.tensors:
        problems, diffs = _tensor_diffs(new.tensors[label], a.tensors[label])
        failures += [f"new vs A, {label}: {p}" for p in problems]
        for key, diff in diffs.items():
            check(f"{label}/{key}", diff, floor["tensors"][label].get(key, 0.0))
    if [s["step"] for s in new.steps] != [s["step"] for s in a.steps]:
        failures.append(f"logged steps differ: new {[s['step'] for s in new.steps]} vs legacy {[s['step'] for s in a.steps]}")
    else:
        for sn, sa, noise in zip(new.steps, a.steps, floor["steps"]):
            for key, diff in _number_diffs({k: sn[k] for k in sa}, sa).items():
                if key != "step":
                    check(f"step {sa['step']} {key}", diff, noise[key])
    for key, diff in _number_diffs({k: new.final[k] for k in a.final}, a.final).items():
        check(f"final {key}", diff, floor["final"][key])
    for label in a.jsons:
        drop = a.provenance.get(label, set())
        if _strip(new.jsons[label], drop) != _strip(a.jsons[label], drop):
            failures.append(f"{label} differs from legacy outside the provenance keys {sorted(drop)}")
    if not exact:
        failures.append("legacy A and B are not bit-identical: rule R4.3's exact checks (initial state, first step, one step from a checkpoint) "
                        "have to be added before this comparison can pass")
    return failures, report


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
