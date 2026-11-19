"""The line-count demonstration (demo/line_count.py) against the original's own harness and its recorded numbers.
The pass/fail rules are R9.1 - R9.5 of tests/EQUIVALENCE_RULES.md (written before the reference or the port was first run)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from conftest import REPO, TEST_OUT, Settings
from legacy_bridge import run_legacy
from reader_checks import MAX_DIFFERING, MAX_REL_DIFF_ELSEWHERE

pytestmark = [pytest.mark.legacy, pytest.mark.gpu, pytest.mark.model]

CELLS = ("count:60", "count:100", "width:100")
HALVES = ("all", "even", "odd")
MAX_SHARE_DIFF = 0.005                 # R9.2, R9.3, R9.4: the original registration's own tolerance for reproducing a share
DEMO = REPO / "demo" / "line_count.py"
ITEMS = REPO / "demo" / "line_count_items.safetensors"
MEASURED = TEST_OUT / "measured" / "demo"
RECORDED_ENERGY_FRACTION = 0.769       # results/demo_line_count/s5.json of the original ("77 %" in the note); the file is not in snapshot 3


def _reference(settings: Settings, run: str) -> Path:
    release = settings.need_release2()
    args = ["--mlp8-dir", str(release / "mlp" / "L8"), "--mlp9-dir", str(release / "mlp" / "L9"),
            "--vocab8-dir", str(release / "state" / "L8"), "--vocab9-dir", str(release / "state" / "L9"), "--run", run]
    return run_legacy("demo_reference.py", args, settings, snapshot=settings.need_snapshot3())


@pytest.fixture(scope="module")
def reference(settings: Settings) -> tuple[dict[str, torch.Tensor], dict]:
    out = _reference(settings, "a")
    return load_file(str(out / "reference.safetensors")), json.loads((out / "info.json").read_text())


@pytest.fixture(scope="module")
def reference_again(settings: Settings) -> dict[str, torch.Tensor]:
    """a second run of the same reference script: how far the original code repeats itself"""
    return load_file(str(_reference(settings, "b") / "reference.safetensors"))


@pytest.fixture(scope="module")
def demo(settings: Settings) -> tuple[dict[str, torch.Tensor], dict]:
    """the public script, run as a user runs it, in its own process; cached by the reader's source, the script and the items"""
    release = settings.need_release2()
    key = hashlib.sha256()
    for path in (REPO / "src" / "statescope" / "reader.py", REPO / "src" / "statescope" / "lm.py", REPO / "src" / "statescope" / "config.py",
                 DEMO, ITEMS, release / "MANIFEST.json"):
        key.update(hashlib.sha256(path.read_bytes()).digest())
    key.update(settings.device.encode())
    out = TEST_OUT / "demo" / f"line_count-{key.hexdigest()[:16]}"
    if not (out / "DONE").is_file():
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(torch.device(settings.device).index or 0), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
        with open(out / "log.txt", "w") as log:
            proc = subprocess.run([sys.executable, str(DEMO), "--release", str(release), "--device", "cuda:0", "--out", str(out)],
                                  cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if proc.returncode != 0:
            tail = "\n".join((out / "log.txt").read_text().splitlines()[-30:])
            pytest.fail(f"demo/line_count.py failed (exit {proc.returncode}); last lines of {out / 'log.txt'}:\n{tail}", pytrace=False)
        (out / "DONE").write_text("done\n")
    return load_file(str(out / "result.safetensors")), json.loads((out / "result.json").read_text())


def test_items(reference) -> None:
    """R9.1: the item file in the repository is the frozen code's build_items output"""
    ref, _ = reference
    items = load_file(str(ITEMS))
    assert sorted(items) == sorted(f"{cell}/{f}" for cell in CELLS for f in ("ids", "cor", "pi"))
    for cell in CELLS:
        for f in ("ids", "cor", "pi"):
            assert items[f"{cell}/{f}"].shape == (64, 256)
            assert torch.equal(items[f"{cell}/{f}"], ref[f"items/{cell}/{f}"]), f"{cell}/{f}"


def test_shares(reference, demo) -> None:
    """R9.2: every arm, cell and half within 0.005 of the reference"""
    _, ref_info = reference
    _, new = demo
    worst, bad = 0.0, []
    for cell in CELLS:
        assert sorted(new["shares"][cell]) == sorted(ref_info["shares"][cell]), "the two sides measured different arms"
        for arm, by_half in ref_info["shares"][cell].items():
            for half in HALVES:
                diff = abs(new["shares"][cell][arm][half][0] - by_half[half][0])
                worst = max(worst, diff)
                if diff > MAX_SHARE_DIFF:
                    bad.append((cell, arm, half, new["shares"][cell][arm][half][0], by_half[half][0]))
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "shares_vs_reference.json").write_text(json.dumps({"largest_share_difference": worst, "limit": MAX_SHARE_DIFF}, indent=1) + "\n")
    assert not bad, bad


LINK_RECORD = "true_mixer_link.json"   # the original project's result file of the group-4885 link, measured after snapshot 3 was frozen
LINK_RECORD_SHA256 = "56a2bec724743b5449ff57f5c8315895527485dda60d2df57891e71837cab9c3"   # the file is not in a snapshot: a rerun by the original project must not move the comparison unnoticed
LINK_NAMES = {"full": "full", "g4885": "proj_g4885", "minus_g4885": "minus_g4885"}


def _recorded(settings: Settings) -> dict[str, dict[str, dict[str, float]]]:
    """cell -> arm -> half -> the share in the original's result files (snapshot 3; the link arms from the original's results folder)"""
    folder = settings.need_snapshot3() / "results" / "demo_line_count"
    gates, s2, s3, s4 = (json.loads((folder / f"{name}.json").read_text()) for name in ("gates", "s2", "s3", "s4"))
    link_file = settings.need_legacy_results() / "demo_line_count" / LINK_RECORD
    if not link_file.is_file():
        pytest.fail(f"the original project's record of the group-4885 link is missing: {link_file}", pytrace=False)
    if hashlib.sha256(link_file.read_bytes()).hexdigest() != LINK_RECORD_SHA256:
        pytest.fail(f"{link_file} is not the file the link arms were first compared with (sha256 differs)", pytrace=False)
    link = json.loads(link_file.read_text())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cell in CELLS:
        rows = {
            "layer_y": gates["cells"][cell]["share"]["layer_y"], "head_o": gates["cells"][cell]["share"]["head_o"],
            "l8/full_post8": s3["cells"][cell]["ceil"]["full_post8/true"], "l8/mlp8_true": s3["cells"][cell]["ceil"]["mlp8_true/true"],
            "l8/molt8_all": s3["cells"][cell]["ceil"]["molt8_all/true"], "l8/t189": s3["cells"][cell]["sets"]["1"]["true"],
            "l8/g11586": s4["cells"][cell]["resid_post8_group_swaps"]["g11586"]["true"],
            "l9/mlp9_y": s2["cells"][cell]["mlp9_y"], "l9/molt9_all": s2["cells"][cell]["M9_whole"],
            "l9/t410": s2["cells"][cell]["singles"]["410"], "l9/t1749": s2["cells"][cell]["singles"]["1749"],
        }
        share = link["cells"][cell]["share"]
        for base in ("mix", "h10o"):
            for mine, theirs in LINK_NAMES.items():
                for route in ("down", "F2", "T2"):
                    rows[f"link/{base}/{mine}/{route}"] = share[f"{base}/{theirs}"][route]
        for j in range(5):
            for route in ("F2", "T2"):
                rows[f"link/mix/rand{j}/{route}"] = share[f"mix/proj_rand{j}"][route]
        out[cell] = {arm: {half: row[half][0] for half in HALVES} for arm, row in rows.items()}
    return out


def test_recorded_shares(settings: Settings, demo) -> None:
    """R9.3: within 0.005 of the shares the original recorded on 2026-11-18"""
    _, new = demo
    recorded = _recorded(settings)
    table, bad = {}, []
    for cell in CELLS:
        assert sorted(recorded[cell]) == sorted(new["shares"][cell]), "an arm of the demonstration has no record (or the reverse)"
        for arm, by_half in recorded[cell].items():
            for half in HALVES:
                got = new["shares"][cell][arm][half][0]
                table[f"{cell} {arm} {half}"] = {"new": got, "recorded": by_half[half]}
                if abs(got - by_half[half]) > MAX_SHARE_DIFF:
                    bad.append((cell, arm, half, got, by_half[half]))
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "shares_vs_record.json").write_text(json.dumps(table, indent=1) + "\n")
    assert not bad, bad


def test_gates_and_descriptive_numbers(settings: Settings, reference, demo) -> None:
    """R9.4: gate values under R0 per 4,096 tokens; firing rates and the frame-energy fraction within 0.005"""
    ref, ref_info = reference
    new_t, new = demo
    for k_new, k_ref in (("chars", "c"), ("newline", "nl"), ("line_first", "ls"), ("known", "known")):
        assert torch.equal(new_t[f"label/{k_new}"], ref[f"label/{k_ref}"]), f"token labels differ: {k_new}"
    for name in ("gate/l8", "gate/l9"):
        a, b = new_t[name].float().flatten(0, 1), ref[name].float().flatten(0, 1)        # (tokens, transforms)
        assert a.shape == b.shape
        for s in range(0, a.shape[0], 4096):
            x, y = a[s:s + 4096], b[s:s + 4096]
            differing = (x > 0) != (y > 0)
            assert int(differing.sum()) <= MAX_DIFFERING, f"{name}, tokens {s}-: {int(differing.sum())} gate decisions differ"
            clean = ~differing.any(1)
            assert float((x[clean] - y[clean]).abs().max() / y.abs().max()) <= MAX_REL_DIFF_ELSEWHERE, f"{name}, tokens {s}-"
    # firing rates, recomputed from the reference's own gates and labels with the original's masks (s3_mlp8.py)
    n = 128
    known, nl, ls = ref["label/known"][:n].bool(), ref["label/nl"][:n].bool(), ref["label/ls"][:n].bool()
    ok = known.clone()
    ok[:, 0] = False
    where = {"line_first": ok & ls, "newline": ok & nl, "other": ok & ~ls & ~nl}
    recorded_fire = json.loads((settings.need_snapshot3() / "results" / "demo_line_count" / "s3.json").read_text())["transforms_on_text"]["189"]["fire"]
    for label, gate in (("L8 transform 189", ref["gate/l8"][:n, :, 0]), ("L9 transform 410", ref["gate/l9"][:n, :, 0]),
                        ("L9 transform 1749", ref["gate/l9"][:n, :, 1])):
        for k, sel in where.items():
            want = float((gate[sel] > 0).float().mean())
            assert abs(new["firing_rate"][label][k] - want) <= MAX_SHARE_DIFF, (label, k)
            if label == "L8 transform 189":
                assert abs(new["firing_rate"][label][k] - recorded_fire[k]) <= MAX_SHARE_DIFF, (label, k, "record")
    got = new["energy_fraction_t189_swap"]
    want = ref_info["energy_fraction_t189_swap"]
    assert abs(got["group"] - want["g11586"]) <= MAX_SHARE_DIFF
    assert abs(got["group"] - RECORDED_ENERGY_FRACTION) <= MAX_SHARE_DIFF
    for j in range(5):
        assert abs(got[f"random{j}"] - want[f"rand{j}"]) <= MAX_SHARE_DIFF


def test_report(reference, reference_again, demo) -> None:
    """R9.5 (reported, no pass/fail): per-item margin differences, new against reference and reference against itself"""
    ref, ref_info = reference
    new_t, _ = demo
    rows = {}
    for key in sorted(k for k in ref if k.startswith("margin/")):
        rows[key[len("margin/"):]] = {"new_vs_reference": float((new_t[key] - ref[key]).abs().max()),
                                      "reference_vs_reference": float((reference_again[key] - ref[key]).abs().max())}
    report = {"largest_new_vs_reference": max(r["new_vs_reference"] for r in rows.values()),
              "largest_reference_vs_reference": max(r["reference_vs_reference"] for r in rows.values()),
              "unpatched_margins_bit_identical": all(torch.equal(new_t[f"margin/{cell}/{m}"], ref[f"margin/{cell}/{m}"]) for cell in CELLS for m in ("m_c", "m_x")),
              "reference_peak_gpu_gb": ref_info["peak_gpu_gb"], "per_arm": rows}
    MEASURED.mkdir(parents=True, exist_ok=True)
    (MEASURED / "margins.json").write_text(json.dumps(report, indent=1) + "\n")
