"""Command lines of the short legacy training runs: the original launcher's arguments (logs/mlp_track/run.sh and the lane scripts in the
snapshot) with a small token budget, absolute input paths, and the run folder placeholder of tests/legacy_scripts/train_reference.py."""
from __future__ import annotations

from conftest import Settings

MODEL = "qwen3.5-0.8b"
SHORT_STEPS = 300                     # x 4,096 tokens per step; long enough to pass the 1M-token dead-atom window once
LOG_EVERY = 100


DEAD_VARIANT = ["--dead-tokens", "40960"]      # 10 steps instead of 1M tokens: makes the dead-atom auxiliary loss run within a short run


def _stream_args(settings: Settings, layer: int, tokens: int, extra: list[str]) -> list[str]:
    """what run.sh passes to the stream trainer, plus the two corpora by absolute path (the originals are defaults relative to the repository)"""
    results = settings.need_legacy_results()
    calib = settings.calib_dir_08b(layer)
    args = ["--model", MODEL, "--calib-dir", str(calib), "--log-every", str(LOG_EVERY), "--probe", "0", "--micro", "4",
            "--buffer-device", "cpu", "--host-uv-states", "1", "--export-format", "tiers"]
    if (calib / "train_index_kept.json").is_file():
        args += ["--ref-index", str(calib / "train_index_kept.json")]
    args += ["--strict-ref", "0", "--view-corpus", str(results / "site_train" / "corpus_as_of_L13"),
             "--full-corpus", str(results / "molt_qwen" / "corpus"), "--tokens", str(tokens), *extra, "--out", "{RUN}"]
    return args


def legacy_molt_run(settings: Settings, layer: int, rep: str, steps: int = SHORT_STEPS, trace: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    """MLP dictionary, recipe M1w (lane_m1w_layers.sh)"""
    fixmap = settings.need_legacy_results() / "mlp_track" / "fixedmap" / f"L{layer}.safetensors"
    head = ["--module", "measure.mlp_track.train_fixwhite", "--rep", rep] + (["--trace-main"] if trace else [])
    return head + ["--", "--layer", str(layer), "--mem-gb", str(settings.train_gb), "--molt", "M1w=Qfix", "--fixmap", str(fixmap),
                   "--molt-export-tokens=", "--"] + _stream_args(settings, layer, steps * 4096, list(extra))


def legacy_vocab_run(settings: Settings, layer: int, rep: str, steps: int = SHORT_STEPS, trace: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    """state dictionary V0, seed 0 (lane_vocab.sh)"""
    spec = settings.need_legacy_snapshot() / "results" / "mlp_track" / "specs" / "v0.json"
    head = ["--module", "train.layer_recipe", "--rep", rep] + (["--trace-main"] if trace else [])
    return head + ["--", "--layer", str(layer), "--mem-gb", str(settings.train_gb), "--vocabs", str(spec), "--vocab-only", "V0s0",
                   "--vocab-export-tokens=", "--"] + _stream_args(settings, layer, steps * 4096, list(extra))
