"""Produce the legacy short-training references (two runs per configuration) ahead of the tests, one process at a time.
The tests ask the bridge for the same runs and get them from its cache.

    python devtools/make_training_references.py [molt|vocab] [base|dead] ...
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import conftest  # noqa: E402
import legacy_bridge  # noqa: E402
import train_args  # noqa: E402

settings = conftest.settings.__wrapped__()
kinds = [k for k in sys.argv[1:] if k in ("molt", "vocab")] or ["molt", "vocab"]
variants = [v for v in sys.argv[1:] if v in ("base", "dead")] or ["base", "dead"]
for kind in kinds:
    make = {"molt": train_args.legacy_molt_run, "vocab": train_args.legacy_vocab_run}[kind]
    for variant in variants:
        extra = tuple(train_args.DEAD_VARIANT) if variant == "dead" else ()
        for rep in ("A", "B"):
            out = legacy_bridge.run_legacy("train_reference.py", make(settings, 19, rep, extra=extra), settings, timeout_s=6 * 3600, training=True)
            print(kind, variant, rep, out, flush=True)
