"""Run the two legacy trainers briefly under line coverage (what the refactor has to carry over). Development aid, not a test.

    python devtools/trace_legacy_training.py molt|vocab [layer]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import conftest  # noqa: E402
import legacy_bridge  # noqa: E402
import train_args  # noqa: E402

kind = sys.argv[1]
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 19
settings = conftest.settings.__wrapped__()
make = {"molt": train_args.legacy_molt_run, "vocab": train_args.legacy_vocab_run}[kind]
out = legacy_bridge.run_legacy("train_reference.py", make(settings, layer, "trace", trace=True), settings, timeout_s=7200, training=True, coverage=True)
print(out)
