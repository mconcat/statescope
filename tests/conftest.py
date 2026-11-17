"""Test settings and shared fixtures.

Machine-specific paths come from the environment, or from `tests/local.env` (not committed; see `tests/local.env.example`). A test that
needs a value that is not set FAILS with a message naming it; nothing is skipped silently. Tests that need private data carry the
markers `release` / `legacy`, so they can be left out on purpose with `-m "not release and not legacy"`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True      # tests import the frozen reader by path; no bytecode may appear inside the snapshot
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # device indices as nvidia-smi shows them; must be set before CUDA starts

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEST_OUT = REPO / "test_out"


def _read_local_env() -> None:
    """KEY=VALUE lines of tests/local.env become environment variables unless already set."""
    path = REPO / "tests" / "local.env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if value:
            os.environ.setdefault(key.strip(), value.strip())


_read_local_env()
# The language model is read from one named cache, offline. A separate variable, because a shell-wide HF_HOME (a lab default that points
# somewhere else) must not silently decide which copy of the model the tests load.
if os.environ.get("STATESCOPE_HF_HOME"):
    os.environ["HF_HOME"] = os.environ["STATESCOPE_HF_HOME"]
    os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass(frozen=True)
class Settings:
    release_root: Path | None       # holds layers/Lxx/{vocab,molt}/
    legacy_results: Path | None     # `results` folder of the original repository (read only)
    legacy_snapshot: Path           # frozen copy of the original code
    device: str
    gpu_gb: float
    train_device: str               # device of the short training runs (one process at a time)
    train_gb: float                 # their memory cap (the original runs used 16 for the MLP dictionary, 9 for the state dictionary)
    train_lock: Path | None         # if set, training runs queue on this lock file with flock (a GPU shared with other work)
    snapshot2: Path = REPO / "legacy_snapshot2"   # frozen reference code of the second design
    molt2_root: Path | None = None  # second design, MLP dictionaries: holds L<layer>/export/{weights.safetensors, config.yaml}

    def need_release_root(self) -> Path:
        if self.release_root is None or not (self.release_root / "layers").is_dir():
            pytest.fail("STATESCOPE_RELEASE_ROOT is not set or has no layers/ folder (see tests/local.env.example)", pytrace=False)
        return self.release_root

    def need_legacy_results(self) -> Path:
        if self.legacy_results is None or not self.legacy_results.is_dir():
            pytest.fail("STATESCOPE_LEGACY_RESULTS is not set or is not a folder (see tests/local.env.example)", pytrace=False)
        return self.legacy_results

    def need_legacy_snapshot(self) -> Path:
        if not (self.legacy_snapshot / "SHA256SUMS").is_file():
            pytest.fail(f"no frozen legacy snapshot at {self.legacy_snapshot} (make one with devtools/freeze_legacy.py)", pytrace=False)
        return self.legacy_snapshot

    def need_snapshot2(self) -> Path:
        if not (self.snapshot2 / "SHA256SUMS").is_file():
            pytest.fail(f"no frozen snapshot of the second design at {self.snapshot2} (make one with devtools/freeze_design2.py)", pytrace=False)
        return self.snapshot2

    def vocab2_dir(self, layer: int) -> Path:
        """second design: export folder of a layer's state dictionary in the original results"""
        return self.need_legacy_results() / "mlp_track" / "q08c" / "train1" / f"L{layer}" / "B1" / "vocab_B1s0" / "final"

    def calib2_dir(self, layer: int) -> Path:
        """second design: calibration folder of a layer (statistics + held-out activations of the new corpus)"""
        return self.need_legacy_results() / "mlp_track" / "q08c" / "calib1" / f"calib_q08_L{layer}"

    def molt2_dir(self, layer: int) -> Path:
        if self.molt2_root is None or not self.molt2_root.is_dir():
            pytest.fail("STATESCOPE_MOLT2_ROOT is not set or is not a folder (see tests/local.env.example)", pytrace=False)
        return self.molt2_root / f"L{layer}"

    def layer_dir(self, layer: int, kind: str) -> Path:
        """release folder of one dictionary: kind is "vocab" or "molt" """
        return self.need_release_root() / "layers" / f"L{layer:02d}" / kind

    def calib_dir_08b(self, layer: int) -> Path:
        """calibration folder (statistics + held-out activations) of a 0.8B layer in the original results; L9 used an older folder"""
        res = self.need_legacy_results()
        return res / "molt_qwen_L9" / "cache_allpos" if layer == 9 else res / "mlp_track" / "calib" / f"calib_q08_L{layer}"


def _path_or_none(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(value) if value else None


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        release_root=_path_or_none("STATESCOPE_RELEASE_ROOT"),
        legacy_results=_path_or_none("STATESCOPE_LEGACY_RESULTS"),
        legacy_snapshot=_path_or_none("STATESCOPE_LEGACY_SNAPSHOT") or REPO / "legacy_snapshot",
        device=os.environ.get("STATESCOPE_DEVICE", "cuda:0"),
        gpu_gb=float(os.environ.get("STATESCOPE_GPU_GB", "12")),
        train_device=os.environ.get("STATESCOPE_TRAIN_DEVICE", os.environ.get("STATESCOPE_DEVICE", "cuda:0")),
        train_gb=float(os.environ.get("STATESCOPE_TRAIN_GB", "16")),
        train_lock=_path_or_none("STATESCOPE_TRAIN_GPU_LOCK"),
        snapshot2=_path_or_none("STATESCOPE_SNAPSHOT2") or REPO / "legacy_snapshot2",
        molt2_root=_path_or_none("STATESCOPE_MOLT2_ROOT"),
    )


@pytest.fixture(scope="session")
def device(settings: Settings) -> str:
    """the CUDA device of the tests, with this process capped at settings.gpu_gb"""
    if not torch.cuda.is_available():
        pytest.fail("this test needs a CUDA device", pytrace=False)
    index = torch.device(settings.device).index or 0
    total_gb = torch.cuda.get_device_properties(index).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(settings.gpu_gb / total_gb, index)
    return settings.device
