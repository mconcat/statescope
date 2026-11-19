"""Test settings and shared fixtures.

Machine-specific paths come from the environment, or from `tests/local.env` (not committed; see `tests/local.env.example`). A test that
needs a value that is not set FAILS with a message naming it; nothing is skipped silently. Tests that need private data (the frozen
original code, the original project's results, the staged weight folders) carry the marker `legacy`, so they can be left out on
purpose with `-m "not legacy"`.

Names. The suffix 2 (snapshot2, molt2_*, release2, test_reader2.py, test_release2.py) means the 0.8B set, which
tests/EQUIVALENCE_RULES.md calls the second design; the suffix 4b means the 4B set. snapshot3 and test_demo.py are the line-count
demonstration. `legacy_snapshot` without a suffix is the first frozen copy of the original code; only the capture test uses it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True      # tests import the reader by path; no bytecode may appear inside a snapshot
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
    legacy_results: Path | None     # `results` folder of the original repository (read only)
    legacy_snapshot: Path           # first frozen copy of the original code (the capture test)
    device: str
    gpu_gb: float
    snapshot2: Path                 # frozen reference code of the two readers
    snapshot3: Path                 # frozen reference code of the line-count demonstration
    molt2_root: Path | None         # 0.8B set, MLP dictionaries as trained: holds L<layer>/export/{weights.safetensors, config.yaml}
    molt2_reexports: Path | None    # holds the two layer-9 exports in the later format (old_corpus_L9_pub128{,_post}/L9/export)
    mlp_copy_08b: Path | None       # 0.8B set, the original project's path-free copy of the MLP dictionaries: L<layer>/, MANIFEST.json
    release2: Path | None           # the 0.8B set as published: state/L<L>/, mlp/L<L>/, MANIFEST.json
    molt4b_root: Path | None        # 4B set, MLP dictionaries as trained: holds L<layer>/export/ for every layer but 16
    molt4b_l16_root: Path | None    # 4B set: holds L16/export/ (layer 16 comes from a separate run)
    mlp_copy_4b: Path | None        # 4B set, the original project's path-free copy of the MLP dictionaries: L<layer>/, MANIFEST.json
    release4b: Path | None          # the 4B set as published: state/L<L>/, mlp/L<L>/, MANIFEST.json

    def _need(self, value: Path | None, name: str, what: str) -> Path:
        if value is None or not value.is_dir():
            pytest.fail(f"{name} is not set or is not a folder: {what} (see tests/local.env.example)", pytrace=False)
        return value

    def need_legacy_results(self) -> Path:
        return self._need(self.legacy_results, "STATESCOPE_LEGACY_RESULTS", "the original project's results")

    def _need_snapshot(self, snapshot: Path, tool: str) -> Path:
        if not (snapshot / "SHA256SUMS").is_file():
            pytest.fail(f"no frozen snapshot at {snapshot} (make one with devtools/{tool})", pytrace=False)
        return snapshot

    def need_legacy_snapshot(self) -> Path:
        return self._need_snapshot(self.legacy_snapshot, "freeze_legacy.py")

    def need_snapshot2(self) -> Path:
        return self._need_snapshot(self.snapshot2, "freeze_design2.py")

    def need_snapshot3(self) -> Path:
        return self._need_snapshot(self.snapshot3, "freeze_demo.py")

    def _need_published(self, folder: Path | None, name: str) -> Path:
        if folder is None or not (folder / "MANIFEST.json").is_file():
            pytest.fail(f"{name} is not set or has no MANIFEST.json (make the folder with devtools/stage_release2.py)", pytrace=False)
        return folder

    def need_release2(self) -> Path:
        return self._need_published(self.release2, "STATESCOPE_RELEASE2")

    def need_release4b(self) -> Path:
        return self._need_published(self.release4b, "STATESCOPE_RELEASE_4B")

    # ---- 0.8B set: the original files

    def vocab2_dir(self, layer: int) -> Path:
        """export folder of a layer's state dictionary in the original results"""
        return self.need_legacy_results() / "mlp_track" / "q08c" / "train1" / f"L{layer}" / "B1" / "vocab_B1s0" / "final"

    def calib2_dir(self, layer: int) -> Path:
        """calibration folder of a layer (held-out activations)"""
        return self.need_legacy_results() / "mlp_track" / "q08c" / "calib1" / f"calib_q08_L{layer}"

    def molt2_dir(self, layer: int) -> Path:
        return self._need(self.molt2_root, "STATESCOPE_MOLT2_ROOT", "the 0.8B MLP dictionaries as trained") / f"L{layer}"

    def molt2_reexport(self, name: str) -> Path:
        root = self._need(self.molt2_reexports, "STATESCOPE_MOLT2_REEXPORTS", "the two layer-9 exports in the later format")
        return self._need(root / name, "STATESCOPE_MOLT2_REEXPORTS", f"its folder {name}/") / "L9" / "export"

    def mlp_copy_08b_dir(self, layer: int) -> Path:
        return self._need(self.mlp_copy_08b, "STATESCOPE_MLP_COPY_08B", "the path-free copy of the 0.8B MLP dictionaries") / f"L{layer}"

    # ---- 4B set: the original files

    def vocab4b_dir(self, layer: int) -> Path:
        """export folder of a layer's state dictionary in the original results"""
        return self.need_legacy_results() / "mlp_track" / "q4b" / "train3" / f"L{layer}" / "B1" / "vocab_B1s0" / "final"

    def calib4b_dir(self, layer: int) -> Path:
        """calibration folder of a layer (held-out activations)"""
        return self.need_legacy_results() / "mlp_track" / "q4b" / "calib3" / f"calib_q4b_L{layer}"

    def molt4b_dir(self, layer: int) -> Path:
        """training export of a layer's MLP dictionary (with its own postnorm.safetensors)"""
        if layer == 16:
            return self._need(self.molt4b_l16_root, "STATESCOPE_4B_MOLT_L16_ROOT", "the 4B layer-16 MLP dictionary as trained") / "L16" / "export"
        return self._need(self.molt4b_root, "STATESCOPE_4B_MOLT_ROOT", "the 4B MLP dictionaries as trained") / f"L{layer}" / "export"

    def mlp_copy_4b_dir(self, layer: int) -> Path:
        return self._need(self.mlp_copy_4b, "STATESCOPE_MLP_COPY_4B", "the path-free copy of the 4B MLP dictionaries") / f"L{layer}"


def _path_or_none(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        legacy_results=_path_or_none("STATESCOPE_LEGACY_RESULTS"),
        legacy_snapshot=_path_or_none("STATESCOPE_LEGACY_SNAPSHOT") or REPO / "legacy_snapshot",
        device=os.environ.get("STATESCOPE_DEVICE", "cuda:0"),
        gpu_gb=float(os.environ.get("STATESCOPE_GPU_GB", "12")),
        snapshot2=_path_or_none("STATESCOPE_SNAPSHOT2") or REPO / "legacy_snapshot2",
        snapshot3=_path_or_none("STATESCOPE_SNAPSHOT3") or REPO / "legacy_snapshot3",
        molt2_root=_path_or_none("STATESCOPE_MOLT2_ROOT"),
        molt2_reexports=_path_or_none("STATESCOPE_MOLT2_REEXPORTS"),
        mlp_copy_08b=_path_or_none("STATESCOPE_MLP_COPY_08B"),
        release2=_path_or_none("STATESCOPE_RELEASE2"),
        molt4b_root=_path_or_none("STATESCOPE_4B_MOLT_ROOT"),
        molt4b_l16_root=_path_or_none("STATESCOPE_4B_MOLT_L16_ROOT"),
        mlp_copy_4b=_path_or_none("STATESCOPE_MLP_COPY_4B"),
        release4b=_path_or_none("STATESCOPE_RELEASE_4B"),
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
