"""Compare two result folders of demo/line_count.py: are they the same measurement, bit for bit?

The guard for restructuring the demonstration script. The script's output must not move at all, and the tests' tolerance (0.005 on a
share) would hide a slip, so this tool asks for identity:
  - result.safetensors: the same keys; per key the same dtype and shape, and torch.equal. For a key whose values differ, the largest
    absolute difference is printed.
  - result.json: the parsed objects are equal (shares, firing rates, energy fractions).
For information it also says whether the two files are the same bytes (that covers the order of the JSON keys too).

    python devtools/compare_demo_runs.py <folder_a> <folder_b>

Exit code 0 only if everything is identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

TENSOR_FILE = "result.safetensors"
JSON_FILE = "result.json"


def compare_tensors(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    """One line per difference between two tensor files; an empty list means identical."""
    problems: list[str] = []
    for key in sorted(set(a) - set(b)):
        problems.append(f"{key}: only in the first folder")
    for key in sorted(set(b) - set(a)):
        problems.append(f"{key}: only in the second folder")
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if x.dtype != y.dtype:
            problems.append(f"{key}: dtype {x.dtype} against {y.dtype}")
        elif x.shape != y.shape:
            problems.append(f"{key}: shape {tuple(x.shape)} against {tuple(y.shape)}")
        elif not torch.equal(x, y):
            difference = (x.double() - y.double()).abs()
            n_differing = int((x != y).sum())
            problems.append(f"{key}: {n_differing} of {x.numel()} values differ, largest absolute difference {float(difference.max()):.6g}")
    return problems


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_bytes(a: Path, b: Path) -> bool:
    return sha256_of(a) == sha256_of(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder_a", type=Path, help="result folder of one run (holds result.safetensors and result.json)")
    ap.add_argument("folder_b", type=Path, help="result folder of the other run")
    args = ap.parse_args()
    folder_a, folder_b = args.folder_a.resolve(), args.folder_b.resolve()
    print(f"first folder:  {folder_a}")
    print(f"second folder: {folder_b}")
    if folder_a == folder_b:
        print("NOT COMPARED: both arguments name the same folder")
        sys.exit(1)

    tensors_a, tensors_b = load_file(str(folder_a / TENSOR_FILE)), load_file(str(folder_b / TENSOR_FILE))
    tensor_problems = compare_tensors(tensors_a, tensors_b)
    for line in tensor_problems:
        print(f"  {line}")
    print(f"{TENSOR_FILE}: {len(tensors_a)} and {len(tensors_b)} tensors, "
          + ("all identical (keys, dtypes, shapes, values)" if not tensor_problems else f"{len(tensor_problems)} differences")
          + f"; same bytes: {'yes' if same_bytes(folder_a / TENSOR_FILE, folder_b / TENSOR_FILE) else 'no'}")

    json_a, json_b = json.loads((folder_a / JSON_FILE).read_text()), json.loads((folder_b / JSON_FILE).read_text())
    json_equal = json_a == json_b
    print(f"{JSON_FILE}: parsed objects " + ("equal" if json_equal else "DIFFER")
          + f"; same bytes: {'yes' if same_bytes(folder_a / JSON_FILE, folder_b / JSON_FILE) else 'no'}")

    identical = not tensor_problems and json_equal
    print("IDENTICAL" if identical else "DIFFERENT")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
