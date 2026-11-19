"""Debug measurement, not a pass/fail test: is the state dictionary's held-out evaluation repeatable on this GPU?

Loads an exported dictionary (weights.safetensors of any state-dictionary export) into the training-form model and, on the first
--tokens held-out pile tokens of a calibration folder, repeats each stage of statescope.earlier_design.vocab.evaluate.heldout_fvu on identical inputs:

  basis       write_basis() (batched fp32 QR on the GPU), --repeats times
  reconstruct reconstruct() with one fixed basis (its last operation is an index_add_ with repeated rows), --repeats times
  ordered     the same sum with a fixed order (each token's k group writes reshaped to (n, k, d) and summed), --repeats times

Writes <out>/determinism.json.
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file

from statescope.heldout import load_heldout
from statescope.earlier_design.vocab.model import StateDictionary
from statescope.earlier_design.vocab.recipe import VocabRecipe


@torch.no_grad()
def reconstruct_ordered(model: StateDictionary, xs: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """StateDictionary.reconstruct with the per-token sum in a fixed order (nonzero() lists each token's k groups together)"""
    xt = xs - model.b_dec
    z = model.encode(xt).float()
    mask = model.select(model.written_norm(z))[0]
    rows, groups = mask.nonzero(as_tuple=True)
    codes = torch.einsum("nr,nrs->ns", z[rows, groups], R[groups])
    writes = (codes[:, None, :] @ Q.index_select(0, groups)).squeeze(1)
    return writes.view(xs.shape[0], model.recipe.active_groups, -1).sum(1) + model.b_dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mem-gb", type=float, default=8.0)
    a = ap.parse_args()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / total_gb)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("highest")
    cfg = json.load(open(os.path.join(a.export_dir, "cfg.json")))
    weights = load_file(os.path.join(a.export_dir, "weights.safetensors"))
    recipe = VocabRecipe(groups=cfg["n_units"], group_size=cfg["unit_size"], active_groups=cfg["k_topk"], seed=cfg["init_seed"])
    model = StateDictionary(cfg["d_in"], cfg["sites"], recipe, device)
    with torch.no_grad():
        for name in ("D", "E", "b_dec", "site_mu", "site_scale"):
            getattr(model, name).copy_(weights[name].to(device))
    x, _ = load_heldout(a.calib_dir, "pile", a.tokens)
    xs = model.standardize(x.to(device), 0)

    def spread(results: list[torch.Tensor]) -> dict:
        """how much repeated results differ from the first one"""
        diffs = [float((r - results[0]).abs().max()) for r in results[1:]]
        return {"bit_identical": all(torch.equal(r, results[0]) for r in results[1:]), "max_abs_diff": max(diffs),
                "max_abs_value": float(results[0].abs().max())}

    bases = [model.write_basis() for _ in range(a.repeats)]
    Q, R = bases[0]
    indexed = [model.reconstruct(xs, Q, R) for _ in range(a.repeats)]
    ordered = [reconstruct_ordered(model, xs, Q, R) for _ in range(a.repeats)]
    sse = [float(((xs - r) ** 2).sum()) for r in indexed]
    report = {"gpu": torch.cuda.get_device_name(0), "tokens": int(xs.shape[0]), "repeats": a.repeats,
              "basis_Q": spread([q for q, _ in bases]), "basis_R": spread([r for _, r in bases]),
              "reconstruct_index_add": spread(indexed), "reconstruct_ordered_sum": spread(ordered),
              "index_add_vs_ordered_max_abs_diff": float((indexed[0] - ordered[0]).abs().max()),
              "sse_index_add_repeats": sse, "sse_relative_spread": (max(sse) - min(sse)) / sse[0]}
    json.dump(report, open(os.path.join(a.out, "determinism.json"), "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
