"""Read one layer of Qwen3.5-0.8B with its two dictionaries and print how well they reconstruct a few texts.

This example reads the EARLIER design (folders layers/Lxx/{vocab,molt}, MoltOps). For the published set (state/L<L>, mlp/L<L>,
PublicMolt) see the README's quick start and demo/line_count.py.

    python examples/reconstruct_texts.py --layer 19 --root <release folder> [--model Qwen/Qwen3.5-0.8B] [--device cuda]

Needs torch, safetensors, transformers and flash-linear-attention (without it transformers runs a slow reference GatedDeltaNet).
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from statescope.reader import MoltOps, StateVocab, capture, fvu

TEXTS = {
    "english": "The committee met on Thursday to review the budget for the coming year. Three members argued that the library should stay "
               "open on weekends, while the treasurer pointed out that the heating bill had doubled since January. After two hours they "
               "agreed to postpone the vote until the auditors had finished their report, which was expected by the end of the month.",
    "code": "def moving_average(values, window):\n    if window <= 0:\n        raise ValueError(\"window must be positive\")\n"
            "    out, total = [], 0.0\n    for i, v in enumerate(values):\n        total += v\n        if i >= window:\n"
            "            total -= values[i - window]\n        if i >= window - 1:\n            out.append(total / window)\n    return out\n",
    "korean": "오늘 아침에는 비가 조금 내렸지만 점심 무렵에는 하늘이 개었다. 우리는 강가를 따라 걸으며 다음 주에 있을 발표 순서를 정했다. "
              "첫째는 실험 설계, 둘째는 측정 결과, 셋째는 남은 문제였다. 발표가 끝나면 모두 함께 저녁을 먹기로 했다.",
}


@torch.no_grad()
def reconstruct(model, tokenizer, vocab: StateVocab, molt: MoltOps, layer: int, text: str) -> dict:
    """one text -> token count, FVU of the state dictionary at both sites, FVU of the MLP dictionary, mean active atoms, and the activations"""
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer(text, add_special_tokens=True)["input_ids"]], device=device)
    acts = capture(model, layer, ids)
    row: dict = {"tokens": int(acts["resid_mid"].shape[0])}
    for kind in ("resid_mid", "resid_post"):
        groups, codes = vocab.encode(acts[kind], f"{kind}.{layer}")
        row[f"fvu_{kind}"] = fvu(acts[kind], vocab.decode(groups, codes, f"{kind}.{layer}"))
    gate, y_hat = molt.forward(acts["resid_mid"])
    row["fvu_mlp"] = fvu(acts["mlp_out"], y_hat)
    row["atoms_on"] = float((gate > 0).sum(1).float().mean())
    row["acts"], row["gate"] = acts, gate
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--root", default=".", help="folder that holds layers/Lxx/{vocab,molt}/")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    layer, device = args.layer, args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device).eval()
    vocab = StateVocab(os.path.join(args.root, f"layers/L{layer:02d}/vocab"), device)
    molt = MoltOps(os.path.join(args.root, f"layers/L{layer:02d}/molt"), device)
    print(f"layer {layer}: vocab {vocab.G} groups x {vocab.b}, {vocab.k} active; molt {molt.n_atoms} atoms")
    print(f"{'text':10s} {'tokens':>6s} {'FVU resid_mid':>14s} {'FVU resid_post':>15s} {'FVU MLP (molt)':>15s} {'atoms on':>9s}")
    row: dict = {}
    for name, text in TEXTS.items():
        row = reconstruct(model, tokenizer, vocab, molt, layer, text)
        print(f"{name:10s} {row['tokens']:6d} {row['fvu_resid_mid']:14.3f} {row['fvu_resid_post']:15.3f} {row['fvu_mlp']:15.3f} {row['atoms_on']:9.1f}")
    # the last token of the last text: its strongest state groups and MLP atoms
    groups, codes = vocab.encode(row["acts"]["resid_post"][-1:], f"resid_post.{layer}")
    order = codes[0].norm(dim=-1).argsort(descending=True)[:5]
    print("strongest groups at the last token (group, written norm in standardized units):",
          [(int(groups[0, i]), round(float(codes[0, i].norm()), 2)) for i in order])
    top = row["gate"][-1].topk(5)
    print("strongest atoms at the last token (atom, rank, gate value):",
          [(int(i), int(molt.rank[i]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
