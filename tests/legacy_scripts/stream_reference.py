"""The first --units units of the ORIGINAL training token stream, as hashes (runs inside the frozen snapshot; CPU, no model).

The stream is set up as train.molt_stream.main does: MixStream(extend=True), check_view_prefix(), excl = prefix_set(reference_seqs()).
Writes <out>/stream.json.
"""
import argparse
import hashlib
import json
import os

import torch

from _hermetic import assert_hermetic


def digest(t: torch.Tensor) -> list:
    return [int(len(t)), hashlib.sha256(t.to(torch.int64).numpy().tobytes()).hexdigest()]


def state_record(st: dict) -> dict:
    return {"pile_pos": int(st["pos"]), "pile_buf": digest(st["pile_buf"].reshape(-1)), "pile_rows_used": int(st["pile_seq"]),
            "cursor": {k: int(v) for k, v in st["cur"].items()}, "unit": int(st["shard"]), "onpolicy_exhausted": bool(st["onpol_done"])}


def unit_record(si, st, parts, info) -> dict:
    return {"unit": int(si), "state": state_record(st), "parts": [[name, [digest(s) for s in seqs]] for name, seqs in parts],
            "info": json.loads(json.dumps(info, default=str))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--view-corpus", required=True)
    ap.add_argument("--full-corpus", required=True)
    ap.add_argument("--units", type=int, default=200)
    ap.add_argument("--replay-from", type=int, default=120)
    ap.add_argument("--replay-units", type=int, default=5)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    import train.molt_stream_data as SD

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")        # what measure.models.load_model gives as lm.tok
    ms = SD.MixStream(tok, a.view_corpus, a.full_corpus, extend=True)
    view_check = ms.check_view_prefix()
    refs = SD.reference_seqs(ms, a.full_corpus)
    ms.excl = SD.prefix_set(refs)
    units, replay_state = [], None
    for _ in range(a.units):
        si, st, parts, info = ms.next_unit()
        if si == a.replay_from:
            replay_state = st
        units.append(unit_record(si, st, parts, info))
    ms.restore(replay_state)
    replay = [unit_record(*ms.next_unit()) for _ in range(a.replay_units)]
    out = {"view_check": view_check, "reference": {k: [digest(s) for s in v] for k, v in refs.items()}, "excluded_prefixes": sorted(ms.excl),
           "units": units, "replay": replay, "legacy_modules": assert_hermetic()}
    json.dump(out, open(os.path.join(a.out, "stream.json"), "w"))


if __name__ == "__main__":
    main()
