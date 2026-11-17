"""Command line of the package: one subcommand per step of the pipeline.

    statescope verify    --manifest release/qwen35-08b/manifest.json --root <release folder>
    statescope calibrate --model qwen3.5-0.8b --layer 19 --out <calib dir>            <data paths>
    statescope fixedmap  --model qwen3.5-0.8b --layer 19 --calib-dir <calib dir> --out <file.safetensors>   <data paths>
    statescope foldcheck --molt <export dir> --calib-dir <calib dir> [--fold-err 3.1e-3]
    statescope splice    --model qwen3.5-0.8b --root <release folder> --layers 9,19 --corpus <jsonl> --out <json>

    statescope train-vocab --model qwen3.5-0.8b --layer 19 --calib-dir <calib dir> --tokens 50e6 --out <run dir>     <data paths>
    statescope train-molt  --model qwen3.5-0.8b --layer 19 --calib-dir <calib dir> --fixmap <file> --tokens 29.3e6 --out <run dir>   <data paths>

<data paths> = --pile-dir --ultrachat-dir --view-corpus --full-corpus (see statescope.config.DataPaths).
The order for one layer: calibrate, fixedmap, then the two trainers (independent of each other), then foldcheck and splice on the exports.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch

from statescope.config import MIX_08B, MODELS, DataPaths


def _add_data_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pile-dir", required=True, help="the 64 parquet shards of the pre-tokenized Pile")
    parser.add_argument("--ultrachat-dir", required=True, help="UltraChat 200k parquet files")
    parser.add_argument("--view-corpus", required=True, help="the model's own text as it was when calibration was made")
    parser.add_argument("--full-corpus", required=True, help="the same corpus, later state (the training stream reads this one)")


def _data_paths(args: argparse.Namespace) -> DataPaths:
    return DataPaths(args.pile_dir, args.ultrachat_dir, args.view_corpus, args.full_corpus)


def _cap_gpu(device: str, gb: float | None) -> None:
    """limit this process's memory on the device, so a co-located job fails itself instead of its neighbours"""
    if gb is None:
        return
    index = torch.device(device).index or 0
    total = torch.cuda.get_device_properties(index).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(gb / total, index)


def cmd_verify(args: argparse.Namespace) -> None:
    from statescope.manifest import load_manifest, verify_files
    problems = verify_files(load_manifest(args.manifest), args.root, workers=args.workers)
    for line in problems:
        print(line)
    if problems:
        raise SystemExit(f"{len(problems)} file(s) do not match the manifest")
    print("every listed file has the listed size and sha256")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from statescope.lm import load_language_model
    from statescope.prep.calibrate import calibrate
    _cap_gpu(args.device, args.mem_gb)
    lm = load_language_model(MODELS[args.model], args.device)
    print(json.dumps(calibrate(lm, args.layer, _data_paths(args), MIX_08B, args.out), indent=1))


def cmd_fixedmap(args: argparse.Namespace) -> None:
    from statescope.data.mix import training_stream
    from statescope.lm import load_language_model
    from statescope.prep.fixedmap import fixed_map
    _cap_gpu(args.device, args.mem_gb)
    lm = load_language_model(MODELS[args.model], args.device)
    stream = training_stream(lm.tokenizer, _data_paths(args), MIX_08B)
    print(json.dumps(fixed_map(lm, args.layer, stream, args.calib_dir, args.out), indent=1))


def _training_setup(args: argparse.Namespace):
    """what both trainers need: the capped device, calibration statistics, the language model and the token stream"""
    from statescope.calibration import load_calibration
    from statescope.data.mix import training_stream
    from statescope.lm import load_language_model
    _cap_gpu(args.device, args.mem_gb)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    stats = load_calibration(args.calib_dir, torch.device(args.device))
    lm = load_language_model(MODELS[args.model], args.device)
    stream = training_stream(lm.tokenizer, _data_paths(args), MIX_08B)
    return stats, lm, stream


def cmd_train_vocab(args: argparse.Namespace) -> None:
    from statescope.train.loop import LoopSettings
    from statescope.vocab.recipe import VocabRecipe
    from statescope.vocab.train import train_vocab
    stats, lm, stream = _training_setup(args)
    loop = LoopSettings(tokens=int(args.tokens), seed=args.seed)
    fvu = train_vocab(lm, args.layer, stream, stats, args.calib_dir, VocabRecipe(seed=args.seed), loop, args.out, args.log_every)
    print(json.dumps({"heldout_fvu": fvu}, indent=1))


def cmd_train_molt(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file

    from statescope.lm import Capturer
    from statescope.molt.recipe import MoltRecipe
    from statescope.molt.train import train_molt
    stats, lm, stream = _training_setup(args)
    capturer = Capturer(lm, args.layer, MIX_08B.min_pos)
    skip_map = load_file(args.fixmap)["W_skip_std"]
    recipe = dataclasses.replace(MoltRecipe(), seed=args.seed)
    final = train_molt(lm, capturer, stream, stats, args.calib_dir, skip_map, args.out, recipe, tokens=int(args.tokens),
                       log_every=args.log_every, device=torch.device(args.device))
    print(json.dumps(final, indent=1))


def cmd_foldcheck(args: argparse.Namespace) -> None:
    from statescope.foldcheck import fold_error_verdict, gate_flip_check
    from statescope.heldout import load_heldout
    from statescope.reader import MoltOps
    _cap_gpu(args.device, args.mem_gb)
    x, _ = load_heldout(args.calib_dir, "pile", args.tokens)
    result = gate_flip_check(MoltOps(args.molt, args.device), x.to(args.device), args.fold_err)
    if args.fold_err is not None:
        result["verdict"] = fold_error_verdict(args.fold_err, result)
    print(json.dumps(result, indent=1))


def cmd_splice(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from statescope.manifest import layer_dir
    from statescope.reader import MoltOps, StateVocab
    from statescope.splice import load_documents, splice_layer
    _cap_gpu(args.device, args.mem_gb)
    spec = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch.bfloat16, device_map=args.device).eval()
    ids = load_documents(tokenizer, args.corpus, args.docs, args.doc_tokens, args.first_row)
    results = {}
    for layer in [int(x) for x in args.layers.split(",")]:
        vocab = StateVocab(layer_dir(args.root, layer, "vocab"), args.device)
        molt = MoltOps(layer_dir(args.root, layer, "molt"), args.device)
        results[str(layer)] = splice_layer(model, layer, ids, vocab, molt, batch=args.batch)
        print(json.dumps(results[str(layer)]), flush=True)
        del vocab, molt
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="statescope", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def gpu(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--mem-gb", type=float, default=None, help="cap this process's memory on the device (GiB)")

    p = sub.add_parser("verify", help="compare the release files with the manifest (size and sha256)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True, help="folder that holds layers/Lxx/{vocab,molt}/")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("calibrate", help="write a layer's calibration folder (statistics + held-out activations)")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", required=True)
    _add_data_paths(p)
    gpu(p)
    p.set_defaults(run=cmd_calibrate)

    p = sub.add_parser("fixedmap", help="write a layer's fixed mean-gate map (the MLP dictionary's frozen skip matrix)")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--out", required=True, help="the .safetensors file to write")
    _add_data_paths(p)
    gpu(p)
    p.set_defaults(run=cmd_fixedmap)

    def training(p: argparse.ArgumentParser, default_tokens: float) -> None:
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        p.add_argument("--layer", type=int, required=True)
        p.add_argument("--calib-dir", required=True)
        p.add_argument("--tokens", type=float, default=default_tokens, help="training budget in tokens")
        p.add_argument("--seed", type=int, default=0, help="only seed 0 has been compared with the original code")
        p.add_argument("--log-every", type=int, default=250, help="steps between log records")
        p.add_argument("--out", required=True, help="run folder to write (export/, logs, final numbers)")
        _add_data_paths(p)
        gpu(p)

    p = sub.add_parser("train-vocab", help="train a layer's state dictionary")
    training(p, 50e6)
    p.set_defaults(run=cmd_train_vocab)

    p = sub.add_parser("train-molt", help="train a layer's MLP dictionary (recipe M1w)")
    training(p, 29.3e6)
    p.add_argument("--fixmap", required=True, help="the fixed mean-gate map written by `statescope fixedmap`")
    p.set_defaults(run=cmd_train_molt)

    p = sub.add_parser("foldcheck", help="gate decisions of an exported MLP dictionary in fp32 vs fp64")
    p.add_argument("--molt", required=True, help="the dictionary's export folder")
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--fold-err", type=float, default=None, help="the trainer's recorded fold error, to look for the pair that explains it")
    gpu(p)
    p.set_defaults(run=cmd_foldcheck)

    p = sub.add_parser("splice", help="next-token loss with a layer's dictionaries spliced into the forward pass")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--root", required=True)
    p.add_argument("--layers", required=True, help="comma-separated")
    p.add_argument("--corpus", required=True, help='JSON lines with a "text" field')
    p.add_argument("--docs", type=int, default=100)
    p.add_argument("--doc-tokens", type=int, default=256)
    p.add_argument("--first-row", type=int, default=0)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--out", required=True)
    gpu(p)
    p.set_defaults(run=cmd_splice)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
