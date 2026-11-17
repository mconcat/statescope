"""Run one ORIGINAL trainer / preparation module from the frozen snapshot, as its launcher script did (runs inside the snapshot).

    train_reference.py --out <dir> --module measure.mlp_track.train_fixwhite [--trace-main] -- <the module's own arguments>

`{RUN}` in the module's arguments is replaced by <dir>/run (the module's own output folder). The module is started with
runpy.run_module(..., run_name="__main__"), i.e. like `python -m <module>`; the working directory is the sandbox, so every relative
path the original code writes lands there.

--trace-main is for line-coverage runs only: train.layer_recipe compiles an edited copy of train.molt_stream.main and executes it under
molt_stream.py's file name, which makes its lines indistinguishable from the module's own. With the flag the edited source is written to
<dir>/molt_stream_main_edited.py and compiled under that name instead. Reference runs used for comparisons do not use the flag.
"""
import argparse
import json
import os
import runpy
import sys

from _hermetic import assert_hermetic


def main() -> None:
    argv = sys.argv[1:]
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--trace-main", action="store_true")
    ap.add_argument("--rep", default="A", help="only distinguishes repeated runs (separate cache entries)")
    a = ap.parse_args(argv[:cut])
    run_dir = os.path.join(a.out, "run")
    module_args = [x.replace("{RUN}", run_dir) for x in argv[cut + 1:]]

    if a.trace_main:
        import inspect

        import train.layer_recipe as LR
        import train.molt_stream as MS

        def edited_main():
            src = inspect.getsource(MS.main)
            for old, new in LR.EDITS:
                if src.count(old) != 1:
                    raise RuntimeError(f"edit does not apply exactly once: {old[:70]!r}")
                src = src.replace(old, new)
            src = src.replace("def main():", "def _main_layer_recipe():", 1)
            path = os.path.join(a.out, "molt_stream_main_edited.py")
            with open(path, "w") as f:
                f.write(src)
            ns = MS.__dict__
            ns["LRX"] = LR.LRX
            exec(compile(src, path, "exec"), ns)
            return ns["_main_layer_recipe"]
        LR.edited_main = edited_main

    sys.argv = [a.module] + module_args
    if a.trace_main and a.module == "train.layer_recipe":
        LR.main()               # the imported module (whose edited_main was replaced above), not a second copy run as __main__
    else:
        runpy.run_module(a.module, run_name="__main__", alter_sys=True)
    json.dump({"module": a.module, "args": module_args, "legacy_modules": assert_hermetic()}, open(os.path.join(a.out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
