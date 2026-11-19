"""The earlier design: the code that prepares, trains, checks and lists the per-layer dictionaries of an earlier, unpublished set.

That design uses an untied state dictionary and an MLP dictionary of recipe "M1w" (gated low-rank atoms on top of a frozen linear map).
This code reproduces how that set was trained. It did NOT train the published weights; their trainers are not in this repository. It is
kept working and tested against the original research code (tests/EQUIVALENCE_RULES.md, rules R2 to R7).

Folders, in the order one layer goes through them:

  data/           the training token stream: readers of the four token sources (sources.py) and their fixed mixture (mix.py)
  prep/           before training: a layer's calibration folder (calibrate.py) and the fixed mean-gate map (fixedmap.py)
  train/          the loop shared by both trainers: token stream -> activations of one layer -> shuffle buffer -> batches (loop.py)
  vocab/          trainer of the state dictionary: recipe, model, calibration, evaluation, export, training run
  molt/           trainer of the MLP dictionary: recipe, model, training step, evaluation, export, training run

Modules:

  calibration.py  calibration statistics of one layer and the changes of coordinates built on them
  foldcheck.py    how sensitive an exported MLP dictionary's hard gate is to float rounding (fp32 against fp64)
  manifest.py     reading a release manifest and checking the listed files (size, sha256)
  release.py      building the release manifest from the exported files and the measured numbers

Two more modules belong to this design but keep their place one level up, because something outside the code names their path:

  statescope/cli.py     the `statescope` command line; the installed console script points at statescope.cli:main
  statescope/splice.py  next-token loss with a layer's dictionaries spliced in; the manifest of release/qwen35-08b names this file

The two dictionary packages vocab/ and molt/ do not import each other (tests/test_repo_rules.py checks it). The code here uses the
shared modules one level up: config.py, lm.py, heldout.py, metrics.py and the reader (reader.py), which reads this design's exports too.
"""
