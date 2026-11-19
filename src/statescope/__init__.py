"""statescope: per-layer dictionaries of Qwen3.5 (a state dictionary of the residual stream, an MLP dictionary of transforms).

The package has three groups of modules. This file imports nothing; import the module you need.

Published set
  reader.py         the reader of the published dictionaries (and of the earlier design's exports). One self-contained file that
                    imports nothing else from the package, so it can be copied next to the weights and used alone.

Shared (used by the demonstration, the measuring tools in devtools/, the tests and the earlier design)
  config.py         every setting in one place: model names, token-source paths, the training mixture
  lm.py             loading the language model and capturing one layer's MLP input and output
  heldout.py        reading the held-out activations of a calibration folder
  metrics.py        held-out reconstruction numbers with the trainers' definitions

Earlier design (trainers and tools of an earlier, unpublished set; this code did not train the published weights)
  earlier_design/   token stream, preparation, both trainers, manifest and gate check; see its own docstring
  cli.py            the `statescope` command line of that design; kept here because the console script names statescope.cli:main
  splice.py         next-token loss with that design's dictionaries spliced in; kept here because the manifest of
                    release/qwen35-08b names statescope/splice.py
"""
