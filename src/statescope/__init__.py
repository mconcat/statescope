"""statescope: per-layer dictionaries of Qwen3.5 (a state dictionary of the residual stream, an MLP dictionary of transforms).

This file imports nothing; import the module you need.

  reader.py     the reader of the published dictionaries. One self-contained file that imports nothing else from the package, so it
                can be copied next to the weights and used alone.
  config.py     the models the dictionaries exist for
  lm.py         loading the language model
  heldout.py    reading the held-out activations of a calibration folder (used by the measuring tools and the tests)
  metrics.py    the state dictionary's held-out reconstruction number with its trainer's definition
"""
