"""Trainer of the state dictionary ("vocabulary") of one layer: groups of directions that reconstruct the residual stream at the two
sites of the layer, resid_mid.L (the MLP's input) and resid_post.L (the layer's output).

  recipe.py      every hyperparameter (VocabRecipe); the defaults are the released dictionaries
  model.py       the dictionary: parameters, initialisation, standardization, selection, losses, balancing
  calibrate.py   per-site mean and scale from the calibration sequences
  evaluate.py    held-out FVU
  export.py      the released file format (cfg.json, weights.safetensors, usage.safetensors)
  train.py       one training step (VocabTrainer) and the whole run (train_vocab)

The exported folder is read by statescope.reader.StateVocab.
"""
