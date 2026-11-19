"""Formula-level tests of the state-dictionary trainer on a tiny dictionary: the balancing pass, the learning-rate schedule, and that
an export is read by the released reader. No checkpoint, no language model, no GPU."""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file

from statescope.reader import StateVocab
from statescope.earlier_design.vocab.model import StateDictionary, balance_encoder_decoder
from statescope.earlier_design.vocab.recipe import VocabRecipe
from statescope.earlier_design.vocab.train import VocabTrainer

D_MODEL = 32
SITES = ["resid_mid.3", "resid_post.3"]
TINY = VocabRecipe(groups=48, group_size=4, active_groups=6, balance_every=2, dead_window_tokens=300)


def test_balancing_keeps_each_groups_map_and_equalizes_the_grams():
    g = torch.Generator().manual_seed(0)
    E, D = torch.randn(40, 4, D_MODEL, generator=g), 3.0 * torch.randn(40, 4, D_MODEL, generator=g)
    maps_before = E.double().transpose(1, 2) @ D.double()                    # (G, d, d): E_g^T D_g
    balance_encoder_decoder(E, D, chunk=16)
    maps_after = E.double().transpose(1, 2) @ D.double()
    assert torch.allclose(maps_after, maps_before, rtol=0, atol=1e-4 * float(maps_before.abs().max()))
    gram_e, gram_d = E.double() @ E.double().transpose(1, 2), D.double() @ D.double().transpose(1, 2)
    assert torch.allclose(gram_e, gram_d, rtol=0, atol=1e-4 * float(gram_d.abs().max()))


def test_learning_rate_is_constant_then_linear_to_the_end_factor():
    recipe = VocabRecipe()
    factors = [recipe.lr_factor(step, 100) for step in range(101)]
    assert factors[:81] == [1.0] * 81                                        # the decay runs over the last 20 of 100 steps, from step 80
    assert all(a > b for a, b in zip(factors[80:100], factors[81:101]))
    assert abs(factors[100] - recipe.lr_end_factor) < 1e-12
    assert recipe.lr_factor(10_000, 100) == factors[100]                     # past the schedule's end the factor stays


def _trained_tiny(steps: int = 5) -> VocabTrainer:
    g = torch.Generator().manual_seed(1)
    model = StateDictionary(D_MODEL, SITES, TINY, "cpu")
    model.site_mu.copy_(0.1 * torch.randn(2, D_MODEL, generator=g))
    model.site_scale.copy_(0.5 + torch.rand(2, generator=g))
    trainer = VocabTrainer(model, schedule_steps=steps)
    for _ in range(steps):
        trainer.step({site: torch.randn(200, D_MODEL, generator=g) for site in SITES})
    return trainer


def test_dead_groups_get_the_auxiliary_loss():
    trainer = _trained_tiny(steps=1)
    trainer.since[::5] = TINY.dead_window_tokens + 1
    g = torch.Generator().manual_seed(2)
    record = trainer.step({site: torch.randn(200, D_MODEL, generator=g) for site in SITES})
    assert record["n_dead"] == len(range(0, TINY.groups, 5)) and record["L_aux"] > 0.0


def test_export_is_read_by_the_released_reader(tmp_path):
    trainer = _trained_tiny()
    model = trainer.model
    trainer.export(str(tmp_path), trainer.cfg({"n_tokens_per_site": 0, "site_scale": {}, "sha256": ""}, batch=200))
    cfg = json.loads((tmp_path / "cfg.json").read_text())
    assert (cfg["n_units"], cfg["unit_size"], cfg["k_topk"], cfg["steps"], cfg["tokens"]) == (48, 4, 6, 5, 1000)
    usage = load_file(str(tmp_path / "usage.safetensors"))
    assert usage["fire"].shape == (48, 2) and int(usage["n_tokens"][0]) == 1000 and int(usage["fire"][:, 0].sum()) == 1000 * 6

    vocab = StateVocab(str(tmp_path))
    x = torch.randn(64, D_MODEL, generator=torch.Generator().manual_seed(3))
    Q, R = model.write_basis()
    for j, site in enumerate(SITES):
        xs = model.standardize(x, j)
        selected = model.select(model.written_norm(model.encode(xs - model.b_dec).float()))[0]          # (n, G) bool
        groups, codes = vocab.encode(x, site)
        assert torch.equal(torch.zeros_like(selected).scatter_(-1, groups, True), selected)             # the same group sets
        trained_form = model.site_mu[j] + model.site_scale[j] * model.reconstruct(xs, Q, R)
        assert torch.allclose(vocab.decode(groups, codes, site), trained_form, rtol=0, atol=1e-4 * float(x.abs().max()))
