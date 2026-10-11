"""Parity of Ajax's DreamerV3 world model with the reference code itself.

The fixtures were produced by the real ``danijar/dreamerv3@29eb964`` (the
2411f7d algorithm plus the upstream replay-context fix) with
``docs/world_models/parity/dreamerv3_world_model_fixtures.py``: a tiny float32
world model (vector obs 5, deter 64, RSSM width 12, 4 latents x 3 classes, 8
blocks, MLP width 16) with every other hyperparameter at the reference's
default, one ``Agent.train`` call on a ``[2, 9]`` replay batch with episode
boundaries, three runs: continuous action 2 and discrete action 3 at the
reference's initial parameters, and the continuous run at perturbed
parameters, where the reward head is no longer zero, so that reward gradients
reach its hidden layer, the RSSM and the encoder.

Ajax runs with its own defaults at the fixture's widths, and
:func:`test_default_hyperparameters_are_the_references` pins the world
model's against the reference's (test_dreamerv3_train_parity.py pins the
rest, and that the mapped parameter trees match Ajax's). Each other test
maps the reference's parameters onto Ajax's tree (:func:`reference_to_ajax`; dreamerv3_spec 2.17), runs
:func:`world_model_loss` with the Gumbel noise the reference drew, and
compares the samples (exactly), the posterior entries, the loss terms, the
KL and the gradient of the weighted world-model loss with respect to every
parameter. The achieved errors are printed (``pytest -s``) and asserted
below the bounds in :data:`TOLERANCES`.
"""

from __future__ import annotations

import functools
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.networks import (
    POSTERIOR_LAYERS,
    PRIOR_LAYERS,
    WorldModel,
)
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.world_model import LOSS_TERMS, world_model_loss

from .common import (
    FIXTURES,
    REFERENCE_TERMS,
    fixture_batch,
    fixture_config,
    get_path,
    parity_error,
    reference_to_ajax,
    reference_trees,
)

RUNS = ("continuous_init", "discrete_init", "continuous_perturbed")
#: Bounds on ``max |ajax - ref| / (1 + max |ref|)`` per compared quantity.
TOLERANCES = {
    "values": 1e-6,  # deter, logits, tokens, losses, KL
    "grads": 1e-5,  # per parameter tensor, relative to its largest entry
}


@functools.lru_cache(maxsize=None)
def load(run: str) -> dict:
    """Fixture arrays of ``run``, split by prefix."""
    with np.load(FIXTURES / f"dreamerv3_wm_{run}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    out: dict = {"meta": meta}
    for key, value in arrays.items():
        group, name = key.split("/", 1)
        out.setdefault(group, {})[name] = value
    return out


@functools.lru_cache(maxsize=None)
def ajax_run(run: str):
    """Ajax's world-model loss and gradients on the fixture of ``run``."""
    fixture = load(run)
    # Only the sizes come from the fixture (the MLP width read off the
    # reference's encoder); every other field keeps Ajax's default, pinned
    # by test_default_hyperparameters_are_the_references.
    units = fixture["param"]["agent/enc/mlp0/kernel"].shape[-1]
    config = fixture_config(fixture["meta"]["rssm"], units)
    discrete = run.startswith("discrete")
    raw = fixture["batch"]
    # The reference keeps the context row's stored latent only
    # (agent.py:175-176, ``data.pop(k)[:, :K]``).
    batch = fixture_batch(
        raw, 3 if discrete else None, raw["deter"][:, 0], raw["stoch"][:, 0]
    )
    model = WorldModel(config, obs_dim=batch.obs.shape[-1])
    params = reference_trees(
        fixture["param"], discrete, config.bins, dropped_must_be_zero=False
    )["world_model"]
    noise = jnp.asarray(fixture["out"]["noise"])

    @jax.jit
    def loss_and_grad(params):
        def weighted(params):
            out = world_model_loss(model, params, batch, noise)
            return out.weighted, out

        return jax.value_and_grad(weighted, has_aux=True)(params)

    with jax.default_matmul_precision("highest"):
        (_, out), grads = loss_and_grad(params)
    return fixture, config, model, params, jax.device_get(out), jax.device_get(grads)


def _layers(params: dict[str, np.ndarray], module: str) -> int:
    """Number of hidden layers of the reference ``module``."""
    pattern = rf"agent/{module}/(mlp|h)\d+/kernel"
    return sum(bool(re.fullmatch(pattern, name)) for name in params)


def test_default_hyperparameters_are_the_references():
    """Ajax's world-model defaults equal the reference's (2411f7d
    ``configs.yaml``; the fixtures override widths only): latent unimix
    1 %, free bits 1 nat, the head depths and the RSSM's prior and
    posterior layers. The defaults the whole training step shares (the
    return horizon, the loss scales, 255 bins) are pinned by
    test_dreamerv3_train_parity.py; they do not depend on the run."""
    fixture = load("continuous_init")
    meta, params = fixture["meta"], fixture["param"]
    ours = DreamerV3Config()
    assert ours.unimix == meta["rssm"]["unimix"] == 0.01
    assert ours.free_nats == meta["free"] == 1.0
    depths = (ours.enc_layers, ours.dec_layers, ours.rew_layers, ours.con_layers)
    assert depths == tuple(meta["layers"][k] for k in ("enc", "dec", "rew", "con"))
    assert depths == tuple(_layers(params, k) for k in ("enc", "dec", "rew", "con"))
    assert (meta["rssm"]["imglayers"], meta["rssm"]["obslayers"]) == (
        PRIOR_LAYERS,
        POSTERIOR_LAYERS,
    )


@pytest.mark.parametrize("run", RUNS)
def test_samples_and_entries_match(run):
    """The forced posterior samples equal the reference's draws exactly, and
    the write-back entries, logits and tokens match."""
    fixture, config, _, _, out, _ = ajax_run(run)
    ref = fixture["out"]
    # No near-tie among the draws (smallest top-2 gap 2.9e-3 in these
    # fixtures, ~1e3 times the logits' error), so the exact match does not
    # hinge on float rounding.
    scores = np.sort(ref["post_logit"] + ref["noise"], -1)
    assert np.min(scores[..., -1] - scores[..., -2]) > 1e-3
    np.testing.assert_array_equal(out.entries.stoch, ref["sample"])
    np.testing.assert_array_equal(out.entries.stoch, ref["entry_stoch"])
    one_hot = np.asarray(out.posterior.stoch)
    np.testing.assert_array_equal(one_hot, np.eye(config.classes)[ref["sample"]])
    mixed = lambda logits: np.log(  # noqa: E731
        (1 - config.unimix) * np.asarray(jax.nn.softmax(logits, -1))
        + config.unimix / config.classes
    )
    errors = {
        "deter": parity_error(out.entries.deter, ref["deter"]),
        "post_logit": parity_error(mixed(out.posterior.logits), ref["post_logit"]),
        "prior_logit": parity_error(mixed(out.prior_logits), ref["prior_logit"]),
        "tokens": parity_error(out.tokens, ref["embed"]),
    }
    print(f"\n{run} values: " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))
    assert max(errors.values()) < TOLERANCES["values"], errors


@pytest.mark.parametrize("run", RUNS)
def test_loss_terms_and_kl_match(run):
    fixture, _, _, _, out, _ = ajax_run(run)
    errors = {
        term: parity_error(out.losses[term], fixture["loss"][REFERENCE_TERMS[term]])
        for term in LOSS_TERMS
    }
    errors["kl"] = parity_error(out.kl, fixture["out"]["kl"])
    errors["weighted"] = parity_error(out.weighted, fixture["loss"]["weighted"])
    print(f"\n{run} losses: " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))
    assert max(errors.values()) < TOLERANCES["values"], errors
    for term in LOSS_TERMS:
        assert out.losses[term].shape == fixture["loss"]["vector"].shape


@pytest.mark.parametrize("run", RUNS)
def test_gradients_match(run):
    """The gradient of the weighted world-model loss, parameter by parameter.

    The error of each tensor is relative to its largest reference entry
    (``max |g_ajax - g_ref| / max |g_ref|``, or absolute where the reference
    gradient is 0). The reference's dropped 256th reward logit gets no
    gradient.
    """
    fixture, config, _, _, _, grads = ajax_run(run)
    errors = {}
    for name, theirs in fixture["grad"].items():
        path = reference_to_ajax(name)
        if path[:2] == ("rew", "out"):
            assert not np.any(theirs[..., config.bins :]), name
            theirs = theirs[..., : config.bins]
        ours = np.asarray(get_path(grads, path), np.float64)
        scale = np.max(np.abs(theirs))
        errors[name] = float(np.max(np.abs(ours - theirs)) / (scale if scale else 1.0))
    worst = max(errors, key=errors.__getitem__)
    nonzero = sum(bool(np.any(g)) for g in fixture["grad"].values())
    print(
        f"\n{run} grads: {len(errors)} tensors ({nonzero} non-zero), "
        f"max relative error {errors[worst]:.1e} ({worst})"
    )
    assert errors[worst] < TOLERANCES["grads"], (worst, errors[worst])
