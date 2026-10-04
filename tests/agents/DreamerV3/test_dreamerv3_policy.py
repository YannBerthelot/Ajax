"""The acting step (:func:`policy_step`; dreamerv3_spec 7.1, 7.3).

``Agent.policy`` (``29eb964:dreamerv3/agent.py:129-164``): one posterior
filter step from the carried ``(h, z)`` and previous action, reset where
``is_first`` (``:136-137``); then a **sample** of the actor in every mode;
the carry keeps the raw sample (``:164``) and the replay the posterior just
computed (``:142-144``). Small networks, no training: a few seconds.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.networks import init_actor, init_world_model
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.train_DreamerV3 import (
    PolicyCarry,
    initial_policy_carry,
    policy_step,
)

from .common import OBS_DIM, TINY, perturbed

#: One class per latent: every posterior sample is the same one-hot, so
#: identical observations after a reset give identical features in all envs.
ONE_CLASS = dataclasses.replace(TINY, classes=1)
#: Discrete: 2 actions; continuous: 2 dimensions.
ACTIONS = 2


def params(config: DreamerV3Config, discrete: bool, seed: int = 0):
    """Perturbed world-model and actor parameters (no degenerate layer: one
    observe step from a reset gives a non-zero ``deter``)."""
    wm_key, actor_key = jax.random.split(jax.random.PRNGKey(seed))
    world_model = init_world_model(wm_key, config, OBS_DIM, ACTIONS)
    actor = init_actor(actor_key, config, ACTIONS, discrete)
    return perturbed(world_model, seed + 1), perturbed(actor, seed + 2)


_policy_step = jax.jit(
    policy_step, static_argnames=("config", "action_dim", "discrete")
)


def act(config, discrete, carry, obs, is_first, key, seed=0):
    world_model, actor = params(config, discrete, seed)
    return _policy_step(
        world_model,
        actor,
        carry,
        obs,
        is_first,
        key,
        config=config,
        action_dim=ACTIONS,
        discrete=discrete,
    )


def noisy_carry(config: DreamerV3Config, n: int, discrete: bool) -> PolicyCarry:
    """A carry far from the initial (zero) one."""
    rng = np.random.default_rng(0)
    zero = initial_policy_carry(config, n, ACTIONS, discrete)
    classes = rng.integers(0, config.classes, zero.stoch.shape[:-1])
    prevact = (
        jnp.asarray(rng.integers(0, ACTIONS, n), jnp.int32)
        if discrete
        else jnp.asarray(rng.normal(0.0, 1.5, (n, ACTIONS)), jnp.float32)
    )
    return PolicyCarry(
        deter=jnp.asarray(rng.normal(size=zero.deter.shape), jnp.float32),
        stoch=jax.nn.one_hot(classes, config.classes),
        prevact=prevact,
    )


@pytest.mark.parametrize("discrete", [True, False])
def test_the_policy_samples_the_actor(discrete):
    """Identical features in 256 envs: the actions still vary across envs
    (the reference samples in every mode, dreamerv3_spec 7.3)."""
    n = 256
    carry = initial_policy_carry(ONE_CLASS, n, ACTIONS, discrete)
    obs = jnp.ones((n, OBS_DIM))
    action, new, _ = act(
        ONE_CLASS, discrete, carry, obs, jnp.ones(n, bool), jax.random.PRNGKey(1)
    )
    np.testing.assert_array_equal(
        new.deter, np.broadcast_to(new.deter[0], new.deter.shape)
    )
    action = np.asarray(action)
    if discrete:
        assert action.dtype == np.int32 and action.shape == (n,)
        assert 0.05 < action.mean() < 0.95, action.mean()
    else:
        assert action.shape == (n, ACTIONS)
        assert np.all(action.std(0) > 0.2), action.std(0)


@pytest.mark.parametrize("discrete", [True, False])
def test_is_first_resets_the_carry(discrete):
    """Where ``is_first``, the step starts from the initial carry (``deter``,
    ``stoch`` and the previous action zeroed), whatever is carried."""
    n = 64
    obs = jnp.asarray(np.random.default_rng(1).normal(size=(n, OBS_DIM)), jnp.float32)
    key = jax.random.PRNGKey(3)
    zero = initial_policy_carry(TINY, n, ACTIONS, discrete)
    noisy = noisy_carry(TINY, n, discrete)
    first = jnp.ones(n, bool)
    from_zero = act(TINY, discrete, zero, obs, first, key)
    from_noisy = act(TINY, discrete, noisy, obs, first, key)
    for ours, theirs in zip(jax.tree.leaves(from_noisy), jax.tree.leaves(from_zero)):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)
    # Without is_first the carry is used.
    carried = act(TINY, discrete, noisy, obs, jnp.zeros(n, bool), key)
    assert not np.allclose(carried[1].deter, from_zero[1].deter)


@pytest.mark.parametrize("discrete", [True, False])
def test_the_carry_keeps_the_raw_sample_and_the_replay_the_new_posterior(discrete):
    """The next previous action is the action exactly as sampled (unclipped
    for a continuous actor, ``29eb964:dreamerv3/agent.py:164``); the latent
    for the replay is the posterior of this step -- the new carry's -- as
    class indices, not the carried one."""
    n = 64
    obs = jnp.asarray(np.random.default_rng(2).normal(size=(n, OBS_DIM)), jnp.float32)
    carry = noisy_carry(TINY, n, discrete)
    action, new, (deter, stoch) = act(
        TINY, discrete, carry, obs, jnp.zeros(n, bool), jax.random.PRNGKey(4)
    )
    if not discrete:
        assert bool((jnp.abs(action) > 1).any())  # samples beyond the bounds
    np.testing.assert_array_equal(new.prevact, action)
    assert new.prevact.dtype == carry.prevact.dtype
    np.testing.assert_array_equal(deter, new.deter)
    np.testing.assert_array_equal(stoch, jnp.argmax(new.stoch, -1))
    assert stoch.dtype == jnp.int32 and stoch.shape == (n, TINY.stoch)
    assert not np.allclose(deter, carry.deter)
    assert not np.array_equal(stoch, jnp.argmax(carry.stoch, -1))
