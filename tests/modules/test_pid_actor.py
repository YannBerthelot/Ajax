"""Tests for ajax.modules.pid_actor — PIDActorNetwork and PIDActorConfig."""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.SAC.utils import SquashedNormal
from ajax.modules.pid_actor import PIDActorConfig, PIDActorNetwork


def _build_network(action_dim=2, derivative_idx=None, penultimate_normalization=False):
    return PIDActorNetwork(
        input_architecture=("16", "relu"),
        action_dim=action_dim,
        obs_current_idx=0,
        obs_target_idx=1,
        obs_derivative_idx=derivative_idx,
        penultimate_normalization=penultimate_normalization,
    )


@pytest.mark.parametrize("derivative_idx,expected_terms", [(None, 1), (2, 2)])
def test_pid_actor_n_terms(derivative_idx, expected_terms):
    net = _build_network(derivative_idx=derivative_idx)
    assert net.n_terms == expected_terms


def test_pid_actor_returns_squashed_normal():
    net = _build_network(derivative_idx=2)
    obs = jnp.array([[0.1, 0.5, 0.0, 0.2]])
    params = net.init(jax.random.PRNGKey(0), obs)
    dist = net.apply(params, obs)
    assert isinstance(dist, SquashedNormal)


def test_pid_actor_zero_init_gives_zero_mean():
    # With zero-initialized gains, the pre-squash mean must be exactly zero
    # regardless of the PID error terms.
    net = _build_network(action_dim=3, derivative_idx=2)
    obs = jnp.array([[0.1, 0.5, 0.3, 0.9]])
    params = net.init(jax.random.PRNGKey(0), obs)
    dist = net.apply(params, obs)
    assert jnp.allclose(dist.distribution.loc, jnp.zeros((1, 3)))


def test_pid_actor_log_std_bounded_initially():
    net = _build_network(action_dim=2)
    obs = jnp.array([[0.1, 0.5]])
    params = net.init(jax.random.PRNGKey(0), obs)
    dist = net.apply(params, obs)
    # log_std is zero-init + constant(-1.0) bias → std = exp(-1).
    assert jnp.allclose(dist.distribution.scale, jnp.exp(jnp.array(-1.0)))


def test_pid_actor_handles_batched_input():
    net = _build_network(action_dim=2, derivative_idx=2)
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    params = net.init(jax.random.PRNGKey(0), obs)
    dist = net.apply(params, obs)
    assert dist.distribution.loc.shape == (5, 2)
    assert dist.distribution.scale.shape == (5, 2)


def test_pid_actor_penultimate_normalization_runs():
    net = _build_network(
        action_dim=2, derivative_idx=None, penultimate_normalization=True
    )
    obs = jnp.ones((1, 4))
    params = net.init(jax.random.PRNGKey(0), obs)
    dist = net.apply(params, obs)
    assert dist.distribution.loc.shape == (1, 2)


def test_pid_actor_config_defaults():
    cfg = PIDActorConfig(obs_current_idx=2, obs_target_idx=5)
    assert cfg.obs_derivative_idx is None
    assert cfg.obs_current_idx == 2
    assert cfg.obs_target_idx == 5


def test_pid_actor_gains_contribute_when_non_zero():
    """After manually perturbing the gains, the mean responds to obs error."""
    net = _build_network(action_dim=1, derivative_idx=None)
    obs = jnp.array([[0.0, 2.0]])  # error = target - current = 2.0
    params = net.init(jax.random.PRNGKey(0), obs)
    mutable = jax.tree_util.tree_map(lambda x: x, params)
    gains_params = mutable["params"]["gains"]
    mutable["params"]["gains"] = {
        "kernel": jnp.ones_like(gains_params["kernel"]),
        "bias": jnp.ones_like(gains_params["bias"]),
    }
    dist_perturbed = net.apply(mutable, obs)
    # Non-zero gains must yield a non-zero pre-squash mean.
    assert not jnp.allclose(dist_perturbed.distribution.loc, 0.0)
