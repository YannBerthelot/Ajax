"""Unit tests for REDQ's random-subset target computation.

REDQ's defining mechanic is the min-over-random-subset target:
given an ensemble of N critics, sample ``subset_size`` of them and take
the min Q-value over that subset as the bellman target. This file
exercises ``compute_redq_td_target`` directly at the boundary values
(subset_size = 1 and subset_size = num_critics).
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest

from ajax.agents.REDQ.train_REDQ import compute_redq_td_target, init_REDQ
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import get_state_action_shapes
from ajax.state import (
    AlphaConfig,
    BufferConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def env_config():
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def REDQ_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["32", "relu"],
        critic_architecture=["32", "relu"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    buffer = get_buffer(
        **{
            **BufferConfig(
                buffer_size=256, batch_size=16, n_envs=env_config.n_envs
            ).__dict__
        }
    )
    return init_REDQ(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
        number_of_critics=10,
    )


def _call_target(REDQ_state, env_config, subset_size, seed=1):
    obs_shape, _ = get_state_action_shapes(env_config.env)
    rng = jax.random.PRNGKey(seed)
    next_obs = jnp.zeros((env_config.n_envs, *obs_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    rewards = jnp.ones((env_config.n_envs, 1))

    target, _ = compute_redq_td_target(
        actor_state=REDQ_state.actor_state,
        critic_states=REDQ_state.critic_state,
        rng=rng,
        next_observations=next_obs,
        dones=dones,
        rewards=rewards,
        gamma=0.99,
        alpha=jnp.array(0.1),
        recurrent=False,
        subset_size=subset_size,
        reward_scale=1.0,
    )
    return target


def test_subset_size_one_is_finite(env_config, REDQ_state):
    target = _call_target(REDQ_state, env_config, subset_size=1)
    assert jnp.all(jnp.isfinite(target))


def test_subset_size_full_ensemble_is_finite(env_config, REDQ_state):
    target = _call_target(REDQ_state, env_config, subset_size=10)
    assert jnp.all(jnp.isfinite(target))


def test_different_subset_sizes_produce_different_targets(env_config, REDQ_state):
    """subset_size=1 picks a single Q; subset_size=num_critics picks min
    over all. These should (in general) differ."""
    t_one = _call_target(REDQ_state, env_config, subset_size=1, seed=1)
    t_full = _call_target(REDQ_state, env_config, subset_size=10, seed=1)
    # Not a strict inequality — equality is possible if all Q-values
    # happen to match — but on random init they should differ.
    assert not jnp.allclose(t_one, t_full)
