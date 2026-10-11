"""Unit tests for REDQ's random-subset target computation.

REDQ's defining mechanic is the min-over-random-subset target:
given an ensemble of N critics, sample ``subset_size`` of them and take
the min Q-value over that subset as the bellman target. This file
exercises ``compute_redq_td_target`` directly at the boundary values
(subset_size = 1 and subset_size = num_critics).
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.REDQ.train_REDQ import compute_redq_td_target, init_REDQ
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import get_state_action_shapes
from ajax.state import AlphaConfig, NetworkConfig, OptimizerConfig


@pytest.fixture
def REDQ_state(gymnax_env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["32", "relu"],
        critic_architecture=["32", "relu"],
        squash=True,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    return init_REDQ(
        key=key,
        env_args=gymnax_env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=get_buffer(buffer_size=256, batch_size=16, n_envs=1),
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
        subset_size=subset_size,
        reward_scale=1.0,
    )
    return target


def test_the_whole_ensembles_target_is_finite_and_lowest(gymnax_env_config, REDQ_state):
    """At one key (one next action) subset_size=1 takes one critic's value,
    subset_size=num_critics the min over all ten: both finite, the whole
    ensemble's at most the single critic's, and (on a random init) apart.
    ``not allclose`` alone would pass on NaN, hence the finiteness."""
    t_one = _call_target(REDQ_state, gymnax_env_config, subset_size=1)
    t_full = _call_target(REDQ_state, gymnax_env_config, subset_size=10)
    assert jnp.all(jnp.isfinite(t_one)) and jnp.all(jnp.isfinite(t_full))
    assert jnp.all(t_full <= t_one)
    assert not jnp.allclose(t_one, t_full)
