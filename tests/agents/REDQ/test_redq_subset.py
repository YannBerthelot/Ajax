"""Unit tests for REDQ's critic step: the random-subset target and the loss.

REDQ's defining mechanic is the min-over-random-subset target:
given an ensemble of N critics, sample ``subset_size`` of them and take
the min Q-value over that subset as the bellman target. This file
exercises ``compute_redq_td_target`` directly at the boundary values
(subset_size = 1 and subset_size = num_critics).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.recurrent import q_values
from ajax.agents.REDQ.train_REDQ import (
    compute_redq_td_target,
    init_REDQ,
    value_loss_function,
)
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


def test_the_critic_loss_sums_each_critics_mean_squared_error(
    gymnax_env_config, REDQ_state
):
    """REDQ's official ``mse_criterion(q_prediction_cat, y_q) * num_Q``
    (watchernyu/REDQ ``redq/algos/redq_sac.py:207``): the sum over the ten
    critics of each one's batch-mean squared error, with no other scale."""
    obs_shape, action_shape = get_state_action_shapes(gymnax_env_config.env)
    obs_key, action_key = jax.random.split(jax.random.PRNGKey(0))
    obs = jax.random.normal(obs_key, (16, *obs_shape))
    actions = jax.random.uniform(action_key, (16, *action_shape), minval=-1)
    target = jnp.linspace(-1.0, 1.0, 16)[:, None]
    critic = REDQ_state.critic_state
    loss, _ = value_loss_function(
        critic.params, critic, obs, actions, target, jnp.zeros((16, 1))
    )
    q = np.asarray(q_values(critic, critic.params, obs, actions, None))
    assert q.shape == (10, 16, 1)
    want = sum(np.mean((q[i] - np.asarray(target)) ** 2) for i in range(10))
    np.testing.assert_allclose(loss, want, rtol=1e-6)
