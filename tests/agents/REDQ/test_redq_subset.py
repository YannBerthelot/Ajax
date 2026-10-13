"""Unit tests for REDQ's critic step: the random-subset target and the loss.

REDQ's defining mechanic is the min-over-random-subset target:
given an ensemble of N critics, sample ``subset_size`` of them and take
the min Q-value over that subset as the bellman target. This file
exercises ``compute_redq_td_target`` directly at the boundary values
(subset_size = 1 and subset_size = num_critics), and against the whole
ensemble's values indexed afterwards (it runs only the sampled critics).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.recurrent import (
    RecurrentCarries,
    previous_actions,
    q_values,
    stream_positions,
)
from ajax.agents.REDQ.train_REDQ import (
    compute_redq_td_target,
    init_REDQ,
    value_loss_function,
)
from ajax.agents.SAC import core
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig
from ajax.state import AlphaConfig, NetworkConfig, OptimizerConfig

# Measured drift 3.4 eps of the values' scale at most (mamba's carries):
# the subset's matmuls round each critic's value by an ulp of it.
TOL = 16 * float(np.finfo(np.float32).eps)


def _init(env_config, number_of_critics, memory=None):
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["32", "relu"],
        critic_architecture=["32", "relu"],
        squash=True,
        memory=memory,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    return init_REDQ(
        key=jax.random.PRNGKey(0),
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=get_buffer(buffer_size=256, batch_size=16, n_envs=1),
        number_of_critics=number_of_critics,
    )


@pytest.fixture
def REDQ_state(gymnax_env_config):
    return _init(gymnax_env_config, 10)


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


def _jitter(tree, key):
    """``tree`` plus Gaussian noise on its float leaves, its flags set (a
    transformer carry's whole window valid): target critics and carries
    apart from the online ones and from each other."""
    leaves, treedef = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))
    noisy = [
        leaf + 0.5 * jax.random.normal(k, leaf.shape, leaf.dtype)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
        else jnp.ones_like(leaf)
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return treedef.unflatten(noisy)


def _assert_close(new, old):
    scale = float(jnp.abs(old).max())
    np.testing.assert_allclose(new, old, rtol=TOL, atol=TOL * scale)


def _check_against_all_critics(state, next_obs, subset_size, carries=None):
    """``compute_redq_td_target`` against its oracle, REDQ's target as it
    was before it ran only the sampled critics: every target critic's
    value, then the min over the sampled ones."""
    critic = state.critic_state.replace(
        target_params=_jitter(state.critic_state.target_params, jax.random.PRNGKey(5))
    )
    rows = np.arange(np.prod(next_obs.shape[:-1])).reshape(next_obs.shape[:-1] + (1,))
    rewards, dones = jnp.sin(rows), jnp.asarray(rows % 3 == 0, jnp.float32)
    for seed in range(4):
        rng = jax.random.PRNGKey(seed)
        new, _ = compute_redq_td_target(
            state.actor_state, critic, rng, next_obs, dones, rewards, 0.99,
            jnp.array(0.1), subset_size, 1.0, carries,
        )  # fmt: skip
        sample_key, idx_key = jax.random.split(rng)
        next_actions, log_probs = core.sample_next_actions(
            state.actor_state, next_obs, sample_key, carries
        )
        q = q_values(
            critic, critic.target_params, next_obs, next_actions, carries,
            bootstrap=True,
        )  # fmt: skip
        idx = jax.random.choice(idx_key, q.shape[0], (subset_size,), replace=False)
        min_q = jnp.min(q[idx], axis=0)
        _assert_close(new, rewards + 0.99 * (1.0 - dones) * (min_q - 0.1 * log_probs))


@pytest.mark.parametrize("subset_size", [1, 2, 10])
def test_the_sampled_critics_alone_give_the_whole_ensembles_subset_min(
    gymnax_env_config, REDQ_state, subset_size
):
    """Running only the sampled target critics gives the min over the whole
    ensemble's values at the same sampled indexes, to float rounding."""
    obs_shape, _ = get_state_action_shapes(gymnax_env_config.env)
    next_obs = jax.random.normal(jax.random.PRNGKey(3), (16, *obs_shape))
    _check_against_all_critics(REDQ_state, next_obs, subset_size)


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_a_recurrent_ensemble_runs_its_sampled_critics_on_their_carries(
    gymnax_env_config, kind
):
    """With memory each sampled critic runs from its own carry: the target
    and the online values match the whole ensemble's at those indexes."""
    memory = MemoryConfig(kind=kind, hidden_size=8, window=8)
    state = _init(gymnax_env_config, 4, memory)
    critic = state.critic_state
    keys = jax.random.split(jax.random.PRNGKey(7), 5)
    # Mid-episode at the window's start: the burned-in carries are read.
    resets = jnp.zeros((4, 1), bool).at[2].set(True)
    actions = jax.random.uniform(keys[0], (4, 1, 1), minval=-1)
    carries = RecurrentCarries(
        resets=resets,
        next_resets=jnp.zeros((4, 1), bool).at[1].set(True),
        actor_hidden=state.actor_state.hidden_state,
        critic_hidden=_jitter(critic.hidden_state, keys[1]),
        target_critic_hidden=_jitter(critic.hidden_state, keys[2]),
        obs=jax.random.normal(keys[3], (4, 1, 3)),
        actions=actions,
        prev_actions=previous_actions(actions, resets),
        positions=stream_positions(jnp.zeros((4, 1), jnp.int32)),
    )
    for subset_size in (1, 2, 4):
        next_obs = jax.random.normal(keys[4], (4, 1, 3))
        _check_against_all_critics(state, next_obs, subset_size, carries)
    members = jnp.array([3, 1])
    online = q_values(critic, critic.params, carries.obs, actions, carries)
    subset = q_values(
        critic, critic.params, carries.obs, actions, carries, members=members
    )
    _assert_close(subset, online[members])
