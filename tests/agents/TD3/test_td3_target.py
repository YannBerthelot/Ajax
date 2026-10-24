"""Unit tests for TD3's target policy smoothing.

TD3 differs from DDPG by:
  1. Using twin critics with min aggregation in the target.
  2. Adding clipped Gaussian noise to the target action (smoothing).
  3. Taking an action via the deterministic actor mean (no entropy term).

This file pins those properties at the function-level boundary.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.TD3.train_TD3 import compute_td3_td_target, init_TD3
from ajax.buffers.utils import get_buffer
from ajax.environments.interaction import get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.distributions import Deterministic
from ajax.state import NetworkConfig, OptimizerConfig


@pytest.fixture
def td3_state(gymnax_env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["32", "relu"],
        critic_architecture=["32", "relu"],
        squash=True,
    )
    return init_TD3(
        key=key,
        env_args=gymnax_env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        buffer=get_buffer(buffer_size=256, batch_size=16, n_envs=1),
        num_critics=2,
    )


def _call_target(
    td3_state,
    env_config,
    target_policy_noise,
    target_noise_clip,
    seed=1,
    done=0.0,
    reward=1.0,
):
    obs_shape, _ = get_state_action_shapes(env_config.env)
    return compute_td3_td_target(
        actor_state=td3_state.actor_state,
        critic_state=td3_state.critic_state,
        rng=jax.random.PRNGKey(seed),
        next_observations=jnp.zeros((env_config.n_envs, *obs_shape)),
        dones=jnp.full((env_config.n_envs, 1), done),
        rewards=jnp.full((env_config.n_envs, 1), reward),
        gamma=0.99,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        reward_scale=1.0,
    )[0]


def test_target_finite(gymnax_env_config, td3_state):
    target = _call_target(td3_state, gymnax_env_config, 0.2, 0.5)
    assert jnp.all(jnp.isfinite(target))


def test_target_smoothing_changes_target(gymnax_env_config, td3_state):
    """Non-zero target noise should produce a different target than zero noise."""
    no_noise = _call_target(td3_state, gymnax_env_config, 0.0, 0.0, seed=1)
    with_noise = _call_target(td3_state, gymnax_env_config, 0.5, 0.5, seed=1)
    assert not jnp.allclose(no_noise, with_noise)


def test_actor_is_deterministic(gymnax_env_config, td3_state):
    """Two forward passes on the same obs (different keys) must agree."""
    obs_shape, _ = get_state_action_shapes(gymnax_env_config.env)
    obs = jnp.zeros((gymnax_env_config.n_envs, *obs_shape))
    pi1, _ = get_pi(td3_state.actor_state, td3_state.actor_state.params, obs)
    pi2, _ = get_pi(td3_state.actor_state, td3_state.actor_state.params, obs)
    assert isinstance(pi1, Deterministic)
    assert jnp.array_equal(pi1.mean(), pi2.mean())
    # tanh-bounded
    a = pi1.mean()
    assert jnp.all(a >= -1.0) and jnp.all(a <= 1.0)


def test_terminal_state_has_no_bootstrap(gymnax_env_config, td3_state):
    """When done=1, target = r * reward_scale (gamma·(1-d) zeroes bootstrap)."""
    target = _call_target(
        td3_state, gymnax_env_config, 0.2, 0.5, seed=0, done=1.0, reward=2.5
    )
    assert jnp.allclose(target, 2.5)
