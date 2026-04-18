"""AVG-specific training tests.

Tests the unique AVG mechanic: the running-average value updates that
AVG uses instead of a target network (``update_AVG_values``). Shared
behaviors (loss shapes, updates, training loop, make_train) are covered
by the probing suite and the smoke test in ``test_AVG.py``.
"""
import gymnax
import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env

from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import init_AVG, update_AVG_values
from ajax.environments.interaction import Transition
from ajax.environments.utils import get_state_action_shapes
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(
        env=env,
        env_params=None,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def gymnax_env_config():
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture(params=["fast_env_config", "gymnax_env_config"])
def env_config(request, fast_env_config, gymnax_env_config):
    return fast_env_config if request.param == "fast_env_config" else gymnax_env_config


@pytest.fixture
def avg_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
        penultimate_normalization=True,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)

    avg_state = init_AVG(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
    )
    obs_shape, action_shape = get_state_action_shapes(env_config.env)
    transition = Transition(
        obs=jnp.ones((env_config.n_envs, *obs_shape)),
        action=jnp.ones((env_config.n_envs, *action_shape)),
        next_obs=jnp.ones((env_config.n_envs, *obs_shape)),
        reward=jnp.ones((env_config.n_envs, 1)),
        terminated=jnp.ones((env_config.n_envs, 1)),
        truncated=jnp.ones((env_config.n_envs, 1)),
        log_prob=jnp.ones((env_config.n_envs, *action_shape)),
    )
    collector_state = avg_state.collector_state.replace(rollout=transition)
    avg_state = avg_state.replace(collector_state=collector_state)
    return avg_state


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_AVG_values(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    rollout = Transition(
        obs=jnp.ones((env_config.n_envs, *observation_shape)),
        action=jnp.ones((env_config.n_envs, *action_shape)),
        next_obs=jnp.ones((env_config.n_envs, *observation_shape)),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[0.0]]),
        truncated=jnp.array([[0.0]]),
        log_prob=jnp.array([[-1.0]]),
    )
    agent_config = AVGConfig(gamma=0.99, target_entropy=-1.0)

    updated_state = update_AVG_values(avg_state, rollout, agent_config)
    log_alpha = updated_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    assert updated_state.reward.count[0] > avg_state.reward.count[0]
    assert updated_state.gamma.count[0] > avg_state.gamma.count[0]
    assert not (updated_state.G_return.count[0] > avg_state.G_return.count[0])
    assert jnp.allclose(updated_state.reward.mean, jnp.array([[1.0 - alpha * -1]]))
    assert jnp.allclose(updated_state.gamma.mean, jnp.array([[0.99]]))


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_AVG_values_terminal(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    rollout = Transition(
        obs=jnp.ones((env_config.n_envs, *observation_shape)),
        action=jnp.ones((env_config.n_envs, *action_shape)),
        next_obs=jnp.ones((env_config.n_envs, *observation_shape)),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[1.0]]),
        truncated=jnp.array([[0.0]]),
        log_prob=jnp.array([[-1.0]]),
    )
    agent_config = AVGConfig(gamma=0.99, target_entropy=-1.0)

    updated_state = update_AVG_values(avg_state, rollout, agent_config)
    log_alpha = updated_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    assert updated_state.reward.count[0] > avg_state.reward.count[0]
    assert updated_state.gamma.count[0] > avg_state.gamma.count[0]
    assert updated_state.G_return.count[0] > avg_state.G_return.count[0]
    assert jnp.allclose(updated_state.reward.mean, jnp.array([[1.0 - alpha * -1]]))
    assert jnp.allclose(updated_state.gamma.mean, jnp.array([[0.0]]))
