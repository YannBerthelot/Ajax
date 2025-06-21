import gymnax
import jax
import pytest
from brax.envs import create as create_brax_env
from flax.serialization import to_state_dict

from ajax.agents.sac.train_sac import init_sac
from ajax.buffers.utils import get_buffer
from ajax.evaluate import evaluate
from ajax.state import (
    AlphaConfig,
    BufferConfig,
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
def sac_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    buffer = get_buffer(
        **to_state_dict(
            BufferConfig(buffer_size=1000, batch_size=32, n_envs=env_config.n_envs)
        )
    )
    return init_sac(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
    )


def test_evaluate_with_fast_env(env_config, sac_state):
    num_episodes = 5
    rng = jax.random.PRNGKey(0)

    rewards, avg_entropy = evaluate(
        env=env_config.env,
        actor_state=sac_state.actor_state,
        num_episodes=num_episodes,
        rng=rng,
        env_params=env_config.env_params,
        recurrent=False,
        lstm_hidden_size=None,
    )

    # Assertions
    assert rewards.shape == ()
    assert avg_entropy.shape == ()


def test_evaluate_with_gymnax_env(env_config, sac_state):
    num_episodes = 3
    rng = jax.random.PRNGKey(42)

    rewards, avg_entropy = evaluate(
        env=env_config.env,
        actor_state=sac_state.actor_state,
        num_episodes=num_episodes,
        rng=rng,
        env_params=env_config.env_params,
        recurrent=False,
        lstm_hidden_size=128,
    )

    # Assertions
    assert rewards.shape == ()
    assert avg_entropy.shape == ()
