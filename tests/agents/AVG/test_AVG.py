import pytest

from ajax.agents.AVG.AVG import AVG
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig


@pytest.mark.parametrize(
    "env_id",
    [
        "Pendulum-v1",
        "fast",
    ],
)
def test_avg_initialization(env_id):
    """Test AVG agent initialization with default parameters."""

    avg_agent = AVG(env_id=env_id)

    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
            "alpha_args",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
            AlphaConfig,
        ),
    ):
        assert hasattr(avg_agent, expected_attr)
        assert isinstance(getattr(avg_agent, expected_attr), expected_type)


def test_avg_initialization_with_discrete_env():
    """Test AVG agent initialization fails with a discrete environment."""
    env_id = "CartPole-v1"
    with pytest.raises(ValueError, match="AVG only supports continuous action spaces."):
        AVG(env_id=env_id)


@pytest.mark.parametrize(
    "env_id",
    [
        "Pendulum-v1",
        "fast",
    ],
)
def test_avg_train_single_seed(env_id):
    """Test AVG agent's train method with a single seed."""
    avg_agent = AVG(env_id=env_id, learning_starts=10)
    avg_agent.train(seed=42, n_timesteps=100)


@pytest.mark.parametrize(
    "env_id",
    [
        "Pendulum-v1",
        "fast",
    ],
)
def test_avg_train_multiple_seeds(env_id):
    """Test AVG agent's train method with multiple seeds using jax.vmap."""
    avg_agent = AVG(env_id=env_id, learning_starts=10)
    seeds = [42, 43, 44]
    n_timesteps = 100
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)


@pytest.mark.parametrize(
    "env_id",
    [
        "Pendulum-v1",
        "fast",
    ],
)
def test_avg_train_multiple_envs(env_id):
    """Test AVG agent's train method with multiple seeds using jax.vmap."""
    avg_agent = AVG(env_id=env_id, learning_starts=10, n_envs=2)
    seeds = [42]
    n_timesteps = 100
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)
