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
    "env_id, seeds, n_envs",
    [
        ["Pendulum-v1", 42, 1],
        ["Pendulum-v1", [42, 43], 1],
        ["Pendulum-v1", [42, 43], 2],
        ["fast", 42, 1],
        ["fast", [42, 43], 1],
        ["fast", [42, 43], 2],
    ],
)
def test_avg_train_all_modes(env_id, seeds, n_envs):
    n_timesteps = 50  # keep small for speed
    learning_starts = 10

    avg_agent = AVG(env_id=env_id, learning_starts=learning_starts, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)
