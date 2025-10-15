import pytest

from ajax.agents.REDQ.REDQ import REDQ
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


def test_REDQ_initialization():
    """Test REDQ agent initialization with default parameters."""
    env_id = "Pendulum-v1"
    REDQ_agent = REDQ(env_id=env_id)

    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
            "alpha_args",
            "buffer",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
            AlphaConfig,
            BufferType,
        ),
    ):
        assert hasattr(REDQ_agent, expected_attr)
        assert isinstance(getattr(REDQ_agent, expected_attr), expected_type)


def test_REDQ_initialization_with_discrete_env():
    """Test REDQ agent initialization fails with a discrete environment."""
    env_id = "CartPole-v1"
    with pytest.raises(
        ValueError, match="REDQ only supports continuous action spaces."
    ):
        REDQ(env_id=env_id)


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

    avg_agent = REDQ(env_id=env_id, learning_starts=learning_starts, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)
