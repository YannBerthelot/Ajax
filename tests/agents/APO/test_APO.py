import pytest

from ajax.agents.APO.APO import APO
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


@pytest.mark.parametrize(
    "env_id",
    ["fast", "Pendulum-v1", "CartPole-v1"],
)
def test_APO_initialization(env_id):
    """Test SAC agent initialization with default parameters."""
    APO_agent = APO(env_id=env_id)

    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
        ),
    ):
        assert hasattr(APO_agent, expected_attr)
        assert isinstance(getattr(APO_agent, expected_attr), expected_type)


@pytest.mark.parametrize(
    "env_id, seeds, n_envs",
    [
        ["Pendulum-v1", 42, 1],
        ["Pendulum-v1", [42, 43], 1],
        ["Pendulum-v1", [42, 43], 2],
        ["CartPole-v1", 42, 1],
        ["CartPole-v1", [42, 43], 1],
        ["CartPole-v1", [42, 43], 2],
        ["fast", 42, 1],
        ["fast", [42, 43], 1],
        ["fast", [42, 43], 2],
    ],
)
def test_avg_train_all_modes(env_id, seeds, n_envs):
    n_timesteps = 50  # keep small for speed

    avg_agent = APO(env_id=env_id, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)
