import pytest

from ajax.agents.APO.APO import APO
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


@pytest.mark.parametrize(
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
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
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
)
def test_APO_train_single_seed(env_id):
    """Test SAC agent's train method with a single seed."""
    # env_id = "Pendulum-v1"
    APO_agent = APO(env_id=env_id, n_steps=10, batch_size=5)
    APO_agent.train(seed=42, n_timesteps=100)


@pytest.mark.parametrize(
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
)
def test_APO_train_multiple_seeds(env_id):
    """Test SAC agent's train method with multiple seeds using jax.vmap."""
    # env_id = "Pendulum-v1"
    APO_agent = APO(env_id=env_id, n_steps=10, batch_size=5)
    seeds = [42, 43, 44]
    n_timesteps = 100
    APO_agent.train(seed=seeds, n_timesteps=n_timesteps)
