import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


@pytest.mark.parametrize(
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
)
def test_PPO_initialization(env_id):
    """Test SAC agent initialization with default parameters."""
    PPO_agent = PPO(env_id=env_id)

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
        assert hasattr(PPO_agent, expected_attr)
        assert isinstance(getattr(PPO_agent, expected_attr), expected_type)


@pytest.mark.parametrize(
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
)
def test_PPO_train_single_seed(env_id):
    """Test SAC agent's train method with a single seed."""
    # env_id = "Pendulum-v1"
    PPO_agent = PPO(env_id=env_id, n_steps=10, batch_size=5)
    PPO_agent.train(seed=42, n_timesteps=100)


@pytest.mark.parametrize(
    "env_id",
    ["Pendulum-v1", "CartPole-v1"],
)
def test_PPO_train_multiple_seeds(env_id):
    """Test SAC agent's train method with multiple seeds using jax.vmap."""
    # env_id = "Pendulum-v1"
    PPO_agent = PPO(env_id=env_id, n_steps=10, batch_size=5)
    seeds = [42, 43, 44]
    n_timesteps = 100
    PPO_agent.train(seed=seeds, n_timesteps=n_timesteps)
