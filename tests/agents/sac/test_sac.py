import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


def test_SAC_initialization():
    """Test SAC agent initialization with default parameters."""
    env_id = "Pendulum-v1"
    SAC_agent = SAC(env_id=env_id)

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
        assert hasattr(SAC_agent, expected_attr)
        assert isinstance(getattr(SAC_agent, expected_attr), expected_type)


def test_SAC_initialization_with_discrete_env():
    """Test SAC agent initialization fails with a discrete environment."""
    env_id = "CartPole-v1"
    with pytest.raises(ValueError, match="SAC only supports continuous action spaces."):
        SAC(env_id=env_id)


def test_SAC_train_single_seed():
    """Test SAC agent's train method with a single seed."""
    env_id = "Pendulum-v1"
    SAC_agent = SAC(env_id=env_id, learning_starts=10)
    SAC_agent.train(seed=42, n_timesteps=100)
    # try:
    #     SAC_agent.train(seed=42, n_timesteps=1000)
    #     success = True  # Placeholder: Add assertions or checks as needed
    # except Exception as e:
    #     success = False
    #     print(f"Training failed with single seed: {e}")
    # assert success, "Training failed for the single seed"


def test_SAC_train_multiple_seeds():
    """Test SAC agent's train method with multiple seeds using jax.vmap."""
    env_id = "Pendulum-v1"
    SAC_agent = SAC(env_id=env_id, learning_starts=10)
    seeds = [42, 43, 44]
    n_timesteps = 100
    SAC_agent.train(seed=seeds, n_timesteps=n_timesteps)
    # try:
    #     SAC_agent.train(seed=seeds, n_timesteps=1000)
    #     success = True  # Placeholder: Add assertions or checks as needed
    # except Exception as e:
    #     success = False
    #     print(f"Training failed with multiple seeds: {e}")
    # assert success, "Training failed for one or more seeds"
