import jax.numpy as jnp
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


@pytest.mark.parametrize(
    "env_id, seeds, n_envs",
    [
        ["fast", 42, 1],
        ["fast", [42, 43], 2],
    ],
)
def test_avg_train_all_modes(env_id, seeds, n_envs):
    n_timesteps = 50  # keep small for speed
    learning_starts = 10

    avg_agent = SAC(env_id=env_id, learning_starts=learning_starts, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)


@pytest.mark.parametrize("alpha_init", [1.0, 0.2])
def test_fixed_alpha_keeps_temperature_at_alpha_init(alpha_init):
    """``fixed_alpha=True`` skips the temperature gradient step entirely.

    ``alpha_update_start=0`` and ``learning_starts=10`` guarantee that the
    temperature update *would* run for most of the 60 steps — the sibling
    run below confirms alpha does move once ``fixed_alpha`` is off — so an
    unchanged ``log_alpha`` proves the flag is honoured rather than the
    update never having been reached.
    """
    common = {
        "env_id": "fast",
        "learning_starts": 10,
        "alpha_update_start": 0,
        "alpha_init": alpha_init,
        "alpha_learning_rate": 1e-2,
    }
    expected_log_alpha = jnp.log(alpha_init)

    fixed_state, _ = SAC(**common, fixed_alpha=True).train(seed=42, n_timesteps=60)
    assert jnp.allclose(fixed_state.alpha.params["log_alpha"], expected_log_alpha)

    learned_state, _ = SAC(**common, fixed_alpha=False).train(seed=42, n_timesteps=60)
    assert not jnp.allclose(
        learned_state.alpha.params["log_alpha"], expected_log_alpha
    ), "sanity check: alpha should move when it is not fixed"
