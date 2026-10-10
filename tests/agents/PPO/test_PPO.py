import jax
import numpy as np
import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.train_PPO import _force_reset, _normalisation_info
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


@pytest.mark.parametrize(
    "env_id",
    ["fast", "Pendulum-v1", "CartPole-v1"],
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
    "env_id, seeds, n_envs",
    [
        ["fast", 42, 1],
        ["fast", [42, 43], 2],
        ["CartPole-v1", [42, 43], 2],
    ],
)
def test_avg_train_all_modes(env_id, seeds, n_envs):
    n_timesteps = 50  # keep small for speed

    avg_agent = PPO(env_id=env_id, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)


@requires_playground
def test_PPO_train_playground_smoke():
    """Smoke test: PPO trains briefly on a playground env (covers playground
    code path end-to-end including seed vmap and eval)."""
    agent = PPO(env_id="PendulumSwingup", n_envs=2)
    agent.train(seed=42, n_timesteps=50)


@pytest.mark.parametrize("env_id, mode", [("Pendulum-v1", "gymnax"), ("fast", "brax")])
def test_a_forced_reset_keeps_the_normaliser_statistics(env_id, mode):
    """num_resets_per_eval resets the envs, not the normaliser's running
    statistics (each gymnax env restarted its own from the reset)."""
    agent = PPO(
        env_id=env_id,
        n_envs=2,
        n_steps=8,
        normalize_observations=True,
        normalize_rewards=True,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=32)[0])
    saved = _normalisation_info(state.collector_state.env_state, mode)
    after = _force_reset(state, agent.env_args, mode).collector_state
    kept = _normalisation_info(after.env_state, mode)
    jax.tree.map(np.testing.assert_array_equal, kept, saved)
