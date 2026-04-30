"""End-to-end tests for the UDRL agent."""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.UDRL.UDRL import UDRL
from ajax.agents.UDRL.state import UDRLConfig
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


def test_UDRL_initialization():
    """The UDRL agent exposes the standard Ajax config attributes."""
    agent = UDRL(env_id="Pendulum-v1", n_envs=1)

    for attr, expected_type in zip(
        ("env_args", "actor_optimizer_args", "network_args", "agent_config"),
        (EnvironmentConfig, OptimizerConfig, NetworkConfig, UDRLConfig),
    ):
        assert hasattr(agent, attr)
        assert isinstance(getattr(agent, attr), expected_type)


def test_UDRL_initialization_discrete():
    """UDRL supports discrete action envs (CartPole)."""
    agent = UDRL(env_id="CartPole-v1", n_envs=1)
    assert agent.env_args.continuous is False


@pytest.mark.parametrize(
    "env_id, seeds, n_envs",
    [
        ["fast", 42, 1],
        ["fast", [42, 43], 2],
        ["Pendulum-v1", 42, 1],
        ["CartPole-v1", [42, 43], 2],
    ],
)
def test_UDRL_train_smoke(env_id, seeds, n_envs):
    """Train briefly under all relevant configurations: seed scalar/list, env modes, action spaces."""
    agent = UDRL(env_id=env_id, n_envs=n_envs, n_steps=8, batch_size=8, n_epochs=1)
    agent.train(seed=seeds, n_timesteps=64)


def test_UDRL_train_updates_actor_params():
    """A short training run should change actor parameters (sanity that gradients flow)."""
    agent = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=8, batch_size=8, n_epochs=2)
    state, _ = agent.train(seed=42, n_timesteps=32)
    # Compare to freshly-initialised params at the same seed.
    fresh = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=8, batch_size=8, n_epochs=2)
    fresh_state, _ = fresh.train(seed=42, n_timesteps=0)

    def any_diff(a, b):
        leaves_a = jax.tree_util.tree_leaves(a)
        leaves_b = jax.tree_util.tree_leaves(b)
        return any(
            not jnp.allclose(la, lb) for la, lb in zip(leaves_a, leaves_b)
        )

    assert any_diff(state.actor_state.params, fresh_state.actor_state.params)


def test_UDRL_obs_dim_is_augmented():
    """Actor's last_obs in the collector should have width = env obs_dim + command_dim (=2)."""
    from ajax.environments.utils import get_state_action_shapes

    agent = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=4, batch_size=4, n_epochs=1)
    state, _ = agent.train(seed=42, n_timesteps=0)

    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    raw_obs_dim = obs_shape[0]
    assert state.collector_state.last_obs.shape[-1] == raw_obs_dim + 2
