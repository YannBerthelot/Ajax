"""Gain mode (ExpertGains) on every continuous agent with an expert: the
agent acts in the expert's gain space, as wide as its learnable gains, and
its replay or rollout holds the gains (that the agents learn them is
tests/probing's)."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APG, UDRL
from ajax.extensions.expert import ExpertGains
from tests.agents.test_agent_contract import CASES
from tests.environments import paid_action

AGENTS = ("SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC", "AVG", "PPO", "APO", "IQL")


class TwoGains:
    """A PD expert on Pendulum's cos and sin (env units), two learnable
    gains: the agent's action is 2-wide, the env's 1."""

    learnable_fields, anchor_gains = ("kp", "kd"), jnp.array([1.0, 0.5])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state: jax.Array, obs: jax.Array, gains: Any) -> tuple:
        action = -gains[..., :1] * obs[..., 1:2] - gains[..., 1:] * obs[..., 2:3]
        return jnp.clip(action, -2.0, 2.0), state + 1.0


@pytest.mark.parametrize("name", AGENTS)
def test_an_agent_in_gain_mode_acts_and_stores_in_the_gain_space(name):
    case = CASES[name]
    cls: Any = case.cls  # each agent's own keywords
    takes = inspect.signature(cls).parameters
    prefill = {"expert_buffer_n_steps": 0} if "expert_buffer_n_steps" in takes else {}
    rollout = {"expose_recent_rollout": True} if case.rollout is not None else {}
    agent = cls(
        env_id="Pendulum-v1",
        n_envs=2,
        expert_policy=TwoGains(),
        extensions=(ExpertGains(),),
        **case.kwargs,
        **prefill,
        **rollout,
    )
    assert agent.env_args.policy_action_dim == 2
    state, _ = agent.train(seed=0, n_timesteps=case.n_timesteps)
    buffer = state.collector_state.buffer_state
    stored = (
        state.last_rollout.action if buffer is None else buffer.experience["action"]
    )
    assert stored.shape[-1] == 2  # the gains, not Pendulum's torque
    assert np.isfinite(np.asarray(stored)).all()
    for part in (state.actor_state.params, state.critic_state.params):
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(part))
    # The expert stepped with the chosen gains: its state counts the steps.
    assert float(np.asarray(state.collector_state.expert_state).max()) > 0


def test_udrl_and_apg_act_in_the_gain_space():
    """UDRL's collector and APG's differentiable rollout actuate the gains
    through the expert: their policies are two-wide."""
    udrl = UDRL(
        "Pendulum-v1",
        n_envs=2,
        n_steps=8,
        batch_size=8,
        n_updates_per_iter=1,
        buffer_capacity=8,
        actor_architecture=("16", "relu"),
        expert_policy=TwoGains(),
        extensions=(ExpertGains(),),
    )
    state, _ = udrl.train(seed=0, n_timesteps=32, num_episode_test=1)
    assert state.buffer.actions.shape[-1] == 2
    env, params = paid_action.make()
    apg = APG(
        env,
        env_params=params,
        n_envs=2,
        horizon=4,
        actor_architecture=("8", "tanh"),
        pid=None,
        expert_policy=TwoGains(),
        extensions=(ExpertGains(),),
    )
    state, _ = apg.train(seed=0, n_timesteps=32)
    leaves = jax.tree.leaves(state.actor_state.params)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert apg.env_args.policy_action_dim == 2
