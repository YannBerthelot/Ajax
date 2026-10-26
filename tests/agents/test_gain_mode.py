"""Gain mode (ExpertGains) on every continuous agent with an expert: the
agent acts in the expert's gain space, as wide as its learnable gains, and
its replay or rollout holds the gains (that the agents learn them is
tests/probing's)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APG, SAC, UDRL
from ajax.extensions.capabilities import NotWired
from ajax.extensions.expert import ExpertGains, ImitationLoss
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import CloningPretrain, MCPretrain
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
    rollout = {"expose_recent_rollout": True} if case.rollout is not None else {}
    agent = cls(
        env_id="Pendulum-v1",
        n_envs=2,
        expert_policy=TwoGains(),
        extensions=(ExpertGains(),),
        **case.kwargs,
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


class OneGainTwoActions:
    """One learnable gain setting a 2-wide env action."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state: jax.Array, obs: jax.Array, gains: Any) -> tuple:
        action = jnp.clip(-gains[..., :1] * obs[..., :1], -2.0, 2.0)
        return jnp.concatenate([action, action / 2], axis=-1), state + 1.0


@pytest.mark.parametrize("expert", [OneGainTwoActions(), TwoGains()])
def test_the_replay_keeps_the_experts_env_action_beside_the_gains(expert):
    """The replay's action column is as wide as the gains, its a_expert as the
    env's action: with fewer gains than actions collection used to crash, with
    more the expert's action was copied into every gain's column."""
    env_width = 2 if isinstance(expert, OneGainTwoActions) else 1
    env, params = paid_action.make(length=8, dim=env_width)
    small = {"actor_architecture": ("16", "relu"), "critic_architecture": ("16",)}
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        expert_policy=expert,
        extensions=(ExpertGains(),),
        learning_starts=8,
        batch_size=8,
        buffer_size=64,
        **small,
    )
    state, _ = agent.train(seed=0, n_timesteps=32)
    experience = state.collector_state.buffer_state.experience
    assert experience["action"].shape[-1] == len(expert.learnable_fields)
    # Every stored row: the expert's action at its anchor gains (the
    # observation is constant), in agent units (the box is [-2, 2]).
    anchor_action, _ = expert(expert.init_state(1), paid_action.OBS[None])
    stored = np.asarray(experience["a_expert"])[0, :, :16]
    assert stored.shape[-1] == env_width
    np.testing.assert_allclose(stored, np.broadcast_to(anchor_action / 2, stored.shape))


def zero_expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros(obs.shape[:-1] + (1,))


@pytest.mark.parametrize(
    "extension",
    [
        ImitationLoss(zero_expert),
        CloningPretrain(zero_expert),
        MCPretrain(zero_expert),
        EDGEExploration(),
    ],
    ids=["ImitationLoss", "CloningPretrain", "MCPretrain", "EDGEExploration"],
)
def test_gain_mode_refuses_an_extension_using_expert_actions_as_the_policys(extension):
    """An expert's actions are env actions, the policy's are gains: cloning
    them into the actor, scoring them with a critic or executing them in
    the policy's place (EDGE) mixes the two."""
    name = type(extension).__name__
    with pytest.raises(NotWired, match=name) as refused:
        SAC(
            "Pendulum-v1",
            expert_policy=TwoGains(),
            extensions=(ExpertGains(), extension),
        )
    reason = "uses the expert's env actions where the policy's gains go"
    assert refused.value.unwired == {name: reason}


def test_gain_mode_takes_imitation_of_the_replayed_gains():
    """ImitationLoss's replayed target clones the stored actions, which are
    the policy's own gains."""
    imitation = ImitationLoss(None, target="replay")
    SAC("Pendulum-v1", expert_policy=TwoGains(), extensions=(ExpertGains(), imitation))
