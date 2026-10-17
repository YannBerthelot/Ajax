"""Agents declare the extension phases they fold; others are rejected."""

import importlib
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from ajax import PPO, SAC
from ajax.agents.base import ActorCritic
from ajax.agents.loop import LOOP_PHASES
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionStack,
    check_extension_phases,
)
from ajax.extensions.expert import JSRLCurriculum, OnlineBC, ResidualPolicy
from ajax.extensions.pretrain import MCPretrain, PhiRefresh
from ajax.extensions.target_mods import CriticBlend, MCVarianceCorrection, ValueBox

AGENTS = "APG APO ASAC AVG DQN DreamerV3 PPO PQN REDQ SAC TD3 TDMPC2 UDRL".split()


@dataclass(frozen=True)
class TargetTweak(Extension):
    name: str = "target-tweak"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {}


@dataclass(frozen=True)
class Instrument(Extension):
    name: str = "instrument"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {}

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state


@dataclass(frozen=True)
class EveryPhase(Extension):
    """Overrides every phase (with the base behaviour)."""

    name: str = "every-phase"

    def pretrain(self, agent_state, ext_state, ctx):
        return agent_state, ext_state

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        return 0.0

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return 0.0

    def action(self, agent_state, ext_state, obs, rng, ctx):
        return None

    def eval_action(self, agent_state, ext_state, obs, rng, ctx):
        return None

    def transform_action(self, actions, a_expert):
        return actions

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {}


class LatentAgent(ActorCritic):
    supported_extension_phases = frozenset({"pretrain", "post_update", "eval_metrics"})

    def __init__(self, extensions=()):
        super().__init__(env_id="CartPole-v1", n_envs=1, extensions=extensions)


def test_unsupported_phases_are_rejected_at_construction():
    with pytest.raises(ValueError) as error:
        LatentAgent(extensions=[Instrument(), TargetTweak()])
    message = str(error.value)
    assert "LatentAgent" in message
    assert "'target-tweak'" in message and "on_target" in message
    assert "instrument" not in message  # its phases are supported
    LatentAgent(extensions=[Instrument()])  # supported phases are accepted


def test_agents_accept_only_the_phases_they_fold():
    """The base accepts none; every agent the loop's, no agent
    ``eval_action``, only SAC ``action`` (its pipeline's slots) and
    ``transform_action`` (its residual policy)."""
    assert ActorCritic.supported_extension_phases == frozenset()
    assert EveryPhase().implemented_phases() == frozenset(PHASES)
    unsupported = r"\['action', 'eval_action', 'transform_action'\]"
    with pytest.raises(ValueError, match=unsupported):
        PPO("CartPole-v1", n_envs=1, extensions=[EveryPhase()])
    for name in AGENTS:
        agent = getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)
        phases = agent.supported_extension_phases
        assert LOOP_PHASES <= phases and "eval_action" not in phases, name
        assert ("action" in phases) == (name == "SAC"), name
        assert ("transform_action" in phases) == (name == "SAC"), name


@dataclass(frozen=True)
class Unslotted(Extension):
    name: str = "unslotted"

    def action(self, agent_state, ext_state, obs, rng, ctx):
        return None


def _expert(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


def test_sac_dispatches_action_only_to_its_pipeline_slots():
    """SAC's pipeline runs ``action`` only for an extension of a slot and only
    with an expert policy: any other is rejected at construction."""
    slotted = JSRLCurriculum()
    with pytest.raises(ValueError, match="'unslotted'"):
        SAC("Pendulum-v1", expert_policy=_expert, extensions=[Unslotted()])
    with pytest.raises(ValueError, match="expert_policy"):
        SAC("Pendulum-v1", extensions=[slotted])
    SAC("Pendulum-v1", expert_policy=_expert, extensions=[slotted])


def test_sac_applies_a_residual_policy_only_through_its_pipeline():
    """ResidualPolicy needs the pipeline SAC runs with an expert policy and
    without the gain policy; otherwise it is rejected at construction."""
    residual = ResidualPolicy(scale=0.5)
    assert residual.implemented_phases() == {"transform_action"}
    for kwargs in ({}, {"expert_policy": _expert, "use_pid_policy": True}):
        with pytest.raises(ValueError, match="'residual_policy'"):
            SAC("Pendulum-v1", extensions=[residual], **kwargs)
    SAC("Pendulum-v1", expert_policy=_expert, extensions=[residual])


@pytest.mark.parametrize("name", sorted(set(AGENTS) - {"SAC", "DreamerV3", "TDMPC2"}))
def test_an_agent_without_a_residual_transform_refuses_a_residual_policy(name):
    """Only SAC runs ``transform_action``: any other agent would train a
    plain policy and report it as residual, so it refuses one."""
    module = importlib.import_module(f"ajax.agents.{name}.{name}")
    agent = getattr(module, name)
    env = "CartPole-v1" if name in ("DQN", "PQN", "UDRL") else "Pendulum-v1"
    with pytest.raises(ValueError, match=r"'residual_policy'.*transform_action"):
        agent(env, extensions=[ResidualPolicy()])


def test_online_bc_needs_the_agents_expert_policy():
    """OnlineBC clones the actions of the agent's own expert_policy: without
    one it would add nothing all run, so SAC refuses it at construction,
    whatever expert MCPretrain holds."""
    stack = [MCPretrain(expert_policy=_expert), OnlineBC()]
    with pytest.raises(ValueError, match="^OnlineBC read the actions"):
        SAC("Pendulum-v1", extensions=stack)
    SAC("Pendulum-v1", expert_policy=_expert, extensions=stack)


def test_the_phase_check_applies_to_any_stack():
    """The check ``ActorCritic`` runs at construction, also used by agents
    that are not ``ActorCritic`` (the offline multi-task TD-MPC2)."""
    supported = frozenset({"post_update", "eval_metrics"})
    check_extension_phases("Offline", ExtensionStack([Instrument()]), supported)
    check_extension_phases("Offline", [], supported)
    with pytest.raises(ValueError) as error:
        check_extension_phases("Offline", [Instrument(), TargetTweak()], supported)
    assert "Offline does not support" in str(error.value)
    assert "'target-tweak'" in str(error.value) and "on_target" in str(error.value)


#: The extensions reading the frozen expert critic phi*, which MCPretrain fits.
PHI_STAR_READERS = {
    "OnlineBC": OnlineBC(),
    "CriticBlend": CriticBlend(expert_policy=_expert),
    "MCVarianceCorrection": MCVarianceCorrection(threshold=1.0),
    "ValueBox": ValueBox(),
    "PhiRefresh": PhiRefresh(expert_policy=_expert),
}


@pytest.mark.parametrize("name", list(PHI_STAR_READERS))
def test_a_reader_of_phi_star_needs_mc_pretrain(name: str) -> None:
    """Refused in a stack without MCPretrain (it would add nothing all run,
    or fail mid-trace), accepted with one."""
    reader, fields = PHI_STAR_READERS[name], frozenset({"raw_next_obs"})
    sac = ("SAC", frozenset(PHASES), fields, _expert)
    with pytest.raises(ValueError, match=f"^{name} read the frozen expert critic"):
        check_extension_phases(sac[0], [reader], *sac[1:])
    check_extension_phases(
        sac[0], [MCPretrain(expert_policy=_expert), reader], *sac[1:]
    )


def test_an_agent_refuses_a_reader_of_phi_star_without_mc_pretrain() -> None:
    with pytest.raises(ValueError, match="SAC's extensions hold no MCPretrain"):
        SAC("Pendulum-v1", expert_policy=_expert, extensions=[OnlineBC()])
