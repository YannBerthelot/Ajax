"""Agents declare the extension phases they fold; others are rejected."""

from dataclasses import dataclass

import pytest

from ajax import PPO
from ajax.agents.base import ActorCritic
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionStack,
    check_extension_phases,
)


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

    def on_obs(self, obs, ext_state, ctx):
        return obs

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


def test_existing_agents_support_every_phase_by_default():
    assert ActorCritic.supported_extension_phases == frozenset(PHASES)
    assert EveryPhase().implemented_phases() == frozenset(PHASES)
    agent = PPO("CartPole-v1", n_envs=1, extensions=[EveryPhase()])
    assert len(agent.extension_stack) == 1


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
