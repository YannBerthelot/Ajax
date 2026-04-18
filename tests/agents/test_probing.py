"""Probing environment tests for Ajax agents.

These tests use ProbingEnvironments to verify that each agent's core RL
components (value loss, backprop, discounting, advantage, actor-critic
coupling) are working correctly. This is the TDD baseline for the
refactoring — if these pass, the agent is functionally correct.
"""
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO import PPO
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from probing_environments.adaptors.ajax import (
    get_action,
    get_gamma,
    get_value,
    init_agent,
    train_agent,
)
from probing_environments.checks import (
    check_actor_and_critic_coupling_continuous,
    check_advantage_policy_continuous,
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)


# All agents
ALL_AGENTS = [SAC, REDQ, PPO, APO, ASAC, AVG]
# Average-reward agents: their critic learns *differential* V (≈0 for constant
# reward), not absolute V — so the V≈1 expectation in value-net checks and the
# discount-ratio expectation in reward-discounting check don't apply.
_AVG_REWARD = {"ASAC", "APO"}
DISCOUNTED_AGENTS = [a for a in ALL_AGENTS if a.__name__ not in _AVG_REWARD]
VALUE_NET_AGENTS = DISCOUNTED_AGENTS
# Coupling test asserts V≥0.8 — same differential-V issue as value-net checks.
COUPLING_AGENTS = DISCOUNTED_AGENTS

# Only SAC and PPO run by default. Other agents are too slow on probing envs
# (AVG especially — Vasan 2024 noted its sample inefficiency). TODO: speed up.
_FAST = {"SAC", "PPO"}
_SKIP_REASON = "probing test too slow for this agent — TODO: speed up"


def _params(agents):
    return [
        pytest.param(a)
        if a.__name__ in _FAST
        else pytest.param(a, marks=pytest.mark.skip(reason=_SKIP_REASON))
        for a in agents
    ]


BUDGET_VALUE = int(1e4)
BUDGET_POLICY = int(1e4)
# Coupling on PolicyAndValueEnv requires learning the obs→action-sign mapping —
# PPO with n_envs=1 needs more rollouts and a higher LR to move the actor.
BUDGET_COUPLING = int(6e4)
LR_COUPLING = 5e-3


@pytest.mark.parametrize("agent_cls", _params(VALUE_NET_AGENTS), ids=lambda c: c.__name__)
class TestProbingValueNet:
    """Value network probing checks (critic only)."""

    def test_loss_or_optimizer(self, agent_cls):
        check_loss_or_optimizer_value_net(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )

    def test_backprop(self, agent_cls):
        check_backprop_value_net(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )


@pytest.mark.parametrize("agent_cls", _params(DISCOUNTED_AGENTS), ids=lambda c: c.__name__)
class TestProbingDiscounting:
    """Reward-discounting check — only meaningful for discounted agents."""

    def test_reward_discounting(self, agent_cls):
        check_reward_discounting(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            get_gamma=get_gamma,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )


@pytest.mark.parametrize("agent_cls", _params(ALL_AGENTS), ids=lambda c: c.__name__)
class TestProbingPolicy:
    """Policy network probing checks (actor + critic)."""

    def test_advantage_policy(self, agent_cls):
        check_advantage_policy_continuous(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            budget=BUDGET_POLICY,
            gymnax=True,
        )


@pytest.mark.parametrize("agent_cls", _params(COUPLING_AGENTS), ids=lambda c: c.__name__)
class TestProbingCoupling:
    """Actor-critic coupling — value assertion excludes avg-reward agents."""

    def test_actor_critic_coupling(self, agent_cls):
        check_actor_and_critic_coupling_continuous(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            get_value=get_value,
            budget=BUDGET_COUPLING,
            learning_rate=LR_COUPLING,
            gymnax=True,
        )
