"""Probing environment tests for Ajax agents.

These tests use ProbingEnvironments to verify that each agent's core RL
components (value loss, backprop, discounting, advantage, actor-critic
coupling) are working correctly. This is the TDD baseline for the
refactoring — if these pass, the agent is functionally correct.
"""
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.agents.REDQ.REDQ import REDQ
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


# Q-function agents that pass all probing checks
Q_AGENTS = [SAC, REDQ]
BUDGET_VALUE = int(1e4)
BUDGET_POLICY = int(1e4)


@pytest.mark.parametrize("agent_cls", Q_AGENTS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("agent_cls", Q_AGENTS, ids=lambda c: c.__name__)
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

    def test_actor_critic_coupling(self, agent_cls):
        check_actor_and_critic_coupling_continuous(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            get_value=get_value,
            budget=BUDGET_POLICY,
            gymnax=True,
        )
