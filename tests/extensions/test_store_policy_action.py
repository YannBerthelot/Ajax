"""StorePolicyAction (SAC's former store_policy_action), an off-policy
correctness ablation every replay agent on the family takes: the replay
stores the action the policy chose instead of the one executed. It acts in
the controller alone, so the learner holds nothing for it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ajax import AVG, DQN, PPO, TD3, DAgger
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.expert import StorePolicyAction
from ajax.networks.memory import MemoryConfig
from tests.extensions.test_expert_warmup import WARMUP_AGENTS, push, warmup_rows

REPLAY_AGENTS = {name: WARMUP_AGENTS[name] for name in ("TD3", "CrossQ", "IQL")}


@pytest.mark.parametrize("name", sorted(REPLAY_AGENTS))
def test_the_replay_stores_the_policys_action_instead_of_the_executed_one(name):
    """The warm-up executes uniform actions; with StorePolicyAction its rows
    store the policy's instead, beside the rewards and next observations
    the executed ones produced, which stay as they were (no key moves;
    after it the policy learns from other rows)."""
    cls, networks = REPLAY_AGENTS[name]
    executed = warmup_rows(cls, **networks)
    stored = warmup_rows(cls, extensions=(StorePolicyAction(),), **networks)
    for field in ("obs", "next_obs", "reward"):
        warmup = stored[field][:, :8], executed[field][:, :8]
        np.testing.assert_array_equal(*warmup, err_msg=field)
    assert not np.allclose(stored["action"][:, :8], executed["action"][:, :8])


def test_the_learner_holds_nothing_for_it_and_a_recurrent_agent_takes_it():
    networks: dict[str, Any] = WARMUP_AGENTS["TD3"][1]
    agent = TD3("Pendulum-v1", extensions=(StorePolicyAction(),), **networks)
    assert not agent.extension_stack.extensions
    assert agent.env_args.replay_action_choosers == (StorePolicyAction(),)
    gru = MemoryConfig(kind="gru", hidden_size=4)
    recurrent = TD3(
        "Pendulum-v1",
        memory=gru,
        learning_starts=40,
        batch_size=4,
        buffer_size=64,
        extensions=(StorePolicyAction(),),
        **networks,
    )
    recurrent.train(seed=0, n_timesteps=8)


@pytest.mark.parametrize("cls", [AVG, PPO], ids=lambda c: c.__name__)
def test_an_agent_without_a_replay_refuses_it(cls: Any) -> None:
    with pytest.raises(MissingCapability, match="replay"):
        cls("Pendulum-v1", extensions=(StorePolicyAction(),))


def test_dqn_stores_its_policys_index_in_the_warm_up():
    """DQN's warm-up executes uniform indices; with StorePolicyAction its
    rows store the epsilon-greedy policy's own, an (n, 1) int column,
    beside the rewards and observations the executed ones produced."""

    def rows(*extensions: Any) -> Any:
        agent = DQN(
            "CartPole-v1",
            n_envs=2,
            learning_starts=32,
            batch_size=8,
            buffer_size=64,
            architecture=("8", "relu"),
            extensions=extensions,
        )
        state, _ = agent.train(seed=0, n_timesteps=32)
        buffer_state = state.collector_state.buffer_state
        assert buffer_state is not None
        return {k: np.asarray(v[0]) for k, v in buffer_state.experience.items()}

    executed, stored = rows(), rows(StorePolicyAction())
    for field in ("obs", "next_obs", "reward"):
        np.testing.assert_array_equal(stored[field], executed[field], err_msg=field)
    assert stored["action"].dtype == executed["action"].dtype
    assert (stored["action"][:, :16] != executed["action"][:, :16]).any()


def test_dagger_has_nothing_for_it_to_ablate():
    """DAgger learns from its expert's labels alone, no stored action."""
    with pytest.raises(MissingCapability, match="reward_learning"):
        DAgger("Pendulum-v1", expert_policy=push, extensions=(StorePolicyAction(),))
