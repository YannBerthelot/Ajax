"""TD3's update order, as sfujim/TD3's (34770cc, TD3.py train): the update is
counted first, and the actor and the targets step when the count is a
multiple of policy_delay, so the first update moves the critics only."""

from __future__ import annotations

import jax
import numpy as np

from ajax.agents.TD3 import train_TD3
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.base import ExtensionStack
from tests.agents.TD3.test_td3_metrics import KEY, NET, _batch


def _moved(before, after) -> bool:
    return any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=False)
    )


def test_the_actor_and_targets_first_step_on_the_policy_delay_th_update(
    monkeypatch,
) -> None:
    agent = TD3(
        "Pendulum-v1", actor_architecture=NET, critic_architecture=NET, policy_delay=2
    )
    state = train_TD3.init_TD3(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.buffer,
    )
    batch = _batch(agent)
    monkeypatch.setattr(
        train_TD3, "sample_replay", lambda *args, **kwargs: (batch, None)
    )
    moved = []
    for _ in range(4):
        new, _ = train_TD3.update_agent(
            state, agent.buffer, False, agent.agent_config, ExtensionStack(()), 1
        )
        moved.append(
            (
                _moved(state.critic_state.params, new.critic_state.params),
                _moved(state.actor_state.params, new.actor_state.params),
                _moved(
                    state.critic_state.target_params, new.critic_state.target_params
                ),
            )
        )
        state = new
    # (critics, actor, targets) per update: the actor and targets on updates 2 and 4.
    assert moved == [
        (True, False, False),
        (True, True, True),
        (True, False, False),
        (True, True, True),
    ]
