"""TD3's update order, as sfujim/TD3's (34770cc, TD3.py train): the update is
counted first, and the actor and the targets step when the count is a
multiple of policy_delay, so the first update moves the critics only."""

from __future__ import annotations

import jax
import numpy as np

import ajax.agents.TD3.TD3 as td3_module
from ajax.agents.TD3.TD3 import TD3
from tests.agents.TD3.test_td3_metrics import KEY, NET, _batch, make_context


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
    context = make_context(agent)
    state = agent.init(KEY, context)
    batch = _batch(agent)
    monkeypatch.setattr(
        td3_module, "sample_replay", lambda *args, **kwargs: (batch, None)
    )
    moved = []
    for _ in range(4):
        new, _ = agent.update(state, context)
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
