"""The keys an agent's steps draw on: ``BaseAgentState.next_keys``."""

from typing import Any

import jax
import numpy as np

from ajax.state import BaseAgentState


def test_next_keys_hands_out_keys_and_moves_the_state_past_them() -> None:
    key = jax.random.PRNGKey(0)
    nets: Any = None  # the networks play no part
    state = BaseAgentState(
        rng=key, actor_state=nets, critic_state=nets, collector_state=nets, eval_rng=key
    )
    state, a, b = state.next_keys(2)
    np.testing.assert_array_equal(np.stack([a, b, state.rng]), jax.random.split(key, 3))
    state, c = state.next_keys()
    assert not (np.asarray(c) == np.asarray(b)).all()
