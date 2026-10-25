"""The keys an agent's steps draw on: BaseAgentState.get_keys_and_new_state
and the extensions' key a step derives
(ajax.agents.update_steps.derive_extensions_key)."""

from typing import Any

import jax
import numpy as np

from ajax.agents.update_steps import derive_extensions_key
from ajax.state import BaseAgentState


def test_drawing_keys_hands_them_out_and_moves_the_new_state_past_them() -> None:
    key = jax.random.PRNGKey(0)
    nets: Any = None  # the networks play no part
    agent_state = BaseAgentState(
        rng=key, actor_state=nets, critic_state=nets, collector_state=nets, eval_rng=key
    )
    first_key, second_key, new_state = agent_state.get_keys_and_new_state(2)
    drawn = np.stack([first_key, second_key, new_state.rng])
    np.testing.assert_array_equal(drawn, jax.random.split(key, 3))
    # The state drawn from is frozen: it still holds its key.
    np.testing.assert_array_equal(agent_state.rng, key)
    third_key, _ = new_state.get_keys_and_new_state()
    assert not (np.asarray(third_key) == np.asarray(second_key)).all()


def test_the_extensions_key_is_no_key_a_split_hands_out() -> None:
    """fold_in(k, i) is split(k, n)[i] under JAX's partitionable threefry:
    a small tag would hand the extensions a key the step splits off."""
    key = jax.random.PRNGKey(3)
    splits = np.asarray(jax.random.split(key, 64))
    np.testing.assert_array_equal(jax.random.fold_in(key, 5), splits[5])
    assert not (splits == np.asarray(derive_extensions_key(key))).all(axis=1).any()
