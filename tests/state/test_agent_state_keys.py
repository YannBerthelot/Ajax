"""The keys an agent's steps draw on: ``BaseAgentState.next_keys`` and the
extensions' key a step derives (``ajax.agents.loop.extension_key``)."""

from typing import Any

import jax
import numpy as np

from ajax.agents.loop import extension_key
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


def test_the_extensions_key_is_no_key_a_split_hands_out() -> None:
    """fold_in(k, i) is split(k, n)[i] under JAX's partitionable threefry:
    a small tag would hand the extensions a key the step splits off."""
    key = jax.random.PRNGKey(3)
    splits = np.asarray(jax.random.split(key, 64))
    np.testing.assert_array_equal(jax.random.fold_in(key, 5), splits[5])
    assert not (splits == np.asarray(extension_key(key))).all(axis=1).any()
