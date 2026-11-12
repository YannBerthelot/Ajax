"""update_steps.scan_on_indexed_keys, the inner loop of SAC's and REDQ's
critic steps and of DQN's gradient steps.

Its contract (the docstring, and #421's restructure): length steps of
step(state, None) -> (state, aux); step i runs on the updates' key stream
restarted from the stream's next key folded with i (jax.random.fold_in);
the loop returns the last step's aux and the state whose stream has
advanced by that one key, whatever the length; a length below 1 raises.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.update_steps import scan_on_indexed_keys
from ajax.state import BaseAgentState

KEY = jax.random.PRNGKey(0)
ROWS = 5
#: The networks and the collector, which the loop never reads.
UNREAD: Any = None


def _state() -> BaseAgentState:
    """A state whose updates' key and two counters are all the loop reads
    and writes: n_updates counts the steps, ext_state holds each step's key
    at its index."""
    return BaseAgentState(
        rng=KEY,
        actor_state=UNREAD,
        critic_state=UNREAD,
        collector_state=UNREAD,
        eval_rng=KEY,
        n_updates=jnp.asarray(0),
        ext_state=(jnp.zeros((ROWS, *KEY.shape), KEY.dtype),),
    )


def _record_key(state: BaseAgentState, _: None) -> tuple:
    """A step that writes the key it runs on at its own index and returns
    that key as its aux."""
    seen = state.ext_state[0].at[state.n_updates].set(state.rng)
    return state.replace(n_updates=state.n_updates + 1, ext_state=(seen,)), state.rng


def _steps_key() -> jax.Array:
    """The stream's next key, which the loop's steps fold their index into."""
    return _state().get_keys_and_new_state()[0]


def test_step_i_runs_on_the_streams_next_key_folded_with_i() -> None:
    """Catches every critic step drawing the same key (an unfolded key),
    or a step skipped or repeated by the loop."""
    stepped, _ = scan_on_indexed_keys(_record_key, _state(), ROWS)
    expected = [jax.random.fold_in(_steps_key(), i) for i in range(ROWS)]
    np.testing.assert_array_equal(stepped.ext_state[0], np.stack(expected))
    assert int(stepped.n_updates) == ROWS


@pytest.mark.parametrize("length", [1, 3])
def test_it_returns_the_last_steps_aux(length: int) -> None:
    """Catches the logs of an earlier step reported as the update's (the
    loop keeps the last step's aux alone)."""
    _, aux = scan_on_indexed_keys(_record_key, _state(), length)
    np.testing.assert_array_equal(aux, jax.random.fold_in(_steps_key(), length - 1))


@pytest.mark.parametrize("length", [1, 2, ROWS])
def test_the_stream_advances_by_one_key_whatever_the_length(length: int) -> None:
    """Catches a stream advanced once per step, which would move every later
    draw of a run whenever its count of critic steps changes."""
    stepped, _ = scan_on_indexed_keys(_record_key, _state(), length)
    after_one_draw = _state().get_keys_and_new_state()[1]
    np.testing.assert_array_equal(stepped.rng, after_one_draw.rng)


@pytest.mark.parametrize("length", [0, -1])
def test_a_length_below_one_raises(length: int) -> None:
    """Catches an update with no critic step at all passing silently."""
    with pytest.raises(ValueError, match="at least 1"):
        scan_on_indexed_keys(_record_key, _state(), length)
