"""A replayed window of the recurrent off-policy agents (recurrent.py): a
critic's memory reads each row's observation and the action taken before
it; the action a loss asks about is a query at the critic's head, never
fed to its carry."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents.recurrent import RecurrentCarries, previous_actions, q_values
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig, zeros_carry_like
from ajax.networks.networks import MultiCritic, init_network_state

S, B, OBS, ACT = 4, 3, 2, 1
_KEYS = jax.random.split(jax.random.PRNGKey(0), 5)
OBS_SEQ = jax.random.normal(_KEYS[0], (S, B, OBS))
NEXT_OBS = jax.random.normal(_KEYS[1], (S, B, OBS))
TAKEN = jax.random.uniform(_KEYS[2], (S, B, ACT), minval=-1.0, maxval=1.0)
QUERIES = jax.random.uniform(_KEYS[3], (S, B, ACT), minval=-1.0, maxval=1.0)
# Env 0 ends an episode at row 1 and starts one at row 2.
RESETS = jnp.zeros((S, B), bool).at[2, 0].set(True)
NEXT_RESETS = jnp.zeros((S, B), bool).at[1, 0].set(True)


def _critic(kind: str) -> Any:
    memory = MemoryConfig(kind, hidden_size=8, window=4)
    critic = MultiCritic(
        input_architecture=("16", "relu"), num=2, memory=memory, query_dim=ACT
    )
    return init_network_state(
        jnp.zeros((1, OBS + 2 * ACT)), critic, _KEYS[4], optax.sgd(0.1), memory=memory
    )


def _carries(state: Any, taken: jax.Array) -> RecurrentCarries:
    """A window whose rows took ``taken``, started from fresh carries."""
    hidden = zeros_carry_like(state.hidden_state, B, batch_axis=1)
    rows_and_next = jnp.concatenate([taken, jnp.zeros_like(taken[:1])])
    starts = jnp.concatenate([RESETS, NEXT_RESETS[-1:]])
    previous = previous_actions(rows_and_next, starts)
    return RecurrentCarries(
        resets=RESETS,
        next_resets=NEXT_RESETS,
        actor_hidden=None,
        actor_next_hidden=None,
        critic_hidden=hidden,
        target_critic_hidden=hidden,
        prev_actions=previous[:-1],
        next_prev_actions=previous[1:],
    )


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_a_queried_action_moves_its_own_rows_value_only(
    kind: str, bootstrap: bool
) -> None:
    state = _critic(kind)
    carries, obs = _carries(state, TAKEN), NEXT_OBS if bootstrap else OBS_SEQ

    def q(queries: jax.Array) -> jax.Array:
        return q_values(state, state.params, obs, queries, carries, bootstrap=bootstrap)

    values, moved = q(QUERIES), q(QUERIES.at[1].add(0.5))
    rows = np.arange(S) != 1
    np.testing.assert_array_equal(values[:, rows], moved[:, rows])
    assert not np.allclose(values[:, 1], moved[:, 1])


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_the_memory_reads_the_action_taken_before_each_row(kind: str) -> None:
    """Row 1's taken action reaches the later rows of its episode (envs 1
    and 2), not row 1 itself nor env 0's next episode (row 2 on)."""
    state = _critic(kind)
    taken = q_values(state, state.params, OBS_SEQ, QUERIES, _carries(state, TAKEN))
    changed = _carries(state, TAKEN.at[1].add(0.5))
    moved = q_values(state, state.params, OBS_SEQ, QUERIES, changed)
    np.testing.assert_array_equal(taken[:, :2], moved[:, :2])
    np.testing.assert_array_equal(taken[:, 2:, 0], moved[:, 2:, 0])
    assert not np.allclose(taken[:, 2:, 1:], moved[:, 2:, 1:])
