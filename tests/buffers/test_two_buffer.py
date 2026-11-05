"""The two-buffer sampler: a replay's ring beside a ring of retained rows.

Contract, from RLPD (Ball et al. 2023, Algorithm 1, lines 12-14) and DDPGfD
(Vecerik et al. 2017, sec. 3): a fixed share gives N/2 rows from each buffer
in every batch; rows in the retained ring outlive every row collection
writes; with no fixed share each row is drawn over all the rows held. The
rows here say where they come from: a positive reward is a retained row.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.buffers.utils import (
    TwoBufferState,
    empty_ring_like,
    get_batch_from_buffer,
    get_buffer,
    with_retained_rows,
)
from ajax.types import NormalizationInfo

SCHEMA = {
    field: jnp.zeros(width)
    for field, width in (
        ("obs", 2),
        ("action", 1),
        ("reward", 1),
        ("terminated", 1),
        ("truncated", 1),
        ("next_obs", 2),
    )
}
BATCH = 64


def add_rows(buffer: Any, state: Any, rewards: jax.Array) -> Any:
    """rows of reward rewards, one add per row, an env at a time."""
    for reward in rewards:
        row = {k: jnp.full((1, *v.shape), reward) for k, v in SCHEMA.items()}
        state = buffer.add(state, row)
    return state


def rings(share: Any, online: int = 8, retained: int = 4, capacity: int = 16) -> tuple:
    """A buffer and its state: rows of reward -1, ..., -online in a ring of
    capacity rows, rows of reward 1, ..., retained kept."""
    buffer = get_buffer(128, BATCH)
    state = empty_ring_like(buffer.init(SCHEMA), capacity)
    kept = jnp.arange(1.0, retained + 1)
    kept = add_rows(buffer, empty_ring_like(state, retained + 1), kept)
    state = add_rows(buffer, state, -jnp.arange(1.0, online + 1))
    return buffer, with_retained_rows(state, kept, share)


def test_a_fixed_share_is_exact_in_every_batch() -> None:
    """A share drawn per row would give a batch 32 +- 6 rows of each: RLPD's
    N/2 + N/2 varies by none."""
    buffer, state = rings(0.5)
    for key in jax.random.split(jax.random.PRNGKey(0), 32):
        rewards = np.asarray(buffer.sample(state, key).experience["reward"])
        assert (rewards > 0).sum() == BATCH // 2
        assert (rewards < 0).sum() == BATCH // 2


def test_each_ring_gives_only_rows_it_holds() -> None:
    """An unwritten slot's zeros reaching a batch would teach the critics a
    state nothing visited: both rings stop at their written rows."""
    buffer, state = rings(0.5, online=1)
    rewards = np.asarray(
        buffer.sample(state, jax.random.PRNGKey(1)).experience["reward"]
    )
    assert set(rewards[rewards < 0]) == {-1.0}
    assert set(rewards[rewards > 0]).issubset({1.0, 2.0, 3.0, 4.0})


def test_retained_rows_outlive_every_row_collection_writes() -> None:
    """DDPGfD "keeps all transitions forever": a run writing more rows than
    the ring holds evicts its own rows and none of the retained ones."""
    buffer, state = rings(0.5)
    state = add_rows(buffer, state, -jnp.arange(10.0, 60.0))
    assert isinstance(state, TwoBufferState) and bool(state.is_full)
    np.testing.assert_array_equal(
        state.retained.experience["reward"][0, :4, 0], [1.0, 2.0, 3.0, 4.0]
    )
    rewards = np.asarray(
        buffer.sample(state, jax.random.PRNGKey(2)).experience["reward"]
    )
    assert set(rewards[rewards > 0]) == {1.0, 2.0, 3.0, 4.0}
    # The ring of 16 holds the newest of those 50 rows, -44 to -59.
    assert rewards[rewards < 0].max() <= -44.0


@pytest.mark.parametrize(("online", "retained"), [(4, 4), (12, 4), (2, 8)])
def test_no_fixed_share_draws_over_all_rows_held(online: int, retained: int) -> None:
    """A row is retained with probability retained / (retained + online). Over
    64 batches of 64 rows the share's standard error is at most 0.5 / 64, so
    0.04 is five standard errors: a wrong ratio of the two rings fails it."""
    buffer, state = rings(None, online, retained)
    keys = jax.random.split(jax.random.PRNGKey(3), 64)
    rewards = jax.vmap(lambda k: buffer.sample(state, k).experience["reward"])(keys)
    share = float(jnp.mean(rewards > 0))
    assert abs(share - retained / (retained + online)) < 0.04


def test_retained_rows_keep_their_rewards_up_to_the_agents_normaliser() -> None:
    """RLPD keeps the logged rewards: a replayed batch divides the rows of
    both rings by one set of statistics, 2 here, and changes them no other
    way. Dividing only one ring would leave two reward scales in a batch."""
    buffer, state = rings(0.5)
    var = jnp.array([[4.0]])
    stats = NormalizationInfo(
        var=var,
        count=jnp.ones((1, 1)),
        mean=jnp.zeros((1, 1)),
        mean_2=var,
        returns=None,
    )
    plain = get_batch_from_buffer(buffer, state, jax.random.PRNGKey(6))
    divided = get_batch_from_buffer(buffer, state, jax.random.PRNGKey(6), stats)
    assert set(np.asarray(plain.reward)[plain.reward > 0]) == {1.0, 2.0, 3.0, 4.0}
    np.testing.assert_allclose(divided.reward, plain.reward / 2.0, rtol=1e-3)
