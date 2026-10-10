"""The DreamerV3 stream replay (``replay.py``; DESIGN 6.3, dreamerv3_spec 5).

Rows are tagged with their absolute index in their env's stream so that
every read can be checked against the row it must be: windows across the
ring's wrap, items never crossing the write head, the online queue skipping
overwritten items, the batch annotation, the replay-context alignment of
Algorithm J (against a transcription run through the world model) and the
latent write-back against a sequential NumPy loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.networks import RSSM, RSSMState, observe
from ajax.agents.DreamerV3.replay import (
    RING_FIELDS,
    BatchIndex,
    ReplayState,
    StreamReplay,
    bytes_per_row,
    grow_rings,
    stoch_dtype,
)
from ajax.agents.DreamerV3.world_model import (
    PosteriorEntries,
    draw_posterior_noise,
    world_model_loss,
)
from ajax.environments.row_collector import Row

from .common import NUM_ACTIONS, OBS_DIM, TINY, tiny_model

N_ENVS, CAPACITY, BATCH, LENGTH = 3, 11, 4, 4  # windows of L = 5 rows
REPLAY = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)


def tag(env, row):
    """The value stored for row ``row`` of env ``env``."""
    return 1000.0 * env + row


def filled(
    rows: int,
    replay: StreamReplay = REPLAY,
    deter: int = 2,
    stoch: int = 2,
    classes: int = 7,
    flags=None,
) -> ReplayState:
    """``rows`` rows per env added tick by tick; every field encodes its row.

    ``obs = [tag, -tag]``, ``reward = tag``, ``action`` the row's index
    (discrete), ``deter = tag``, ``stoch = (row mod classes, env)``;
    ``flags(env, row) -> (is_first, is_last, is_terminal)`` sets the flags.
    """
    state = replay.init(2, 9, True, deter, stoch, classes)
    add = jax.jit(replay.add)
    for t in range(rows):
        envs = np.arange(replay.n_envs)
        tags = tag(envs, t).astype(np.float32)
        f = np.array([flags(e, t) if flags else (False, False, False) for e in envs])
        row = Row(
            obs=jnp.asarray(np.stack([tags, -tags], -1)),
            reward=jnp.asarray(tags),
            is_first=jnp.asarray(f[:, 0]),
            is_last=jnp.asarray(f[:, 1]),
            is_terminal=jnp.asarray(f[:, 2]),
            action=jnp.full((replay.n_envs,), t % 9, jnp.int32),
            extras=(
                jnp.asarray(np.repeat(tags[:, None], deter, -1)),
                jnp.asarray(
                    np.stack([np.full(len(envs), t % classes), envs], -1)[:, :stoch]
                ),
            ),
        )
        state = add(state, row, jnp.asarray(t))
    return state


def test_rows_are_written_in_lockstep_slots_and_wrap():
    state = filled(14)  # rows 3..13 kept in an 11-row ring
    for env in range(N_ENVS):
        for row in range(3, 14):
            assert float(state.reward[env, row % CAPACITY]) == tag(env, row)
            assert int(state.action[env, row % CAPACITY]) == row % 9
    assert state.stoch.dtype == jnp.uint8


def test_windows_are_read_across_the_wrap():
    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    state = filled(14, replay)
    start = jnp.array([9, 7, 3, 4], jnp.int32)  # 9 and 7 wrap (slots 9, 10, 0, ...)
    env = jnp.array([0, 2, 1, 2], jnp.int32)
    window = replay.gather(state, BatchIndex(env, start, jnp.zeros(4, bool)), None)
    expected = tag(np.asarray(env)[:, None], np.asarray(start)[:, None] + np.arange(5))
    np.testing.assert_array_equal(window.reward, expected)
    np.testing.assert_array_equal(window.obs[..., 0], expected)
    np.testing.assert_array_equal(window.obs[..., 1], -expected)
    np.testing.assert_array_equal(window.context_deter[:, 0], expected[:, 0])
    np.testing.assert_array_equal(window.context_stoch[:, 0], np.asarray(start) % 7)
    np.testing.assert_array_equal(window.context_stoch[:, 1], env)
    assert window.context_stoch.dtype == jnp.int32


@pytest.mark.parametrize("rows", [5, 6, 9, 11, 12, 17, 30])
def test_sampled_windows_never_cross_the_write_head(rows):
    """Uniform rows start in ``[max(0, rows - C), rows - L]``: fully written,
    not overwritten; every such item is drawn; each window reads ``L``
    consecutive rows of one env."""
    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    # The online queue drained: every row is uniform.
    state = filled(rows, replay).replace(popped=jnp.asarray(10**6, jnp.int32))
    keys = jax.random.split(jax.random.PRNGKey(rows), 400)
    _, index = jax.vmap(lambda k: replay.sample(state, jnp.asarray(rows), k))(keys)
    assert not bool(index.online.any())
    start, env = np.asarray(index.start).ravel(), np.asarray(index.env).ravel()
    oldest = max(0, rows - CAPACITY)
    assert start.min() >= oldest and start.max() <= rows - (LENGTH + 1)
    items = {(e, s) for e in range(N_ENVS) for s in range(oldest, rows - LENGTH)}
    assert set(zip(env.tolist(), start.tolist())) == items
    window = jax.vmap(lambda i: replay.gather(state, i, None))(index)
    expected = tag(
        np.asarray(index.env)[..., None],
        np.asarray(index.start)[..., None] + np.arange(LENGTH + 1),
    )
    np.testing.assert_array_equal(window.reward, expected)


def test_uniform_rows_are_drawn_with_replacement():
    """Each batch row is an independent uniform draw (2411f7d
    ``selectors.py:34-36``, one ``_sample`` per row, ``replay.py:268``): with
    6 items and batches of 4, a batch repeats an item with probability
    ``1 - 6 * 5 * 4 * 3 / 6**4 = 0.72``."""
    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    state = filled(6, replay).replace(popped=jnp.asarray(10**6, jnp.int32))
    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    _, index = jax.vmap(lambda k: replay.sample(state, jnp.asarray(6), k))(keys)
    items = np.asarray(index.env) * 100 + np.asarray(index.start)
    repeated = np.array([len(set(batch)) < BATCH for batch in items.tolist()])
    assert 0.65 < repeated.mean() < 0.79, repeated.mean()


def test_item_counts_and_online_pushes():
    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    rows = jnp.arange(0, 30)
    items = np.asarray(jax.vmap(replay.items_per_env)(rows))
    np.testing.assert_array_equal(items, np.clip(np.minimum(rows, 11) - 4, 0, None))
    # Each env pushes at its rows 5, 10, ... (0-indexed): after r rows,
    # floor((r - 1) / 5) items per env.
    pushed = np.asarray(jax.vmap(replay.online_pushed)(rows))
    np.testing.assert_array_equal(pushed, N_ENVS * (np.maximum(rows - 1, 0) // 5))
    q = jnp.arange(7)
    env, start = replay.online_item(q)
    np.testing.assert_array_equal(env, [0, 1, 2, 0, 1, 2, 0])
    np.testing.assert_array_equal(start, [1, 1, 1, 6, 6, 6, 11])


def test_online_queue_pops_pending_items_first_then_skips_overwritten_ones():
    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    state = filled(12, replay)  # pushed: starts 1 and 6 of every env = 6 items
    state, index = replay.sample(state, jnp.asarray(12), jax.random.PRNGKey(0))
    np.testing.assert_array_equal(index.online, [True] * 4)
    np.testing.assert_array_equal(index.env, [0, 1, 2, 0])
    np.testing.assert_array_equal(index.start, [1, 1, 1, 6])
    state, index = replay.sample(state, jnp.asarray(12), jax.random.PRNGKey(1))
    np.testing.assert_array_equal(index.online, [True, True, False, False])
    np.testing.assert_array_equal(index.start[:2], [6, 6])
    assert int(state.popped) == 6
    # 19 rows: start 6 is overwritten (oldest kept row: 8) and start 11 is
    # live; a fresh queue skips the dead items, as the reference's KeyError
    # retry skips evicted ones (replay.py:178-191).
    state = filled(19, replay)
    state, index = replay.sample(state, jnp.asarray(19), jax.random.PRNGKey(2))
    np.testing.assert_array_equal(index.online, [True, True, True, False])
    np.testing.assert_array_equal(index.start[:3], [11, 11, 11])
    assert int(state.popped) == 9


def test_batch_annotation_marks_the_first_row_and_abandoned_episodes():
    """``is_first[:, 0] = True`` and ``is_last |= next is_first``, the last
    column excluded (2411f7d ``replay.py:302-317``)."""

    def flags(env, row):
        return (row in (3, 8), row == 6, row == 6)

    replay = StreamReplay(N_ENVS, CAPACITY, BATCH, LENGTH)
    state = filled(12, replay, flags=flags)
    start = jnp.array([0, 2, 4, 6], jnp.int32)
    window = replay.gather(
        state, BatchIndex(jnp.zeros(4, jnp.int32), start, start > 9), None
    )
    rows = np.asarray(start)[:, None] + np.arange(5)
    stored_first = np.isin(rows, [3, 8])
    is_first = stored_first.copy()
    is_first[:, 0] = True
    next_first = np.concatenate([is_first[:, 1:], np.zeros((4, 1), bool)], 1)
    np.testing.assert_array_equal(window.is_first, is_first)
    np.testing.assert_array_equal(window.is_last, (rows == 6) | next_first)
    np.testing.assert_array_equal(window.is_terminal, rows == 6)
    # Row 2 precedes the stored is_first at 3: marked last in window 0 and 2.
    assert bool(window.is_last[0, 2]) and bool(window.is_last[1, 0])
    # The row before window start 4's forced is_first is outside the window.
    assert not bool(window.is_last[2, -1])


def test_replay_context_batch_follows_algorithm_j():
    """Carry from the latent stored with row 0, observations and flags of
    rows ``1..T``, previous actions ``action[0:T]`` (29eb964
    ``agent.py:172-183``): the world-model loss on the replay's batch gives
    exactly what a transcription of Algorithm J computes."""
    model, params, _ = tiny_model(discrete=True)
    config = TINY
    replay = StreamReplay(2, 20, 3, 6)
    rng = np.random.default_rng(0)
    state = replay.init(OBS_DIM, NUM_ACTIONS, True, config.deter, config.stoch, 3)
    for t in range(20):
        row = Row(
            obs=jnp.asarray(rng.normal(size=(2, OBS_DIM)), jnp.float32),
            reward=jnp.asarray(rng.normal(size=2), jnp.float32),
            is_first=jnp.asarray([t in (0, 9), t == 0]),
            is_last=jnp.asarray([t == 8, False]),
            is_terminal=jnp.zeros(2, bool),
            action=jnp.asarray(rng.integers(0, NUM_ACTIONS, 2), jnp.int32),
            extras=(
                jnp.asarray(rng.normal(size=(2, config.deter)), jnp.float32),
                jnp.asarray(rng.integers(0, config.classes, (2, config.stoch))),
            ),
        )
        state = replay.add(state, row, jnp.asarray(t))
    index = BatchIndex(
        env=jnp.array([0, 1, 0], jnp.int32),
        start=jnp.array([5, 2, 12], jnp.int32),
        online=jnp.zeros(3, bool),
    )
    batch = replay.gather(state, index, NUM_ACTIONS)
    noise = draw_posterior_noise(jax.random.PRNGKey(3), config, (3, 6))
    entries = world_model_loss(model, params, batch, noise).entries

    # Algorithm J, transcribed: index the stored stream directly.
    env, start = np.asarray(index.env), np.asarray(index.start)
    rows = start[:, None] + np.arange(7)

    def stored(field):
        return np.asarray(field)[env[:, None], rows]

    carry = RSSMState(
        deter=jnp.asarray(np.asarray(state.deter)[env, start]),
        stoch=jax.nn.one_hot(np.asarray(state.stoch)[env, start], config.classes),
    )
    obs = stored(state.obs)[:, 1:]
    prevact = jax.nn.one_hot(stored(state.action)[:, :-1], NUM_ACTIONS)
    is_first = stored(state.is_first)[:, 1:]
    tokens = model.apply({"params": params}, obs, method=type(model).encode)
    _, post = observe(
        RSSM(config), params["rssm"], carry, tokens, prevact, is_first, noise
    )
    np.testing.assert_array_equal(entries.deter, post.deter)
    np.testing.assert_array_equal(entries.stoch, np.argmax(post.stoch, -1))
    # The episode start inside window 0 (row 9) resets the carry there.
    assert bool(batch.is_first[0, 4]) and bool(batch.is_last[0, 3])


def test_write_back_matches_a_sequential_numpy_loop():
    """Duplicated and overlapping windows, one across the wrap: rows are
    written in batch order, later rows win, row 0 of a window is never
    written by that window (2411f7d ``replay.py:158-167``)."""
    replay = StreamReplay(2, CAPACITY, 6, LENGTH)
    state = filled(14, replay, deter=3, stoch=2)
    index = BatchIndex(
        env=jnp.array([0, 0, 1, 0, 1, 0], jnp.int32),
        start=jnp.array([4, 4, 9, 6, 9, 5], jnp.int32),  # dup, dup, overlaps
        online=jnp.zeros(6, bool),
    )
    rng = np.random.default_rng(1)
    entries = PosteriorEntries(
        deter=jnp.asarray(rng.normal(size=(6, LENGTH, 3)), jnp.float32),
        stoch=jnp.asarray(rng.integers(0, 7, (6, LENGTH, 2)), jnp.int32),
    )
    new = jax.jit(replay.write_back)(state, index, entries)

    deter, stoch = np.array(state.deter), np.array(state.stoch)
    for b in range(6):
        env, start = int(index.env[b]), int(index.start[b])
        for k in range(LENGTH):
            slot = (start + 1 + k) % CAPACITY
            deter[env, slot] = entries.deter[b, k]
            stoch[env, slot] = entries.stoch[b, k]
    np.testing.assert_array_equal(new.deter, deter)
    np.testing.assert_array_equal(new.stoch, stoch)
    # Row 4 of env 0 is the context of rows 0 and 1 and of no other window.
    np.testing.assert_array_equal(new.deter[0, 4], state.deter[0, 4])
    # The other fields are untouched.
    np.testing.assert_array_equal(new.obs, state.obs)


@pytest.mark.parametrize("classes", [7, 256, 300])
def test_stochastic_latents_round_trip_as_class_indices(classes):
    assert stoch_dtype(classes) == (jnp.uint8 if classes <= 256 else jnp.int32)
    replay = StreamReplay(1, 6, 1, 2)
    state = replay.init(1, 1, False, 1, 3, classes)
    indices = jnp.array([[0, classes - 1, classes // 2]], jnp.int32)
    row = Row(
        obs=jnp.zeros((1, 1)),
        reward=jnp.zeros(1),
        is_first=jnp.ones(1, bool),
        is_last=jnp.zeros(1, bool),
        is_terminal=jnp.zeros(1, bool),
        action=jnp.zeros((1, 1)),
        extras=(jnp.zeros((1, 1)), indices),
    )
    state = replay.add(state, row, jnp.asarray(0))
    assert state.stoch.dtype == stoch_dtype(classes)
    window = replay.gather(
        state,
        BatchIndex(
            jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32), jnp.zeros(1, bool)
        ),
        None,
    )
    np.testing.assert_array_equal(window.context_stoch, indices)
    entries = PosteriorEntries(jnp.zeros((1, 2, 1)), jnp.full((1, 2, 3), classes - 1))
    state = replay.write_back(
        state,
        BatchIndex(
            jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32), jnp.zeros(1, bool)
        ),
        entries,
    )
    np.testing.assert_array_equal(state.stoch[0, 1:3], classes - 1)


def test_continuous_actions_are_stored_raw_and_passed_through():
    replay = StreamReplay(1, 6, 1, 2)
    state = replay.init(1, 2, False, 1, 1, 2)
    action = jnp.array([[1.7, -3.0]])  # outside [-1, 1]: stored unclipped
    row = Row(
        obs=jnp.zeros((1, 1)),
        reward=jnp.zeros(1),
        is_first=jnp.ones(1, bool),
        is_last=jnp.zeros(1, bool),
        is_terminal=jnp.zeros(1, bool),
        action=action,
        extras=(jnp.zeros((1, 1)), jnp.zeros((1, 1), jnp.int32)),
    )
    state = replay.add(state, row, jnp.asarray(0))
    index = BatchIndex(
        jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32), jnp.zeros(1, bool)
    )
    batch = replay.gather(state, index, None)
    np.testing.assert_array_equal(batch.action[0, 0], action[0])


def test_a_grown_ring_is_the_ring_of_that_length():
    """Before any row is overwritten, a ring grown to ``C'`` rows is the
    ring a ``C'``-row replay holds after the same rows (row ``a`` at slot
    ``a``), with the seed axis kept and the online counter unchanged -- so
    every later add, draw and gather is the longer replay's."""
    short, long = (
        StreamReplay(N_ENVS, 9, BATCH, LENGTH),
        StreamReplay(N_ENVS, 15, BATCH, LENGTH),
    )
    seeds = 2

    def batched(state):
        return jax.tree.map(lambda x: jnp.stack([x] * seeds), state)

    state = batched(filled(7, short)).replace(popped=jnp.array([3, 3], jnp.int32))
    grown = grow_rings(state, 7, 15)
    expected = batched(filled(7, long)).replace(popped=jnp.array([3, 3], jnp.int32))
    for name in (*RING_FIELDS, "popped"):
        ours, theirs = getattr(grown, name), getattr(expected, name)
        assert ours.dtype == theirs.dtype, name
        np.testing.assert_array_equal(ours, theirs, err_msg=name)
    # Growing a full but unwrapped ring is exact too; a wrapped one cannot grow.
    np.testing.assert_array_equal(
        grow_rings(filled(9, short), 9, 15).reward, filled(9, long).reward
    )
    with pytest.raises(ValueError, match="overwritten"):
        grow_rings(filled(10, short), 10, 15)
    with pytest.raises(ValueError, match="shrink"):
        grow_rings(filled(7, long), 7, 9)


def test_ring_geometry_is_validated_and_sized():
    with pytest.raises(ValueError, match="fewer than one window"):
        StreamReplay(2, 4, 1, 4)
    with pytest.raises(ValueError, match="batch_size"):
        StreamReplay(2, 8, 0, 4)
    # 12m on a 24-dim observation with 6 actions: 8.3 KB per row.
    assert bytes_per_row(24, 6, False, 2048, 32, 16) == 8351
