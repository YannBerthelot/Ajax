"""The DreamerV3 update schedule against a port of the 2411f7d driver loop.

``TrainRatio`` (``train_DreamerV3.py``) is static arithmetic on the tick; the
oracle is the transition-level reference loop (``reference_loop.py``:
``train.py`` + ``when.Ratio`` + ``Replay.add``). Pinned: the first update's
tick, the per-tick counts and the totals for ``n_envs`` in ``{1, 4, 16,
32}``, and the online items each batch pops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.replay import ReplayState, StreamReplay
from ajax.agents.DreamerV3.train_DreamerV3 import TrainRatio

from .reference_loop import Ratio, reference_train_loop

#: (batch_size, batch_length, train_ratio): the paper's DMC-proprio setting
#: (2411f7d defaults with ``dmc_proprio``'s ratio 512), the default ratio 32,
#: and small geometries, one at one and one at two updates per transition.
#: Every ratio ``r`` is a power of two, as in every reference configuration,
#: so that the reference's float bookkeeping ``prev += repeats / r`` is exact
#: (:func:`test_inexact_ratios_differ_by_at_most_one_update` covers the rest).
SETTINGS = [
    (16, 64, 512),
    (16, 64, 32),
    (4, 8, 64),
    (3, 5, 15),
]


def ticks_for(schedule: TrainRatio) -> int:
    """Enough ticks to cover the gate and a few hundred transitions after."""
    return schedule.first_update_tick + 1 + max(400 // schedule.n_envs, 8)


@pytest.mark.parametrize("n_envs", [1, 4, 16, 32])
@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: "B%d-T%d-r%g" % s)
def test_updates_per_tick_match_the_reference_loop(n_envs, setting):
    batch_size, batch_length, train_ratio = setting
    schedule = TrainRatio(n_envs, batch_size, batch_length, train_ratio)
    ticks = ticks_for(schedule)
    reference, _ = reference_train_loop(
        n_envs, batch_size, batch_length, train_ratio, ticks
    )
    ours = jax.vmap(schedule.updates_in_tick)(jnp.arange(ticks))
    np.testing.assert_array_equal(np.asarray(ours), reference)
    first = int(np.flatnonzero(reference)[0])
    assert schedule.first_update_tick == first
    for t in (first, first + 1, ticks // 2, ticks):
        assert schedule.total_updates(t) == sum(reference[:t])


def test_paper_protocol_first_update_is_at_tick_64():
    """16 envs, batch 16 x 64: the 16th item arrives with the last env's row
    of tick 64 (step 1040), where ``Ratio`` returns exactly 1."""
    schedule = TrainRatio(16, 16, 64, 512)
    assert schedule.first_update_step == 1040
    assert schedule.first_update_tick == 64
    counts = np.asarray(jax.vmap(schedule.updates_in_tick)(jnp.arange(70)))
    assert counts[:64].sum() == 0
    assert counts[64] == 1
    # 0.5 updates per transition afterwards: 8 per tick of 16 rows.
    assert list(counts[65:70]) == [8] * 5


def test_ratio_port_returns_one_on_its_first_call():
    ratio = Ratio(0.5)
    assert ratio(1040) == 1
    assert [ratio(1040 + k) for k in range(1, 6)] == [0, 1, 0, 1, 0]


def test_large_ticks_do_not_overflow():
    """The closed form never multiplies the step by the ratio's numerator."""
    schedule = TrainRatio(16, 16, 64, 512)
    tick = 50_000_000  # 8e8 rows
    expected = schedule.total_updates(tick + 1) - schedule.total_updates(tick)
    assert int(schedule.updates_in_tick(jnp.asarray(tick))) == expected == 8


@pytest.mark.parametrize("setting", [(4, 5, 6), (16, 64, 100)], ids=str)
def test_inexact_ratios_differ_by_at_most_one_update(setting):
    """Where ``1 / r`` is not a float (0.3, 100 / 1024), the reference's
    ``prev += repeats / r`` rounds and sometimes runs an update one
    transition late; the exact closed form is then at most one update ahead,
    never behind (deviation D26)."""
    batch_size, batch_length, train_ratio = setting
    for n_envs in (1, 16):
        schedule = TrainRatio(n_envs, batch_size, batch_length, train_ratio)
        ticks = schedule.first_update_tick + 3000
        reference, _ = reference_train_loop(
            n_envs, batch_size, batch_length, train_ratio, ticks
        )
        ours = np.asarray(jax.vmap(schedule.updates_in_tick)(jnp.arange(ticks)))
        lead = np.cumsum(ours) - np.cumsum(reference)
        assert lead.min() == 0 and lead.max() <= 1
        assert (ours != reference).any()  # the rounding does show


def test_schedule_validates_its_configuration():
    with pytest.raises(ValueError, match="train_ratio"):
        TrainRatio(4, 16, 64, 0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainRatio(4, 0, 64, 512)
    with pytest.raises(ValueError, match="int32"):
        TrainRatio(4, 16, 64, 0.1)  # 0.1's binary fraction is not small


@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_online_queue_pops_match_the_reference_replay(n_envs):
    """The replay's online pops, batch after batch, are the reference's.

    The reference loop runs with each tick's updates after its adds (D3), so
    both sample from the same rows; the schedule's counts drive our replay.
    A low ratio lets the queue fill up between updates, so batches mix
    online and uniform rows.
    """
    batch_size, batch_length, train_ratio = 4, 5, 5
    schedule = TrainRatio(n_envs, batch_size, batch_length, train_ratio)
    ticks = schedule.first_update_tick + 60
    _, reference = reference_train_loop(
        n_envs, batch_size, batch_length, train_ratio, ticks, updates_after_tick=True
    )
    replay = StreamReplay(n_envs, ticks, batch_size, batch_length)
    state = ReplayState(
        **{
            k: jnp.zeros((n_envs, ticks), jnp.float32)
            for k in ("obs", "action", "reward", "is_first", "is_last")
        },
        is_terminal=jnp.zeros((n_envs, ticks), bool),
        deter=jnp.zeros((n_envs, ticks, 1)),
        stoch=jnp.zeros((n_envs, ticks, 1), jnp.uint8),
        popped=jnp.zeros((), jnp.int32),
    )
    sample = jax.jit(replay.sample)
    ours = []
    key = jax.random.PRNGKey(0)
    for tick in range(ticks):
        for _ in range(int(schedule.updates_in_tick(jnp.asarray(tick)))):
            key, sub = jax.random.split(key)
            state, index = sample(state, jnp.asarray(tick + 1), sub)
            ours.append(
                [
                    (int(e), int(s)) if bool(o) else None
                    for e, s, o in zip(index.env, index.start, index.online)
                ]
            )
    assert len(ours) == len(reference)
    assert ours == reference
    online = sum(row is not None for batch in ours for row in batch)
    assert 0 < online < len(ours) * batch_size  # both kinds of rows occur
