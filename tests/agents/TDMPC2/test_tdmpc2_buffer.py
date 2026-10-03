"""TD-MPC2 episode ring (``ajax.agents.TDMPC2.buffer``): slice conversion,
committed rounds, the staging slot, wrap-around, sampling, capacity, and the
adoption of a carried ring by a resumed run.

The reference is ``nicklashansen/tdmpc2@b67b21c:tdmpc2/common/buffer.py``
(whole episodes, ``RandomSampler`` + ``RandomCropTensorDict(H + 1)`` +
``DataPrepTransform``) fed by ``trainer/online_trainer.py:50-65, 93-102``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2.buffer import (
    EpisodeBuffer,
    replay_capacity,
    slice_batch,
)

T, H, OBS, ACT = 6, 3, 2, 1


def reference_episode(obs, actions, rewards):
    """b67b21c's rows of one finished episode (``online_trainer.py:50-65``):
    row 0 = ``(o_0, empty, NaN)`` (``to_td(obs)`` after the reset), row
    ``k >= 1`` = ``(o_k, a_{k-1}, r_{k-1})``."""
    dummy_action = np.full((1, actions.shape[-1]), np.nan, np.float32)
    return (
        obs,
        np.concatenate([dummy_action, actions]),
        np.concatenate([[np.nan], rewards]).astype(np.float32),
    )


def data_prep(episode, start, horizon):
    """``RandomCropTensorDict(H + 1)`` at ``start`` then ``DataPrepTransform``
    (``buffer.py:26-28``): ``obs``, ``action[1:]``, ``reward[1:]``."""
    obs, action, reward = episode
    crop = slice(start, start + horizon + 1)
    return obs[crop], action[crop][1:], reward[crop][1:]


def collector_rows(obs, actions, rewards):
    """The same episode as the row collector emits it (``DESIGN.md`` §5.2):
    row ``k`` = ``(o_k, a_k, r_{k-1})``, action 0 on the final row, reward 0
    on the reset row."""
    zero_a = np.zeros((1, actions.shape[-1]), np.float32)
    return (
        obs,
        np.concatenate([actions, zero_a]),
        np.concatenate([[0.0], rewards]).astype(np.float32),
    )


def random_episode(rng, t=T):
    return (
        rng.normal(size=(t + 1, OBS)).astype(np.float32),
        rng.uniform(-1, 1, size=(t, ACT)).astype(np.float32),
        rng.normal(size=(t,)).astype(np.float32),
    )


def write_round(buffer, state, episodes, round_index):
    """Write one round (one episode per env) tick by tick, as the agent does."""
    start = round_index * buffer.period
    return write_ticks(
        buffer, state, {round_index: episodes}, start, start + buffer.period
    )


def write_ticks(buffer, state, rounds, start, stop):
    """Write ticks ``start .. stop - 1`` of the rounds ``{index: episodes}``."""
    period = buffer.period
    for tick in range(start, stop):
        rows = [collector_rows(*ep) for ep in rounds[tick // period]]
        k = tick % period
        state = buffer.add(
            state,
            jnp.stack([r[0][k] for r in rows]),
            jnp.stack([r[1][k] for r in rows]),
            jnp.stack([r[2][k] for r in rows]),
            tick,
        )
    return state


def test_slices_are_data_prep_transform_of_the_reference_rows():
    """Every offset of a stored episode gives b67b21c's training slice."""
    rng = np.random.default_rng(0)
    buffer = EpisodeBuffer.create(
        capacity=T, n_envs=1, episode_length=T, obs_dim=OBS, action_dim=ACT
    )
    episode = random_episode(rng)
    state = write_round(buffer, buffer.init(), [episode], 0)
    reference = reference_episode(*episode)
    for start in range(T - H + 1):
        batch = slice_batch(state, jnp.array([0]), jnp.array([start]), H)
        obs, action, reward = data_prep(reference, start, H)
        np.testing.assert_array_equal(batch.obs[:, 0], obs)
        np.testing.assert_array_equal(batch.action[:, 0], action)
        np.testing.assert_array_equal(batch.reward[:, 0], reward)
    assert batch.obs.shape == (H + 1, 1, OBS)
    assert batch.action.shape == (H, 1, ACT)
    assert batch.reward.shape == (H, 1)


def test_rounds_commit_on_their_held_tick_and_staging_is_excluded():
    n = 2
    buffer = EpisodeBuffer.create(
        capacity=2 * T * n, n_envs=n, episode_length=T, obs_dim=OBS, action_dim=ACT
    )
    assert buffer.n_rounds == 3  # two committed rounds + staging
    period = buffer.period
    assert buffer.num_episodes(period - 2) == 0  # round 0 not finished
    assert buffer.num_episodes(period - 1) == n  # its held tick written
    assert buffer.num_episodes(2 * period - 2) == n  # round 1 in progress
    assert buffer.num_episodes(2 * period - 1) == 2 * n
    assert buffer.num_episodes(10 * period) == 2 * n  # R - 1 rounds at most
    # Traced ticks give the same counts.
    ticks = jnp.arange(4 * period)
    np.testing.assert_array_equal(
        jax.vmap(buffer.num_episodes)(ticks),
        [buffer.num_episodes(int(t)) for t in ticks],
    )

    # Mid-round 1: only round 0's slots are drawn, never the staging round.
    rng = np.random.default_rng(1)
    state = write_round(buffer, buffer.init(), [random_episode(rng)] * n, 0)
    tick = period + 2
    slots = buffer.episode_slot(tick, jnp.arange(buffer.num_episodes(tick)))
    np.testing.assert_array_equal(slots, [0, 1])
    batch = buffer.sample(state, jax.random.PRNGKey(0), tick, 512, H)
    assert np.all(np.isfinite(batch.obs))
    # Mid-round 3 with a full ring: rounds 2, 1 committed (newest first);
    # round 0's slot is the staging slot being overwritten.
    tick = 3 * period + 2
    slots = buffer.episode_slot(tick, jnp.arange(buffer.num_episodes(tick)))
    np.testing.assert_array_equal(slots, [4, 5, 2, 3])


def test_the_ring_wraps_and_keeps_the_newest_rounds():
    """Five rounds through a ring of R = 3: rounds 3 and 4 are committed
    (slots 0, 1 and 2, 3); the staging slot of round 5 still holds round 2
    and is never drawn."""
    n = 2
    buffer = EpisodeBuffer.create(
        capacity=2 * T * n, n_envs=n, episode_length=T, obs_dim=OBS, action_dim=ACT
    )
    state = buffer.init()
    rng = np.random.default_rng(2)
    episodes = {}
    for r in range(5):
        episodes[r] = [random_episode(rng) for _ in range(n)]
        state = write_round(buffer, state, episodes[r], r)
    last_tick = 5 * buffer.period - 1
    assert buffer.committed_rounds(last_tick) == 2
    expected = {}  # slot -> reference episode
    for j in range(buffer.num_episodes(last_tick)):
        r = 4 - j // n
        expected[int(buffer.episode_slot(last_tick, jnp.array(j)))] = episodes[r][j % n]
    assert sorted(expected) == [0, 1, 2, 3]  # rounds 3 (slots 0, 1) and 4 (2, 3)
    for slot, episode in expected.items():
        reference = reference_episode(*episode)
        for start in range(T - H + 1):
            batch = slice_batch(state, jnp.array([slot]), jnp.array([start]), H)
            obs, action, reward = data_prep(reference, start, H)
            np.testing.assert_array_equal(batch.obs[:, 0], obs)
            np.testing.assert_array_equal(batch.action[:, 0], action)
            np.testing.assert_array_equal(batch.reward[:, 0], reward)
    # Slots 4, 5 hold round 2 (the staging slot of round 5), never sampled.
    batch = buffer.sample(state, jax.random.PRNGKey(3), last_tick, 4096, H)
    sampled_first_obs = np.asarray(batch.obs[0])
    staged = np.stack([ep[0] for ep in episodes[2]])  # [n, T + 1, OBS]
    assert not np.isin(sampled_first_obs[:, 0], staged[..., 0]).any()


def test_sampling_is_uniform_over_committed_episodes_and_offsets():
    n = 3
    buffer = EpisodeBuffer.create(
        capacity=4 * T * n, n_envs=n, episode_length=T, obs_dim=OBS, action_dim=ACT
    )
    state = buffer.init()
    # obs = (episode id, row index), so a sample identifies its episode and offset
    for r in range(2):
        episodes = []
        for e in range(n):
            ident = np.full((T + 1, 1), r * n + e, np.float32)
            rows = np.arange(T + 1, dtype=np.float32)[:, None]
            episodes.append(
                (
                    np.concatenate([ident, rows], axis=1),
                    np.zeros((T, ACT), np.float32),
                    np.zeros(T, np.float32),
                )
            )
        state = write_round(buffer, state, episodes, r)
    tick = 2 * buffer.period + 1  # round 2 in progress: 2 rounds committed
    draws = 60_000
    batch = jax.jit(buffer.sample, static_argnums=(3, 4))(
        state, jax.random.PRNGKey(4), tick, draws, H
    )
    ident = np.asarray(batch.obs[0, :, 0]).astype(int)
    start = np.asarray(batch.obs[0, :, 1]).astype(int)
    rows = np.asarray(batch.obs[:, :, 1]) - start  # consecutive rows s .. s + H
    np.testing.assert_array_equal(
        rows, np.broadcast_to(np.arange(H + 1)[:, None], rows.shape)
    )
    episodes = 2 * n
    counts = np.bincount(ident, minlength=episodes)
    assert counts.size == episodes
    np.testing.assert_allclose(counts / draws, 1 / episodes, atol=0.01)
    offsets = np.bincount(start, minlength=T - H + 1)
    assert offsets.size == T - H + 1  # s in [0, T - H]
    np.testing.assert_allclose(offsets / draws, 1 / (T - H + 1), atol=0.01)


def test_zero_committed_episodes_cannot_be_sampled_from():
    buffer = EpisodeBuffer.create(
        capacity=100, n_envs=2, episode_length=T, obs_dim=OBS, action_dim=ACT
    )
    for tick in range(T):  # before the first held tick
        with pytest.raises(ValueError, match="before any episode is committed"):
            buffer.check_sampleable_from(tick)
    buffer.check_sampleable_from(T)  # round 0's held tick
    with pytest.raises(ValueError, match="horizon"):
        buffer.sample(buffer.init(), jax.random.PRNGKey(0), T, 4, T + 1)


def test_capacity_and_memory():
    assert replay_capacity(1_000_000, 300_000) == 300_000
    assert replay_capacity(1_000_000, 3_000_000) == 1_000_000
    assert replay_capacity(50, 0) == 1  # the n_timesteps=0 skeleton: smallest
    with pytest.raises(ValueError):
        replay_capacity(0, 10)
    # b67b21c DMC: 1e6 // 500 = 2000 episodes, + the staging round.
    walker = EpisodeBuffer.create(
        capacity=1_000_000, n_envs=1, episode_length=500, obs_dim=24, action_dim=6
    )
    assert walker.n_rounds == 2001
    assert walker.nbytes == 2001 * 501 * (24 + 6 + 1) * 4  # ~124 MB per seed
    # ceil: a partial round of capacity still gets a full round.
    assert (
        EpisodeBuffer.create(
            capacity=7, n_envs=2, episode_length=3, obs_dim=1, action_dim=1
        ).n_rounds
        == 3
    )
    with pytest.raises(ValueError, match="n_rounds"):
        EpisodeBuffer(n_envs=1, episode_length=3, n_rounds=1, obs_dim=1, action_dim=1)


# ---------------------------------------------------------------------------
# A resumed run adopts the ring it carries
# ---------------------------------------------------------------------------


def _ring(n_rounds, n=2):
    return EpisodeBuffer.create(
        capacity=(n_rounds - 1) * T * n,
        n_envs=n,
        episode_length=T,
        obs_dim=OBS,
        action_dim=ACT,
    )


def test_a_chunked_ring_replays_what_the_uninterrupted_ring_does():
    """A first chunk sized for its own 2.5 rounds (R = 3) is resumed mid
    round 2 into the ring of the whole run (R = 5): after five rounds the
    adopted ring holds and samples exactly what the uninterrupted ring does
    (rounds 1-4), while the first chunk's ring alone would have evicted
    rounds 1 and 2."""
    small, big = _ring(3), _ring(5)
    assert (small.n_rounds, big.n_rounds) == (3, 5)
    rng = np.random.default_rng(5)
    rounds = {r: [random_episode(rng) for _ in range(2)] for r in range(5)}
    resume, end = 2 * big.period + 3, 5 * big.period
    carried = write_ticks(small, small.init(), rounds, 0, resume)
    adopted = write_ticks(big, big.adopt(carried, resume), rounds, resume, end)
    reference = write_ticks(big, big.init(), rounds, 0, end)
    last = end - 1
    assert big.committed_rounds(last) == 4
    committed = np.asarray(big.episode_slot(last, jnp.arange(big.num_episodes(last))))
    for field in ("obs", "action", "reward"):
        np.testing.assert_array_equal(
            getattr(adopted, field)[committed], getattr(reference, field)[committed]
        )
    key = jax.random.PRNGKey(6)
    for x, y in zip(
        jax.tree.leaves(big.sample(adopted, key, last, 64, H)),
        jax.tree.leaves(big.sample(reference, key, last, 64, H)),
        strict=True,
    ):
        np.testing.assert_array_equal(x, y)
    # The carried round in progress keeps its written rows.
    np.testing.assert_array_equal(
        big.adopt(carried, resume).obs[4:6, :3], carried.obs[4:6, :3]
    )


def test_adopting_keeps_seed_axes_and_shares_an_equal_ring():
    small, big = _ring(3), _ring(5)
    rng = np.random.default_rng(7)
    rounds = {r: [random_episode(rng) for _ in range(2)] for r in range(2)}
    tick = small.period + 2
    state = write_ticks(small, small.init(), rounds, 0, tick)
    assert small.adopt(state, tick) is state  # same size: unchanged
    seeds = jax.tree.map(lambda x: jnp.stack([x, 2 * x]), state)
    adopted = big.adopt(seeds, tick)
    assert adopted.obs.shape == (2, big.n_slots, T + 1, OBS)
    for i in range(2):
        one = big.adopt(jax.tree.map(lambda x, i=i: x[i], seeds), tick)
        for x, y in zip(jax.tree.leaves(one), jax.tree.leaves(adopted)):
            np.testing.assert_array_equal(x, y[i])
    # A smaller ring keeps the newest rounds it reads.
    shrunk = EpisodeBuffer.create(
        capacity=T * 2, n_envs=2, episode_length=T, obs_dim=OBS, action_dim=ACT
    ).adopt(adopted, tick)
    assert shrunk.obs.shape[1] == 2 * 2


def test_adopting_refuses_evicted_rounds_and_foreign_layouts():
    small, big = _ring(3), _ring(5)
    rng = np.random.default_rng(8)
    rounds = {r: [random_episode(rng) for _ in range(2)] for r in range(4)}
    # Four rounds through R = 3: rounds 0 and 1 are gone, but R = 5 reads
    # rounds 0-3 at tick 4 P (a smaller buffer_size built the carried ring).
    state = write_ticks(small, small.init(), rounds, 0, 4 * small.period)
    with pytest.raises(ValueError, match="smaller buffer_size"):
        big.adopt(state, 4 * small.period)
    # Episodes of another length, or slots that are not whole rounds.
    other = EpisodeBuffer.create(
        capacity=8, n_envs=2, episode_length=T + 1, obs_dim=OBS, action_dim=ACT
    )
    with pytest.raises(ValueError, match=r"T \+ 1"):
        big.adopt(other.init(), 0)
    with pytest.raises(ValueError, match="rounds"):
        big.adopt(_ring(3, n=3).init(), 0)
    # Writing or sampling a ring of another size without adopting it.
    with pytest.raises(ValueError, match="adopted"):
        big.add(state, jnp.zeros((2, OBS)), jnp.zeros((2, ACT)), jnp.zeros(2), 0)
    with pytest.raises(ValueError, match="adopted"):
        big.sample(state, jax.random.PRNGKey(0), small.period, 4, H)
