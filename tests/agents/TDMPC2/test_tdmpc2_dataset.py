"""Offline multi-task datasets (M8): export, pooling, schema, sampling.

:mod:`ajax.agents.TDMPC2.dataset` (``docs/world_models/DESIGN.md`` §7;
tdmpc2_spec 4.18-4.19): a tiny single-task TD-MPC2 run on the counter env
(whose rows are recoverable from its actions) is exported, pooled with a
second task of other dims and read back; the export's slot order is checked
on a wrapped ring; the sampler is b67b21c's uniform episode x uniform crop
over the pooled episodes.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2
from ajax.agents.TDMPC2.buffer import EpisodeBuffer, EpisodeBufferState, slice_batch
from ajax.agents.TDMPC2.dataset import (
    MultiTaskDataset,
    TaskEpisodes,
    concatenate_episodes,
    export_episodes,
    pool_tasks,
)

from .toy_envs import CounterEnv

T = 6  # rows T + 1 = 7
TINY = {
    "enc_dim": 16,
    "mlp_dim": 16,
    "latent_dim": 8,
    "num_q": 2,
    "batch_size": 4,
    "num_samples": 8,
    "num_elites": 2,
    "num_pi_trajs": 2,
    "iterations": 1,
}


def synthetic(n, obs_dim, action_dim, name, rows=T + 1, seed=0):
    """Episodes in the schema's layout: action 0 on the last row, reward 0
    on row 0."""
    rng = np.random.default_rng(seed)
    action = rng.uniform(-1, 1, (n, rows, action_dim)).astype(np.float32)
    action[:, -1] = 0.0
    reward = rng.normal(size=(n, rows)).astype(np.float32)
    reward[:, 0] = 0.0
    return TaskEpisodes(
        obs=rng.normal(size=(n, rows, obs_dim)).astype(np.float32),
        action=action,
        reward=reward,
        episode_length=rows - 1,
        name=name,
    )


@pytest.fixture(scope="module")
def run():
    """Two seeds, 2 envs, 20 env steps per env: 3 complete rounds (6
    episodes per seed) and 2 steps of a fourth, the planner acting after
    the seed phase."""
    agent = TDMPC2(CounterEnv(length=T), n_envs=2, seed_steps=12, **TINY)
    state, _ = agent.train(seed=[0, 1], n_timesteps=40)
    return agent, jax.device_get(state)


def test_export_reads_every_committed_episode_in_order(run):
    agent, state = run
    episodes = export_episodes(agent, state, seed_index=0, name="counter")
    assert (episodes.num_episodes, episodes.rows) == (6, T + 1)
    assert (episodes.obs_dim, episodes.action_dim) == (2, 1)
    assert episodes.episode_length == T and episodes.name == "counter"
    # The ring never wrapped (5 rounds): its first 6 slots, rounds in order.
    np.testing.assert_array_equal(episodes.obs, state.buffer_state.obs[0, :6])
    # The obs-aligned rows of the counter env (obs [t, x], reward (t + 1) a):
    # row k holds o_k, a_k (none on the last row) and r_{k-1} (none on row 0).
    np.testing.assert_array_equal(
        episodes.obs[..., 0], np.broadcast_to(np.arange(T + 1), (6, T + 1))
    )
    a = episodes.action[..., 0]
    np.testing.assert_allclose(
        episodes.obs[:, 1:, 1], np.cumsum(np.clip(a[:, :-1], -1, 1), axis=1), atol=1e-5
    )
    np.testing.assert_allclose(
        episodes.reward[:, 1:],
        np.arange(1, T + 1) * np.clip(a[:, :-1], -1, 1),
        atol=1e-5,
    )
    np.testing.assert_array_equal(a[:, -1], 0.0)
    np.testing.assert_array_equal(episodes.reward[:, 0], 0.0)
    # Seeds are exported separately.
    other = export_episodes(agent, state, seed_index=1, name="counter")
    assert not np.array_equal(other.action, episodes.action)
    with pytest.raises(ValueError, match="seed_index"):
        export_episodes(agent, state)


def test_pooled_dataset_round_trips_and_samples_the_exports(run):
    """Pool the counter env's two seeds with a task of other dims: padding
    with zeros at the end, task ids in order, the per-task metadata, the
    unpooled episodes identical to the exports, and the slices of the pooled
    dataset those of the exports."""
    agent, state = run
    counter = concatenate_episodes(
        [export_episodes(agent, state, seed_index=i, name="counter") for i in (0, 1)]
    )
    other = synthetic(4, obs_dim=3, action_dim=2, name="other")
    dataset = pool_tasks([counter, other])
    assert dataset.obs.shape == (16, T + 1, 3)
    assert dataset.action.shape == (16, T + 1, 2)
    np.testing.assert_array_equal(dataset.task, [0] * 12 + [1] * 4)
    assert dataset.obs_dims == (2, 3) and dataset.action_dims == (1, 2)
    assert dataset.episode_lengths == (T, T) and dataset.names == ("counter", "other")
    np.testing.assert_array_equal(dataset.episode_counts(), [12, 4])
    np.testing.assert_array_equal(dataset.obs[:12, :, 2:], 0.0)
    np.testing.assert_array_equal(dataset.action[:12, :, 1:], 0.0)
    for task, source in ((0, counter), (1, other)):
        back = dataset.task_episodes(task)
        for field in ("obs", "action", "reward"):
            np.testing.assert_array_equal(getattr(back, field), getattr(source, field))
        assert (back.episode_length, back.name) == (source.episode_length, source.name)

    key = jax.random.PRNGKey(0)
    batch, task = jax.jit(lambda k: dataset.sample(k, 32, 3))(key)
    episode_key, start_key = jax.random.split(key)
    episode = jax.random.randint(episode_key, (32,), 0, 16)
    start = jax.random.randint(start_key, (32,), 0, T + 1 - 3)
    np.testing.assert_array_equal(task, dataset.task[episode])
    direct = slice_batch(
        EpisodeBufferState(dataset.obs, dataset.action, dataset.reward),
        episode,
        start,
        3,
    )
    for field in ("obs", "action", "reward"):
        np.testing.assert_array_equal(getattr(batch, field), getattr(direct, field))
    # A counter slice: obs rows s .. s + 3, actions s .. s + 2, rewards s + 1 .. s + 3.
    i = int(np.flatnonzero(np.asarray(task) == 0)[0])
    e, s = int(episode[i]), int(start[i])
    np.testing.assert_array_equal(batch.obs[:, i, :2], counter.obs[e, s : s + 4])
    np.testing.assert_array_equal(batch.action[:, i, :1], counter.action[e, s : s + 3])
    np.testing.assert_array_equal(batch.reward[:, i], counter.reward[e, s + 1 : s + 4])


def test_export_follows_a_wrapped_ring_and_refuses_evictions():
    """Slots of the newest rounds, oldest first, on a ring that wrapped; the
    evicted episodes make the export raise unless allowed."""
    ring = EpisodeBuffer(
        n_envs=2, episode_length=3, n_rounds=3, obs_dim=1, action_dim=1
    )
    # 5 complete rounds (ticks 0 .. 19): rounds 3, 4 held in slots 0, 1.
    slots, evicted = ring.committed_slots(19)
    np.testing.assert_array_equal(slots, [0, 1, 2, 3])
    assert evicted == 6
    # Mid-round of the sixth (tick 22): still rounds 3 and 4.
    np.testing.assert_array_equal(ring.committed_slots(22)[0], [0, 1, 2, 3])
    # Unwrapped: every complete round from 0.
    np.testing.assert_array_equal(ring.committed_slots(7)[0], [0, 1, 2, 3])
    assert ring.committed_slots(7)[1] == 0
    assert ring.committed_slots(2)[0].size == 0

    # Slot r holds round (3 + r) for r < 2 after 5 rounds: tag the obs.
    obs = np.zeros((6, 4, 1), np.float32)
    obs[0:2], obs[2:4], obs[4:6] = 3.0, 4.0, 2.0  # round of each slot
    obs[1::2] += 0.5  # env 1
    state = SimpleNamespace(
        buffer_state=EpisodeBufferState(
            obs=obs, action=np.zeros((6, 4, 1)), reward=np.zeros((6, 4))
        ),
        collector_state=SimpleNamespace(rows=np.int32(2 * 21)),  # 21 ticks
    )
    agent = SimpleNamespace(env_args=SimpleNamespace(n_envs=2), agent_episode_length=3)
    with pytest.raises(ValueError, match="evicted the 6 oldest"):
        export_episodes(agent, state)
    held = export_episodes(agent, state, allow_evicted=True)
    np.testing.assert_array_equal(held.obs[:, 0, 0], [3.0, 3.5, 4.0, 4.5])
    with pytest.raises(ValueError, match="no seed axis"):
        export_episodes(agent, state, seed_index=0)
    state.collector_state.rows = np.int32(2 * 3)
    with pytest.raises(ValueError, match="no complete episode"):
        export_episodes(agent, state)


def test_sampling_is_uniform_over_the_pooled_episodes():
    """b67b21c's RandomSampler over all episodes: a task's share of a batch
    is its share of the episodes (no per-task balancing, spec 4.19); crops
    are uniform over the ``L - H`` offsets."""
    dataset = pool_tasks(
        [synthetic(12, 2, 1, "a", seed=1), synthetic(4, 3, 2, "b", seed=2)]
    )
    _, task = dataset.sample(jax.random.PRNGKey(1), 8192, 3)
    share = float(np.mean(np.asarray(task) == 0))
    assert share == pytest.approx(0.75, abs=0.02)
    batch, _ = dataset.sample(jax.random.PRNGKey(2), 8192, 3)
    # Row 0 of a slice is its offset in the (synthetic) episode: recover it.
    first = np.asarray(batch.obs[0])
    offsets = np.argmax(
        np.all(dataset.obs[None, :, :, :] == first[:, None, None, :], axis=-1).any(1),
        axis=-1,
    )
    counts = np.bincount(offsets, minlength=T + 1)
    assert counts[T - 2 :].sum() == 0  # s <= L - 1 - H = 3
    np.testing.assert_allclose(counts[: T - 2] / 8192, 0.25, atol=0.03)
    with pytest.raises(ValueError, match="horizon"):
        dataset.sample(jax.random.PRNGKey(0), 4, T + 1)


def test_schema_is_validated():
    a = synthetic(3, 2, 1, "a")
    with pytest.raises(ValueError, match="same number of rows"):
        pool_tasks([a, synthetic(3, 2, 1, "b", rows=5)])
    with pytest.raises(ValueError, match="unique"):
        pool_tasks([a, synthetic(3, 2, 1, "a", seed=1)])
    with pytest.raises(ValueError, match="cannot concatenate"):
        concatenate_episodes([a, synthetic(3, 3, 1, "a")])
    with pytest.raises(ValueError, match="pool at least one"):
        pool_tasks([])
    with pytest.raises(ValueError, match="episode_length"):
        TaskEpisodes(a.obs, a.action, a.reward, episode_length=2)
    dataset = pool_tasks([a, synthetic(3, 4, 3, "b")])
    dataset.check()
    junk_obs = dataset.replace(obs=dataset.obs.at[0, 0, 3].set(1.0))
    with pytest.raises(ValueError, match="zero beyond each task's obs"):
        junk_obs.check()
    junk_action = dataset.replace(action=dataset.action.at[1, 2, 2].set(0.5))
    with pytest.raises(ValueError, match="zero beyond each task's action"):
        junk_action.check()
    with pytest.raises(ValueError, match="task ids"):
        dataset.replace(task=dataset.task.at[0].set(2)).check()
    with pytest.raises(ValueError, match="every task needs episodes"):
        dataset.replace(task=jnp.zeros_like(dataset.task)).check()
    assert isinstance(dataset, MultiTaskDataset)
    assert dataset.nbytes == sum(
        x.nbytes for x in (dataset.obs, dataset.action, dataset.reward, dataset.task)
    )
