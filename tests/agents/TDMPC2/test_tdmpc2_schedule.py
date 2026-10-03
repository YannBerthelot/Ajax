"""TD-MPC2 schedule and control flow against a port of the b67b21c loop.

``reference_online_loop`` transcribes ``OnlineTrainer.train`` of
``nicklashansen/tdmpc2@b67b21c:tdmpc2/trainer/online_trainer.py:67-117``
(unchanged at 5f6fade) with the agent and buffer replaced by recorders and
the env by :class:`toy_envs.CounterEnv`'s dynamics. The static
:class:`~ajax.agents.TDMPC2.train_TDMPC2.Schedule` must reproduce it for one
env (``DESIGN.md`` §4.4), and the real row collector + episode ring must
store exactly the reference's episodes. ``n_envs > 1`` is checked against a
port of the documented generalisation (deviation T7).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2.buffer import EpisodeBuffer, slice_batch
from ajax.agents.TDMPC2.train_TDMPC2 import Schedule
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.state import EnvironmentConfig

from .toy_envs import CounterEnv

H = 3


def reference_online_loop(steps, seed_steps, episode_length, actions):
    """``online_trainer.py:67-117`` (b67b21c), transcribed.

    ``actions[step]`` stands for ``agent.act`` / ``env.rand_act()``; the env
    is CounterEnv (``t += 1``, ``x += a``, ``r = t_before_step_plus_1 * a``,
    done after ``episode_length`` steps). Returns per step ``(random,
    n_updates, episodes in the buffer when the updates run)`` and the
    buffer's episodes as ``(obs, action, reward)`` rows (``to_td``, row 0 a
    dummy action / NaN reward).
    """
    trace, buffer, tds = [], [], []
    step, done = 0, True
    t = x = 0.0
    while step <= steps:  # :70
        if done:  # :77
            if step > 0:  # :84
                buffer.append(tuple(np.array(v, np.float32) for v in zip(*tds)))  # :91
            t, x = 0.0, 0.0  # :93 env.reset()
            tds = [([t, x], [np.nan], np.nan)]  # :94 to_td(obs)
        random = not step > seed_steps  # :97-100
        a = float(actions[step])
        reward = (t + 1) * a  # :101 env.step(action)
        t, x = t + 1, x + a
        done = t == episode_length
        tds.append(([t, x], [a], reward))  # :102
        n_updates = 0
        if step >= seed_steps:  # :105
            n_updates = seed_steps if step == seed_steps else 1  # :106-110
        trace.append((random, n_updates, len(buffer)))
        step += 1  # :115
    return trace, buffer


def generalised_loop(steps_per_env, seed_steps, episode_length, n_envs):
    """The documented ``n_envs > 1`` rule (``DESIGN.md`` §4.4), as a loop.

    Lockstep envs; whole episodes are added when they end (one round of
    ``n_envs``); the seed phase ends on the first vector step after which
    the total env steps exceed ``seed_steps`` with at least one round in the
    buffer; then a burst of ``seed_steps`` updates, then ``n_envs`` updates
    per vector step. Returns per vector step ``(random, n_updates,
    episodes)``.
    """
    trace, episodes, seeded = [], 0, False
    for k in range(steps_per_env):
        if k > 0 and k % episode_length == 0:
            episodes += n_envs  # the previous round ended
        if not seeded:
            seeded = n_envs * (k + 1) > seed_steps and episodes >= n_envs
            trace.append((True, seed_steps if seeded else 0, episodes))
        else:
            trace.append((False, n_envs, episodes))
    return trace


def ajax_trace(schedule, buffer, steps_per_env):
    """Per stepping tick: (random, n_updates, committed episodes)."""
    ticks = np.arange(schedule.num_ticks(steps_per_env * schedule.n_envs))
    stepping = ticks[~np.asarray(schedule.is_held(ticks))]
    assert stepping.size == steps_per_env
    random = np.asarray(schedule.random_phase(stepping))
    n_updates = np.asarray(schedule.n_updates(stepping))
    episodes = np.asarray(buffer.num_episodes(stepping))
    return [
        (bool(r), int(u), int(e))
        for r, u, e in zip(random, n_updates, episodes, strict=True)
    ]


@pytest.mark.parametrize(
    "episode_length, seed_steps",
    [
        (500, 2500),  # DMC: T = 500, S = max(1000, 5 T)
        (200, 1000),  # Pendulum-v1
        (100, 1000),  # Meta-World / MyoSuite
        (7, 30),  # S not a multiple of T
        (10, 29),  # S + 1 a multiple of T: step S ends an episode
    ],
)
def test_one_env_reproduces_the_reference_loop(episode_length, seed_steps):
    steps = seed_steps + 3 * episode_length + 5  # reference: steps + 1 env steps
    trace, _ = reference_online_loop(
        steps, seed_steps, episode_length, np.zeros(steps + 1)
    )
    schedule = Schedule(n_envs=1, episode_length=episode_length, seed_steps=seed_steps)
    buffer = EpisodeBuffer.create(
        capacity=10**7, n_envs=1, episode_length=episode_length, obs_dim=2, action_dim=1
    )
    assert ajax_trace(schedule, buffer, steps + 1) == trace
    # S + 1 random actions at steps 0 .. S, the burst right after step S with
    # floor(S / T) finished episodes, then one update per step.
    random = [r for r, _, _ in trace]
    assert random == [True] * (seed_steps + 1) + [False] * (steps - seed_steps)
    assert trace[seed_steps][1:] == (seed_steps, seed_steps // episode_length)
    assert schedule.seed_step == seed_steps
    total = sum(u for _, u, _ in trace)
    assert total == steps == (steps + 1) - 1  # total updates = env steps - 1


@pytest.mark.parametrize(
    "n_envs, episode_length, seed_steps",
    [(4, 200, 1000), (16, 500, 2500), (3, 10, 31), (2, 7, 30), (8, 5, 1000)],
)
def test_many_envs_follow_the_documented_generalisation(
    n_envs, episode_length, seed_steps
):
    steps_per_env = max(seed_steps // n_envs, episode_length) + 2 * episode_length + 3
    trace = generalised_loop(steps_per_env, seed_steps, episode_length, n_envs)
    schedule = Schedule(
        n_envs=n_envs, episode_length=episode_length, seed_steps=seed_steps
    )
    buffer = EpisodeBuffer.create(
        capacity=10**7,
        n_envs=n_envs,
        episode_length=episode_length,
        obs_dim=2,
        action_dim=1,
    )
    assert ajax_trace(schedule, buffer, steps_per_env) == trace
    burst = [k for k, (random, u, _) in enumerate(trace) if random and u > 0]
    assert burst == [schedule.seed_step]
    assert trace[schedule.seed_step][2] >= n_envs  # a committed round


def test_short_seed_phases_wait_for_the_first_episode():
    """S < T would make the reference sample an empty buffer; Ajax keeps
    acting randomly until the first round is committed."""
    schedule = Schedule(n_envs=1, episode_length=10, seed_steps=3)
    assert schedule.seed_step == 10
    assert schedule.seed_tick == 11  # the first step of the second episode
    assert int(schedule.n_updates(11)) == 3


def test_held_ticks_and_tick_counts():
    schedule = Schedule(n_envs=2, episode_length=4, seed_steps=8)
    ticks = np.arange(20)
    np.testing.assert_array_equal(np.asarray(schedule.is_held(ticks)), ticks % 5 == 4)
    assert all(bool(schedule.random_phase(t)) for t in (4, 9, 14))  # held: no planning
    assert [schedule.tick_of_step(k) for k in (0, 3, 4, 8, 9)] == [0, 3, 5, 10, 11]
    assert [schedule.steps_before(t) for t in (0, 3, 4, 5, 9, 10, 11)] == [
        0,
        3,
        4,
        4,
        8,
        8,
        9,
    ]
    assert schedule.num_ticks(16) == 10  # 8 steps per env: two whole episodes
    assert schedule.num_ticks(14) == 8  # 7 steps per env, ends mid-episode
    # A split run covers the ticks of an uninterrupted one.
    for first in range(0, 30, 2):
        start = schedule.num_ticks(first)
        assert start + schedule.num_ticks(30 - first, start) == schedule.num_ticks(30)
    with pytest.raises(ValueError):
        Schedule(n_envs=0, episode_length=4, seed_steps=1)


@pytest.mark.parametrize("n_envs", [1, 2])
def test_collector_and_ring_store_the_reference_episodes(n_envs):
    """The real row collector (static mode) on CounterEnv writes, through the
    episode ring, exactly the episodes the reference loop adds to its buffer
    (with the same actions), the training slices are DataPrepTransform's,
    and the seed phase / burst / per-step updates line up step for step."""
    T, S, steps_per_env = 5, 12, 33
    env = CounterEnv(length=T)
    env_args = EnvironmentConfig(
        env=env, env_params=env.default_params, n_envs=n_envs, continuous=True
    )
    schedule = Schedule(n_envs=n_envs, episode_length=T, seed_steps=S)
    buffer = EpisodeBuffer.create(
        capacity=steps_per_env * n_envs,
        n_envs=n_envs,
        episode_length=T,
        obs_dim=2,
        action_dim=1,
    )

    def policy(carry, obs, is_first, key):  # stands for the planner
        return jnp.tanh(0.3 * obs[:, :1] - 0.5), carry, None

    def tick_fn(carry, tick):
        cs, store = carry
        cs, row = collect_row(
            cs,
            tick,
            policy,
            env_args=env_args,
            reset_mode="static",
            episode_length=T,
            random_phase=schedule.random_phase(tick),
            timestep_unit="env_steps",
        )
        store = buffer.add(store, row.obs, row.action, row.reward, tick)
        return (cs, store), row

    n_ticks = schedule.num_ticks(steps_per_env * n_envs)
    init = (init_row_collector_state(jax.random.PRNGKey(0), env_args), buffer.init())
    (cs, store), rows = jax.jit(
        lambda c: jax.lax.scan(tick_fn, c, jnp.arange(n_ticks))
    )(init)
    assert int(cs.timestep) == steps_per_env * n_envs
    assert int(cs.n_offschedule_dones) == 0

    held = np.asarray(schedule.is_held(np.arange(n_ticks)))
    actions = np.asarray(rows.action)[~held, :, 0]  # [steps_per_env, n_envs]
    seed_tick = schedule.seed_tick
    stepping_ticks = np.flatnonzero(~held)
    # Random in the seed phase (uniform draws), the policy's afterwards.
    after = stepping_ticks > seed_tick
    obs_t = np.asarray(rows.obs)[~held, :, 0]
    np.testing.assert_allclose(
        actions[after], np.tanh(0.3 * obs_t[after] - 0.5), rtol=1e-6
    )
    assert not np.allclose(actions[~after], np.tanh(0.3 * obs_t[~after] - 0.5))

    last_tick = n_ticks - 1
    for e in range(n_envs):
        if n_envs == 1:
            trace, episodes = reference_online_loop(
                steps_per_env - 1, S, T, actions[:, e]
            )
            assert ajax_trace(schedule, buffer, steps_per_env) == trace
        else:
            _, episodes = reference_online_loop(steps_per_env - 1, S, T, actions[:, e])
        rounds = buffer.committed_rounds(last_tick)
        assert len(episodes) == rounds  # nothing evicted at this capacity
        for r, episode in enumerate(episodes):
            slot = (r % buffer.n_rounds) * n_envs + e
            for start in range(T - H + 1):
                batch = slice_batch(store, jnp.array([slot]), jnp.array([start]), H)
                crop = slice(start, start + H + 1)
                obs, action, reward = (v[crop] for v in episode)
                np.testing.assert_allclose(batch.obs[:, 0], obs, rtol=1e-6)
                np.testing.assert_allclose(batch.action[:, 0], action[1:], rtol=1e-6)
                np.testing.assert_allclose(batch.reward[:, 0], reward[1:], rtol=1e-6)
