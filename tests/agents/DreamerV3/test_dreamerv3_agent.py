"""The DreamerV3 agent end to end (milestone M7; DESIGN 6, 5.5, 5.7).

Tiny configurations, trained once per module: gymnax CartPole-v1 (discrete,
terminating episodes) with extensions and logging, Pendulum-v1 (continuous,
torque bounds ``[-2, 2]``) and mujoco_playground CartpoleBalance (action
repeat 2, short episodes). Checked on the same runs: finite outputs and
per-seed shapes, the optimizer step counters against the schedule, the
replay contents, the logging cadence and metrics, resume, extensions; and,
through spies on the CartPole run (``jax.debug.callback`` records of every
training step, batch draw and write-back), what each update trains on: rows
already written, fresh noise, the online items of 2411f7d's loop, and its
posterior written back.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax import DreamerV3
from ajax.agents import loop
from ajax.agents.DreamerV3 import train_DreamerV3
from ajax.agents.DreamerV3.replay import ReplayState, StreamReplay
from ajax.agents.DreamerV3.state import MODEL_SIZES
from ajax.agents.DreamerV3.train_DreamerV3 import train_metrics
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import LoggingConfig

from .reference_loop import reference_train_loop

#: A tiny model (d = 16) and batch (4 windows of 8 + 1 rows), imagination 3
#: steps, warmup 10 updates: a few ms per update on CPU.
TINY: dict[str, Any] = {
    "model_size": "1m",
    "units": 16,
    "hidden": 16,
    "deter": 32,
    "classes": 4,
    "stoch": 4,
    "blocks": 4,
    "imag_horizon": 3,
    "batch_size": 4,
    "batch_length": 8,
    "warmup": 10,
}
SEEDS = [0, 1]
N_ENVS = 4
ROWS = 480  # 120 ticks
#: A 100-row ring per env: it wraps within the 120-tick run, and a first
#: half of 60 ticks sizes its ring by its own length (60 rows), which the
#: resumed second half must grow.
CAPACITY = N_ENVS * 100
TRAIN_RATIO = 32  # 1 update per transition with 4 x 8 batches: 4 per tick
LOG_EVERY = N_ENVS * 30  # rows


def _playground_available() -> bool:
    try:
        import mujoco_playground  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Counter(Extension):
    """Counts the ``post_update`` folds."""

    name: str = "counter"

    def init_state(self, agent_state, rng):
        return jnp.zeros((), jnp.int32)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1


@dataclass(frozen=True)
class Metric(Extension):
    """Adds one evaluation metric."""

    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"ext/x": jnp.asarray(1.0)}


@dataclass(frozen=True)
class TargetTweak(Extension):
    """A phase DreamerV3 does not fold."""

    name: str = "target"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target


def cartpole_agent(**kwargs) -> DreamerV3:
    return DreamerV3(
        "CartPole-v1",
        n_envs=N_ENVS,
        train_ratio=TRAIN_RATIO,
        replay_capacity=CAPACITY,
        **TINY,
        **kwargs,
    )


def _recorder(records: list) -> Any:
    """A host callback appending its arguments, as NumPy arrays, to ``records``."""

    def append(*args):
        records.append(tuple(np.asarray(a) for a in args))

    return append


@pytest.fixture(scope="module")
def cartpole():
    """CartPole with a Counter and a Metric, logging every 30 ticks.

    The logging worker is not started and the log callback records what it
    is sent, so the run is hermetic. Spies record, per update and per seed
    (the seed ``vmap`` unrolls a callback over the seeds, in no guaranteed
    order within an update): the trained observations and the first noise
    leaf (``train_step``), the rows and the drawn windows (``sample``) and
    the written-back windows and latents (``write_back``). They call the
    originals, so the run is the unspied one.
    """
    agent = cartpole_agent(extensions=[Counter(), Metric()])
    logged: list = []
    spied = SimpleNamespace(batches=[], samples=[], writes=[])
    train_step = train_DreamerV3.train_step
    sample, write_back = StreamReplay.sample, StreamReplay.write_back

    def record(metrics, index, run_ids):
        logged.append((int(index), {k: np.asarray(v) for k, v in metrics.items()}))

    def train_step_spy(state, batch, noise, *, config):
        jax.debug.callback(
            _recorder(spied.batches), batch.obs, jax.tree.leaves(noise)[0]
        )
        return train_step(state, batch, noise, config=config)

    def sample_spy(self, state, rows, key):
        state, index = sample(self, state, rows, key)
        jax.debug.callback(
            _recorder(spied.samples), rows, index.env, index.start, index.online
        )
        return state, index

    def write_back_spy(self, state, index, entries):
        jax.debug.callback(
            _recorder(spied.writes), index.env, index.start, entries.deter
        )
        return write_back(self, state, index, entries)

    config = LoggingConfig(
        config={}, use_wandb=False, use_tensorboard=False, log_frequency=LOG_EVERY
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loop, "vmap_log", record)
        patch.setattr(train_DreamerV3, "train_step", train_step_spy)
        patch.setattr(StreamReplay, "sample", sample_spy)
        patch.setattr(StreamReplay, "write_back", write_back_spy)
        state, metrics = agent.train(
            seed=SEEDS, n_timesteps=ROWS, num_episode_test=3, logging_config=config
        )
        jax.effects_barrier()
    return SimpleNamespace(
        agent=agent, state=state, metrics=metrics, logged=logged, spied=spied
    )


def test_cartpole_trains_with_finite_outputs_and_per_seed_shapes(cartpole):
    state, agent = cartpole.state, cartpole.agent
    c = agent.dreamer_config
    assert state.world_model_state.params["rssm"]["dynhid0"]["kernel"].shape[0] == 2
    for leaf in jax.tree.leaves(
        (state.world_model_state.params, state.actor_state.params, state.critic_state)
    ):
        assert leaf.shape[0] == len(SEEDS)
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isfinite(leaf)))
    replay = state.replay_state
    assert agent.replay_rows_per_env == 100
    assert replay.obs.shape == (2, N_ENVS, 100, 4)
    assert replay.action.shape == (2, N_ENVS, 100)  # discrete: int32 indices
    assert replay.deter.shape == (2, N_ENVS, 100, c.deter)
    assert replay.stoch.shape == (2, N_ENVS, 100, c.stoch)
    assert replay.stoch.dtype == jnp.uint8
    assert state.collector_state.policy_carry.stoch.shape == (
        2,
        N_ENVS,
        c.stoch,
        c.classes,
    )
    np.testing.assert_array_equal(state.collector_state.rows, [ROWS, ROWS])
    np.testing.assert_array_equal(state.collector_state.timestep, [ROWS, ROWS])
    # Terminating episodes: CartPole's random-ish policy falls within ~20 steps.
    assert np.all(np.isfinite(state.collector_state.episodic_mean_return))
    assert bool(replay.is_terminal.any()) and bool(replay.is_first.any())


def test_every_optimizer_steps_once_per_scheduled_update(cartpole):
    state, agent = cartpole.state, cartpole.agent
    expected = agent.schedule.total_updates(ROWS // N_ENVS)
    assert expected > 100
    for steps in (
        state.n_updates,
        state.world_model_state.step,
        state.actor_state.step,
        state.critic_state.step,
    ):
        np.testing.assert_array_equal(steps, [expected, expected])
    # The slow critic left the critic after its first (hard-copy) update.
    assert not np.allclose(
        jax.tree.leaves(state.critic_state.params)[0],
        jax.tree.leaves(state.critic_state.target_params)[0],
    )


def test_the_replay_holds_the_last_rows_and_their_latents(cartpole):
    """The ring holds rows 20..119 of every env; their stored ``deter`` were
    refreshed by the write-back or set by the policy (never all zero), and
    the online queue consumed every pushed window (4 updates per tick pop
    faster than 4 envs push one window per 9 rows)."""
    replay = cartpole.state.replay_state
    assert bool(jnp.all(jnp.abs(replay.deter).sum(-1) > 0))
    pushed = N_ENVS * ((ROWS // N_ENVS - 1) // 9)
    np.testing.assert_array_equal(replay.popped, [pushed, pushed])


def test_logs_at_the_cadence_with_house_and_train_metrics(cartpole):
    logged, metrics = cartpole.logged, cartpole.metrics
    every = LOG_EVERY // N_ENVS
    ticks = [t for t in range(ROWS // N_ENVS) if (t + 1) % every == 0]
    timesteps = sorted({int(m["timestep"]) for _, m in logged})
    assert timesteps == [(t + 1) * N_ENVS for t in ticks]
    assert sorted({index for index, _ in logged}) == [0, 1]
    last = logged[-1][1]
    for key in (
        "timestep",
        "env_frames",
        "Train/n_updates",
        "Train/episodic mean reward",
        "Eval/episodic mean reward",
        "Eval/mean episodic length",
        "Train/opt_loss",
        "Train/rec_loss",
        "Train/actor_loss",
        "ext/x",
    ):
        assert key in last, key
    assert int(last["env_frames"]) == int(last["timestep"])  # action_repeat 1
    # The returned evaluations: what was logged, one row per log in order.
    np.testing.assert_array_equal(
        metrics["timestep"], [[(t + 1) * N_ENVS for t in ticks]] * 2
    )
    assert np.all(np.isfinite(np.asarray(metrics["Eval/episodic mean reward"])))
    np.testing.assert_array_equal(np.asarray(metrics["ext/x"]), 1.0)
    # Episodes last at least one step and at most CartPole's 500.
    length = np.asarray(metrics["Eval/mean episodic length"])
    assert np.all((length >= 1) & (length <= 500))
    expected = [cartpole.agent.schedule.total_updates(t + 1) for t in ticks]
    np.testing.assert_array_equal(metrics["Train/n_updates"], [expected, expected])
    # Train metrics are means over the updates since the previous log (every
    # logged window has updates), and the accumulator restarts at each log:
    # the run ends on a logging tick, so it holds nothing.
    assert np.all(np.isfinite(np.asarray(metrics["Train/opt_loss"])))
    np.testing.assert_array_equal(cartpole.state.train_metrics.count, [0, 0])


def _per_update(records: list, n_updates: int) -> list:
    """Records of both seeds, grouped by update (two per update)."""
    assert len(records) == len(SEEDS) * n_updates
    return [records[i : i + 2] for i in range(0, len(records), 2)]


def test_updates_train_on_written_rows_with_fresh_noise(cartpole):
    """Every update samples after its tick's rows were added (deviation D3):
    no trained row is an unwritten ring slot (CartPole observations are
    never exactly zero; an empty slot is); and each draws fresh noise."""
    total = cartpole.agent.schedule.total_updates(ROWS // N_ENVS)
    batches = _per_update(cartpole.spied.batches, total)
    for pair in batches:
        for obs, _ in pair:
            assert np.all(np.any(obs != 0, -1)), "an update read an unwritten row"
    noise = {eps.tobytes() for pair in batches for _, eps in pair}
    assert len(noise) == len(SEEDS) * total


def test_online_items_follow_the_reference_loop(cartpole):
    """Per update: the rows in the replay (every row of the update's tick)
    and the online items popped, against the Python port of 2411f7d's loop
    (``train.py`` + ``Replay.add`` / ``_sample`` + ``when.Ratio``) with each
    tick's updates after its adds; the uniform rows are the rest."""
    ticks = ROWS // N_ENVS
    counts, reference = reference_train_loop(
        N_ENVS, 4, 8, TRAIN_RATIO, ticks, updates_after_tick=True
    )
    samples = _per_update(cartpole.spied.samples, len(reference))
    tick_of_update = np.repeat(np.arange(ticks), counts)
    for update, (pair, expected) in enumerate(zip(samples, reference)):
        for rows, env, start, online in pair:
            assert int(rows) == tick_of_update[update] + 1
            np.testing.assert_array_equal(online, [k is not None for k in expected])
            popped = list(zip(env[online].tolist(), start[online].tolist()))
            assert popped == [k for k in expected if k is not None], update
    assert any(k is not None for batch in reference for k in batch)  # exercised


def test_the_last_update_writes_back_its_posterior(cartpole):
    """The last update's posterior latents of rows ``1..T`` of its windows
    are in the final ring, later batch rows winning (nothing is added after
    a tick's updates). The spy records the two seeds in either order."""
    deter = np.asarray(cartpole.state.replay_state.deter)
    capacity = deter.shape[2]
    writes = _per_update(
        cartpole.spied.writes, cartpole.agent.schedule.total_updates(ROWS // N_ENVS)
    )

    def stored_by(seed, record):
        env, start, entries = record
        expected = {}
        for b in range(len(env)):
            for k in range(entries.shape[1]):
                slot = int((start[b] + 1 + k) % capacity)
                expected[int(env[b]), slot] = entries[b, k]  # later rows win
        return all(
            np.array_equal(deter[seed][e, t], v) for (e, t), v in expected.items()
        )

    matches = [[stored_by(seed, record) for record in writes[-1]] for seed in (0, 1)]
    assert matches in ([[True, False], [False, True]], [[False, True], [True, False]])


def test_extensions_count_every_update(cartpole):
    state = cartpole.state
    np.testing.assert_array_equal(state.ext_state[0], state.n_updates)
    assert state.ext_state[1] == ()


def test_resume_continues_the_schedule_and_matches_an_uninterrupted_run(cartpole):
    """Half the rows, then the other half from the returned state: the
    absolute tick continues (the training-start gate is not re-applied, the
    queue and ratio go on), and the first half's ring -- sized by its own
    60 rows per env -- grows to the 100 rows of the whole run's, so the
    updates, the replay, the parameters and the extensions' states are those
    of one run of all the rows, with the same extensions (whose
    ``post_update`` keys come from the agent's stream; the logging of the
    uninterrupted run does not touch the training state)."""
    agent = cartpole_agent(extensions=[Counter(), Metric()])
    half = agent.train(seed=SEEDS, n_timesteps=ROWS // 2)
    assert half[0].replay_state.obs.shape == (2, N_ENVS, 60, 4)
    assert agent.resume_iteration_offset(half[0]) == ROWS // 2 // N_ENVS
    state, _ = agent.train(seed=SEEDS, n_timesteps=ROWS // 2, initial_state=half)
    full = cartpole.state
    assert agent.replay_rows_per_env == 100
    np.testing.assert_array_equal(state.n_updates, full.n_updates)
    np.testing.assert_array_equal(state.world_model_state.step, full.n_updates)
    np.testing.assert_array_equal(state.collector_state.rows, full.collector_state.rows)
    np.testing.assert_array_equal(state.replay_state.popped, full.replay_state.popped)
    jax.tree.map(np.testing.assert_array_equal, state.ext_state, full.ext_state)
    for ours, theirs in zip(
        jax.tree.leaves(state.world_model_state.params),
        jax.tree.leaves(full.world_model_state.params),
    ):
        np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(state.replay_state.obs, full.replay_state.obs)
    np.testing.assert_allclose(state.replay_state.deter, full.replay_state.deter)
    # A checkpoint skeleton (a 0-tick run) gets the latest run's ring.
    assert agent._resolve_replay(0, None) == (100, None)


@struct.dataclass
class _Resumable:
    """The fields of a state that the ring resolution reads."""

    collector_state: Any
    replay_state: ReplayState


def _resumable(rows: int, ring: int) -> _Resumable:
    """Two seeds after ``rows`` rows per env in a ``ring``-row replay (the
    written rows' rewards are 1)."""
    replay = StreamReplay(N_ENVS, ring, 4, 8).init(4, 2, True, 32, 4, 4)
    written = min(rows, ring)
    replay = replay.replace(reward=replay.reward.at[:, :written].set(1.0))
    return _Resumable(
        collector_state=SimpleNamespace(rows=np.full(2, rows * N_ENVS)),
        replay_state=jax.tree.map(lambda x: jnp.stack([x, x]), replay),
    )


def test_each_call_resolves_its_ring():
    """Host-side, no training: a fresh run sizes the ring by its own length
    (not an earlier run's), a 0-tick skeleton takes the latest run's, a
    resumed run grows a run-clamped ring to its total length and keeps a
    ring that is long enough; a wrapped ring cannot grow."""
    agent = DreamerV3(
        "CartPole-v1", n_envs=N_ENVS, replay_capacity=N_ENVS * 1000, **TINY
    )
    minimal = agent.schedule.min_ring_rows()
    assert agent._resolve_replay(0, None) == (minimal, None)
    assert agent.replay_rows_per_env is None
    assert agent._resolve_replay(N_ENVS * 20, None) == (20, None)
    assert agent._resolve_replay(N_ENVS * 300, None) == (300, None)
    assert agent._resolve_replay(N_ENVS * 5000, None) == (1000, None)
    assert agent._resolve_replay(0, None) == (1000, None)

    rows, state = agent._resolve_replay(N_ENVS * 280, _resumable(20, 20))
    assert rows == 300 == agent.replay_rows_per_env
    reward = np.asarray(state.replay_state.reward)
    assert reward.shape == (2, N_ENVS, 300)
    np.testing.assert_array_equal(reward[..., :20], 1.0)
    np.testing.assert_array_equal(reward[..., 20:], 0.0)
    assert state.replay_state.stoch.shape == (2, N_ENVS, 300, 4)

    long = _resumable(1500, 1000)
    rows, state = agent._resolve_replay(N_ENVS * 10, long)
    assert rows == 1000 and state is long
    with pytest.raises(ValueError, match="replay_capacity"):
        agent._resolve_replay(N_ENVS * 10, _resumable(50, 40))


@pytest.fixture(scope="module")
def pendulum():
    agent = DreamerV3("Pendulum-v1", n_envs=2, train_ratio=32, **TINY)
    state, _ = agent.train(seed=0, n_timesteps=400)
    return SimpleNamespace(agent=agent, state=state)


def test_pendulum_trains_on_raw_continuous_actions(pendulum):
    state, agent = pendulum.state, pendulum.agent
    assert agent.replay_rows_per_env == 200  # the whole run: min(capacity, rows)
    replay = state.replay_state
    assert replay.action.shape == (1, 2, 200, 1)
    # The replay stores the raw samples (std ~0.9 at init: some beyond +-1);
    # the collector clipped and mapped them to the torque bounds [-2, 2].
    assert float(jnp.abs(replay.action).max()) > 1.0
    expected = agent.schedule.total_updates(200)
    np.testing.assert_array_equal(state.actor_state.step, [expected])
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    # 200 rows per env are one unfinished 200-step episode.
    assert bool(replay.is_first[0, :, 0].all())
    assert not bool(replay.is_first[0, :, 1:].any() | replay.is_last.any())


@pytest.mark.skipif(not _playground_available(), reason="mujoco_playground missing")
def test_playground_cartpole_with_action_repeat():
    """Action repeat 2 and 40 simulator steps: 20 agent steps per episode,
    so rows ``0, 21, 42, ...`` start episodes and ``20, 41, ...`` end them
    (dynamic reset mode: the terminal row, then the reset row)."""
    agent = DreamerV3(
        "CartpoleBalance",
        n_envs=2,
        action_repeat=2,
        episode_length=40,
        train_ratio=32,
        **TINY,
    )
    state, _ = agent.train(seed=0, n_timesteps=200)
    replay = jax.tree.map(lambda x: x[0], state.replay_state)
    rows = np.arange(100)
    np.testing.assert_array_equal(
        replay.is_first[:, rows], np.tile(rows % 21 == 0, (2, 1))
    )
    np.testing.assert_array_equal(
        replay.is_last[:, rows], np.tile(rows % 21 == 20, (2, 1))
    )
    assert not bool(replay.is_terminal.any())
    assert float(jnp.abs(replay.action).max()) > 1.0  # raw samples stored
    np.testing.assert_array_equal(state.n_updates, [agent.schedule.total_updates(100)])
    logged = train_metrics(jax.tree.map(lambda x: x[0], state), None, action_repeat=2)
    assert int(logged["env_frames"]) == 2 * 200
    assert np.isfinite(float(logged["Train/episodic mean reward"]))


def test_unsupported_extension_phases_are_rejected():
    with pytest.raises(ValueError, match="on_target"):
        cartpole_agent(extensions=[TargetTweak()])
    assert DreamerV3.supported_extension_phases == frozenset(
        {"pretrain", "post_update", "eval_metrics"}
    )


def test_construction_resolves_the_configuration():
    agent = DreamerV3(
        "CartPole-v1", n_envs=2, model_size="25m", deter=96, return_horizon=50
    )
    c = agent.dreamer_config
    d = MODEL_SIZES["25m"]
    assert (c.units, c.hidden, c.deter, c.classes) == (d, d, 96, d // 16)
    assert c.gamma == pytest.approx(0.98)
    assert agent.config["algo_name"] == "DreamerV3"
    assert agent.config["model_size"] == "25m"
    assert agent.agent_config.train_ratio == 512
    assert agent.replay_rows_per_env is None and agent.replay_bytes_per_seed == 0
    with pytest.raises(ValueError, match="replay_capacity"):
        DreamerV3("CartPole-v1", n_envs=2, replay_capacity=100)
    with pytest.raises(ValueError, match="model_size"):
        DreamerV3("CartPole-v1", model_size="3m")


def test_resume_offset_needs_one_tick_for_all_seeds():
    agent = cartpole_agent()
    state = SimpleNamespace(collector_state=SimpleNamespace(rows=np.array([8, 12])))
    with pytest.raises(ValueError, match="cannot resume"):
        agent.resume_iteration_offset(state)
    state = SimpleNamespace(collector_state=SimpleNamespace(rows=np.array([8, 8])))
    assert agent.resume_iteration_offset(state) == 2
