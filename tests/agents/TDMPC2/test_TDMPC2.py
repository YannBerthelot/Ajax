"""TD-MPC2 agent: surface, acting, smoke runs, update counts, resume,
extensions, evaluation / logging and the termination guard.

Module-scoped runs train once at a tiny size and are asserted many ways
(``docs/world_models/DESIGN.md`` §10, "Agent"): gymnax Pendulum-v1 (T = 200,
torque bounds [-2, 2] mapped from the agent's [-1, 1]) as a first run, its
resumption from a checkpoint by a new agent (the new-process flow of
``ajax.checkpoint``) and an uninterrupted run of the same total length;
playground CartpoleBalance with a short episode and action repeat 2 (static
fresh resets) with logging captured in-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2
from ajax.agents.TDMPC2 import core, train_TDMPC2
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2State
from ajax.agents.TDMPC2.train_TDMPC2 import Schedule, evaluate_tdmpc2, planner_policy
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.extensions.base import Extension, ExtensionContext
from ajax.logging.wandb_logging import LoggingConfig

from .toy_envs import ConstantRewardEnv, TerminatingEnv

# A tiny model and planner; everything else at the paper defaults. More
# elites than policy-prior trajectories, so that the MPPI statistics (and
# with them the warm start) shape the decisions of an untrained model.
TINY = {
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "num_q": 2,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 8,
    "num_pi_trajs": 4,
    "iterations": 2,
}
SEEDS = [0, 1]
N_ENVS = 2
# Pendulum: T = 200. S = 400 puts the seed tick on per-env step 200 (the
# first step of the second episode), so the first run (260 steps per env)
# holds the burst and 59 regular ticks; its ring holds the 2 rounds of its
# 520 steps. The resumed run (350 more steps per env) completes a third
# round, which only a ring sized for the whole run keeps.
PENDULUM_S, FIRST, RESUMED = 400, 2 * 260, 2 * 350
SEED_STEP = 200  # Schedule.seed_step: max(400 // 2, T)


@dataclass(frozen=True)
class CounterExt(Extension):
    """Records what ``pretrain`` and ``post_update`` see: how often each runs,
    the ``step`` of the first and last update and the planner's warm start
    at the first update."""

    name: str = "counter"

    def init_state(self, agent_state, rng):
        return {
            "pretrain": jnp.asarray(0, jnp.int32),
            "updates": jnp.asarray(0, jnp.int32),
            "first_step": jnp.asarray(-1, jnp.int32),
            "last_step": jnp.asarray(-1, jnp.int32),
            "carry_at_first": jnp.asarray(-1.0, jnp.float32),
        }

    def pretrain(self, agent_state, ext_state, ctx: ExtensionContext):
        return agent_state, {**ext_state, "pretrain": ext_state["pretrain"] + 1}

    def post_update(self, agent_state, ext_state, ctx: ExtensionContext):
        first = ext_state["updates"] == 0
        carry = jnp.abs(agent_state.collector_state.policy_carry).sum()
        return agent_state, {
            **ext_state,
            "updates": ext_state["updates"] + 1,
            "first_step": jnp.where(first, ctx.step, ext_state["first_step"]),
            "last_step": jnp.asarray(ctx.step, jnp.int32),
            "carry_at_first": jnp.where(first, carry, ext_state["carry_at_first"]),
        }


@dataclass(frozen=True)
class MetricExt(Extension):
    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx: ExtensionContext):
        return {"ext_smoke/x": jnp.asarray(1.0)}


@dataclass(frozen=True)
class TargetTweak(Extension):
    name: str = "target-tweak"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target


def _scheduled_updates(schedule: Schedule, n_ticks: int) -> int:
    return int(np.sum(np.asarray(schedule.n_updates(np.arange(n_ticks)))))


def _pendulum_agent(**kwargs):
    return TDMPC2(
        "Pendulum-v1",
        n_envs=N_ENVS,
        seed_steps=PENDULUM_S,
        extensions=[CounterExt()],
        **TINY,
        **kwargs,
    )


@pytest.fixture(scope="module")
def pendulum(tmp_path_factory):
    agent = _pendulum_agent()
    first, first_metrics = agent.train(
        seed=SEEDS, n_timesteps=FIRST, num_episode_test=2
    )
    first = jax.device_get(first)
    first_capacity = agent.replay_capacity
    # Resume in a "new process": a fresh agent restores the checkpoint into
    # its n_timesteps=0 skeleton (ajax.checkpoint's flow).
    path = str(tmp_path_factory.mktemp("tdmpc2") / "first.pkl")
    save_checkpoint(first, path)
    resumer = _pendulum_agent()
    skeleton, _ = resumer.train(seed=SEEDS, n_timesteps=0)
    skeleton_slots = skeleton.buffer_state.obs.shape[1]
    resumed, resumed_metrics = resumer.train(
        seed=SEEDS,
        n_timesteps=RESUMED,
        num_episode_test=2,
        initial_state=restore_into(skeleton, path),
    )
    full_agent = _pendulum_agent()
    full, _ = full_agent.train(
        seed=SEEDS, n_timesteps=FIRST + RESUMED, num_episode_test=2
    )
    schedule = Schedule(n_envs=N_ENVS, episode_length=200, seed_steps=PENDULUM_S)
    return SimpleNamespace(
        agent=agent,
        resumer=resumer,
        full_agent=full_agent,
        schedule=schedule,
        first=first,
        first_metrics=first_metrics,
        first_capacity=first_capacity,
        skeleton_slots=skeleton_slots,
        resumed=resumed,
        resumed_metrics=resumed_metrics,
        full=full,
    )


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_episode_length_derived_defaults():
    agent = TDMPC2("Pendulum-v1")
    assert agent.agent_episode_length == 200
    assert agent.gamma == pytest.approx(0.975)  # (40 - 1) / 40
    assert agent.seed_steps == 1000  # max(1000, 5 T)
    assert agent.agent_config == TDMPC2Config.from_model_size(5)
    small = TDMPC2("Pendulum-v1", model_size=1, mlp_dim=64, gamma=0.9, seed_steps=7)
    assert (small.agent_config.latent_dim, small.agent_config.mlp_dim) == (128, 64)
    assert (small.gamma, small.seed_steps) == (0.9, 7)
    # The 1000 floor of the seed phase (T = 10: 5 T = 50) and the discount
    # heuristic's arguments (T / denom, clipped to [min, max]).
    short = TDMPC2(ConstantRewardEnv(length=10))
    assert short.seed_steps == 1000
    assert short.gamma == pytest.approx(0.95)  # (2 - 1) / 2 clipped up
    assert TDMPC2("Pendulum-v1", discount_denom=10).gamma == pytest.approx(0.95)
    assert TDMPC2("Pendulum-v1", discount_max=0.97).gamma == pytest.approx(0.97)
    assert TDMPC2("Pendulum-v1", discount_min=0.98).gamma == pytest.approx(0.98)
    assert agent.supported_extension_phases == {
        "pretrain",
        "post_update",
        "eval_metrics",
    }


def test_every_hyperparameter_reaches_the_config():
    """Each static kwarg of DESIGN 4.1 lands in TDMPC2Config; the schedulable
    learning rate and the optimizer constants reach make_train; the values
    actually used are in the run config next to the raw arguments."""
    overrides = {
        "simnorm_dim": 4,
        "num_bins": 51,
        "vmax": 5.0,
        "dropout": 0.02,
        "horizon": 4,
        "rho": 0.4,
        "consistency_coef": 10.0,
        "reward_coef": 0.2,
        "value_coef": 0.3,
        "grad_clip_norm": 7.0,
        "tau": 0.05,
        "entropy_coef": 2e-4,
        "log_std_min": -8.0,
        "log_std_max": 1.0,
        "batch_size": 64,
        "buffer_size": 12345,
        "iterations": 3,
        "num_samples": 128,
        "num_elites": 16,
        "num_pi_trajs": 8,
        "min_std": 0.1,
        "max_std": 1.5,
        "temperature": 0.7,
    }
    agent = TDMPC2("Pendulum-v1", **overrides)
    for name, value in overrides.items():
        assert getattr(agent.agent_config, name) == value, name

    def lr(step):
        return 1e-3

    keywords = (
        TDMPC2("Pendulum-v1", learning_rate=lr, enc_lr_scale=0.5, pi_eps=1e-6)
        .get_make_train()
        .keywords
    )
    assert keywords["learning_rate"] is lr
    assert (keywords["enc_lr_scale"], keywords["pi_eps"]) == (0.5, 1e-6)

    config = TDMPC2("Pendulum-v1", model_size=1).config
    assert config["algo_name"] == "TDMPC2"
    assert (config["gamma"], config["seed_steps"], config["enc_dim"]) == (
        None,
        None,
        None,
    )
    resolved = {k: v for k, v in config.items() if k.startswith("resolved_")}
    assert resolved == {
        "resolved_gamma": pytest.approx(0.975),
        "resolved_seed_steps": 1000,
        "resolved_iterations": 6,
        "resolved_enc_dim": 256,
        "resolved_mlp_dim": 384,
        "resolved_latent_dim": 128,
        "resolved_num_enc_layers": 2,
        "resolved_num_q": 2,
    }
    assert config["agent_episode_length"] == 200


def test_playground_defaults_follow_the_paper_dmc_protocol():
    pytest.importorskip("mujoco_playground")
    agent = TDMPC2("CartpoleBalance", episode_length=1000, action_repeat=2)
    assert agent.agent_episode_length == 500
    assert agent.gamma == pytest.approx(0.99)
    assert agent.seed_steps == 2500


def test_discrete_actions_are_rejected():
    with pytest.raises(ValueError, match="continuous"):
        TDMPC2("CartPole-v1")


def test_unsupported_extension_phases_are_rejected():
    with pytest.raises(ValueError, match="on_target"):
        TDMPC2("Pendulum-v1", extensions=[TargetTweak()])
    TDMPC2("Pendulum-v1", extensions=[CounterExt(), MetricExt()])


def test_episodes_shorter_than_the_horizon_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        TDMPC2(TerminatingEnv(length=2), horizon=3)


def test_resume_offset_needs_equal_tick_counts():
    agent = TDMPC2("Pendulum-v1", n_envs=2)
    state = SimpleNamespace(collector_state=SimpleNamespace(rows=np.array([8, 8])))
    assert agent.resume_iteration_offset(state) == 4
    for rows in ([8, 10], [7, 7]):
        state.collector_state.rows = np.array(rows)
        with pytest.raises(ValueError, match="cannot resume"):
            agent.resume_iteration_offset(state)


# ---------------------------------------------------------------------------
# Acting: the collector's policy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def decide():
    """A jitted planner_policy of an untrained tiny model (Pendulum shapes)."""
    config = TDMPC2Config.from_model_size(1, **TINY)
    learner = core.create_update_state(jax.random.PRNGKey(0), config, 3, 1)

    @jax.jit
    def run(carry, obs, is_first, key):
        return planner_policy(
            carry,
            obs,
            is_first,
            key,
            wm_params=learner.world_model_state.params,
            pi_params=learner.actor_state.params,
            config=config,
            gamma=0.99,
            eval_mode=False,
        )

    return config, run


def test_the_warm_start_is_reset_where_an_episode_starts(decide):
    """``t0 = is_first`` per env: a new episode plans from a zero mean, not
    the previous episode's (``5f6fade:tdmpc2/tdmpc2.py:130-133``)."""
    config, run = decide
    obs = jnp.tile(jnp.array([[0.3, -0.2, 0.5]]), (2, 1))
    stale = jnp.full((2, config.horizon, 1), 0.8)
    zero = jnp.zeros_like(stale)
    is_first = jnp.array([True, False])
    key = jax.random.PRNGKey(7)
    action_stale, mean_stale, _ = run(stale, obs, is_first, key)
    action_zero, mean_zero, _ = run(zero, obs, is_first, key)
    np.testing.assert_allclose(action_stale[0], action_zero[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mean_stale[0], mean_zero[0], rtol=1e-5, atol=1e-6)
    assert not np.allclose(mean_stale[1], mean_zero[1], atol=1e-4)


def test_each_env_plans_with_its_own_draws(decide):
    config, run = decide
    obs = jnp.tile(jnp.array([[0.3, -0.2, 0.5]]), (2, 1))
    carry = jnp.zeros((2, config.horizon, 1))
    action, mean, _ = run(carry, obs, jnp.ones(2, bool), jax.random.PRNGKey(3))
    assert not np.allclose(action[0], action[1])
    assert not np.allclose(mean[0], mean[1])


# ---------------------------------------------------------------------------
# Pendulum: smoke, schedule, resume, extensions
# ---------------------------------------------------------------------------


def test_smoke_outputs_are_finite_and_shaped_per_seed(pendulum):
    state = pendulum.first
    # Without a logging config nothing is evaluated: no per-tick metrics.
    assert pendulum.first_metrics is None
    for leaf in jax.tree.leaves(state):
        assert np.shape(leaf)[0] == len(SEEDS)
    for tree in (state.world_model_state, state.actor_state, state.update_metrics):
        for leaf in jax.tree.leaves(tree):
            assert np.all(np.isfinite(leaf))
    # The ring: R = ceil(520 / (200 * 2)) + 1 = 3 rounds of 2 episodes.
    assert state.buffer_state.obs.shape == (len(SEEDS), 3 * N_ENVS, 201, 3)
    assert pendulum.first_capacity == FIRST
    cs = state.collector_state
    np.testing.assert_array_equal(cs.timestep, FIRST)
    np.testing.assert_array_equal(cs.n_offschedule_dones, 0)
    np.testing.assert_array_equal(state.n_terminations, 0)
    assert np.all(np.isfinite(cs.episodic_mean_return))  # one episode finished
    assert np.all(cs.episodic_mean_return < 0)  # Pendulum's costs
    # The planner ran after the seed phase: its warm start is not zero.
    assert np.all(np.abs(cs.policy_carry).sum(axis=(1, 2, 3)) > 0)
    # The ring stores the agent's raw [-1, 1] actions; the env got them
    # mapped to Pendulum's [-2, 2] torque. Replay round 0 (env 0, slot 0)
    # through gymnax Pendulum's dynamics (g = 10, m = l = 1, dt = 0.05):
    # thdot' = clip(thdot + (15 sin(th) + 3 u) dt, -8, 8).
    actions = np.asarray(state.buffer_state.action)
    assert np.abs(actions).max() <= 1.0 and np.abs(actions).max() > 0.9
    obs = np.asarray(state.buffer_state.obs[0, 0])  # [T + 1, (cos, sin, thdot)]
    a = actions[0, 0, :-1, 0]

    def next_thdot(u):
        return np.clip(obs[:-1, 2] + (15 * obs[:-1, 1] + 3 * u) * 0.05, -8, 8)

    np.testing.assert_allclose(next_thdot(2 * a), obs[1:, 2], atol=1e-4)
    assert not np.allclose(next_thdot(a), obs[1:, 2], atol=1e-2)


def test_adam_steps_equal_the_scheduled_updates(pendulum):
    state, schedule = pendulum.first, pendulum.schedule
    expected = _scheduled_updates(schedule, schedule.num_ticks(FIRST))
    # The burst of S at the seed tick, then n_envs per stepping tick.
    assert expected == PENDULUM_S + N_ENVS * (260 - (SEED_STEP + 1))
    for count in (
        state.n_updates,
        state.world_model_state.step,
        state.actor_state.step,
        state.ext_state[0]["updates"],  # post_update after every update
    ):
        np.testing.assert_array_equal(count, expected)


def test_extension_hooks_follow_the_schedule(pendulum):
    first = pendulum.first.ext_state[0]
    resumed = pendulum.resumed.ext_state[0]
    # pretrain runs once, on the fresh run only.
    np.testing.assert_array_equal(first["pretrain"], 1)
    np.testing.assert_array_equal(resumed["pretrain"], 1)
    # post_update's step is the collector's env-step count: the burst runs
    # right after the seed tick, env step SEED_STEP of every env.
    np.testing.assert_array_equal(first["first_step"], N_ENVS * (SEED_STEP + 1))
    np.testing.assert_array_equal(first["last_step"], FIRST)
    np.testing.assert_array_equal(resumed["last_step"], FIRST + RESUMED)
    # The seed tick acts randomly: the planner has never run when the burst
    # starts, so its warm start is still zero.
    np.testing.assert_array_equal(first["carry_at_first"], 0.0)


def test_a_new_agent_resumes_a_checkpoint_as_the_uninterrupted_run(pendulum):
    first, resumed, full = pendulum.first, pendulum.resumed, pendulum.full
    schedule = pendulum.schedule
    total = _scheduled_updates(schedule, schedule.num_ticks(FIRST + RESUMED))
    # No second seed phase or burst: the resumed run adds n_envs updates
    # per env step.
    assert total == int(first.n_updates[0]) + RESUMED
    for state in (resumed, full):
        np.testing.assert_array_equal(state.n_updates, total)
        np.testing.assert_array_equal(state.world_model_state.step, total)
        np.testing.assert_array_equal(state.actor_state.step, total)
        np.testing.assert_array_equal(state.ext_state[0]["updates"], total)
        np.testing.assert_array_equal(state.collector_state.timestep, FIRST + RESUMED)
    np.testing.assert_array_equal(
        resumed.collector_state.rows, full.collector_state.rows
    )
    # Every call sizes the ring for the run so far: the skeleton's is the
    # smallest (R = 2), the resumed and the uninterrupted runs' hold
    # ceil(1220 / 400) + 1 = 5 rounds, of which 3 are complete.
    assert pendulum.skeleton_slots == 2 * N_ENVS
    assert pendulum.resumer.replay_capacity == FIRST + RESUMED
    assert pendulum.full_agent.replay_capacity == FIRST + RESUMED
    assert resumed.buffer_state.obs.shape == full.buffer_state.obs.shape
    assert full.buffer_state.obs.shape[1] == 5 * N_ENVS
    # The split run computes what the uninterrupted one computes: the same
    # ticks, random streams and replayed episodes.
    complete = 3 * N_ENVS
    for a, b in (
        (resumed.world_model_state.params, full.world_model_state.params),
        (resumed.world_model_state.target_params, full.world_model_state.target_params),
        (resumed.actor_state.params, full.actor_state.params),
        (resumed.collector_state.last_obs, full.collector_state.last_obs),
        (resumed.q_scale.value, full.q_scale.value),
        (resumed.buffer_state.obs[:, :complete], full.buffer_state.obs[:, :complete]),
    ):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_evaluation_runs_the_planner_for_whole_episodes(pendulum):
    agent = pendulum.agent
    state = jax.tree.map(lambda x: x[0], pendulum.resumed)
    evaluate = jax.jit(
        lambda s, k: evaluate_tdmpc2(
            s,
            k,
            env_args=agent.env_args,
            config=agent.agent_config,
            gamma=agent.gamma,
            num_episodes=2,
        )
    )
    key = jax.random.PRNGKey(0)
    out = evaluate(state, key)
    assert float(out["Eval/mean episodic length"]) == 200
    assert -2000 < float(out["Eval/episodic mean reward"]) < 0
    # Deterministic per state and key; each evaluation (n_logs) starts from
    # fresh initial states.
    again = evaluate(state, key)
    assert float(again["Eval/episodic mean reward"]) == float(
        out["Eval/episodic mean reward"]
    )
    later = evaluate(state.replace(n_logs=state.n_logs + 1), key)
    assert float(later["Eval/episodic mean reward"]) != float(
        out["Eval/episodic mean reward"]
    )


# ---------------------------------------------------------------------------
# Playground: action repeat, static resets, logging
# ---------------------------------------------------------------------------


def test_playground_cartpole_trains_and_logs(monkeypatch):
    pytest.importorskip("mujoco_playground")
    logged = []

    def capture(metrics, index, run_ids, logging_config):
        logged.append({k: np.asarray(v) for k, v in metrics.items()})

    monkeypatch.setattr(train_TDMPC2, "vmap_log", capture)
    monkeypatch.setattr(train_TDMPC2, "start_async_logging", lambda: None)

    # Trace-time record of the planner's mode and env count per call site.
    planner_calls = set()

    def planner_spy(carry, obs, is_first, key, **kwargs):
        planner_calls.add((kwargs["eval_mode"], obs.shape[0]))
        return planner_policy(carry, obs, is_first, key, **kwargs)

    monkeypatch.setattr(train_TDMPC2, "planner_policy", planner_spy)

    # Run-time record of every update's batch and noise, per seed.
    updates: dict = {}
    update = core.update

    def record(index, obs, td_eps):
        updates.setdefault(int(index), []).append((np.array(obs), np.array(td_eps)))

    def update_spy(state, batch, noise, **kwargs):
        if isinstance(state, TDMPC2State):  # not the init's shape-only call
            jax.debug.callback(record, state.index, batch.obs, noise.td_eps)
        return update(state, batch, noise, **kwargs)

    monkeypatch.setattr(core, "update", update_spy)

    # 40 simulator steps at repeat 2: T = 20 agent steps; S = 40 puts the
    # seed tick on per-env step 20; 60 steps per env = 3 episodes.
    agent = TDMPC2(
        "CartpoleBalance",
        n_envs=N_ENVS,
        episode_length=40,
        action_repeat=2,
        seed_steps=40,
        extensions=[MetricExt()],
        **TINY,
    )
    assert agent.agent_episode_length == 20
    logging_config = LoggingConfig(
        config={}, use_wandb=False, use_tensorboard=False, log_frequency=80
    )
    state, metrics = agent.train(
        seed=SEEDS, n_timesteps=120, num_episode_test=3, logging_config=logging_config
    )
    jax.effects_barrier()

    schedule = Schedule(n_envs=N_ENVS, episode_length=20, seed_steps=40)
    expected = _scheduled_updates(schedule, schedule.num_ticks(120))
    assert expected == 40 + N_ENVS * (60 - 21)
    np.testing.assert_array_equal(state.n_updates, expected)
    np.testing.assert_array_equal(state.world_model_state.step, expected)
    np.testing.assert_array_equal(state.actor_state.step, expected)
    np.testing.assert_array_equal(state.collector_state.timestep, 120)
    np.testing.assert_array_equal(state.collector_state.n_offschedule_dones, 0)
    np.testing.assert_array_equal(state.n_terminations, 0)
    for leaf in jax.tree.leaves(state.world_model_state.params):
        assert np.all(np.isfinite(leaf))
    # Static resets: every episode starts from a fresh random state. The
    # ring has ceil(120 / (20 * 2)) + 1 = 4 rounds; rounds 0-2 are written.
    assert state.buffer_state.obs.shape[1] == 4 * N_ENVS
    first_obs = np.asarray(state.buffer_state.obs[:, : 3 * N_ENVS, 0])
    for seed_obs in first_obs:
        assert len({tuple(np.round(o, 6)) for o in seed_obs}) == len(seed_obs)
    # One summed reward per agent step: up to 2 per row on CartpoleBalance.
    assert float(np.max(state.buffer_state.reward)) > 1.0

    # Training explores (eval_mode off) on the n_envs training envs; the
    # evaluation plans in eval_mode on num_episode_test envs.
    assert planner_calls == {(False, N_ENVS), (True, 3)}
    # Every update samples a fresh batch with fresh noise: no two updates of
    # a seed share either, within the burst or across ticks.
    assert sorted(updates) == list(range(len(SEEDS)))
    for records in updates.values():
        assert len(records) == expected
        for i, (obs, eps) in enumerate(records):
            for other_obs, other_eps in records[:i]:
                assert not np.array_equal(obs, other_obs)
                assert not np.array_equal(eps, other_eps)

    # log_frequency 80 env steps = 40 per env = 2 episodes: one log per seed,
    # on the held tick after the second episode.
    assert len(logged) == len(SEEDS)
    np.testing.assert_array_equal(state.n_logs, 1)
    for entry in logged:
        assert int(entry["timestep"]) == 80
        assert int(entry["env_frames"]) == 160
        assert float(entry["Eval/mean episodic length"]) == 20
        assert 0 < float(entry["Eval/episodic mean reward"]) <= 40
        assert np.isfinite(entry["Train/episodic mean reward"])
        assert float(entry["Train/total_loss"]) > 0  # the last update's, kept
        assert int(entry["Train/n_updates"]) == 40 + N_ENVS * (40 - 21)
        assert int(entry["Train/terminations"]) == 0
        assert float(entry["ext_smoke/x"]) == 1.0  # the extension's metric
    evals = np.asarray(metrics["Eval/episodic mean reward"])
    logged_ticks = np.flatnonzero(np.isfinite(evals[0]))
    np.testing.assert_array_equal(logged_ticks, [2 * 21 - 1])


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "terminate_at",
    [3, 10],
    ids=["off-schedule", "on-the-last-step"],
)
def test_terminations_are_refused_after_the_run(terminate_at):
    """A termination is refused wherever it happens, including on the
    episode's scheduled last step (``is_terminal`` on the final row)."""
    agent = TDMPC2(
        TerminatingEnv(length=10, terminate_at=terminate_at), seed_steps=5, **TINY
    )
    with pytest.raises(ValueError, match="terminations are not supported"):
        agent.train(seed=0, n_timesteps=30, num_episode_test=1)
