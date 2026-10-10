"""P4, split-run equivalence: a run checkpointed, restored into a new agent's
fresh skeleton and resumed must be the run never interrupted. Exact contracts
on two seeds: whole states, counters, logged timesteps, an extension's inputs."""

from __future__ import annotations

import dataclasses
import functools
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax import make as make_gymnax_env
from probing_environments.gymnax_envs import PolicyAndValueEnv as PV
from probing_environments.gymnax_envs import continuous_actions as continuous

import ajax
import ajax.log
from ajax.environments import model_reference as mr
from ajax.environments.system_class import UniformPerturbation
from ajax.extensions import target_mods
from ajax.extensions.expert import ExpertGuidance
from ajax.extensions.pretrain import MCPretrain

from . import agents, envs, runs
from . import readouts as R
from .verdict import Case, xfail, xparam

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "44136fa355b3"  # verdict.digest(CASES): no judged case
SEEDS, N32, TEST = (0, 1), ("32", "relu", "32", "relu"), {"num_episode_test": 1}
RD, VL = continuous.RewardDiscountingEnv, continuous.ValueLossOrOptimizerEnv
# Not built: whole-budget schedules after resuming (each call keys on its own total:
# the owner's call), learned values (a resume that lost every state passes 99.5%).


@dataclasses.dataclass(frozen=True)
class Split:
    """``first`` steps, then ``second`` on a new restored agent: one ``budget`` call."""

    build: Callable[[], Any]
    budget: int
    first: int
    second: int
    kw: dict = dataclasses.field(default_factory=dict)


def split_run(s: Split, folder: str) -> tuple[Any, np.ndarray, Any]:
    """The uninterrupted state, the first leg's last step, the split state."""
    agent, kw = s.build(), {**TEST, **s.kw}
    full = runs.train(agent, SEEDS, s.budget, **kw).state
    first = runs.train(agent, SEEDS, s.first, **kw)
    ended = np.asarray(first.state.collector_state.timestep)  # before donation
    return full, ended, runs.resume(first, s.second, folder, s.build(), **kw).state


def _make(name: str, env: type, kw: dict, preset: str = "probe") -> Callable:
    return lambda: agents.make(name, *envs.package(env), preset=preset, **kw)


def _apg() -> Any:
    """tests/agents/APG/test_APG.py's gravity-free pendulum tracking task."""
    env, params = make_gymnax_env("Pendulum-v1")
    ref = mr.StepReference(16, -0.5, 0.5, 4, 8)  # horizon, values, durations
    model = mr.LinearReferenceModel.first_order()
    wrapped = mr.ModelReferenceWrapper(
        env, ref, model, output_fn=lambda obs: jnp.arctan2(obs[1], obs[0])
    )
    system = UniformPerturbation(params.replace(g=0.0), ("m", "l"), scale=0.1)
    kw: dict[str, Any] = {"n_envs": 4, "horizon": 16, "learning_rate": 3e-3}
    kw |= {"system_class": system, "actor_architecture": ("16", "relu")}
    return agents.make("APG", wrapped, system.nominal, preset=None, **kw)


# Every SAC part updates from step 50, in both legs. The discrete agents run the
# coupling env (the discrete discounting env has one action), epsilon constant
# (its default decays over each call's own total, train_DQN.py:135-138); PQN's
# two minibatches make its shuffle matter. PPO and PQN run one iteration more per
# call (train_PPO.py:1446, train_PQN.py:339): their second leg is one shorter.
NETS = {"actor_architecture": N32, "critic_architecture": N32}
STARTS = ("learning_starts", "policy_update_start", "alpha_update_start")
SAC_KW = {**NETS, "buffer_size": 1000, "batch_size": 32, **dict.fromkeys(STARTS, 50)}
EPS = {"architecture": N32, "epsilon_start": 0.3, "epsilon_end": 0.3}
DQN_KW = {**EPS, "buffer_size": 1000, "batch_size": 32}
UDRL_KW = {"n_envs": 1, "actor_architecture": N32, "n_steps": 8, "batch_size": 8}
UDRL_KW |= {"n_epochs": 1, "n_updates_per_iter": 4}
EQUIVALENT = {
    "SAC": Split(_make("SAC", RD, SAC_KW), 400, 200, 200),
    "SafeSAC": Split(_make("SafeSAC", RD, SAC_KW), 400, 200, 200),
    "DQN": Split(_make("DQN", PV, DQN_KW), 400, 200, 200),
    "PQN": Split(_make("PQN", PV, {**EPS, "num_minibatches": 2}), 2560, 1280, 1152),
    "PPO": Split(_make("PPO", RD, NETS), 640, 320, 288),
    "UDRL": Split(_make("UDRL", PV, UDRL_KW), 128, 64, 64),
    # APG restarts its optimizer on resume by design (train_APG.py:467-471).
    "APG": Split(_apg, 256, 128, 128, {"reset_optimizer_on_resume": False}),
}
NOT_RESUMABLE = {"TD3": "train_TD3.py:1012-1013", "REDQ": "train_REDQ.py:1261-1262"}
NOT_RESUMABLE |= {"ASAC": "train_ASAC.py:1379-1380", "APO": "train_APO.py:1000-1001"}
NOT_RESUMABLE |= {"AVG": "AVG.py:175-180"}  # where train takes no initial state
CANNOT = "{}'s train takes no initial state ({}) while agents/base.py:285-306 passes initial_state and resume_from_state; right answer: the resumed call trains on to the uninterrupted run's step, today TypeError"
ELSEWHERE = {  # world models whose own tests assert a resume equals one call
    "DreamerV3": "DreamerV3/test_dreamerv3_agent.py::test_resume_continues_the_schedule_and_matches_an_uninterrupted_run",
    "TDMPC2": "TDMPC2/test_TDMPC2.py::test_a_new_agent_resumes_a_checkpoint_as_the_uninterrupted_run",
    "TDMPC2MultiTask": "TDMPC2/test_TDMPC2MultiTask.py::test_a_resumed_run_computes_the_uninterrupted_run",
}
# Ten log points a run; PPO's, PQN's two iterations apart: none at a leg's overshoot.
LOGGED = {"SAC": 40, "DQN": 40, "PPO": 64, "PQN": 256}
TRAIN_FRAC = Split(_make("SAC", RD, {**SAC_KW, "use_train_frac": True}), 400, 200, 200)
VB_STEPS = 100


@pytest.mark.parametrize("name", list(EQUIVALENT))
def test_a_resumed_run_equals_the_uninterrupted_run(name: str, tmp_path: Path) -> None:
    """Every leaf alike, floats to rtol 1e-5 (programs compiled apart may round
    apart: SAC's worst leaf is at 0.09 of it on macOS ARM; past it, shorten)."""
    full, ended, split = split_run(EQUIVALENT[name], str(tmp_path))
    steps = np.asarray(full.collector_state.timestep)
    np.testing.assert_array_equal(np.asarray(split.collector_state.timestep), steps)
    differing = R.differing_leaves(full, split)
    assert not differing, (f"first leg ended at {ended}", steps, differing[:20])


@pytest.mark.parametrize(
    "name",
    [xparam(n, CANNOT.format(n, w), TypeError) for n, w in NOT_RESUMABLE.items()],
)
def test_the_agent_resumes_from_a_checkpoint(name: str, tmp_path: Path) -> None:
    """On the bookkeeping preset (D2), resumed after 64 of 128 steps, the call
    ends where one call ends (APO runs an extra iteration per call: 32 more)."""
    build, second = _make(name, VL, {}, "bookkeeping"), 32 if name == "APO" else 64
    first = runs.train(build(), SEEDS, 64, **TEST)
    split = runs.resume(first, second, str(tmp_path), build(), **TEST).state
    full = runs.train(first.agent, SEEDS, 128, **TEST).state
    steps = (np.asarray(s.collector_state.timestep) for s in (split, full))
    np.testing.assert_array_equal(*steps)


def test_every_agent_is_checked_for_resume() -> None:
    """Every agent (export or directory) is above or resume-tested by name elsewhere."""
    root = Path(ajax.__file__).parent / "agents"
    dirs = {p.name for p in root.iterdir() if (p / f"{p.name}.py").is_file()}
    listed = sorted([*EQUIVALENT, *NOT_RESUMABLE, *ELSEWHERE])
    assert listed == sorted(set(ajax.__all__) | dirs)
    for where in ELSEWHERE.values():
        path, test = where.split("::")
        text = (Path(runs.REPO) / "tests" / "agents" / path).read_text()
        assert f"def {test}(" in text, where


# --- A resumed run logs and evaluates as the uninterrupted run does ----------
@functools.cache
def logged(name: str) -> dict[str, tuple[list[int], list[int]]]:
    """Per call, sorted: the timesteps it logged at, and every evaluation it
    computed, logged or not (a callback in ajax.log.evaluate, once per seed)."""
    s, kw, out = EQUIVALENT[name], {**TEST, "log_every": LOGGED[name]}, {}
    agent, evaluate, seen = s.build(), ajax.log.evaluate, list[int]()

    def counted(*args: Any, **kwargs: Any) -> Any:
        result, state = evaluate(*args, **kwargs), kwargs["agent_state"]
        t = state.collector_state.timestep
        jax.debug.callback(lambda t, _: seen.append(int(t)), t, state.eval_rng)
        return result

    def call(leg: str, train: Callable[[], runs.Run]) -> runs.Run:
        seen.clear()  # a resume's 0-step skeleton evaluates nothing
        with mock.patch.object(ajax.log, "evaluate", counted):
            run = train()
        out[leg] = (sorted(int(m["timestep"]) for _, m in run.events), sorted(seen))
        return run

    call("full", lambda: runs.train(agent, SEEDS, s.budget, **kw))
    first = call("first", lambda: runs.train(agent, SEEDS, s.first, **kw))
    with tempfile.TemporaryDirectory() as folder:
        call("second", lambda: runs.resume(first, s.second, folder, s.build(), **kw))
    return out


@pytest.mark.parametrize("name", list(LOGGED))
def test_a_fresh_run_logs_and_evaluates_at_every_log_point(name: str) -> None:
    """The readout check of the two tests below: each fresh call logs and
    evaluates once per seed at every multiple of the frequency up to its total."""
    every, s, calls = LOGGED[name], EQUIVALENT[name], logged(name)
    for leg, steps in (("full", s.budget), ("first", s.first)):
        want = sorted(k * every for k in range(1, steps // every + 1) for _ in SEEDS)
        assert calls[leg] == (want, want), (leg, calls[leg])


@xfail(
    "evaluate_and_log logs only while timestep <= the call's own total_timesteps (log.py:281); a resumed call starts past it, so it logs nothing. Right answer: the uninterrupted run's 10 events per seed; today 5 (the first leg's)"
)
@pytest.mark.parametrize("name", list(LOGGED))
def test_a_resumed_run_logs_where_the_uninterrupted_run_logs(name: str) -> None:
    """The two legs together log where the uninterrupted run logs."""
    calls = logged(name)
    counts = {leg: len(c[0]) for leg, c in calls.items()}  # over both seeds
    assert sorted(calls["first"][0] + calls["second"][0]) == calls["full"][0], counts


@xfail(
    "a resumed call vmaps the restored state over seeds (agents/base.py:301), so the timestep and the log flag are batched and the lax.cond at log.py:409 lowers to a select that evaluates on every iteration. Right answer: evaluations only at log events; today every iteration of the resumed leg"
)
@pytest.mark.parametrize("name", list(LOGGED))
def test_a_resumed_run_evaluates_only_when_it_logs(name: str) -> None:
    """The resumed call evaluates (logged or not) only at its log events."""
    logs, evals = logged(name)["second"]
    assert evals == logs, f"{len(evals)} evaluations at {sorted(set(evals))[:6]}..."


# --- Totals and bounds that must survive a resume -----------------------------
@xfail(
    "SAC(use_train_frac=True) does not trace: interaction.py:970-972 picks the raw next observation, which lacks the training-fraction column, as the next last_obs, so the scan carry changes shape (TypeError). Right answer: it trains to 1.0",
    TypeError,
)
def test_sac_trains_with_the_training_fraction_in_its_observation() -> None:
    """A fresh run trains and ends at training fraction 1.0 (the observation
    column's source, state.py:222-226); no other test runs use_train_frac."""
    run = runs.train(TRAIN_FRAC.build(), SEEDS, TRAIN_FRAC.budget, **TEST)
    frac = run.state.collector_state.train_time_fraction
    np.testing.assert_allclose(np.asarray(frac), 1.0, rtol=0, atol=1e-6)


@xfail(
    "max_timesteps is the first call's total (train_SAC.py:1754) and is restored with the state (state.py:191, 222-226), so after resuming train_frac = timestep / (budget/2). Right answer 1.0 at the end, as in the uninterrupted run; 2.0 behind the trace crash (read with it repaired on a scratch copy). Today it fails first on the trace crash above",
    (TypeError, AssertionError),
)
def test_the_training_fraction_ends_where_the_uninterrupted_run_ends(
    tmp_path: Path,
) -> None:
    """The split run ends at one call's training fraction, 1.0, any schedule."""
    full, _, split = split_run(TRAIN_FRAC, str(tmp_path))
    frac = [np.asarray(s.collector_state.train_time_fraction) for s in (full, split)]
    np.testing.assert_allclose(frac[0], 1.0, rtol=0, atol=1e-6)
    assert np.all(np.abs(frac[1] - frac[0]) <= 1e-6), f"split {frac[1]}, one call 1.0"


class ConstantExpert:
    """Plays 0.5, to act (``expert(obs)``) and to evaluate (``expert(state, obs)``)."""

    def __call__(self, *args: Any) -> Any:
        action = jnp.full((*args[-1].shape[:-1], 1), 0.5, dtype=jnp.float32)
        return action if len(args) == 1 else (action, args[0])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1), dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class RecordingValueBox(target_mods.ValueBox):
    """Per step: the bounds handed to ``box_compute_threshold`` (swapped in while
    ValueBox.action is traced: it looks the function up) and stored on the state
    (target_mods.py:350-351), and the threshold's position in the box (0-1)."""

    sink: Any = None  # a function: hashed by identity, so each is traced afresh

    def action(self, agent_state: Any, ext: Any, obs: Any, rng: Any, ctx: Any) -> Any:
        compute = target_mods.box_compute_threshold
        stored = (agent_state.expert_v_min, agent_state.expert_v_max)

        def recorded(v_min: Any, v_max: Any, frac: Any) -> Any:
            unit = compute(0.0, 1.0, frac)
            jax.debug.callback(self.sink, v_min, v_max, *stored, unit)
            return compute(v_min, v_max, frac)

        with mock.patch.object(target_mods, "box_compute_threshold", recorded):
            return super().action(agent_state, ext, obs, rng, ctx)


def value_box_agent(records: list) -> Any:
    """SAC, MC pre-training and the recording ValueBox under a constant
    expert; the expert buffer off (with it on, the expert path crashes)."""
    e = ConstantExpert()
    off: dict[str, Any] = {"expert_buffer_n_steps": 0, "expert_mix_fraction": 0.0}
    mc: dict[str, Any] = {"n_mc_steps": 200, "n_mc_episodes": 4, "n_steps": 20}
    mc |= {"online_light_steps": 5}

    def sink(*values: Any) -> None:
        records.append(tuple(float(np.asarray(v).item()) for v in values))

    exts = (
        ExpertGuidance(expert_policy=e, **off),
        MCPretrain(expert_policy=e, **mc),
        RecordingValueBox(expert_policy=e, sink=sink),
    )
    kw: dict[str, Any] = {**SAC_KW, **dict.fromkeys(STARTS, 20), **off, "n_envs": 1}
    kw |= {"batch_size": 16, "gamma": agents.GAMMA, "expert_policy": e, "use_box": True}
    return agents.make("SAC", *envs.package(RD), preset=None, extensions=exts, **kw)


@functools.cache
def value_box() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per seed and step, (handed v_min, v_max, stored v_min, v_max, position)
    in a fresh leg and in the resumed leg; the bounds stored after the first."""
    fresh, resumed = list[tuple](), list[tuple]()
    first = runs.train(value_box_agent(fresh), SEEDS, VB_STEPS, **TEST)
    stored = np.stack([first.state.expert_v_min, first.state.expert_v_max], -1)
    with tempfile.TemporaryDirectory() as folder:
        runs.resume(first, VB_STEPS, folder, value_box_agent(resumed), **TEST)
    return np.array(fresh), np.array(resumed), stored


def test_the_value_box_readout_sees_what_a_fresh_leg_computes() -> None:
    """The readout check of the two below: a record per seed and step, the stored
    bounds (far from 0) in both legs, handed in the fresh one, a ramp t / 100."""
    first, second, stored = value_box()
    assert first.shape == second.shape == (len(SEEDS) * VB_STEPS, 5), first.shape
    assert np.all(np.abs(stored) > 1e-3), f"degenerate bounds {stored}"
    rows = np.unique(stored, axis=0)
    for leg in (first, second):
        np.testing.assert_array_equal(np.unique(leg[:, 2:4], axis=0), rows)
    np.testing.assert_array_equal(first[:, :2], first[:, 2:4])
    ramp = np.repeat(np.arange(VB_STEPS) / VB_STEPS, len(SEEDS))
    np.testing.assert_allclose(np.sort(first[:, 4]), ramp, rtol=0, atol=1e-6)


@xfail(
    "on resume make_scan_fn sets the value-box bounds to 0.0 (train_SAC.py:1867-1872) instead of the MC-pretrain bounds stored on the state. Right answer: the stored expert_v_min/expert_v_max; today 0.0 and 0.0, so the threshold collapses"
)
def test_the_value_box_keeps_its_bounds_after_resuming() -> None:
    """At every step of the resumed leg ValueBox is handed the stored bounds."""
    _, second, stored = value_box()
    handed = np.unique(second[:, :2], axis=0).tolist()
    assert np.array_equal(second[:, :2], second[:, 2:4]), (handed, stored.tolist())


@xfail(
    "ValueBox ramps on timestep / the call's own total (target_mods.py:350, fed from train_SAC.py:1905), and a resumed call starts at the restored timestep. Right answer: the threshold stays within the box (position at most 1) at every step; today up to 1.99 after resuming"
)
def test_the_value_box_threshold_stays_within_the_box_after_resuming() -> None:
    """The threshold stays in the box (position <= 1) in either leg, any schedule."""
    first, second, _ = value_box()
    tops = first[:, 4].max(), second[:, 4].max()
    assert max(tops) <= 1.0, f"highest position: fresh leg {tops[0]}, resumed {tops[1]}"
