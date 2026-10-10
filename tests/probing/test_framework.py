"""The harness itself: calibrations, faults, pinned presets and answers, the
env builder's contract, the oracles and readers on known values (one 0-step
run), every exported agent probed, and the line caps: per file and 7,000 for
the directory."""

from __future__ import annotations

import functools
import importlib
import math
import os
import subprocess
import sys
import types
from typing import Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probing_environments.gymnax_envs import continuous_actions as continuous

import ajax
from ajax.networks.memory import MemoryConfig

from . import agents, envs, faults, oracles, runs, verdict
from . import readouts as R

# The plan's lines per file; test_framework 145 + 35 for the API's checks,
# faults/ 170 + 30 (the catalogue proves each probe's power).
CAPS = {"verdict.py": 255, "envs.py": 215, "agents.py": 155, "runs.py": 250}
CAPS |= {"readouts.py": 220, "oracles.py": 170, "faults.py": 100, "faults/": 200}
CAPS |= {"test_framework.py": 180, "test_value_chains.py": 1051}
CAPS |= {"test_episode_ends.py": 1007, "test_bookkeeping.py": 1013}
CAPS |= {"test_extensions.py": 762, "test_p4_split_run.py": 360}
CAPS |= {"test_memory.py": 372, "test_world_models.py": 452}


def _modules() -> list:
    """The scenario modules already in the framework (they define CASES)."""
    files = [f for f in sorted(os.listdir(faults.HERE)) if f.startswith("test_")]
    names = [f[:-3] for f in files if "\nCASES: " in open(f"{faults.HERE}/{f}").read()]
    return [importlib.import_module(f".{m}", __package__) for m in names]


def _lines(path: str) -> int:
    paths = [f"{path}/{f}" for f in os.listdir(path)] if os.path.isdir(path) else [path]
    return sum(sum(1 for _ in open(p)) for p in paths if os.path.isfile(p))


def test_every_calibrated_case_lies_within_its_wrong_answers() -> None:
    cases = [c for m in _modules() for c in m.CASES.values()]
    errors = [e for c in cases for e in verdict.calibration_errors(c)]
    assert not errors, errors


def test_presets_and_answers_are_pinned() -> None:
    """A calibration holds for its configuration and its oracle's answers
    only (``python -m tests.probing.verdict answers MODULE`` shows them)."""
    assert agents.digest() == agents.DIGEST, "a preset changed: recalibrate"
    got = {m.__name__: verdict.digest(m.CASES) for m in _modules()}
    assert got == {m.__name__: m.ANSWER_DIGEST for m in _modules()}


def test_choose_takes_the_first_budget_twice_whose_worst_error_fits() -> None:
    q = verdict.Query("V", 1.0, {"off": 0.8})
    case = verdict.Case("c", (q,), lambda seeds, budget: {}, 0)
    by_budget = {100: {"V": [1.0, 0.9]}, 200: {"V": [1.0, 0.97, 1.01]}}
    assert verdict.choose(case, by_budget) == (200, (0.06,))


def test_every_planted_fault_applies_once_and_names_live_nodes() -> None:
    files = [m.__file__ for m in _modules()]
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *files]
    out = subprocess.run(cmd, cwd=faults.REPO, capture_output=True, text=True)
    nodes = {n.removeprefix("tests/probing/") for n in out.stdout.split("\n")}
    for fault in faults.load().values():
        with open(os.path.join(faults.REPO, fault.file)) as fh:
            assert fh.read().count(fault.old) == 1, fault.id
        dead = set(fault.catches) - nodes
        assert fault.catches and not dead, (fault.id, dead)


def test_spec_env_resets_records_and_truncates() -> None:
    """A recording 3-step counter: reward 1, termination on step 3 with its
    final observation, auto-reset, the record kept; a limit of 2 alone."""

    def transition(st: envs.State, a: jax.Array, key: jax.Array) -> tuple:
        return st.replace(s=st.s + 1), 1.0, st.s + 1 >= 3

    key = jax.random.PRNGKey(0)
    env, params = envs.Spec("t", transition, record=(2, 3)).make()
    _, st = env.reset(key, params)
    for t, a in enumerate((0.1, 0.2, 0.3, 0.4)):
        obs, st, r, term, trunc, info = env.step(key, st, jnp.array([a]), params)
        assert float(r) == 1.0 and bool(term) == (t == 2) and not bool(trunc)
        if t == 2:
            np.testing.assert_array_equal(info["final_observation"], [3.0])
    np.testing.assert_allclose(st.first_a, [0.1, 0.2])
    np.testing.assert_allclose(st.last_a, [0.4, 0.2, 0.3])  # a ring: step 4 overwrote 1
    assert int(st.clock) == 4 and int(st.s) == 1 and float(obs[0]) == 1.0
    env, params = envs.Spec("u", transition, limit=2).make()
    _, st = env.reset(key, params)
    for _ in range(2):
        _, st, _, term, trunc, _ = env.step(key, st, jnp.zeros(1), params)
    assert bool(trunc) and not bool(term) and int(st.s) == 0


def test_oracles_and_readers_meet_known_values() -> None:
    """Q6's sigma*, entropy and softmax, P8's and P5's limits; a simulated 2-step
    episode paying 1 at its end, V = (gamma, 1); the leaf diff; rollout rows
    (seeds, envs, steps); the normaliser's (5 - 2) / 2; N(0, 1)'s entropy."""
    assert round(oracles.max_entropy_sigma(), 4) == 0.8744
    assert round(oracles.tanh_gaussian_entropy(0.8744), 5) == 0.68364
    assert round(oracles.softmax_optimum(1.0), 4) == 0.7311
    assert abs(oracles.maxent_mean_action(0.0, 1.0)) < 1e-6  # no slope, no lean
    assert oracles.clipped_square_mean(0.5, 100.0) == pytest.approx(0.25)
    assert oracles.ema_step_rms(1.0, 2.0) == pytest.approx(2.0 * math.sqrt(2.0))

    def episode(s: np.ndarray) -> tuple:  # state, obs, reward, terminated, final
        return 1 - s, s, 1.0 * s, s == 1, 1 - s

    cols = oracles.simulate(episode, np.zeros(2, int), 9)  # 2 envs, 8 rows
    v = oracles.fixed_point(oracles.blocks(cols, 4), 2, 0.5, 0.5)
    np.testing.assert_allclose(v, [0.5, 1.0])
    a, b = ({"k": jax.random.PRNGKey(i), "w": np.full(3, i)} for i in (0, 1))
    assert R.differing_leaves(a, a) == [] and len(R.differing_leaves(a, b)) == 2
    rollout = types.SimpleNamespace(obs=np.arange(30).reshape(2, 5, 3))
    rows = R.rollout_rows(types.SimpleNamespace(last_rollout=rollout), ("obs",))
    np.testing.assert_array_equal(rows["obs"][0, 1], [1, 4, 7, 10, 13])
    full = functools.partial(np.full, (2, 1, 1))  # (envs, 1, obs_dim) leaves
    stats = types.SimpleNamespace(count=full(4), mean=full(2), mean_2=full(16))
    n = R.Nets(None, stats=stats)
    assert float(R.input_error(n, jnp.array([[5.0]]), jnp.array([1.5]))) == 0.0
    soft = R.soft_value(distrax.Normal(0.0, 1.0), 1.0, lambda x: x, R.KEY)
    assert abs(soft - 0.5 * math.log(2 * math.pi * math.e)) < 0.05


def test_recurrent_readers_and_checkpoint_resume(tmp_path: Any) -> None:
    """A recurrent actor stepped reads what its sequence path reads; a resume
    through a checkpoint equals one in memory; the world models build."""
    env, p = envs.package(continuous.ValueLossOrOptimizerEnv)
    for name in ("DreamerV3", "TDMPC2"):
        for preset in ("tiny", "bookkeeping"):
            agents.make(name, env, p, preset=preset)
    memory = MemoryConfig("gru", 8)
    ppo = agents.make("PPO", env, p, preset="bookkeeping", memory=memory)
    run, axes = runs.train(ppo, (0, 1), 0), (0, None, None)
    obs, starts = jax.random.normal(R.KEY, (4, 3, 1)), jnp.eye(4, 3, dtype=bool)
    n = R.nets(run.state)
    seq = jax.vmap(lambda n: R.actor_sequence(n, obs, starts).mean())(n)
    step = jax.vmap(R.step_actor, axes)(n, obs, starts)
    np.testing.assert_allclose(step, seq, atol=1e-6)
    assert jax.vmap(R.critic_sequence, axes)(n, obs, starts).shape == (2, 4, 3)
    through = runs.resume(run, 32, str(tmp_path)).state  # first: the other donates
    assert R.differing_leaves(through, runs.resume(run, 32).state, 0.0, 0.0) == []


def test_every_file_keeps_to_its_cap() -> None:
    over = {f: (_lines(os.path.join(faults.HERE, f)), CAPS[f]) for f in CAPS}
    over = {f: v for f, v in over.items() if v[0] > v[1]}
    assert not over, over


def test_the_directory_keeps_to_its_budget() -> None:
    """Every file git tracks or would track (untracked, not ignored)."""
    cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(
        [*cmd, "tests/probing"], cwd=faults.REPO, capture_output=True, text=True
    )
    total = sum(_lines(os.path.join(faults.REPO, f)) for f in out.stdout.split())
    assert total <= 7000, total


def test_every_exported_agent_is_probed() -> None:
    ids = [f"{c}-" for m in _modules() for c in m.CASES]
    missing = [a for a in ajax.__all__ if not any(f"-{a}-" in i for i in ids)]
    assert not missing, missing
