"""``benchmarks/agent_bench.py``: its presets time training, not only
collection (each one updates within the bench's run, read from the agents'
configs), its CLI, and its execution-time measure on one tiny run. The
bench itself is not run here (its timings need a quiet machine).

The script is not a package module; it is loaded from its path.
"""

import importlib.util
import json
import os
import sys

import jax.numpy as jnp
import pytest

from ajax.agents.loop import TrainLoop

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "benchmarks",
    "agent_bench.py",
)
_SPEC = importlib.util.spec_from_file_location("agent_bench", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent_bench = importlib.util.module_from_spec(_SPEC)
sys.modules["agent_bench"] = agent_bench
_SPEC.loader.exec_module(agent_bench)


def updates_in_run(agent, n_envs: int, timesteps: int) -> int:
    """The updates a run of ``timesteps`` makes, from the agent's config."""
    schedule = getattr(agent, "schedule", None)
    if schedule is not None and hasattr(schedule, "total_updates"):  # DreamerV3
        return schedule.total_updates(timesteps // n_envs)
    if schedule is not None:  # TD-MPC2's seed phase, burst, then per tick
        ticks = jnp.arange(schedule.num_ticks(timesteps))
        return int(schedule.n_updates(ticks).sum())
    config = agent.agent_config
    if hasattr(config, "learning_starts"):
        # One update per iteration (n_envs steps) once the collected steps
        # reach learning_starts (TrainLoop.maybe_update).
        iterations = timesteps // n_envs
        first = max(1, -(-config.learning_starts // n_envs))
        return max(0, iterations - first + 1)
    if hasattr(config, "n_steps"):  # one update per rollout
        loop = TrainLoop.create(agent.env_args, timesteps, num_episode_test=0)
        return loop.n_rollouts(config.n_steps)
    # APG: one update per closed-loop rollout of ``horizon`` steps
    # (train_APG.make_train's num_updates).
    return max(timesteps // (n_envs * config.horizon), 1)


@pytest.mark.parametrize("name", sorted(agent_bench.AGENTS))
def test_every_preset_updates_within_the_run(name):
    env_id, kwargs = agent_bench.AGENTS[name]
    agent = agent_bench._build(name, env_id, kwargs, agent_bench.N_ENVS)
    n_updates = updates_in_run(agent, agent_bench.N_ENVS, agent_bench.TIMESTEPS)
    assert n_updates > 0, f"{name} would time collection only"


def test_capturing_without_out_is_refused(capsys):
    """A capture defaulting to the baseline file overwrote it with the rows
    of the agents named by --only."""
    with pytest.raises(SystemExit) as exit_info:
        agent_bench.main(["--only", "SAC"])
    assert exit_info.value.code == 2
    assert "--out is required" in capsys.readouterr().err


def test_covered_seconds_counts_overlapping_spans_once():
    """A jit traced inside another's trace reports a span inside the outer
    one; the union is what the run spent outside execution."""
    spans = [(5.0, 12.0), (0.0, 10.0), (2.0, 3.0), (20.0, 21.0)]
    assert agent_bench.covered_seconds(spans) == 13.0
    assert agent_bench.covered_seconds([]) == 0.0


def test_a_measured_run_reports_its_execution_without_tracing():
    """A fresh agent retraces in every trial: exec_s leaves that out."""
    (row,) = agent_bench._time_agent(
        "DQN", "CartPole-v1", {}, n_envs=2, warmup=32, timesteps=64, trials=1
    )
    assert 0 < row["exec_s"] < row["measured_s"]
    assert row["exec_steps_per_s"] > row["steps_per_s"]


def write_rows(path, agent, **metrics):
    path.write_text(json.dumps({"agent": agent, **metrics}) + "\n")
    return str(path)


def test_compare_judges_execution_when_both_files_have_it(tmp_path):
    base = write_rows(tmp_path / "b", "SAC", steps_per_s=100.0, exec_steps_per_s=200.0)
    slower = write_rows(
        tmp_path / "n", "SAC", steps_per_s=100.0, exec_steps_per_s=150.0
    )
    assert agent_bench.compare(base, slower, tol=0.1) == 1
    # A baseline recorded before exec_steps_per_s existed: steps_per_s.
    old = write_rows(tmp_path / "o", "SAC", steps_per_s=100.0)
    assert agent_bench.compare(old, slower, tol=0.1) == 0
