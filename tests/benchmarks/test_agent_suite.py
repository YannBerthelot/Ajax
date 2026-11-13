"""``benchmarks/suite/agent_suite.py``, the agent benchmark the cloud sessions
run.

The contract, from the task that ordered it (owner, 2026-11-02) and the
module's docstring:

- every public agent is benchmarked or excluded with a reason; a benchmarked
  agent builds on its env with its defaults but n_envs, and its budget is
  whole evaluation intervals of whole training iterations;
- ``run`` trains a seed range to its budget chunk by chunk: one evaluation
  interval until a chunk has begun past the agent's warm-up, then the whole
  intervals that fill the chunk time at the measured pace of the last chunk
  begun past it (warm-up steps run without updates, faster than training);
  it records each chunk (its evaluations, steps, speed and manifest) and
  publishes it;
- an invocation stops before a chunk predicted to end past its time limit,
  and the next one trains only the chunks not recorded, from the kept state;
  without that state (its machine gone) it starts a new attempt from step 0,
  the records kept;
- disjoint seed ranges keep separate records;
- ``time`` resumes past the warm-up and counts every seed's steps.

The run tests use a fake trainer on a fake clock: a chunk takes COMPILE_S
plus its steps at PACE steps per second (WARM_UP_PACE before its warm-up
ends), and each evaluation returns its own step, so a record's returns show
which steps it evaluated.
"""

import json
import os
import re

import agent_suite
import jax
import numpy as np
import pytest

import ajax
from tests.environments.breaking_env import ALWAYS

EVAL = 100
BUDGET = 1000
PACE = 10.0  # run steps per second
WARM_UP_PACE = 100.0  # steps per second before the updates start
COMPILE_S = 5.0
SEEDS = [0, 1]
#: The (path, message) of every publication, cleared before each test.
published: list = []


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTrainer:
    """Keeps states by step as AjaxTrainer does (the chunk's start and end),
    and refuses to resume from a state it does not hold."""

    def __init__(self, clock: Clock, saved=(), warm_up: int = 0) -> None:
        self.clock, self.saved, self.warm_up = clock, set(saved), warm_up
        self.calls: list[tuple[int, int]] = []

    def saved_steps(self) -> set:
        return set(self.saved)

    def train(self, start: int, end: int) -> dict:
        if start and start not in self.saved:
            raise FileNotFoundError(f"no state at step {start}")
        self.calls.append((start, end))
        warming = max(min(end, self.warm_up) - start, 0)
        training = end - start - warming
        self.clock.now += COMPILE_S + warming / WARM_UP_PACE + training / PACE
        self.saved = {start, end}
        steps = list(range(start + EVAL, end + 1, EVAL))
        return {
            "returns": [steps for _ in SEEDS],
            "non_finite_steps": [[0 for _ in steps] for _ in SEEDS],
            "eval_lengths": [[1000 for _ in steps] for _ in SEEDS],
            "compile_s": COMPILE_S,
        }


def run(trainer, path, clock, seeds=SEEDS, chunk_s=30.0, stop_after_s=1e9, warm_up=0):
    return agent_suite.run_seeds(
        "SAC",
        seeds,
        BUDGET,
        EVAL,
        trainer,
        path,
        publish=lambda path, message: published.append((path, message)),
        chunk_s=chunk_s,
        stop_after_s=stop_after_s,
        clock=clock,
        describe=lambda: {"commit": "abc"},
        warm_up=warm_up,
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def no_publications():
    published.clear()


# --- the table ---------------------------------------------------------------
def test_every_agent_is_benchmarked_or_excluded_with_a_reason():
    benchmarked = {label.split("-")[0] for label in agent_suite.BENCHMARKS}
    assert benchmarked.isdisjoint(agent_suite.EXCLUDED)
    assert sorted(benchmarked.union(agent_suite.EXCLUDED)) == sorted(ajax.__all__)
    assert all(reason.strip() for reason in agent_suite.EXCLUDED.values())


@pytest.mark.parametrize("label", sorted(agent_suite.BENCHMARKS))
def test_every_benchmark_builds_and_chunks_on_whole_iterations(label):
    benchmark = agent_suite.BENCHMARKS[label]
    agent = agent_suite.build(label)  # refuses an env the agent cannot run
    # Default settings apart from n_envs, and options that change no number:
    # DreamerV3's replay latents in host memory (bit-identical, PR #426).
    assert set(benchmark.options).issubset({"n_envs", "replay_on_host"})
    assert benchmark.reason.strip()
    assert benchmark.budget % benchmark.eval_every == 0
    assert benchmark.eval_every % agent_suite.iteration_steps(agent) == 0


def test_a_label_on_a_further_task_runs_its_benchmark_on_that_env():
    # The owner's request (2026-11-03): the same agents, with the same
    # settings, on more tasks.
    base = agent_suite.BENCHMARKS["PPO-MinAtar"]
    further = agent_suite.on_more_tasks(("PPO-MinAtar",), ("Asterix-MinAtar",))
    assert list(further) == ["PPO-MinAtar-Asterix"]
    benchmark = further["PPO-MinAtar-Asterix"]
    assert benchmark.env_id == "Asterix-MinAtar"
    assert (benchmark.budget, benchmark.eval_every, benchmark.options) == (
        base.budget,
        base.eval_every,
        base.options,
    )
    assert agent_suite.BENCHMARKS["APO-WalkerWalk"].options == {"n_envs": 10}


def test_a_gymnasium_mujoco_task_is_labelled_without_its_prefix():
    further = agent_suite.on_more_tasks(("SAC",), ("mujoco:Hopper-v5",))
    assert list(further) == ["SAC-Hopper-v5"]
    assert further["SAC-Hopper-v5"].env_id == "mujoco:Hopper-v5"


def test_tdmpc2_runs_only_the_gymnasium_mujoco_tasks_without_terminations():
    # Paper-era TD-MPC2 refuses episodes that end before the time limit, and
    # Hopper's end when it falls; HalfCheetah's always run to the limit.
    assert "TDMPC2-HalfCheetah-v5" in agent_suite.BENCHMARKS
    assert "TDMPC2-Hopper-v5" not in agent_suite.BENCHMARKS
    assert "SAC-Hopper-v5" in agent_suite.BENCHMARKS


# --- run, on the fake trainer ------------------------------------------------
def test_a_run_trains_its_budget_in_chunks_sized_to_the_measured_pace(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    trainer = FakeTrainer(clock)

    finished = run(trainer, path, clock)

    # The first chunk is one interval: 5 s compiling + 100 steps at 10/s =
    # 15 s, so 10 steps/s once compiled. A 30 s chunk then holds (30 - 5) x
    # 10 = 250 steps: two whole intervals. The last chunk is what remains.
    assert finished
    assert trainer.calls == [
        (0, 100),
        (100, 300),
        (300, 500),
        (500, 700),
        (700, 900),
        (900, 1000),
    ]
    written = records(path)
    assert [(r["start"], r["end"], r["attempt"]) for r in written] == [
        (start, end, 0) for start, end in trainer.calls
    ]
    for record in written:
        assert record["returns"] == [record["eval_steps"]] * len(SEEDS)
        zeros, lengths = (
            [0 for _ in record["eval_steps"]],
            [1000 for _ in record["eval_steps"]],
        )
        assert record["non_finite_steps"] == [zeros] * len(SEEDS)
        assert record["eval_lengths"] == [lengths] * len(SEEDS)
        assert record["run_steps_per_s"] == PACE
        assert record["seeds"] == SEEDS and record["budget"] == BUDGET
    assert [p for p, _ in published] == [path] * len(written)


def test_a_chunk_begun_inside_the_warm_up_does_not_set_the_pace(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    trainer = FakeTrainer(clock, warm_up=150)

    finished = run(trainer, path, clock, warm_up=150)

    # Steps before 150 run at 100/s, the later ones at 10/s. The first chunk
    # (0-100) is all warm-up, 100 steps/s once compiled: sized at that pace, a
    # 30 s chunk would hold the whole rest of the budget, about 90 s of
    # training. The chunk begun at 100 still holds warm-up steps (100-150), so
    # the one begun at 200 is one interval too. From its pace, 10 steps/s, a
    # 30 s chunk holds two whole intervals.
    assert finished
    assert trainer.calls == [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 500),
        (500, 700),
        (700, 900),
        (900, 1000),
    ]


def test_an_interrupted_run_trains_only_the_chunks_not_recorded(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    trainer = FakeTrainer(clock)

    # 60 s per invocation: 15 s, then 25 s; a third 25 s chunk would end at
    # 65 s, so the invocation stops after two.
    first = run(trainer, path, clock, stop_after_s=60.0)
    after_first = list(trainer.calls)
    trainer.calls.clear()
    second = run(trainer, path, clock, stop_after_s=60.0)
    after_second = list(trainer.calls)
    trainer.calls.clear()
    third = run(trainer, path, clock, stop_after_s=60.0)
    after_third = list(trainer.calls)
    trainer.calls.clear()
    again = run(trainer, path, clock, stop_after_s=60.0)

    assert (first, second, third, again) == (False, False, True, True)
    assert after_first == [(0, 100), (100, 300)]
    assert after_second == [(300, 500), (500, 700)]
    assert after_third == [(700, 900), (900, 1000)]
    assert trainer.calls == []
    assert [r["end"] for r in records(path)] == [100, 300, 500, 700, 900, 1000]


def test_a_chunk_whose_record_was_lost_is_trained_again_from_its_start(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    run(FakeTrainer(clock), path, clock, stop_after_s=20.0)  # records 0-100
    # Killed after saving the state at 300, before recording that chunk.
    trainer = FakeTrainer(clock, saved={100, 300})

    run(trainer, path, clock)

    assert trainer.calls[0] == (100, 300)
    assert {r["attempt"] for r in records(path)} == {0}


def test_a_run_whose_state_was_lost_starts_a_new_attempt_from_step_zero(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    run(FakeTrainer(clock), path, clock, stop_after_s=60.0)  # records 0-300
    trainer = FakeTrainer(clock)  # a new machine: no state kept

    finished = run(trainer, path, clock)

    assert finished
    assert trainer.calls[0] == (0, 200)  # at the pace already measured
    written = [(r["attempt"], r["start"], r["end"]) for r in records(path)]
    assert written[:2] == [(0, 0, 100), (0, 100, 300)]
    assert written[2:] == [(1, start, end) for start, end in trainer.calls]


def test_disjoint_seed_ranges_keep_separate_records(tmp_path):
    clock = Clock()
    low, high = [0, 1], [2, 3]
    low_path = tmp_path / f"{agent_suite.range_name(low)}.jsonl"
    high_path = tmp_path / f"{agent_suite.range_name(high)}.jsonl"
    run(FakeTrainer(clock), low_path, clock, seeds=low)
    low_records = low_path.read_text()
    trainer = FakeTrainer(clock)

    run(trainer, high_path, clock, seeds=high)

    assert low_path != high_path
    assert trainer.calls[0] == (0, 100)  # the other range's records aside
    assert low_path.read_text() == low_records
    assert {tuple(r["seeds"]) for r in records(high_path)} == {(2, 3)}


def test_a_run_refuses_records_of_another_budget(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"
    run(FakeTrainer(clock), path, clock, stop_after_s=20.0)

    with pytest.raises(ValueError, match="budget"):
        agent_suite.run_seeds(
            "SAC",
            SEEDS,
            2 * BUDGET,
            EVAL,
            FakeTrainer(clock),
            path,
            publish=lambda *_: None,
            chunk_s=30.0,
            stop_after_s=1e9,
            clock=clock,
        )


def test_each_record_carries_its_manifest(tmp_path):
    clock, path = Clock(), tmp_path / "seeds-0-1.jsonl"

    agent_suite.run_seeds(
        "SAC",
        SEEDS,
        EVAL,
        EVAL,
        FakeTrainer(clock),
        path,
        publish=lambda *_: None,
        chunk_s=30.0,
        stop_after_s=1e9,
        clock=clock,
    )

    manifest = records(path)[0]["manifest"]
    assert re.fullmatch("[0-9a-f]{40}", manifest["commit"])
    assert isinstance(manifest["clean_tree"], bool)
    assert manifest["machine"]["cores"] == os.cpu_count()
    assert manifest["jax"] == jax.__version__


def test_a_manifest_says_which_device_and_xla_flags_a_run_used(monkeypatch):
    """Catches a record that can't say it ran with the XLA workaround, or on
    which GPU: those lived only in pull request descriptions."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_disable_hlo_passes=algsimp")

    manifest = agent_suite.manifest()

    assert manifest["xla_flags"] == "--xla_disable_hlo_passes=algsimp"
    assert manifest["device"] == jax.devices()[0].device_kind


def test_dreamerv3_benchmarks_keep_their_replay_latents_in_host_memory():
    """10 seeds of DreamerV3 fit one GPU job only with the latents on the host."""
    assert agent_suite.BENCHMARKS["DreamerV3"].options["replay_on_host"] is True
    assert agent_suite.BENCHMARKS["DreamerV3-Hopper-v5"].options["replay_on_host"]


def test_a_manifest_says_which_physics_engine_a_run_used():
    """Warp and pure-JAX runs give different runs, so a record must say which."""
    assert agent_suite.manifest()["physics"] == "jax"
    assert agent_suite.manifest("warp")["physics"] == "warp"


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="Warp runs on a GPU")
def test_a_warp_benchmark_builds_its_env_with_warp_physics():
    """The physics choice reaches the env builder: on a CPU, Warp's refusal."""
    with pytest.raises(ValueError, match="warp"):
        agent_suite.build("SAC-Hopper-v5", physics="warp")


def test_a_chunk_reports_each_seeds_non_finite_steps(monkeypatch, tmp_path):
    """The stand-in's episodes diverge at step 3. Each seed's 2 envs, 8 steps
    each, diverge at steps 3 and 6: 4. An evaluation episode lasts those 3."""
    options = {"n_envs": 2, "n_steps": 8, "batch_size": 8, "n_epochs": 1}
    benchmark = agent_suite.Benchmark(ALWAYS, 16, 16, "", options)
    monkeypatch.setitem(agent_suite.BENCHMARKS, "PPO-Breaks", benchmark)
    trainer = agent_suite.AjaxTrainer("PPO-Breaks", SEEDS, 16, 16, tmp_path)
    result = trainer.train(0, 16)
    assert result["non_finite_steps"] == [[4] for _ in SEEDS]
    assert result["eval_lengths"] == [[3] for _ in SEEDS]


# --- the real wiring, on DQN's benchmark at a tiny budget --------------------
def test_ajax_runs_resume_a_new_process_from_the_kept_checkpoint(tmp_path):
    # 128 steps, under DQN's 1000-step warm-up: no update, the wiring alone.
    budget, every, folder = 128, 64, tmp_path / "checkpoints"
    path = tmp_path / "seeds-0-1.jsonl"

    def invocation(stop_after_s):
        trainer = agent_suite.AjaxTrainer("DQN", SEEDS, budget, every, folder)
        finished = agent_suite.run_seeds(
            "DQN",
            SEEDS,
            budget,
            every,
            trainer,
            path,
            publish=lambda *_: None,
            chunk_s=0.0,
            stop_after_s=stop_after_s,
            describe=lambda: {},
        )
        return finished, trainer

    first, _ = invocation(stop_after_s=0.0)  # stops after its first chunk
    second, trainer = invocation(stop_after_s=1e9)

    assert (first, second) == (False, True)
    written = records(path)
    assert [(r["start"], r["end"], r["attempt"]) for r in written] == [
        (0, 64, 0),
        (64, 128, 0),
    ]
    for record in written:
        returns = np.asarray(record["returns"])
        assert returns.shape == (len(SEEDS), 1) and np.isfinite(returns).all()
        assert record["non_finite_steps"] is None  # gymnax has no guard to count
        assert np.asarray(record["eval_lengths"]).shape == returns.shape
    assert trainer.saved_steps() == {64, 128}


def test_an_on_policy_run_ends_each_chunk_where_it_is_asked(tmp_path, monkeypatch):
    """An on-policy chunk of whole evaluation intervals trains its whole
    rollouts and no more: on a tiny PQN (rollouts of 2 envs x 8 steps,
    evaluated every 2 of them), each chunk ends where run_seeds asks."""
    options = {"n_envs": 2, "n_steps": 8, "architecture": ("8", "relu")}
    options.update({"num_minibatches": 4})  # 16 rows a rollout
    tiny = agent_suite.Benchmark("CartPole-v1", 64, 32, "tiny", options)
    monkeypatch.setitem(agent_suite.BENCHMARKS, "PQN", tiny)
    path = tmp_path / "seeds-0-1.jsonl"
    trainer = agent_suite.AjaxTrainer("PQN", SEEDS, 64, 32, tmp_path / "states")
    finished = agent_suite.run_seeds(
        "PQN",
        SEEDS,
        64,
        32,
        trainer,
        path,
        publish=lambda *_: None,
        chunk_s=0.0,  # one evaluation interval a chunk
        stop_after_s=1e9,
        describe=lambda: {},
    )
    assert finished
    assert [(r["start"], r["end"]) for r in records(path)] == [(0, 32), (32, 64)]


# --- time ----------------------------------------------------------------------
def test_time_resumes_past_the_warm_up_and_counts_every_seeds_steps():
    record = agent_suite.time_agent("DQN", n_seeds=2, steps=64)

    # DQN acts on one env (iterations of 1 step) and updates from step 1000.
    assert record["warm_up_steps"] == 1001
    assert record["measured_steps"] == 64
    assert record["env_steps_per_s"] == pytest.approx(
        2 * record["run_steps_per_s"], rel=1e-2
    )
    assert record["compile_s"] > 0 and record["exec_s"] > 0
    assert record["peak_memory_gb"] > 0
    assert record["device_peak_memory_gb"] is None  # a CPU has no device memory
