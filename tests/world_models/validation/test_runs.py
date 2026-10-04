"""The chunked runs and the registries of the M9 validation scripts.

Host-side, no training: :func:`wm_runs.run_chunked` on a counter "agent"
(resume from the last save, records after it dropped, a curve line cut
short dropped, a finished run skipped, another specification refused, the
state kept or deleted), the DreamerV3 training-episode probe on synthetic
return windows, the TD-MPC2 evaluation keys, the specification and
provenance helpers, and the paper-protocol and multi-task registries
against the papers' protocols.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import wm_runs
from flax import struct
from multitask_validation import PROTOCOL, SMOKE, MultiTaskProtocol
from paper_protocol import paper_runs, smoke_runs
from wm_runs import (
    CURVE_FILE,
    RUN_FILE,
    STATE_FILE,
    EvalProbe,
    Probe,
    TrainEpisodeProbe,
)


@struct.dataclass
class Counter:
    """A "state": units done and a value that depends on every chunk."""

    done: np.ndarray
    total: np.ndarray


class Interrupted(Exception):
    """Stands for a killed process (raised after a chunk's save)."""


class Echo(Probe):
    """Records the state's value; counts its starts."""

    def __init__(self) -> None:
        self.starts: list = []

    def start(self, state, progress):
        self.starts.append(None if state is None else progress)

    def measure(self, state, progress):
        return {"value": [float(state.total)]}


def _run(tmp_path, *, budget=6, chunk=2, spec=None, interrupt_at=None, **kwargs):
    calls = []

    def train(state, n):
        calls.append(n)
        if state is None:
            state = Counter(done=np.asarray(0), total=np.asarray(0.0))
        done = int(state.done) + n
        return Counter(done=np.asarray(done), total=state.total * 2 + done)

    def log(message):
        if interrupt_at is not None and f": {interrupt_at}/" in message:
            raise Interrupted

    probe = Echo()
    records, state = wm_runs.run_chunked(
        str(tmp_path),
        spec=spec or {"name": "counter"},
        budget=budget,
        chunk=chunk,
        train=train,
        skeleton=lambda: Counter(done=np.asarray(0), total=np.asarray(0.0)),
        progress_of=lambda s: int(s.done),
        probe=probe,
        log=log,
        **kwargs,
    )
    return records, state, calls, probe


def test_an_interrupted_run_resumes_from_its_last_save(tmp_path):
    full, final, calls, _ = _run(tmp_path / "full", budget=7, chunk=2)
    assert calls == [2, 2, 2, 1]
    assert [r["progress"] for r in full] == [2, 4, 6, 7]
    assert final.total == full[-1]["value"][0]
    assert not os.path.exists(tmp_path / "full" / STATE_FILE)  # deleted at the end

    # Interrupted after chunk 2 (saved after every chunk): resumes at 4.
    with pytest.raises(Interrupted):
        _run(tmp_path / "split", budget=7, chunk=2, save_every_s=0, interrupt_at=4)
    assert os.path.exists(tmp_path / "split" / STATE_FILE)
    records, state, calls, probe = _run(tmp_path / "split", budget=7, chunk=2)
    assert calls == [2, 1] and probe.starts == [4]
    assert records == [
        {**r, "wall_s": s["wall_s"], "provenance": s["provenance"]}
        for r, s in zip(full, records)
    ]
    assert float(state.total) == float(final.total)
    with open(tmp_path / "split" / CURVE_FILE) as f:
        assert [json.loads(line)["progress"] for line in f] == [2, 4, 6, 7]


def test_records_after_the_last_save_are_recomputed(tmp_path):
    """Never saved after the first chunk (a long save interval): an
    interruption at 6 resumes from 2 and drops the records of 4 and 6."""
    with pytest.raises(Interrupted):
        _run(tmp_path, budget=8, chunk=2, save_every_s=0, interrupt_at=2)
    with pytest.raises(Interrupted):
        _run(tmp_path, budget=8, chunk=2, save_every_s=1e9, interrupt_at=6)
    with open(tmp_path / CURVE_FILE) as f:
        assert len(f.readlines()) == 3  # 2, 4, 6 written; state at 2
    records, _, calls, _ = _run(tmp_path, budget=8, chunk=2)
    assert calls == [2, 2, 2]
    assert [r["progress"] for r in records] == [2, 4, 6, 8]


def test_a_curve_line_cut_short_is_dropped_and_recomputed(tmp_path):
    """A process killed while appending a record leaves a partial last
    line: the resume drops it (it came after the save) and rewrites the
    file; a bad line before the last is corruption and raises."""
    full, *_ = _run(tmp_path / "full", budget=6, chunk=2)
    with pytest.raises(Interrupted):
        _run(tmp_path / "split", budget=6, chunk=2, save_every_s=0, interrupt_at=4)
    curve = tmp_path / "split" / CURVE_FILE
    with open(curve, "a") as f:
        f.write('{"progress": 6, "valu')
    assert [r["progress"] for r in wm_runs.read_records(str(curve))] == [2, 4]
    records, _, calls, _ = _run(tmp_path / "split", budget=6, chunk=2)
    assert calls == [2]
    assert [r["value"] for r in records] == [r["value"] for r in full]
    assert [
        json.loads(line)["progress"] for line in curve.read_text().splitlines()
    ] == [
        2,
        4,
        6,
    ]
    curve.write_text('{"progress": 2}\n{"progr\n{"progress": 6}\n')
    with pytest.raises(json.JSONDecodeError):
        wm_runs.read_records(str(curve))
    assert wm_runs.read_records(str(tmp_path / "none.jsonl")) == []
    assert not list(tmp_path.glob("*/*.tmp"))  # atomic writes leave none


def test_a_finished_run_is_skipped_and_a_kept_state_returned(tmp_path):
    _run(tmp_path / "a", budget=4)
    records, state, calls, _ = _run(tmp_path / "a", budget=4)
    assert calls == [] and state is None and len(records) == 2
    _, kept, _, _ = _run(tmp_path / "b", budget=4, keep_state=True)
    assert os.path.exists(tmp_path / "b" / STATE_FILE)
    records, state, calls, _ = _run(tmp_path / "b", budget=4, keep_state=True)
    assert calls == [] and float(state.total) == float(kept.total)


def test_another_specification_is_refused(tmp_path):
    _run(tmp_path, budget=4, spec={"name": "a"})
    with open(tmp_path / RUN_FILE) as f:
        stored = json.load(f)
    assert stored["spec"] == {"name": "a"}
    assert {"git_sha", "jax", "device", "backend"} <= set(stored["provenance"])
    with pytest.raises(ValueError, match="another specification"):
        _run(tmp_path, budget=4, spec={"name": "b"})
    with pytest.raises(ValueError, match="positive"):
        _run(tmp_path / "c", budget=0)


def test_specification_differences_and_provenance_summary():
    expected = {"a": 1, "kw": {"x": (1, 2), "y": 3}, "s": "z"}
    assert wm_runs.spec_differences(expected, {"a": 1, "kw": {"x": [1, 2], "y": 3},
                                               "s": "z"}) == []  # fmt: skip
    assert wm_runs.spec_differences(
        expected, {"a": 2, "kw": {"x": [1], "z": 0}, "t": 1}
    ) == ["a", "kw.x", "kw.y", "kw.z", "s", "t"]
    assert wm_runs.spec_differences(1, 2) == ["<root>"]
    one = {"git_sha": "a", "git_dirty": False, "jax": "0.4", "device": "A100"}
    summary = wm_runs.provenance_summary(
        [{"provenance": one}, {"provenance": {**one, "git_sha": "b", "git_dirty": True}},
         {"provenance": one}]
    )  # fmt: skip
    assert summary == {
        "git_sha": ["a", "b"],
        "dirty": True,
        "jax": ["0.4"],
        "device": ["A100"],
    }


def test_tdmpc2_evaluation_keys_depend_on_seed_and_progress_only():
    """The key of seed ``s`` after ``p`` agent steps is
    ``fold_in(PRNGKey(s), p)``, in a fresh probe as in a resumed one."""
    import jax

    def evaluate(state, keys):  # the keys (raw uint32 pairs) as the "returns"
        keys = np.asarray(keys)
        return {
            "Eval/episodic mean reward": keys[:, 1].astype(np.float64),
            "Eval/mean episodic length": np.zeros(len(keys)),
        }

    fresh, resumed = EvalProbe(evaluate, [0, 1], 2), EvalProbe(evaluate, [0, 1], 2)
    fresh.start(None, 0)
    resumed.start(object(), 40)
    expected = [float(jax.random.fold_in(jax.random.PRNGKey(s), 80)[1]) for s in (0, 1)]
    assert fresh.measure(None, 80)["value"] == expected
    assert resumed.measure(None, 80)["value"] == expected
    assert fresh.measure(None, 40)["value"] != expected
    assert fresh.measure(None, 80)["episodes"] == [2, 2]


def _window_state(index, returns):
    """A state whose collector holds return windows ``[S, W, n_envs]`` with
    write positions ``[S, n_envs]``."""
    window = SimpleNamespace(
        buffer=np.asarray(returns, np.float32)[..., None],
        index=np.asarray(index, np.int8)[..., None],
    )
    return SimpleNamespace(
        collector_state=SimpleNamespace(episodic_return_state=window)
    )


def test_the_training_episode_probe_reads_the_chunk_episodes():
    """T = 2 (3 rows per episode), 2 envs, a 4-slot window, 1 seed: episodes
    end on ticks 1, 4, 7, ...: after t ticks (t + 1) // 3 have ended."""
    probe = TrainEpisodeProbe(n_envs=2, episode_length=2)
    assert [probe.episodes_after(2 * t) for t in (0, 1, 2, 4, 5, 8)] == [
        0,
        0,
        1,
        1,
        2,
        3,
    ]
    probe.start(None, 0)
    # After 5 ticks (10 rows): 2 episodes per env, slots 0 and 1.
    returns = np.zeros((1, 4, 2))
    returns[0, :2] = [[1.0, 10.0], [2.0, 20.0]]
    fields = probe.measure(_window_state([[2, 2]], returns), 10)
    assert fields["episodes"] == [4] and fields["return_sum"] == [33.0]
    assert fields["value"] == [33.0 / 4]
    # One more tick: no episode ended, NaN.
    fields = probe.measure(_window_state([[2, 2]], returns), 12)
    assert fields["episodes"] == [0] and np.isnan(fields["value"][0])
    # After 10 ticks (20 rows): (10 + 1) // 3 = 3 ended, one more per env,
    # written at slot 2.
    returns[0, 2] = [3.0, 30.0]
    fields = probe.measure(_window_state([[3, 3]], returns), 20)
    assert fields["return_sum"] == [33.0] and fields["episodes"] == [2]
    # 9 more ticks: 3 more episodes per env, the window wraps (slots 3, 0, 1).
    returns[0, 3], returns[0, 0], returns[0, 1] = [4, 40], [5, 50], [6, 60]
    fields = probe.measure(_window_state([[2, 2]], returns), 38)
    assert fields["episodes"] == [6] and fields["return_sum"] == [165.0]


def test_the_training_episode_probe_refuses_what_it_cannot_read():
    probe = TrainEpisodeProbe(n_envs=1, episode_length=2)
    probe.start(None, 0)
    with pytest.raises(ValueError, match="T \\+ 1 rows"):  # a termination
        probe.measure(_window_state([[1]], np.zeros((1, 4, 1))), 5)
    probe.start(None, 0)
    with pytest.raises(ValueError, match="return window keeps"):
        probe.measure(_window_state([[0]], np.zeros((1, 4, 1))), 15)
    # A resumed probe counts from the restored state.
    probe.start(_window_state([[1]], np.zeros((1, 4, 1))), 5)
    returns = np.zeros((1, 4, 1))
    returns[0, 1] = 7.0
    assert probe.measure(_window_state([[2]], returns), 8)["return_sum"] == [7.0]


def test_the_paper_protocol_registry():
    """DreamerV3 on the 18 Table 11 tasks (12m, ratio 512, 16 envs, repeat 2,
    250K rows = 500K env steps, 5 seeds); TD-MPC2 on the 22 playground DMC
    tasks of its results (size 5, 1 env, repeat 2, the CSV budget, 3 seeds,
    10 eval episodes every 50K steps)."""
    runs = paper_runs()
    dreamer = {k: v for k, v in runs.items() if v.agent == "DreamerV3"}
    tdmpc2 = {k: v for k, v in runs.items() if v.agent == "TDMPC2"}
    assert len(dreamer) == 18 and len(tdmpc2) == 22
    walker = dreamer["dreamerv3-walker_walk"]
    assert walker.env_id == "WalkerWalk" and walker.reference_task == "dmc_walker_walk"
    assert walker.kwargs == {
        "model_size": "12m",
        "n_envs": 16,
        "train_ratio": 512,
        "action_repeat": 2,
        "episode_length": 1000,
    }
    assert walker.env_frames(walker.budget) == 500_000
    assert walker.seeds == (0, 1, 2, 3, 4) and walker.metric == "train_episode_return"
    # 10K rows = 625 ticks of 16 envs, 20K env steps, at most 2 of the
    # 501-row episodes per env (the probe's window keeps 10).
    assert walker.chunk // 16 == 625 and walker.env_frames(walker.chunk) == 20_000
    assert walker.chunk // 16 // 501 + 1 <= 10
    assert all(v.budget == walker.budget for v in dreamer.values())
    cheetah, humanoid = tdmpc2["tdmpc2-cheetah-run"], tdmpc2["tdmpc2-humanoid-run"]
    assert cheetah.kwargs == {
        "model_size": 5,
        "n_envs": 1,
        "action_repeat": 2,
        "episode_length": 1000,
    }
    assert cheetah.env_frames(cheetah.budget) == 4_000_000
    assert humanoid.env_frames(humanoid.budget) == 14_000_000
    assert cheetah.chunk == 50_000 and cheetah.num_eval_episodes == 10
    assert cheetah.seeds == (0, 1, 2) and cheetah.metric == "eval_return"
    # TD-MPC2 units are agent steps; env steps (the references' x) are
    # twice as many.
    assert cheetah.unit == "agent_steps" and walker.unit == "rows"
    assert cheetah.budget == 2_000_000 and cheetah.env_frames(50_000) == 100_000
    assert tdmpc2["tdmpc2-cup-catch"].env_id == "BallInCup"
    for spec in tdmpc2.values():
        assert spec.chunk % 500 == 0 and spec.budget % spec.chunk == 0
    for spec in smoke_runs().values():
        assert spec.smoke and spec.budget == 2 * spec.chunk


def test_the_multitask_protocol():
    assert PROTOCOL.tasks[:3] == ("walker-stand", "walker-walk", "walker-run")
    assert len(PROTOCOL.tasks) == 19 and PROTOCOL.tasks[-1] == "hopper-hop"
    assert PROTOCOL.source_kwargs["action_repeat"] == 2
    assert PROTOCOL.offline_kwargs["action_repeat"] == 2
    assert PROTOCOL.source_kwargs == paper_runs()["tdmpc2-walker-walk"].kwargs
    assert PROTOCOL.source_kwargs.get("buffer_size", 1_000_000) >= (
        PROTOCOL.source_budget
    )
    assert (PROTOCOL.source_budget, PROTOCOL.source_chunk) == (500_000, 50_000)
    assert PROTOCOL.source_seeds == (0, 1, 2) and PROTOCOL.eval_episodes == 10
    assert PROTOCOL.offline_kwargs["model_size"] == 19
    assert PROTOCOL.offline_kwargs["batch_size"] == 1024
    assert (PROTOCOL.offline_updates, PROTOCOL.offline_chunk) == (1_000_000, 100_000)
    assert PROTOCOL.offline_seeds == (0,)
    order = [PROTOCOL.tasks.index(t) for t in SMOKE.tasks]
    assert order == sorted(order)  # the smoke tasks keep mt30 order
    # The full-history guard at its boundary: a buffer of the run's length.
    budget = PROTOCOL.source_budget
    fields = PROTOCOL.__dict__
    MultiTaskProtocol(**{**fields, "source_kwargs": {"buffer_size": budget}})
    with pytest.raises(ValueError, match="full history"):
        MultiTaskProtocol(**{**fields, "source_kwargs": {"buffer_size": budget - 1}})
