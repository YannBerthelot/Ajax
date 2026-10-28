"""LoggingConfig's backends.

Contract (the owner's design of 2026-10-28, ajax.logging.backends):
- backends given: every run logs to exactly those. Each backend's start(run),
  if it has one, runs in the caller before training; in the logging worker
  one backend object opens every run before its first log, then gets each
  evaluation train returns as log(run_id, step, metrics), NaN metrics left
  out, and close once at the end.
- backends empty: use_wandb (on unless set False) and use_tensorboard (off
  unless set True) name WandbBackend(project_name, group_name) and
  TensorBoardBackend(folder, the working directory when None), as before.
- backends with either flag set: refused, the message naming the backends
  form. A backend that cannot be copied into the worker: refused before
  training; one the worker cannot read stops neither the others nor train.
- config defaults to an empty dict of each LoggingConfig's own.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
from typing import Any

import pytest

from ajax.logging.backends import RunInfo, TensorBoardBackend, WandbBackend
from ajax.logging.wandb_logging import LoggingConfig
from tests.logging import fake_wandb
from tests.logging.test_logging_pins import AGENTS, SEEDS, logged_rows


@dataclasses.dataclass(frozen=True)
class RecordingBackend:
    """Writes each call it gets to a file, with its process and object."""

    path: str

    def _write(self, call: str, **fields: Any) -> None:
        line = {"call": call, "pid": os.getpid(), "object": id(self), **fields}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(line) + "\n")

    def start(self, run: RunInfo) -> None:
        self._write("start", run_id=run.run_id, seed=run.seed)

    def open(self, run: RunInfo) -> None:
        self._write("open", run_id=run.run_id, name=run.name, seed=run.seed)

    def log(self, run_id: str, step: int, metrics: Any) -> None:
        values = {k: v.item() for k, v in metrics.items()}
        self._write("log", run_id=run_id, step=step, metrics=values)

    def close(self) -> None:
        self._write("close")


def train_sac(backends: tuple) -> tuple[dict, list[str]]:
    """A tiny two-seed SAC run logging to backends: its evaluations and run ids."""
    make, n_timesteps, every = AGENTS["SAC"]
    logging = LoggingConfig(run_name="run", log_frequency=every, backends=backends)
    run_ids: list[str] = []
    _, evaluations = make().train(
        seed=SEEDS,
        n_timesteps=n_timesteps,
        num_episode_test=2,
        logging_config=logging,
        on_ids_ready=run_ids.extend,
    )
    return evaluations, run_ids


def as_seen(call: dict) -> tuple[bool, str, dict]:
    """Whether this process made the call, its name and its run's fields."""
    fields = {k: v for k, v in call.items() if k not in ("pid", "call", "object")}
    fields.pop("run_id")
    return call["pid"] == os.getpid(), call["call"], fields


def test_a_custom_backend_receives_every_evaluation_of_each_run(
    tmp_path: Any, wandb_calls: str
) -> None:
    path = str(tmp_path / "calls.jsonl")
    evaluations, run_ids = train_sac((RecordingBackend(path),))

    calls = fake_wandb.read_calls(path)
    in_worker = [c for c in calls if c["pid"] != os.getpid()]
    assert len({c["object"] for c in in_worker}) == 1
    assert [c["call"] for c in calls].count("close") == 1
    assert calls[-1]["call"] == "close"
    for seed_index, (seed, run_id) in enumerate(zip(SEEDS, run_ids, strict=True)):
        rows = logged_rows(evaluations, seed_index)
        expected = [
            (True, "start", {"seed": seed}),
            (False, "open", {"seed": seed, "name": f"run_{run_id}"}),
            *[(False, "log", {"step": t, "metrics": m}) for t, m in rows],
        ]
        got = [as_seen(c) for c in calls if c.get("run_id") == run_id]
        assert got == expected
    assert not os.path.exists(wandb_calls)  # W&B was not named


def test_a_backend_the_worker_cannot_read_stops_neither_the_others_nor_train(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class on this module in this process only, as one defined in a
    # notebook cell: it pickles here, and the worker cannot unpickle it.
    late = type("Late", (RecordingBackend,), {"__module__": __name__})
    monkeypatch.setattr(sys.modules[__name__], "Late", late, raising=False)
    path = str(tmp_path / "calls.jsonl")
    evaluations, run_ids = train_sac((late(path + ".late"), RecordingBackend(path)))

    logged = [c for c in fake_wandb.read_calls(path) if c["call"] == "log"]
    assert len(logged) == len(SEEDS) * len(logged_rows(evaluations, 0))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (LoggingConfig(), (WandbBackend(),)),
        (LoggingConfig(use_wandb=False), ()),
        (
            LoggingConfig(project_name="p", group_name="g", use_tensorboard=True),
            (WandbBackend(project="p", group="g"), TensorBoardBackend(folder=".")),
        ),
        (
            LoggingConfig(use_wandb=False, use_tensorboard=True, folder="runs"),
            (TensorBoardBackend(folder="runs"),),
        ),
        (LoggingConfig(backends=(TensorBoardBackend(),)), (TensorBoardBackend(),)),
    ],
    ids=["default", "off", "both", "tensorboard", "backends"],
)
def test_runs_log_to_the_backends_given_else_to_those_the_flags_name(
    config: LoggingConfig, expected: tuple
) -> None:
    assert config.run_backends == expected


@pytest.mark.parametrize(
    "flag", [{"use_wandb": True}, {"use_wandb": False}, {"use_tensorboard": False}]
)
def test_backends_with_a_flag_set_are_refused(flag: dict) -> None:
    with pytest.raises(ValueError, match=r"backends=\("):
        LoggingConfig(backends=(TensorBoardBackend(),), **flag)


@dataclasses.dataclass(frozen=True)
class Unpicklable(RecordingBackend):
    lock: Any = dataclasses.field(default_factory=threading.Lock)


def test_a_backend_that_cannot_be_pickled_is_refused_before_training(
    tmp_path: Any,
) -> None:
    with pytest.raises(TypeError, match="picklable"):
        train_sac((Unpicklable(str(tmp_path / "calls.jsonl")),))


def test_each_config_gets_an_empty_dict_of_its_own() -> None:
    LoggingConfig().config["seen"] = True
    assert LoggingConfig().config == {}
