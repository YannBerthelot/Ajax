"""A stand-in for the wandb module that records every call it gets.

Each call is one JSON line in the file the AJAX_FAKE_WANDB_LOG environment
variable names, with the process it came from: the caller creates a run
before training, and the spawned logging worker sends it the steps. The
worker finds this module through a one-line wandb.py that the test puts
first on sys.path (a spawned process starts with its parent's sys.path).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

__all__ = ["finish", "init", "log"]

LOG_ENV = "AJAX_FAKE_WANDB_LOG"

# The run that module-level log and finish calls go to: the last one init made.
_current: dict[str, Optional[str]] = {"run_id": None}


def _plain(value: Any) -> Any:
    """numpy arrays and scalars as numbers or lists; anything else as text."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return str(value)


def as_recorded(value: Any) -> Any:
    """value as a recorded call holds it, after its trip through JSON."""
    return json.loads(json.dumps(value, default=_plain))


def _record(run_id: Optional[str], call: str, arguments: dict) -> None:
    line = json.dumps(
        {"pid": os.getpid(), "run_id": run_id, "call": call, "arguments": arguments},
        default=_plain,
    )
    with open(os.environ[LOG_ENV], "a") as handle:
        handle.write(line + "\n")


def read_calls(path: str) -> list[dict]:
    """The recorded calls, in the order they were made."""
    with open(path) as handle:
        return [json.loads(line) for line in handle]


class Run:
    """What init returns: the run's own log and finish."""

    def __init__(self, run_id: Optional[str]) -> None:
        self.run_id = run_id

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        _record(self.run_id, "run.log", {"metrics": metrics, "step": step})

    def finish(self) -> None:
        _record(self.run_id, "run.finish", {})


def init(**kwargs: Any) -> Run:
    _current["run_id"] = kwargs.get("id")
    _record(kwargs.get("id"), "init", kwargs)
    return Run(kwargs.get("id"))


def log(metrics: dict, step: Optional[int] = None) -> None:
    _record(_current["run_id"], "log", {"metrics": metrics, "step": step})


def finish() -> None:
    _record(_current["run_id"], "finish", {})
