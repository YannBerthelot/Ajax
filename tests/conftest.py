"""The two test tiers (CONTRIBUTING.md, Testing).

Everything under tests/probing trains and is the slow tier: its tests are
marked slow here, except the harness's own checks (test_framework.py). A test
elsewhere is fast unless marked slow by hand (the breadth rows of a fast
contract). A fast test trains nothing: the guard reads the gradient updates
every train program a test runs has applied, off the states it returns, and
times the test; the session's summary names the fast tests past the budgets.
It only reports for now; the switch to the two tiers makes it fail them.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import time
from typing import Any

import numpy as np
import pytest

#: Gradient updates a fast test may apply (a split run's 2 + 2).
MAX_UPDATES = int(os.environ.get("AJAX_FAST_MAX_UPDATES", "4"))
#: Seconds a fast test may take, setup to teardown, on CI's runners.
MAX_SECONDS = float(os.environ.get("AJAX_FAST_MAX_SECONDS", "30"))
_START = pytest.StashKey[float]()
#: The guard's state, test infrastructure only: the most updates a state the
#: running test's train programs returned held, and the fast tests past a
#: budget (reported by the process that writes the summary, under xdist too).
_updates = 0
_over: list[tuple[str, dict]] = []


def is_slow_tier(path: str) -> bool:
    return "/tests/probing/" in path and not path.endswith("/test_framework.py")


def updates_of(out: Any) -> int:
    """The most gradient steps a train call's result holds over its seeds
    (``(state, out)`` or a state): its ``n_updates`` or any of its train
    states' ``step`` (SAC, REDQ, ASAC and AVG count only the latter); 0 when
    it holds none or is traced."""
    state = out[0] if isinstance(out, tuple) else out
    counts = [getattr(state, "n_updates", 0)]
    if dataclasses.is_dataclass(state):
        for field in dataclasses.fields(state):
            value: Any = getattr(state, field.name, None)
            if hasattr(value, "step") and hasattr(value, "params"):  # a train state
                counts.append(value.step)
    try:
        return max(int(np.max(np.asarray(count))) for count in counts)
    except Exception:  # a tracer: nothing to read
        return 0


def _counting(fn: Any) -> Any:
    @functools.wraps(fn)
    def counted(*args: Any, **kwargs: Any) -> Any:
        global _updates
        out = fn(*args, **kwargs)
        _updates = max(_updates, updates_of(out))
        return out

    counted.counted = True  # type: ignore[attr-defined]
    return counted


def _counting_loop(train: Any) -> Any:
    """``TrainLoop.train`` builds the train function a test may run itself."""

    @functools.wraps(train)
    def build(*args: Any, **kwargs: Any) -> Any:
        return _counting(train(*args, **kwargs))

    build.counted = True  # type: ignore[attr-defined]
    return build


def pytest_configure(config: pytest.Config) -> None:
    from ajax.agents.base import ActorCritic
    from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
    from ajax.agents.loop import TrainLoop
    from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
    from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask

    if getattr(TrainLoop.train, "counted", False):
        return
    for cls in (ActorCritic, DreamerV3, TDMPC2, TDMPC2MultiTask):
        cls.train = _counting(vars(cls)["train"])  # type: ignore[method-assign]
    TrainLoop.train = _counting_loop(TrainLoop.train)  # type: ignore[method-assign]


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    for item in items:
        if is_slow_tier(item.path.as_posix()):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Any) -> Any:
    global _updates
    _updates = 0
    item.stash[_START] = time.perf_counter()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: Any) -> Any:
    outcome = yield
    if call.when == "teardown" and not item.get_closest_marker("slow"):
        seconds = time.perf_counter() - item.stash[_START]
        measured = {"updates": _updates, "seconds": round(seconds, 1)}
        outcome.get_result().user_properties.append(("fast_tier", measured))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Runs where the summary is written, the xdist controller included: the
    measurement rides on the teardown report."""
    for key, measured in report.user_properties:
        if key != "fast_tier" or not isinstance(measured, dict):
            continue
        if measured["updates"] > MAX_UPDATES or measured["seconds"] > MAX_SECONDS:
            _over.append((report.nodeid, measured))


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if not _over:
        return
    terminalreporter.section("fast tier (report only)")
    terminalreporter.write_line(
        f"{len(_over)} fast tests past {MAX_UPDATES} gradient updates or"
        f" {MAX_SECONDS:.0f} s; a test that trains belongs in tests/probing:"
    )
    for nodeid, measured in sorted(_over, key=lambda o: -o[1]["seconds"]):
        terminalreporter.write_line(
            f"  {measured['updates']:6d} updates {measured['seconds']:6.1f} s  {nodeid}"
        )
