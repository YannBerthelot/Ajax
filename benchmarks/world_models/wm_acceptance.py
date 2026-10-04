"""Acceptance criteria of the world-model validation (M9), fixed before any run.

``docs/world_models/VALIDATION.md`` explains them; this module is their
only definition (``paper_report.py`` and ``multitask_validation.py`` apply
them; the tests pin them). They were set before any GPU run and are not to
be tuned to its results: a change is a new protocol, recorded as such.

**Paper-protocol runs** (DreamerV3, TD-MPC2 single-task). Scores are
averaged over a window of env steps (:data:`WINDOWS`), per seed: the mean
of the curve's points with ``lo < x <= hi`` (Ajax's DreamerV3 points are
weighted by the training episodes each holds, so the window mean is the
mean return of the training episodes that ended in it). A window instead
of a single point absorbs the curves' point-to-point noise (one 10-episode
evaluation, or the training episodes of 20K env steps) and the unknown
binning convention of DreamerV3's bundled curves. At each window:

* **per task**: the Ajax seed mean must reach the lowest reference seed
  minus :data:`TOLERANCE` (``ajax_mean >= min(ref_seeds) - 50``): within
  the reference's own seed range or above it, with a margin for the
  playground ports, whose physics and rewards differ from dm_control's;
* **aggregate**: over the protocol's tasks, the median and the mean of the
  Ajax seed means must each reach the reference's (median and mean of its
  seed means) minus :data:`TOLERANCE`, and at least :data:`TASK_SHARE` of
  the tasks must pass (15 of DreamerV3's 18, 18 of TD-MPC2's 22): a few
  tasks may differ in playground without failing the agent, a systematic
  shortfall may not.

A task's window is judged only when the run follows the protocol (its
specification is the registry's, ``paper_report.py`` checks it), reached
the window's end and has at least :data:`MIN_SEEDS` seeds; otherwise the
window is incomplete (as is its aggregate when a protocol task is
missing). The verdict is ``FAIL`` if
any judged aggregate fails, else ``INCOMPLETE`` if any window is
incomplete, else ``PASS``.

**Multi-task pipeline**: per task the offline model's mean return must
reach :data:`MT_FRACTION` of its source agents' final mean return; tasks
whose source agents stay below :data:`MT_SOURCE_FLOOR` (did not learn) are
reported but not judged; at least :data:`TASK_SHARE` of the tasks must be
judged (else ``INCOMPLETE``: too few sources learned to test anything),
and ``PASS`` when at least :data:`TASK_SHARE` of the judged tasks pass.
This validates the multi-task mechanisms on Ajax's own
data, not the paper's multi-task numbers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

#: Return units (DMC returns lie in [0, 1000]): 5% of the maximum.
TOLERANCE = 50.0
#: Share of the tasks that must pass (per window; multi-task: of the judged
#: tasks, and the share of the tasks that must be judged).
TASK_SHARE = 0.8
#: Seeds a run needs for a verdict (the TD-MPC2 reference has 3).
MIN_SEEDS = 3

#: Offline multi-task return as a fraction of the source agents' (per task).
MT_FRACTION = 0.5
#: Source final returns below this are "did not learn": not judged.
MT_SOURCE_FLOOR = 100.0

PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"


@dataclasses.dataclass(frozen=True)
class Window:
    """``(hi - width, hi]`` env steps; ``end=None`` puts ``hi`` at the task's
    budget (the reference's last point)."""

    label: str
    width: int
    end: Optional[int] = None

    def bounds(self, budget: int) -> tuple[int, int]:
        hi = budget if self.end is None else self.end
        return hi - self.width, hi


#: The windows of each agent (env steps). DreamerV3 (500K env steps): around
#: the 250K midpoint and the last 100K; TD-MPC2: up to 1M (the
#: paper's DMC headline budget, tdmpc2_spec 4.27) and the last 1M.
WINDOWS: dict[str, tuple[Window, ...]] = {
    "DreamerV3": (Window("250K", 100_000, 300_000), Window("500K", 100_000, 500_000)),
    "TDMPC2": (Window("1M", 500_000, 1_000_000), Window("final", 1_000_000)),
}


def window_mean(
    x: Sequence[float],
    y: Sequence[float],
    lo: float,
    hi: float,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """The (weighted) mean of the ``y`` with ``lo < x <= hi``; NaN if none
    (or if their weights sum to 0)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    w = np.ones_like(y) if weights is None else np.asarray(weights, np.float64)
    inside = (x > lo) & (x <= hi) & (w > 0)
    if not inside.any():
        return float("nan")
    return float(np.sum(w[inside] * y[inside]) / np.sum(w[inside]))


@dataclasses.dataclass(frozen=True)
class TaskResult:
    """One task at one window."""

    task: str
    window: str
    ajax: np.ndarray  # per-seed window means (empty: no run)
    reference: np.ndarray  # per-seed window means
    # "pass", "fail", "off protocol", "no run", "not reached", "too few seeds"
    status: str

    @property
    def ajax_mean(self) -> float:
        return float(np.mean(self.ajax)) if self.ajax.size else float("nan")

    @property
    def reference_mean(self) -> float:
        return float(np.mean(self.reference))

    @property
    def bar(self) -> float:
        """The per-task bar: the lowest reference seed minus the tolerance."""
        return float(np.min(self.reference)) - TOLERANCE


def judge_task(
    task: str,
    window: str,
    ajax: Sequence[float],
    reference: Sequence[float],
    *,
    reached: bool = True,
    on_protocol: bool = True,
) -> TaskResult:
    """Judge one task at one window from per-seed window means.

    ``on_protocol`` False (the run's specification is not the protocol's):
    ``"off protocol"``; ``ajax`` empty: ``"no run"``; ``reached`` False or a
    NaN seed (no point in the window): ``"not reached"``; fewer than
    :data:`MIN_SEEDS` seeds: ``"too few seeds"``; else ``"pass"`` iff the
    seed mean reaches ``min(reference) - TOLERANCE``.
    """
    ajax_arr = np.asarray(ajax, np.float64)
    ref = np.asarray(reference, np.float64)
    if ref.size == 0 or np.isnan(ref).any():
        raise ValueError(f"{task} @ {window}: the reference has no point there")
    if not on_protocol:
        status = "off protocol"
    elif ajax_arr.size == 0:
        status = "no run"
    elif not reached or np.isnan(ajax_arr).any():
        status = "not reached"
    elif ajax_arr.size < MIN_SEEDS:
        status = "too few seeds"
    else:
        status = "pass" if ajax_arr.mean() >= ref.min() - TOLERANCE else "fail"
    return TaskResult(task, window, ajax_arr, ref, status)


@dataclasses.dataclass(frozen=True)
class WindowVerdict:
    """The aggregate of one window over the protocol's tasks."""

    window: str
    tasks: tuple[TaskResult, ...]
    status: str  # PASS, FAIL, INCOMPLETE
    ajax_median: float
    ajax_mean: float
    reference_median: float
    reference_mean: float
    share_passed: float


def judge_window(window: str, results: Sequence[TaskResult]) -> WindowVerdict:
    """The aggregate criterion over every protocol task (module docstring).

    ``INCOMPLETE`` unless every task was judged (pass or fail); then
    ``PASS`` iff the Ajax median and mean of the task means reach the
    reference's minus :data:`TOLERANCE` and the share of passing tasks is
    at least :data:`TASK_SHARE`. The aggregates are reported over the
    judged tasks in either case.
    """
    if not results:
        raise ValueError("no task to judge")
    judged = [r for r in results if r.status in ("pass", "fail")]
    ours = np.asarray([r.ajax_mean for r in judged])
    theirs = np.asarray([r.reference_mean for r in judged])
    passed = sum(r.status == "pass" for r in judged)
    share = passed / len(judged) if judged else float("nan")
    stats = (
        (float(np.median(ours)), float(np.mean(ours)))
        if judged
        else (float("nan"),) * 2
    )
    ref_stats = (
        (float(np.median(theirs)), float(np.mean(theirs)))
        if judged
        else (float("nan"),) * 2
    )
    if len(judged) < len(results):
        status = INCOMPLETE
    else:
        ok = (
            stats[0] >= ref_stats[0] - TOLERANCE
            and stats[1] >= ref_stats[1] - TOLERANCE
            and share >= TASK_SHARE
        )
        status = PASS if ok else FAIL
    return WindowVerdict(
        window, tuple(results), status, *stats, *ref_stats, share_passed=share
    )


def overall(statuses: Sequence[str]) -> str:
    """``FAIL`` if any is, else ``INCOMPLETE`` if any is, else ``PASS``."""
    if not statuses:
        return INCOMPLETE
    if FAIL in statuses:
        return FAIL
    if INCOMPLETE in statuses:
        return INCOMPLETE
    return PASS


# ---------------------------------------------------------------------------
# Multi-task pipeline
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MultiTaskResult:
    task: str
    source: float  # mean over the source seeds of their final return
    offline: float  # mean over the offline seeds
    status: str  # "pass", "fail", "not judged", "missing"

    @property
    def ratio(self) -> float:
        return self.offline / self.source if self.source > 0 else float("nan")


def judge_multitask(
    source: Mapping[str, Sequence[float]], offline: Mapping[str, Sequence[float]]
) -> tuple[list[MultiTaskResult], str]:
    """Per-task results and the verdict of the multi-task pipeline.

    ``source[task]``: the source agents' final returns (per seed);
    ``offline[task]``: the offline model's returns (per seed). A task
    missing on either side (or with a NaN) is ``"missing"`` and makes the
    verdict ``INCOMPLETE``; a source mean below :data:`MT_SOURCE_FLOOR` is
    ``"not judged"``; otherwise ``"pass"`` iff ``offline >= MT_FRACTION *
    source``. ``INCOMPLETE`` when fewer than :data:`TASK_SHARE` of the tasks
    are judged (too few sources learned for the comparison to test the
    mechanisms); else ``PASS`` iff at least :data:`TASK_SHARE` of the judged
    tasks pass.
    """
    results = []
    for task in source:
        src = np.asarray(source[task], np.float64)
        off = np.asarray(offline.get(task, []), np.float64)
        if src.size == 0 or off.size == 0 or np.isnan(src).any() or np.isnan(off).any():
            results.append(MultiTaskResult(task, float("nan"), float("nan"), "missing"))
            continue
        s, o = float(src.mean()), float(off.mean())
        if s < MT_SOURCE_FLOOR:
            status = "not judged"
        else:
            status = "pass" if o >= MT_FRACTION * s else "fail"
        results.append(MultiTaskResult(task, s, o, status))
    judged = [r for r in results if r.status in ("pass", "fail")]
    if (
        any(r.status == "missing" for r in results)
        or not judged
        or len(judged) < TASK_SHARE * len(results)
    ):
        return results, INCOMPLETE
    share = sum(r.status == "pass" for r in judged) / len(judged)
    return results, PASS if share >= TASK_SHARE else FAIL
