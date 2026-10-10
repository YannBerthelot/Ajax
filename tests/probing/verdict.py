"""Queries, cases and the two-stage verdict every judged probe shares.

A query states a reading's right answer and the wrong answers named faults
produce; a case is one judged cell: its readings, budget, and calibrated
tolerances or the live defect it pins as a strict xfail. A check trains
seeds 0-7 in one program: 7 or more within tolerance pass, 4 or fewer fail,
otherwise seeds 8-15 are added and 13 of 16 pass. Preconditions raise
RuntimeError, never AssertionError, so a strict xfail cannot absorb them::

    poetry run python -m tests.probing.verdict {ladder|choose|certify|answers} MODULE ...
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

STAGE_1, STAGE_2 = tuple(range(8)), tuple(range(8, 16))
CEILING, FLOOR, MARGIN_FLOOR = 0.1, 0.02, 0.25
Readings = Mapping[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class Query:
    """A margin query only asks the reading to reach its tolerance on the
    truth's side of zero (a sign, or a gap between two actions' values)."""

    name: str
    truth: float
    wrong: Mapping[str, float]
    margin: bool = False

    @property
    def max_tolerance(self) -> float:
        return 0.5 * min(abs(self.truth - w) for w in self.wrong.values())

    def within(self, r: np.ndarray, tol: float) -> np.ndarray:
        if self.margin:
            return r * np.sign(self.truth) >= tol
        return np.abs(r - self.truth) <= tol

    def describe(self, r: np.ndarray, tol: float) -> str:
        want = (
            f"{'>=' if self.truth > 0 else '<='} {np.sign(self.truth) * tol:.3f}"
            if self.margin
            else f"{self.truth:.3f} +- {tol:.3f}"
        )
        line = (
            f"  {self.name}: wants {want}; readings {np.array2string(r, precision=3)}"
        )
        median = float(np.median(r))
        if not self.within(np.array([median]), tol)[0]:
            near = min(self.wrong, key=lambda k: abs(self.wrong[k] - median))
            line += f"; median {median:.3f} is nearest the wrong answer '{near}' ({self.wrong[near]:.3f})"
        return line


@dataclasses.dataclass(frozen=True)
class Case:
    """``tol`` aligns with ``queries``; without it the tolerance is the
    ceiling, capped by half the gap for two-sided queries. ``note`` records
    the calibration."""

    id: str
    queries: tuple[Query, ...]
    readings: Callable[[tuple[int, ...], int], Readings]
    budget: int
    tol: tuple[float, ...] = ()
    defect: str = ""
    raises: Any = AssertionError
    slow: bool = False
    ceiling: float = CEILING
    note: str = ""

    @property
    def tolerances(self) -> dict[str, float]:
        if self.tol:
            return {q.name: t for q, t in zip(self.queries, self.tol, strict=True)}
        cap = {
            q.name: self.ceiling if q.margin else q.max_tolerance for q in self.queries
        }
        return {k: min(self.ceiling, v) for k, v in cap.items()}

    def run(self, seeds: Sequence[int]) -> Readings:
        out = self.readings(tuple(seeds), self.budget)
        return {q.name: np.asarray(out[q.name]) for q in self.queries}


def xfail(defect: str, raises: Any = AssertionError) -> pytest.MarkDecorator:
    """A strict xfail whose reason is the defect's one explanation."""
    return pytest.mark.xfail(
        strict=True, raises=raises, reason=f"LIVE defect: {defect}"
    )


def xparam(
    value: Any, defect: str = "", raises: Any = AssertionError, id: str | None = None
) -> Any:
    marks = [xfail(defect, raises)] if defect else []
    return pytest.param(value, id=id or str(value), marks=marks)


def params(
    cases: Mapping[str, Case], defect: Callable[[Case], str] | None = None
) -> list:
    """One pytest param per case, ``slow`` marked, a strict xfail on its
    defect; ``defect`` replaces the cases' own (an exact test of a case)."""
    out = []
    for c in cases.values():
        why = c.defect if defect is None else defect(c)
        marks = [xfail(why, c.raises)] if why else []
        out.append(pytest.param(c, id=c.id, marks=marks + [pytest.mark.slow] * c.slow))
    return out


def _ok(queries: Sequence[Query], r: Readings, tol: Mapping[str, float]) -> np.ndarray:
    return np.all([q.within(np.asarray(r[q.name]), tol[q.name]) for q in queries], 0)


def judge(
    queries: Sequence[Query],
    run: Callable[[Sequence[int]], Readings],
    tol: Mapping[str, float],
    label: str,
) -> tuple[bool, str]:
    """The two-stage rule; returns (passed, report)."""
    readings = {k: np.asarray(v) for k, v in run(STAGE_1).items()}
    within, seeds = int(_ok(queries, readings, tol).sum()), len(STAGE_1)
    if 4 < within < 7:
        more = run(STAGE_2)
        within += int(_ok(queries, more, tol).sum())
        seeds += len(STAGE_2)
        readings = {k: np.concatenate([readings[k], more[k]]) for k in readings}
    passed = within >= (13 if seeds > len(STAGE_1) else 7)
    lines = [q.describe(readings[q.name], tol[q.name]) for q in queries]
    return passed, "\n".join([f"{label}: {within} of {seeds} seeds within", *lines])


def check(case: Case) -> None:
    label = f"{case.id} (budget {case.budget} steps)"
    passed, report = judge(case.queries, case.run, case.tolerances, label)
    assert passed, report


def calibration_errors(case: Case) -> list[str]:
    """Calibrated tolerances lie in (0, min(ceiling, half the gap)]; margins
    in (0, |truth|)."""
    errors = (
        [] if len(case.tol) in (0, len(case.queries)) else [f"{case.id}: tol length"]
    )
    for q, t in zip(case.queries, case.tol):
        cap = min(q.max_tolerance, case.ceiling) + 1e-9
        if not (0 < t < abs(q.truth) if q.margin else 0 < t <= cap):
            errors.append(f"{case.id} {q.name}: {t} outside its gap")
    return errors


# ---------------------------------------------------------------------------
# Calibration: the ladder on seeds 1000-1031 and 2000-2031 pooled, the
# smallest budget at which twice the worst error of every two-sided query
# is within min(ceiling, half the gap) (tolerance: that, at least FLOOR) and
# half the weakest seed's margin reaches MARGIN_FLOOR; then certify on
# 3000-3031.
# ---------------------------------------------------------------------------


def block(first: int) -> tuple[int, ...]:
    return tuple(range(first, first + 32))


def ladder(case: Case, first: int, budgets: Sequence[int]) -> dict[int, dict]:
    """Readings of a 32-seed block untrained and at every budget."""
    out = {b: case.readings(block(first), b) for b in (0, *budgets)}
    return {
        b: {k: np.round(v, 5).tolist() for k, v in r.items()} for b, r in out.items()
    }


def choose(
    case: Case, by_budget: Mapping[int, Mapping[str, Sequence[float]]]
) -> tuple[int | None, tuple[float, ...]]:
    for budget in sorted(b for b in by_budget if b > 0):
        tol: list[float] = []
        for q in case.queries:
            r = np.asarray(by_budget[budget][q.name])
            if q.margin:
                t = 0.5 * float(np.min(r * np.sign(q.truth)))
                if t < MARGIN_FLOOR:
                    break
            else:
                t = 2 * float(np.max(np.abs(r - q.truth)))
                if t > min(q.max_tolerance, case.ceiling):
                    break
                t = min(q.max_tolerance, max(t, FLOOR))
            tol.append(round(t, 3) if q.margin else min(round(t, 3), q.max_tolerance))
        else:
            return budget, tuple(tol)
    return None, ()


def certify(case: Case, first: int = 3000) -> int:
    """Seeds of the block within the case's tolerances at its budget."""
    r = case.readings(block(first), case.budget)
    return int(_ok(case.queries, r, case.tolerances).sum())


def answers(cases: Mapping[str, Case]) -> dict[str, dict[str, list]]:
    """Every query's truth, wrong answers, tolerance and kind, to 1e-6."""
    out: dict[str, dict[str, list]] = {}
    for c in cases.values():
        for q in c.queries:
            wrong = {k: round(v, 6) for k, v in q.wrong.items()}
            tol = round(c.tolerances[q.name], 6)
            out.setdefault(c.id, {})[q.name] = [round(q.truth, 6), wrong, tol, q.margin]
    return out


def digest(cases: Mapping[str, Case]) -> str:
    text = json.dumps(answers(cases), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


if __name__ == "__main__":
    action, module, *rest = sys.argv[1:]
    cases = importlib.import_module(f"tests.probing.{module}").CASES
    if action == "answers":
        print(json.dumps(answers(cases), indent=1, sort_keys=True))
    elif action == "ladder":  # CASE FIRST_SEED B1,B2,... OUT.json
        budgets = [int(b) for b in rest[2].split(",")]
        with open(rest[3], "w") as fh:
            json.dump(ladder(cases[rest[0]], int(rest[1]), budgets), fh)
    elif action == "choose":  # CASE OUT.json [MORE.json ...]: blocks pooled
        pooled: dict[int, dict[str, list]] = {}
        for path in rest[1:]:
            with open(path) as fh:
                for b, r in json.load(fh).items():
                    for k, v in r.items():
                        pooled.setdefault(int(b), {}).setdefault(k, []).extend(v)
        print(choose(cases[rest[0]], pooled))
    else:  # certify CASE [FIRST_SEED]
        case = cases[rest[0]]
        print(f"{certify(case, *map(int, rest[1:]))}/32 within at {case.budget} steps")
