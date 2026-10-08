"""Calibrate the known-answer probes: training budget and tolerances.

For every agent and probe, train the calibration seeds (1000-1031) in one
program at each budget of a ladder, plus once untrained (budget 0), and
read every query. The budget is the smallest on the ladder at which twice
the largest error over the 32 seeds stays within 0.1 and within half the
gap to the nearest wrong answer, for every query; the tolerance is that
doubled error (at least 0.02). A one-sided query's margin is half
the weakest seed's reading on the truth's side, and must reach 0.25. The
untrained readings show how often a network that learned nothing would
pass.

Run (CPU, about 40 minutes)::

    JAX_PLATFORMS=cpu poetry run python -m tests.probing.calibrate OUT.json [AGENT/probe ...]

then print the ``CALIBRATION`` entries for ``known_answers.py`` (re-applying
the rule to saved readings, later files winning)::

    poetry run python -m tests.probing.calibrate --choose OUT.json [MORE.json ...]

Finally check the choice on fresh seeds (2000-2031)::

    JAX_PLATFORMS=cpu poetry run python -m tests.probing.calibrate --certify
"""

from __future__ import annotations

import json
import sys
import time
from typing import Sequence

import numpy as np

from .known_answers import (
    AGENTS,
    CALIBRATION,
    CALIBRATION_SEEDS,
    CERTIFICATION_SEEDS,
    POOLED_SEEDS,
    Probe,
    read,
    train,
)

LADDER = (1250, 2500, 5000, 10_000, 20_000)
LONG_LADDER = (5000, 10_000, 20_000, 40_000, 80_000)
TOLERANCE_FLOOR = 0.02
TOLERANCE_CEILING = 0.1
MARGIN_FLOOR = 0.25


def ladder(agent_name: str, probe: Probe) -> tuple[int, ...]:
    if agent_name == "PQN":
        return LONG_LADDER
    return LADDER


def choose(probe: Probe, by_budget: dict[int, dict[str, list[float]]]) -> dict:
    for budget in sorted(by_budget):
        readings = {k: np.array(v) for k, v in by_budget[budget].items()}
        tolerances = {}
        for q in probe.queries:
            r = readings[q.name]
            if q.margin:
                # Half the weakest seed's reading on the truth's side.
                margin = 0.5 * float(np.min(r * np.sign(q.truth)))
                if margin < MARGIN_FLOOR:
                    break
                tolerances[q.name] = round(margin, 3)
            else:
                worst = float(np.max(np.abs(r - q.truth)))
                if 2 * worst > min(q.max_tolerance, TOLERANCE_CEILING):
                    break
                tolerances[q.name] = round(
                    min(q.max_tolerance, max(2 * worst, TOLERANCE_FLOOR)), 3
                )
        else:
            return {"budget": budget, "tolerances": tolerances}
    return {"budget": None, "tolerances": None}


def main(
    out: str, only: Sequence[str] = (), seeds: Sequence[int] = CALIBRATION_SEEDS
) -> None:
    """``only`` restricts the run to ``AGENT/probe`` pairs, e.g. SAC/coupling."""
    results = {}
    for agent_name, (agent_cls, probes) in AGENTS.items():
        for probe in probes:
            if only and f"{agent_name}/{probe.name}" not in only:
                continue
            entry: dict = {"by_budget": {}}
            for budget in (0, *ladder(agent_name, probe)):
                start = time.perf_counter()
                state = train(agent_cls, probe.env, seeds, budget)
                readings = read(agent_name, probe, state)
                seconds = time.perf_counter() - start
                entry["by_budget"][budget] = {
                    "seconds": round(seconds, 1),
                    "readings": {
                        k: [round(float(x), 4) for x in v] for k, v in readings.items()
                    },
                }
                print(
                    f"{agent_name} {probe.name} {budget}: {seconds:.0f} s", flush=True
                )
            trained = {b: e["readings"] for b, e in entry["by_budget"].items() if b > 0}
            entry["choice"] = choose(probe, trained)
            tol = entry["choice"]["tolerances"]
            if tol is not None:
                entry["choice"]["untrained_seeds_within"] = untrained_seeds_within(
                    probe, entry["by_budget"][0]["readings"], tol
                )
            results[f"{agent_name}/{probe.name}"] = entry
            print(agent_name, probe.name, entry["choice"], flush=True)
            with open(out, "w") as fh:
                json.dump(results, fh, indent=1)


def untrained_seeds_within(
    probe: Probe, untrained: dict[str, list[float]], tolerances: dict[str, float]
) -> int:
    ok = np.ones(len(CALIBRATION_SEEDS), dtype=bool)
    for q in probe.queries:
        ok &= q.within(np.array(untrained[q.name]), tolerances[q.name])
    return int(ok.sum())


def rechoose(paths: Sequence[str]) -> None:
    """Re-apply the rule to saved readings (later files win) and print the
    ``CALIBRATION`` entries."""
    entries: dict = {}
    for path in paths:
        with open(path) as fh:
            entries.update(json.load(fh))
    probes = {f"{n}/{p.name}": p for n, (_, ps) in AGENTS.items() for p in ps}
    for key, entry in entries.items():
        probe = probes[key]
        by_budget = {int(b): e["readings"] for b, e in entry["by_budget"].items()}
        choice = choose(probe, {b: r for b, r in by_budget.items() if b > 0})
        agent_name = key.split("/")[0]
        if choice["budget"] is None:
            print(f"    # {key}: no budget on the ladder qualifies")
            continue
        within = untrained_seeds_within(probe, by_budget[0], choice["tolerances"])
        print(
            f"    ({agent_name!r}, {probe.name!r}): ({choice['budget']}, "
            f"{choice['tolerances']!r}),  # untrained: {within}/32 within"
        )


def certify(
    seeds: Sequence[int], out: str | None = None, only: Sequence[str] = ()
) -> None:
    """Train ``seeds`` at each calibrated budget and count the seeds within
    tolerance: the per-seed pass rate on seeds the calibration never saw.
    ``out`` keeps the readings (for ``--pool``)."""
    saved = {}
    for agent_name, (agent_cls, probes) in AGENTS.items():
        for probe in probes:
            if only and f"{agent_name}/{probe.name}" not in only:
                continue
            budget, tolerances = CALIBRATION[agent_name, probe.name]
            readings = read(
                agent_name, probe, train(agent_cls, probe.env, seeds, budget)
            )
            ok = np.ones(len(seeds), dtype=bool)
            for q in probe.queries:
                ok &= q.within(readings[q.name], tolerances[q.name])
            print(
                f"{agent_name}/{probe.name}: {int(ok.sum())}/{len(ok)} fresh seeds "
                f"within tolerance at {budget} steps",
                flush=True,
            )
            saved[f"{agent_name}/{probe.name}"] = {
                "budget": budget,
                "readings": {k: [float(x) for x in v] for k, v in readings.items()},
            }
    if out:
        with open(out, "w") as fh:
            json.dump(saved, fh, indent=1)


def choose_pooled(first: str, second: str) -> None:
    """Apply the rule to two seed blocks' ladders pooled (budgets both
    have), for the pairs in both files; print the entries."""
    with open(first) as fh:
        a = json.load(fh)
    with open(second) as fh:
        b = json.load(fh)
    probes = {f"{n}/{p.name}": p for n, (_, ps) in AGENTS.items() for p in ps}
    for key in sorted(set(a) & set(b)):
        probe = probes[key]
        budgets = sorted(set(a[key]["by_budget"]) & set(b[key]["by_budget"]), key=int)
        pooled = {
            int(budget): {
                q: a[key]["by_budget"][budget]["readings"][q]
                + b[key]["by_budget"][budget]["readings"][q]
                for q in a[key]["by_budget"][budget]["readings"]
            }
            for budget in budgets
            if int(budget) > 0
        }
        choice = choose(probe, pooled)
        agent_name = key.split("/")[0]
        print(
            f"    ({agent_name!r}, {probe.name!r}): ({choice['budget']}, "
            f"{choice['tolerances']!r}),"
        )


def pool(extra: str, paths: Sequence[str]) -> None:
    """Re-apply the rule at each calibrated budget to the calibration
    readings plus the extra seeds saved by ``certify``; print the entries."""
    entries: dict = {}
    for path in paths:
        with open(path) as fh:
            entries.update(json.load(fh))
    with open(extra) as fh:
        more = json.load(fh)
    probes = {f"{n}/{p.name}": p for n, (_, ps) in AGENTS.items() for p in ps}
    for key, probe in probes.items():
        agent_name = key.split("/")[0]
        budget = more[key]["budget"]
        first = entries[key]["by_budget"][str(budget)]["readings"]
        pooled = {q: first[q] + more[key]["readings"][q] for q in first}
        choice = choose(probe, {budget: pooled})
        print(
            f"    ({agent_name!r}, {probe.name!r}): ({budget}, "
            f"{choice['tolerances']!r}),"
        )


if __name__ == "__main__":
    if sys.argv[1] == "--certify":
        certify(CERTIFICATION_SEEDS, sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--certify-block":
        certify(POOLED_SEEDS, sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--pool":
        pool(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--ladder-block":
        main(sys.argv[2], sys.argv[3:], seeds=POOLED_SEEDS)
    elif sys.argv[1] == "--choose-pooled":
        choose_pooled(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--choose":
        rechoose(sys.argv[2:])
    else:
        main(sys.argv[1], sys.argv[2:])
