"""Offline training on a known answer: the agent's own updates on a replay
of logged rows, with no environment step (``TrainLoop.offline``).

SignedActionEnv's one-step rows (reward clip(a) s) are logged on a grid of
actions holding the best one, a = s (test_extensions.signed_rows): trained
on them alone, TD3's critic reads Q(s, a) = a s and its actor a = s.
"""

from __future__ import annotations

from typing import Any

import pytest

from . import agents, envs, runs
from . import readouts as R
from .test_extensions import GAMMA, signed_rows
from .verdict import Case, Query, check, params


def offline_readings(name: str) -> Any:
    def build() -> Any:
        return agents.make(name, *envs.package(envs.SignedActionEnv))

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            actions = {f"a({x:g})": R.action(n, x) for x in (-1.0, 1.0)}
            return actions | {"Q(1, 0.5)": R.critic(n, 1.0, 0.5)}

        return R.per_seed(at, R.nets(run.state))

    return runs.readings(build, read, offline_data=signed_rows())


_NONE = {"no update on the logged rows, the networks at initialisation": 0.0}
CASES: dict[str, Case] = {
    "offline-TD3": Case(
        "offline-TD3",
        (
            Query("a(-1)", -1.0, _NONE | {"the logged actions' mean": 0.0}),
            Query("a(1)", 1.0, _NONE | {"the logged actions' mean": 0.0}),
            Query(
                "Q(1, 0.5)",
                0.5,
                _NONE
                | {"bootstrapped past the termination": 0.5 + GAMMA / (1 - GAMMA)},
            ),
        ),
        offline_readings("TD3"),
        250,
        (0.038, 0.075, 0.036),
        note="prior-data-offline-TD3's budget and tolerances (the same updates); ladder"
        " 125-1000 on 1000-1031, 2000-2031: one seed holds a(1) at 0.906 throughout;"
        " cert 32/32",
    ),
}
ANSWER_DIGEST = "519340c823f0"


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
