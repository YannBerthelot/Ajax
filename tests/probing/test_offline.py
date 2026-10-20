"""Offline training on a known answer: the agent's own updates on a replay
of logged rows, with no environment step (``TrainLoop.offline``).

SignedActionEnv's one-step rows (reward clip(a) s) are logged on a grid of
actions holding the best one, a = s (test_extensions.signed_rows): trained
on them alone, TD3's critic reads Q(s, a) = a s and its actor a = s.

TD3+BC (Fujimoto and Gu, 2021) adds the cloning term of ImitationLoss(None,
target="replay", alpha): the actor minimises -lambda Q + E_rows (pi - a)^2,
lambda = alpha / mean|Q|; at each s the rows' actions average 0, so
pi = lambda s / 2 with lambda = alpha / pi: pi^2 = alpha / 2, a(+-1) = +-0.5
at alpha 0.5; with lambda fixed at alpha (no normalisation), +-alpha / 2.

IQL (Kostrikov et al., 2022): V(s) is the expectile of the rows' returns
a s, and the actor's mean the advantage-weighted mean of their actions,
weights min(exp(temperature (a s - V)), 100) (a state-independent std
leaves the weighted likelihood's mean there). At expectile 0.9 and
temperature 20 the clip binds near a = 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ajax.extensions.expert import ImitationLoss
from ajax.networks.networks import predict_value

from . import agents, envs, runs
from . import readouts as R
from .test_extensions import GAMMA, signed_rows
from .verdict import Case, Query, check, params

ALPHA = 0.5


def offline_readings(name: str, extensions: tuple = (), data: Any = None) -> Any:
    def build() -> Any:
        env = envs.package(envs.SignedActionEnv)
        return agents.make(name, *env, extensions=extensions)

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            actions = {f"a({x:g})": R.action(n, x) for x in (-1.0, 1.0)}
            return actions | {"Q(1, 0.5)": R.critic(n, 1.0, 0.5)}

        return R.per_seed(at, R.nets(run.state))

    rows = signed_rows() if data is None else data
    return runs.readings(build, read, offline_data=rows)


GRID = np.linspace(-1.0, 1.0, 21)  # signed_rows' actions at each s


def expectile(x: np.ndarray, tau: float) -> float:
    v = float(x.mean())
    for _ in range(500):  # the fixed point of the asymmetric weighted mean
        w = np.where(x > v, tau, 1.0 - tau)
        v = float((w * x).sum() / w.sum())
    return v


def awr_mean(v: float, temperature: float, clip: bool = True) -> float:
    w = np.exp(temperature * (GRID - v))
    w = np.minimum(w, 100.0) if clip else w
    return float((w * GRID).sum() / w.sum())


TAU, BETA = 0.9, 20.0
V_IQL = expectile(GRID, TAU)


def iql_readings() -> Any:
    def build() -> Any:
        env = envs.package(envs.SignedActionEnv)
        return agents.make("IQL", *env, preset=None, expectile=TAU, temperature=BETA)

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            v = predict_value(n.extra, n.extra.params, R.inp(n, 1.0)).reshape(())
            return {"V(1)": v} | {f"a({x:g})": R.action(n, x) for x in (-1.0, 1.0)}

        return R.per_seed(at, R.nets(run.state, run.state.value_state))

    return runs.readings(build, read, offline_data=signed_rows())


def _awr(x: float) -> Query:
    wrong = {
        "no advantage weights": 0.0,
        "no clip at 100": awr_mean(V_IQL, BETA, False),
    }
    wrong |= {"expectile 0.5 (V the mean)": awr_mean(0.0, BETA)}
    return Query(
        f"a({x:g})", x * awr_mean(V_IQL, BETA), {k: x * w for k, w in wrong.items()}
    )


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
    "td3bc-alpha-TD3": Case(
        "td3bc-alpha-TD3",
        tuple(
            Query(
                f"a({x:g})",
                x * (ALPHA / 2) ** 0.5,
                {
                    "no cloning": x,
                    "cloning alone": 0.0,
                    "lambda = alpha": x * ALPHA / 2,
                },
            )
            for x in (-1.0, 1.0)
        ),
        offline_readings("TD3", (ImitationLoss(None, target="replay", alpha=ALPHA),)),
        1000,
        (0.1, 0.1),
        note="ladder 250-2000 on 1000-1031, 2000-2031: worst 0.062 at 1000 (the"
        " cloning targets' noise); the ceiling; cert 32/32",
    ),
    "iql-expectile-IQL": Case(
        "iql-expectile-IQL",
        (
            Query("V(1)", V_IQL, {"expectile 0.5 (V the mean)": 0.0}),
            _awr(-1.0),
            _awr(1.0),
        ),
        iql_readings(),
        2000,
        (0.09, 0.054, 0.054),
        note="ladder 500-4000 on 1000-1031, 2000-2031: worst 0.044, 0.032, 0.034 at"
        " 2000; the actions' cap, half the gap to the unclipped answer; cert 32/32",
    ),
}
ANSWER_DIGEST = "acc1454b53db"


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
