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

CQL (Kumar et al., 2020) on SAC: the critic's term alpha (logsumexp_a Q(s,
a) - E_data Q(s, a)) balances the data's squared error where 2 mu (Q - r) +
alpha (rho - mu) = 0, with mu = 1/2 the data's density on [-1, 1] and rho
the critic's softmax, rho = exp Q / int exp Q: Q = a s - alpha / 2 (2 rho -
1), solved on a fine grid (cql_fixed_point). The rows' actions are cells'
centres (cell_rows): a grid holding the end points +-1 gives them half a cell
each, twice the data's density there. The cell reads Q inside the data.

ReBRAC's TargetBC (Tarasov et al., 2023) on TD3, two-step episodes paying 0
(chain_rows): the data's next action at s1 is 0, which the cloning term
pins the actor to; the target scores TD3's smoothed next action, 0 plus
noise clipped to +-0.5, so Q(s0, a) = -gamma beta E[clip(0.2 e, +-0.5)^2].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.offline import ConservativeQ, TargetBC
from ajax.networks.networks import predict_value

from . import agents, envs, oracles, runs
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
            return {**actions, "Q(1, 0.5)": R.critic(n, 1.0, 0.5)}

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
            out = {f"a({x:g})": R.action(n, x) for x in (-1.0, 1.0)}
            return {"V(1)": v, **out}

        return R.per_seed(at, R.nets(run.state, run.state.value_state))

    return runs.readings(build, read, offline_data=signed_rows())


def _awr(x: float) -> Query:
    wrong = {
        "no advantage weights": 0.0,
        "no clip at 100": awr_mean(V_IQL, BETA, False),
    }
    wrong.update({"expectile 0.5 (V the mean)": awr_mean(0.0, BETA)})
    return Query(
        f"a({x:g})", x * awr_mean(V_IQL, BETA), {k: x * w for k, w in wrong.items()}
    )


def cql_fixed_point(alpha: float, at: tuple[float, ...]) -> list[float]:
    """Q(1, a) at the actions ``at`` where the data and CQL's term balance."""
    a = np.linspace(-1.0, 1.0, 2001)
    q = a.copy()
    for _ in range(2000):  # a damped fixed-point iteration
        rho = np.exp(q) / np.trapezoid(np.exp(q), a)
        q = (q + a - alpha / 2 * (2 * rho - 1)) / 2
    return [float(np.interp(x, a, q)) for x in at]


CQL_ALPHA, CQL_AT = 1.0, (-0.85, 0.85)


def cell_rows() -> TransitionDataset:
    """One-step rows at s = +-1 (reward a s) whose actions are the centres of
    20 equal cells of [-1, 1]."""
    a = np.tile(np.linspace(-0.95, 0.95, 20), 2)[:, None]
    s = np.repeat([-1.0, 1.0], 20)[:, None]
    ends = np.ones(len(a), bool)
    return TransitionDataset(s, a, (a * s)[:, 0], ends, ~ends, s)


def cql_readings() -> Any:
    def build() -> Any:
        env = envs.package(envs.SignedActionEnv)
        ext = ConservativeQ(alpha=CQL_ALPHA, n_actions=100)
        return agents.make("SAC", *env, extensions=(ext,))

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            return {f"Q(1, {x:g})": R.critic(n, 1.0, x) for x in CQL_AT}

        return R.per_seed(at, R.nets(run.state))

    return runs.readings(build, read, offline_data=cell_rows())


def chain_rows() -> TransitionDataset:
    """Two-step episodes from s0 (obs [1, 0]) to s1 ([0, 1]), paying 0: s0's
    actions on a grid, s1's 0."""
    a0 = np.linspace(-1.0, 1.0, 21)
    obs = np.tile(np.eye(2), (21, 1))
    action = np.stack([a0, np.zeros(21)], 1).reshape(-1, 1)
    terminated = np.tile([False, True], 21)
    next_obs = np.tile(np.eye(2)[[1, 1]], (21, 1))
    return TransitionDataset(
        obs, action, np.zeros(42), terminated, np.zeros(42, bool), next_obs
    )


BETA = 20.0
TARGET_BC = -agents.GAMMA * BETA * oracles.clipped_square_mean(0.2, 0.5)


def rebrac_readings() -> Any:
    def build() -> Any:
        env, params = chain().make()
        exts = (ImitationLoss(None, coef=10.0, target="replay"), TargetBC(beta=BETA))
        return agents.make("TD3", env, params, extensions=exts)

    def chain() -> envs.Spec:
        def transition(st: envs.State, a: Any, key: Any) -> tuple:
            return st.replace(s=st.s + 1), 0.0, st.s == 1

        obs = lambda st: envs.one_hot(st.s, 2)  # noqa: E731
        return envs.Spec("rebrac_chain", transition, obs, obs_dim=2, limit=2)

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            return {"Q(s0, 0)": R.critic(n, envs.one_hot(0, 2), 0.0)}

        return R.per_seed(at, R.nets(run.state))

    return runs.readings(build, read, offline_data=chain_rows())


_NONE = {"no update on the logged rows, the networks at initialisation": 0.0}
CASES: dict[str, Case] = {
    "offline-TD3": Case(
        "offline-TD3",
        (
            Query("a(-1)", -1.0, {**_NONE, "the logged actions' mean": 0.0}),
            Query("a(1)", 1.0, {**_NONE, "the logged actions' mean": 0.0}),
            Query(
                "Q(1, 0.5)",
                0.5,
                {
                    **_NONE,
                    "bootstrapped past the termination": 0.5 + GAMMA / (1 - GAMMA),
                },
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
    "cql-fixed-point-SAC": Case(
        "cql-fixed-point-SAC",
        tuple(
            Query(f"Q(1, {x:g})", q, {"no penalty": x})
            for x, q in zip(CQL_AT, cql_fixed_point(CQL_ALPHA, CQL_AT), strict=True)
        ),
        cql_readings(),
        2000,
        (0.1, 0.1),
        note="ladder 1000-4000 on 1000-1031, 2000-2031: medians at the fixed point,"
        " worst 0.059, 0.096 at 2000; the ceiling; cert 32/32",
    ),
    "rebrac-target-bc-TD3": Case(
        "rebrac-target-bc-TD3",
        (
            Query(
                "Q(s0, 0)",
                TARGET_BC,
                {
                    "no penalty": 0.0,
                    "the penalty undiscounted": TARGET_BC / agents.GAMMA,
                },
            ),
        ),
        rebrac_readings(),
        2000,
        (0.1,),
        note="ladder 500-4000 on 1000-1031, 2000-2031: medians at the answer, worst"
        " 0.104, 0.071 at 2000; the ceiling; cert 32/32",
    ),
}
ANSWER_DIGEST = "86d362d0cad7"


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
