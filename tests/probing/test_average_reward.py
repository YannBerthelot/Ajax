"""RVI Q-learning on reset cycles: RVI-SAC (Hisaki & Ono, ICML 2024), and the two modules it
brought to every agent, ValueCentering and ResetCost, on SAC.

Position i of a cycle shows obs[i] and pays r[i] whatever the action: R -> A -> B (obs 1, 0, -1;
r 3, 1, 0), terminating after B, whose replay row's next observation is then the reset one, R. The
temperature is fixed at 1e-3, so alpha log pi is negligible next to the rewards; the offsets and
the reset rate follow at 0.02 to 0.05 (their default: 0.005), so that they settle within the
budgets. The
reset cost keeps its step, 3e-4: it rises about that much an update for as long as the
termination rate, which no action changes here, exceeds eps, and a slow rise keeps the offset's
lag behind (4 - c) / 3 small.

q1-RVISAC-reset: the target goes on through the termination, charged c, so the values solve
Q(R) = 3 - xi + Q(A), Q(A) = 1 - xi + Q(B), Q(B) = -c - xi + Q(R): xi = (4 - c) / 3 (the RVI fixed
point f(Q) = rho), Q(R) - Q(A) = 3 - xi, Q(B) - Q(R) = -c - xi. Cut at the termination instead, xi
is the mean of the next values Q(A), Q(B), Q(R) = 1 - c - 2 xi, -c - xi, 4 - c - 3 xi: (5 - 3c) / 9,
7/9 below (the rewards put R's first so that the gap, (2 r_R + r_A) / 9, is wide). The reset
critic's rate reads the termination rate, 1/3 (its target without the indicator: 0); c rises while
that rate exceeds eps 1e-3 (stepped the wrong way it stays at 0).

q1-centering-SAC: SAC at gamma 0.62 with ValueCentering's batch-mean rule, the cycle wrapping from
B to R: at the fixed point Q = (I - gamma P)^-1 (r - xi) and xi = mean Q, so xi = rho / (2 - gamma),
rho = 4/3; uncentred, rho / (1 - gamma); shifted by +xi, -rho / gamma; the mean reward (simple
centering), rho.

q1-ResetCost-SAC: SAC at gamma 0.62 with ResetCost on the terminating cycle: the target goes on
through the termination, Q(B) = -c + gamma Q(R); cut there, Q(B) = -c and Q(R) = 3 + gamma (1 -
gamma c), which with c near 0.7 leaves the residual Q(B) - (gamma Q(R) - c) about -2.08. The reset
rate reads 1/3; c rises.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax.numpy as jnp
import pytest

from ajax.agents.centering import RVIBatchMean
from ajax.agents.reset_cost import ResetCostSettings
from ajax.extensions.centering import ValueCentering
from ajax.extensions.reset_cost import ResetCost

from . import agents, envs, runs
from . import readouts as R
from .verdict import Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "da57731143bf"
GAMMA, RHO = agents.GAMMA, 4.0 / 3.0
FIXED_ALPHA: dict[str, Any] = {"fixed_alpha": True, "alpha_init": 1e-3}
RESET = ResetCostSettings(rate_tau=0.05)
GROWS = {"offset not subtracted: grows without bound": float("inf")}
STAYS = {"stepped the wrong way: stays at 0": 0.0}


def cycle(terminal: bool) -> envs.Spec:
    obs, r = (1.0, 0.0, -1.0), (3.0, 1.0, 0.0)

    def transition(st: envs.State, a: Any, key: Any) -> tuple:
        nxt = jnp.minimum(st.s + 1, 2) if terminal else (st.s + 1) % 3
        return st.replace(s=nxt), jnp.asarray(r)[st.s], terminal & (st.s == 2)

    def show(st: envs.State) -> Any:
        return jnp.asarray(obs, jnp.float32)[st.s][None]

    return envs.Spec("reset-cycle", transition, show, obs_box=(-1.0, 1.0))


def values(agent: str) -> Callable[[R.Nets], dict]:
    """Q at R, A and B, each at the policy's mean action, over the ensemble."""

    def read(n: R.Nets) -> dict:
        at = {"R": 1.0, "A": 0.0, "B": -1.0}
        return {f"Q({k})": R.value(agent, n, x) for k, x in at.items()}

    return read


def reset_readings(state: Any) -> dict:
    return {"c": state.cost.params["cost"], "rate": state.rate}


def rvisac() -> Callable:
    def read(run: runs.Run) -> dict:
        q = R.per_seed(values("RVISAC"), R.nets(run.state))
        xi = run.state.offset.reshape(-1)
        out = reset_readings(run.state.reset_cost)
        c = out["c"].reshape(-1)
        return {
            "xi - (4 - c) / 3": xi - (4.0 - c) / 3.0,
            "Q(R) - Q(A) - (3 - xi)": q["Q(R)"] - q["Q(A)"] - (3.0 - xi),
            "Q(B) - Q(R) + c + xi": q["Q(B)"] - q["Q(R)"] + c + xi,
            "reset rate": out["rate"].reshape(-1),
            "c": c,
        }

    build = functools.partial(
        agents.make,
        "RVISAC",
        *cycle(True).make(),
        # xi's spread across seeds is the batch mean's noise times the root
        # of its rate: the authors' batch of 256 (the preset's is 64) and a
        # rate of 0.02 keep it near 0.01, and the authors' learning rate.
        centering=RVIBatchMean(rate=0.02),
        reset_cost=RESET,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        **FIXED_ALPHA,
    )
    return runs.readings(build, read)


def sac(extension: Any, terminal: bool) -> Callable:
    def read(run: runs.Run) -> dict:
        q = R.per_seed(values("SAC"), R.nets(run.state))
        ext = run.state.ext_state[0]
        if isinstance(extension, ValueCentering):
            return {"offset": ext.reshape(-1)}
        out = reset_readings(ext)
        c = out["c"].reshape(-1)
        residual = q["Q(B)"] - (GAMMA * q["Q(R)"] - c)
        return {"Q(B) - (gamma Q(R) - c)": residual, "reset rate": out["rate"], "c": c}

    build = functools.partial(
        agents.make,
        "SAC",
        *cycle(terminal).make(),
        extensions=(extension,),
        **FIXED_ALPHA,
    )
    return runs.readings(build, read)


CASES["q1-RVISAC-reset"] = Case(
    "q1-RVISAC-reset",
    (
        Query("xi - (4 - c) / 3", 0.0, {"cut at the termination": -7.0 / 9.0}),
        Query("Q(R) - Q(A) - (3 - xi)", 0.0, GROWS),
        Query("Q(B) - Q(R) + c + xi", 0.0, {"no reset cost charged": 0.5}),
        Query("reset rate", 1.0 / 3.0, {"the reset target without the indicator": 0.0}),
        Query("c", 1.0, STAYS, margin=True),
    ),
    rvisac(),
    2000,
    (0.064, 0.095, 0.057, 0.02, 0.365),
    note="worst errors 0.032, 0.047, 0.028, 0.007 on 1000-1031 + 2000-2031, c at"
    " least 0.729; xi and Q(R) - Q(A) sit 0.02 high, lagging c's rise; certified 32/32",
)
CASES["q1-centering-SAC"] = Case(
    "q1-centering-SAC",
    (
        Query(
            "offset",
            RHO / (2.0 - GAMMA),
            {
                "uncentred": RHO / (1.0 - GAMMA),
                "shifted by +offset": -RHO / GAMMA,
                "the mean reward (simple centering)": RHO,
            },
        ),
    ),
    sac(ValueCentering(RVIBatchMean(rate=0.05)), terminal=False),
    2000,
    (0.083,),
    note="worst error 0.042 on 1000-1031 + 2000-2031 (2500: 0.056); certified 32/32",
)
CASES["q1-ResetCost-SAC"] = Case(
    "q1-ResetCost-SAC",
    (
        Query("Q(B) - (gamma Q(R) - c)", 0.0, {"cut at the termination": -2.08}),
        Query("reset rate", 1.0 / 3.0, {"the reset target without the indicator": 0.0}),
        Query("c", 1.0, STAYS, margin=True),
    ),
    sac(ResetCost(rate_tau=0.05), terminal=True),
    2000,
    (0.024, 0.032, 0.365),
    note="worst errors 0.012, 0.016; c at least 0.731 on 1000-1031 + 2000-2031; certified 32/32",
)


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
