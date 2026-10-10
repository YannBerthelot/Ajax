"""Value chains: tiny MDPs whose values and actions an agent's own estimator
must reach exactly (P0a three actions, Q11 transition-gradient chain, Q9
alias chain, the package's known answers, P1 normalisation chain, P8 soft
chain, Q6 entropy bonus). Each judged case trains 8 seeds in one program and
reads them against right and named wrong answers (``verdict``); the last
test of the module judges every case.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any, Callable

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probing_environments import gymnax_envs as pe
from probing_environments.gymnax_envs import continuous_actions as continuous

from ajax.agents.APG.train_APG import evaluate_apg
from ajax.agents.obs_norm import apply_obs_norm
from ajax.agents.PPO.utils import _compute_gae
from ajax.agents.PQN.utils import compute_q_lambda_targets
from ajax.environments.differentiable import with_transition_gradients
from ajax.networks.networks import predict_value

from . import agents, envs, oracles, runs
from . import readouts as R
from .verdict import STAGE_1, Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "08eb20de9be9"  # verdict.digest(CASES): every answer, pinned


# --- P0a: max bootstrap and action indexing ---------------------------------
# s0 (obs 0) -> s1 (obs 1) -> end; action a at s1 pays (0, 0.5, 1)[a]; gamma
# 0.8, so Q(s0) = 0.8 max_a Q(s1, a). PQN runs at q_lambda 0: at its default
# 0.65 the mean-bootstrap answer sits 0.14 away and no budget qualifies.

R3, G3 = (0.0, 0.5, 1.0), 0.8


def _three(st: envs.State, a: jax.Array, key: Any) -> tuple:
    paid = jnp.asarray(R3, jnp.float32)[a]
    return st.replace(s=st.s + 1), jnp.where(st.s == 1, paid, 0.0), st.s >= 1


THREE = envs.Spec("three_actions", _three, obs_box=(0.0, 2.0), actions=3)
PERMUTED = {
    "gathered one index up": (1.0, 0.0, 0.5),
    "gathered one index down": (0.5, 1.0, 0.0),
    "actions reversed": (1.0, 0.5, 0.0),
}
Q_S1 = tuple(
    Query(
        f"Q(s1, a={a})", R3[a], {k: v[a] for k, v in PERMUTED.items() if v[a] != R3[a]}
    )
    for a in range(3)
)
Q_S0_WRONG = {"mean bootstrap (D01)": G3 * 0.5, "no discount": 1.0}
GREEDY = Query("greedy action at s1", 2.0, {"readout one up": 0.0, "one down": 1.0})


def _three_q(n: R.Nets) -> dict:
    q0, q1 = R.q_values(n, 0.0), R.q_values(n, 1.0)
    out = {f"Q(s{s}, a={a})": q[a] for s, q in ((0, q0), (1, q1)) for a in range(3)}
    greedy = R.pi(n, 1.0).mode().reshape(()).astype(jnp.float32)
    return out | {"max_a Q(s0, a)": q0.max(), GREEDY.name: greedy}


def _three_ppo(n: R.Nets) -> dict:
    p2 = R.pi(n, 1.0).probs.reshape(-1)[2]
    return {"P(a=2 | s1)": p2, "V(s1)": R.critic(n, 1.0), "V(s0)": R.critic(n, 0.0)}


def _three_run(agent: str, **kw: Any) -> Callable:
    def build() -> Any:
        return agents.make(agent, *THREE.make(), gamma=G3, **kw)

    read = _three_ppo if agent == "PPO" else _three_q
    return runs.readings(build, lambda run: R.per_seed(read, R.nets(run.state)))


CASES["p0a-DQN"] = Case(
    "p0a-DQN",
    (*Q_S1, *(Query(f"Q(s0, a={a})", G3, Q_S0_WRONG) for a in range(3)), GREEDY),
    _three_run("DQN"),
    2500,
    (0.026, 0.021, 0.033, 0.037, 0.032, 0.029, 0.02),
    note="1250 qualified but 2 seeds of 3000-3031 read Q(s0) 0.047 off; cert 32/32",
)
CASES["p0a-PQN"] = Case(
    "p0a-PQN",
    (*Q_S1, Query("max_a Q(s0, a)", G3, Q_S0_WRONG), GREEDY),
    _three_run("PQN", q_lambda=0.0),
    5000,
    (0.068, 0.043, 0.083, 0.085, 0.02),
    note="cert 31/32; reads the greedy Q(s0): the others drift, rarely explored",
)
CASES["p0a-PPO"] = Case(
    "p0a-PPO",
    (
        Query("P(a=2 | s1)", 1.0, {"uniform policy": 1 / 3, "settled elsewhere": 0.0}),
        Query("V(s1)", 1.0, {"uniform policy": 0.5}),
        Query("V(s0)", G3, {"uniform policy": G3 * 0.5, "no discount": 1.0}),
    ),
    _three_run("PPO"),
    10_000,
    (0.02, 0.068, 0.043),
    note="at 5000 one seed read V(s1) 0.672; cert 32/32",
)


def test_p0a_targets_are_the_max_bootstrap_fixed_point() -> None:
    """src's compute_q_lambda_targets gives Q(s0) = 0.8 with the max
    bootstrap and 0.4 with the mean at lambda 0 (0.787 and 0.647 at 0.65,
    where the decayed epsilon-greedy reward at s1, 0.975, weighs in); a
    terminal step's target is its reward whatever follows."""
    for lam, right, wrong in ((0, 0.8, 0.4), (0.65, 0.787, 0.647)):
        for boot, want in ((1.0, right), (0.5, wrong)):
            rows = ([0.0, 0.975], [boot, 123.0], [0.0, 1.0], [0.0, 0.0])
            args = [jnp.array(x).reshape(2, 1, 1) for x in rows]
            t = compute_q_lambda_targets(*args, G3, lam).reshape(-1)
            np.testing.assert_allclose(t, [want, 0.975], atol=1e-6)


# --- Q11: APG through a differentiable transition ---------------------------
# x0 = +-1 drawn at reset (or pinned by ``start``); r0 = -3 a0^2, x1 = x0 + a0;
# r1 = -(x1 - 2)^2, terminal. The optimum a0* = (2 - x0) / 4 is reachable only
# through the transition. APG reads its action through apply_fn, as before.


def _grad_chain(st: envs.State, a: jax.Array, key: Any) -> tuple:
    a, first = a.reshape(()), st.s == 0
    reward = jnp.where(first, -3.0 * a**2, -((st.c - 2.0) ** 2))
    return st.replace(s=st.s + 1, c=jnp.where(first, st.c + a, st.c)), reward, st.s >= 1


def _grad_start(key: jax.Array, p: envs.Params) -> tuple:
    sign = jnp.where(jax.random.bernoulli(key), 1.0, -1.0)
    return 0, jnp.where(p.start == 0.0, sign, p.start)


def _grad_obs(st: envs.State) -> jax.Array:
    return st.c.reshape(1)


_GRAD: dict = {"obs_box": (-3.0, 3.0), "gradients": True}
GRAD_CHAIN = envs.Spec("grad_chain", _grad_chain, _grad_obs, _grad_start, **_GRAD)


def _ret(x0: float, a0: float) -> float:
    return -3.0 * a0**2 - (x0 + a0 - 2.0) ** 2


def _grad_queries(x0: float) -> tuple[Query, Query]:
    actions = {
        "gradient cut / horizon - 1 / done reward masked": 0.0,
        "horizon + 1": (2.0 - x0) / 7.0,
        "observation ignored": 0.5,
        "objective sign flipped (lower bound)": -1.0,
        "objective sign flipped (upper bound)": 1.0,
    }
    best, sign = (2.0 - x0) / 4.0, f"{x0:+.0f}"
    returns = {k: _ret(x0, a) for k, a in actions.items()}
    returns["evaluation drops the last reward"] = -3.0 * best**2
    returns["done reward masked in the shared rollout (a0 = 0)"] = 0.0
    j = Query(f"J({sign})", _ret(x0, best), returns)
    return Query(f"a({sign})", best, actions), j


def _apg_read(run: runs.Run) -> dict:
    def per_start(state: Any, x0: float) -> tuple[jax.Array, jax.Array]:
        a = R.action(R.nets(state), x0, clip=False)
        args = run.agent.env_args
        args = args.replace(env_params=args.env_params.replace(start=x0))
        m = evaluate_apg(state, R.KEY, args, None, 2, 2, stateful=False)
        return a, m["Eval/episodic mean reward"]

    out: dict[str, Any] = {}
    for x0 in (1.0, -1.0):
        a, j = jax.vmap(lambda s, x0=x0: per_start(s, x0))(run.state)
        out |= {f"a({x0:+.0f})": a, f"J({x0:+.0f})": j}
    return out


def _apg() -> Any:
    kw: dict = {"horizon": 2, "learning_rate": 3e-3, "lr_schedule": "warmup_cosine"}
    return agents.make("APG", *GRAD_CHAIN.make(), **kw)


CASES["q11-APG"] = Case(
    "q11-APG",
    tuple(q for pair in zip(_grad_queries(1.0), _grad_queries(-1.0)) for q in pair),
    runs.readings(_apg, _apg_read),
    16_000,
    (0.02, 0.073, 0.02, 0.02),
    note="16k the first rung, cert 32/32; cosine decay, as at a fixed rate Adam jitters",
)


def _episode(a0: jax.Array, env: Any, params: Any) -> tuple:
    _, st = env.reset(R.KEY, params)
    _, st, r0, t0, *_ = env.step(R.KEY, st, a0[None], params)
    _, st, r1, t1, *_ = env.step(R.KEY, st, a0[None], params)
    return r0 + r1, jnp.stack([t0, t1])


def test_q11_chain_passes_its_gradient_only_once_enabled() -> None:
    """d return / d a0 = -6 a0 - 2 (x0 + a0 - 2) with transition gradients
    enabled; gymnax's default detach leaves the myopic -6 a0. The episode
    terminates on its second step."""
    bare, params = GRAD_CHAIN.make()
    enabled, a0 = with_transition_gradients(bare), jnp.asarray(0.4)
    assert enabled.transition_gradients_enabled
    assert not bare.transition_gradients_enabled
    for x0 in (1.0, -1.0):
        p = params.replace(start=x0)
        for env, slope in ((enabled, -2.4 - 2 * (x0 - 1.6)), (bare, -2.4)):
            (total, ends), grad = jax.value_and_grad(_episode, has_aux=True)(a0, env, p)
            np.testing.assert_allclose(total, _ret(x0, 0.4), rtol=1e-6)
            np.testing.assert_allclose(grad, slope, rtol=1e-5)
            np.testing.assert_array_equal(ends, [False, True])


# --- Q9: an estimator's fingerprint under aliasing --------------------------
# Four hidden steps share observation 1.0 and pay (1, 1, -1, 0), the 4th
# terminal; gamma 0.9, lambda 0.3 (neither default). The value there is the
# fixed point of the agent's own estimator: 0.5378 for the lambda-return. PQN
# acts on Discrete(1), so max_a Q is the regressed value itself.

R9, G9, L9 = (1.0, 1.0, -1.0, 0.0), 0.9, 0.3


def _alias(st: envs.State, a: jax.Array, key: Any) -> tuple:
    return st.replace(s=st.s + 1), jnp.asarray(R9, jnp.float32)[st.s], st.s + 1 >= 4


ALIAS = {
    n: envs.Spec("alias", _alias, lambda st: jnp.ones(1), actions=n) for n in (0, 1)
}


def _alias_rows(n_steps: int) -> list[tuple]:
    return [(0, R9[t % 4], 0, 0, int(t % 4 == 3), 0) for t in range(n_steps)]


def alias_value(n_steps: int, gamma: float = G9, lam: float = L9, **rule: Any) -> float:
    """The fixed point of an estimator over rollouts of whole episodes."""
    rows = [(1.0, _alias_rows(n_steps))]
    ro = oracles.stack(rows)
    return float(oracles.fixed_point(ro, 1, gamma, lam, oracles.Rule(**rule))[0])


def forward_view(c: float, gamma: float, lam: float) -> np.ndarray:
    """Each step's lambda-return from its forward-view definition, V = c."""
    out = np.zeros(4)
    for t in range(4):
        sums = np.cumsum([gamma**k * R9[t + k] for k in range(4 - t)])
        cut = [lam ** (n - 1) * (sums[n - 1] + gamma**n * c) for n in range(1, 4 - t)]
        out[t] = (1.0 - lam) * sum(cut) + lam ** (3 - t) * sums[-1]
    return out


Q9_COMMON = {
    "gamma and lambda swapped": alias_value(16, L9, G9),
    "TD(0)": alias_value(16, lam=0.0),
    "Monte Carlo": alias_value(16, lam=1.0),
    "harness gamma 0.62": alias_value(16, gamma=0.62),
}
Q9_WRONG = {
    # P03 (gamma off the GAE trace, i.e. lambda / gamma) reads 0.5095, too
    # close to gate; the src target test catches it.
    "PPO": {
        "PPO lambda not forwarded (default 0.95)": alias_value(32, lam=0.95),
        "GAE scan runs forward": alias_value(32, forward=True),
        "V-trace port wired": alias_value(32, vtrace=True),
    },
    "PQN": {
        "PQN lambda not forwarded (default 0.65)": alias_value(16, lam=0.65),
        "Q01: lambda and 1 - lambda swapped": alias_value(16, lam=1.0 - L9),
        "Q02: Q(lambda) scan runs forward": alias_value(16, forward=True),
    },
}
# PPO without gradient clipping: with the default 0.5 the legacy path's
# critic leaves the fixed point in transient spikes and no budget met the
# rule (worst over 32 seeds 0.075 to 0.104); unclipped, 0.0002 from 625.
_PPO9 = {"gae_lambda": L9, "max_grad_norm": None}
Q9_CELLS: dict[
    str, tuple[str, int, float, dict]
] = {  # n_envs 1 runs PPO's legacy path, 4 with 2 minibatches the other
    "PPO": ("PPO", 625, 0.02, _PPO9),
    "PPO-minibatch": ("PPO", 1250, 0.047, _PPO9 | {"n_envs": 4, "num_minibatches": 2}),
    "PQN": ("PQN", 2500, 0.02, {"q_lambda": L9}),
}


def _q9_read(run: runs.Run) -> dict:
    agent, keys = type(run.agent).__name__, ("reward", "terminated", "truncated")
    out = R.per_seed(lambda n: {"V": R.value(agent, n, 1.0)}, R.nets(run.state))
    return out | {k: v[..., 0] for k, v in R.rollout_rows(run.state, keys).items()}


def _q9_agent(agent: str, kw: dict) -> Callable:
    env, params = ALIAS[int(agent == "PQN")].make()
    return lambda: agents.make(
        agent, env, params, gamma=G9, expose_recent_rollout=True, **kw
    )


for cell, (agent, budget, tol, kw) in Q9_CELLS.items():
    query = Query("V", alias_value(16), Q9_COMMON | Q9_WRONG[agent])
    reads = runs.readings(_q9_agent(agent, kw), _q9_read)
    CASES[f"q9-{cell}"] = Case(f"q9-{cell}", (query,), reads, budget, (tol,))


def test_q9_oracle_is_the_forward_view_fixed_point() -> None:
    """The oracle's right answer is the forward-view lambda-return's fixed
    point, an independent derivation (as TD(0)'s is at lambda 0)."""
    for lam in (L9, 0.0):
        c = 0.0
        for _ in range(2000):
            c = float(np.mean(forward_view(c, G9, lam)))
        assert c == pytest.approx(alias_value(16, lam=lam), abs=1e-9)


@pytest.mark.parametrize("n_steps", [32, 16])
def test_q9_target_functions_return_the_lambda_return(n_steps: int) -> None:
    """PQN's compute_q_lambda_targets and PPO's _compute_gae (advantages +
    values), on the probe's rollout with V = 0.5378, return each step's
    forward-view lambda-return: a wrong recursion (Q01, Q02, P03, a forward
    GAE scan) fails here even where the trained cells cannot see it."""
    c, rows, shape = alias_value(16), np.array(_alias_rows(n_steps)), (n_steps, 1, 1)
    r, term = (jnp.asarray(rows[:, i], jnp.float32).reshape(shape) for i in (1, 4))
    v, zeros = jnp.full(shape, c), jnp.zeros(shape)
    want = np.tile(forward_view(c, G9, L9), n_steps // 4)
    q = compute_q_lambda_targets(r, v, term, zeros, G9, L9)
    g = _compute_gae(r, v, v, term, zeros, G9, L9)[1]
    for targets in (q, g):
        np.testing.assert_allclose(np.asarray(targets).reshape(-1), want, atol=1e-5)


@pytest.mark.parametrize("cell", list(Q9_CELLS))
def test_q9_rollouts_hold_whole_episodes(cell: str) -> None:
    """The geometry the oracle assumes, from the last rollout of the
    trained cell: whole episodes from a reset, rewards (1, 1, -1, 0)
    repeated, terminated on every 4th step only, never truncated."""
    case = CASES[f"q9-{cell}"]
    r = case.readings(STAGE_1, case.budget)
    n_steps = r["reward"].shape[-1]
    assert n_steps == (16 if cell == "PQN" else 32)
    assert (r["reward"] == np.tile(R9, n_steps // 4)).all(), r["reward"]
    assert (r["terminated"] == np.tile([0, 0, 0, 1], n_steps // 4)).all()
    assert not r["truncated"].any()


# --- pilot: the package's known-answer probes -------------------------------
# Every probe an agent's action space fits, gamma 0.62; the continuous policy
# probes declared and clipped to [-1, 1], the coupling one paying clip(a) s.
# Max-entropy agents settle below the reward bound (SAC near 0.91): policy
# readings are margins. A query's name is its reading: V(x), a(x) (the
# clipped mean action), Q(x, a=i) or a difference of two. Ladder on seeds
# 1000-1031 and 2000-2031; 638 of 640 seeds of 3000-3031 within.

G = agents.GAMMA
BLIND, NONE = {"observation ignored": 0.5}, {"no learning": 0.0}
SIGNS = {"observation ignored": 0.0, "reversed": -1.0}
_FLIP = "observation ignored or reversed"
_V = {"policy ignores the observation": 0.0, "reversed": -1.0}
_V0 = {"gamma squared": G**2, "1 - gamma": 1 - G, "no discount": 1.0}
_V0["done flag one step late"] = 0.0  # replay of the 0382f32 buffer bug
Probes = dict[str, tuple[Any, tuple[Query, ...]]]


def _pilot(pkg: Any) -> Probes:
    """The value probes, in either action flavour."""
    done = {"done mask ignored": 1 / (1 - G)}
    late = {"done mask ignored": 1 / (1 - G**2)}
    backprop = (Query("V(0)", 0.0, BLIND), Query("V(1)", 1.0, BLIND))
    return {
        "value": (pkg.ValueLossOrOptimizerEnv, (Query("V(0)", 1.0, done),)),
        "backprop": (pkg.ValueBackpropEnv, backprop),
        "discounting": (
            pkg.RewardDiscountingEnv,
            (Query("V(0)", G, _V0), Query("V(1)", 1.0, late)),
        ),
    }


PILOT = {"continuous": _pilot(continuous), "discrete": _pilot(pe)}
PILOT["continuous"]["advantage"] = (
    envs.symmetric(continuous.AdvantagePolicyLossPolicyUpdateEnv),
    (Query("a(1)", 1.0, NONE | {"policy gradient reversed": -1.0}, True),),
)
PILOT["continuous"]["coupling"] = (
    envs.SignedActionEnv,
    (
        Query("a(+1)", 1.0, NONE | {_FLIP: -1.0}, True),
        Query("a(-1)", -1.0, NONE | {_FLIP: 1.0}, True),
        Query("V(+1)", 1.0, _V, True),
        Query("V(-1)", 1.0, _V, True),
    ),
)
PILOT["discrete"]["advantage"] = (
    pe.AdvantagePolicyLossPolicyUpdateEnv,
    (
        Query("Q(0, a=0)", 1.0, {"actions swapped": 0.0}),
        Query("Q(0, a=0) - Q(0, a=1)", 1.0, NONE | {"actions swapped": -1.0}, True),
    ),
)
PILOT["discrete"]["coupling"] = (
    pe.PolicyAndValueEnv,
    (
        Query("Q(0, a=0)", 1.0, BLIND),
        Query("Q(1, a=1)", 1.0, BLIND),
        Query("Q(0, a=0) - Q(0, a=1)", 1.0, SIGNS, True),
        Query("Q(1, a=1) - Q(1, a=0)", 1.0, SIGNS, True),
    ),
)
PILOT_CAL = {  # budget, tolerances in query order
    "SAC-value": (1250, (0.02,)),
    "SAC-backprop": (1250, (0.037, 0.02)),
    "SAC-discounting": (10_000, (0.093, 0.02)),
    "SAC-advantage": (5000, (0.333,)),
    "SAC-coupling": (5000, (0.333, 0.341, 0.329, 0.342)),
    "PPO-value": (1250, (0.02,)),
    "PPO-backprop": (1250, (0.059, 0.05)),
    "PPO-discounting": (10_000, (0.077, 0.067)),
    "PPO-advantage": (1250, (0.498,)),
    "PPO-coupling": (1250, (0.5, 0.5, 0.486, 0.495)),
    "DQN-value": (1250, (0.02,)),
    "DQN-backprop": (1250, (0.02, 0.02)),
    "DQN-discounting": (1250, (0.02, 0.02)),
    "DQN-advantage": (1250, (0.02, 0.5)),
    "DQN-coupling": (1250, (0.039, 0.02, 0.499, 0.498)),
    "PQN-value": (80_000, (0.02,)),
    "PQN-backprop": (5000, (0.02, 0.02)),
    "PQN-discounting": (5000, (0.02, 0.02)),
    "PQN-advantage": (80_000, (0.02, 0.5)),
    "PQN-coupling": (5000, (0.02, 0.02, 0.499, 0.497)),
}
_TERM = re.compile(r"(V|a|Q)\(([-+]?\d)(?:, a=(\d))?\)")


def _pilot_read(agent: str, queries: tuple[Query, ...]) -> Callable:
    def term(n: R.Nets, kind: str, x: str, i: str) -> jax.Array:
        if kind == "V":
            return R.value(agent, n, float(x))
        return R.action(n, float(x)) if kind == "a" else R.q_values(n, float(x))[int(i)]

    def read(n: R.Nets) -> dict:
        out = {}
        for q in queries:
            v = [term(n, *t) for t in _TERM.findall(q.name)]
            out[q.name] = v[0] - v[1] if len(v) == 2 else v[0]
        return out

    return lambda run: R.per_seed(read, R.nets(run.state))


for key, (steps, tols) in PILOT_CAL.items():
    agent, probe = key.split("-")
    env_cls, qs = PILOT["discrete" if agent in ("DQN", "PQN") else "continuous"][probe]
    build = functools.partial(agents.make, agent, *envs.package(env_cls))
    reads = runs.readings(build, _pilot_read(agent, qs))
    CASES[f"pilot-{key}"] = Case(f"pilot-{key}", qs, reads, steps, tols)


# --- P1: observation normalisation in training, bootstrap and evaluation ----
# s0 -> s1 -> end, raw observations 5 then 1 (1 at the end too): the
# normaliser settles at mean 3, sd 2, so the networks see s0 as +1 and s1 as
# -1; raw s1 reads as normalised s0, twice-normalised s0 as s1. Discrete(3):
# r(s0) = 1[a = 0], r(s1) = 0.3 1[a = 0] + 1[a = 2]; Box: r(s0) = c0 -
# (a + 0.5)^2, r(s1) = 1 - (a - 0.5)^2, c0 = 1 (0 for ASAC, whose
# termination penalty would dwarf r(s1)). Read: the last logged evaluation,
# and a value at s0 against its oracle from the final policy (with alpha,
# TD3's smoothing noise or ASAC's penalty). Ladder on 1000-1031 and
# 2000-2031; 32/32 of 3000-3031 within (PPO-discrete 31/32).

S0, S1 = 5.0, 1.0


def _p1_reward(discrete: bool, c0: float = 1.0) -> Callable:
    def reward(s: Any, a: Any) -> jax.Array:
        if discrete:
            r0 = jnp.where(a == 0, 1.0, 0.0)
            r1 = jnp.where(a == 0, 0.3, 0.0) + jnp.where(a == 2, 1.0, 0.0)
        else:
            a = jnp.clip(a, -1.0, 1.0)
            r0, r1 = c0 - (a + 0.5) ** 2, 1.0 - (a - 0.5) ** 2
        return jnp.where(s == 0, r0, r1)

    return reward


def _p1_chain(reward: Callable, actions: int) -> envs.Spec:
    """The limit outlasts the episode; it is also the evaluation's length."""

    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        return st.replace(s=st.s + 1), reward(st.s, a.reshape(())), st.s >= 1

    def obs(st: envs.State) -> jax.Array:
        return jnp.where(st.s == 0, S0, S1).reshape(1)

    return envs.Spec("p1", transition, obs, obs_box=(0.0, S0), actions=actions, limit=4)


P1_REWARD = {"discrete": _p1_reward(True), "continuous": _p1_reward(False)}
P1_REWARD["asymmetric"] = _p1_reward(False, 0.0)
P1_CHAIN = {k: _p1_chain(r, 3 if k == "discrete" else 0) for k, r in P1_REWARD.items()}
_ON = {"normalize_observations": True}
P1_CELLS: dict[str, tuple[str, str, int, dict]] = {  # agent, chain, budget, overrides
    "DQN": ("DQN", "discrete", 5000, _ON),
    # q_lambda 0, a one-step bootstrap: at 0.65 fault (a)'s gap shrinks to 0.17.
    "PQN": ("PQN", "discrete", 5120, _ON | {"q_lambda": 0.0}),
    "PPO-continuous": ("PPO", "continuous", 2048, _ON),
    "PPO-discrete": ("PPO", "discrete", 2048, _ON),
    "APO-discrete": ("APO", "discrete", 4096, _ON | {"n_steps": 64, "batch_size": 64}),
    "AVG": ("AVG", "continuous", 10_000, {"gamma": G, "n_envs": 1}),  # its defaults
    "SAC": ("SAC", "continuous", 5000, _ON),
    "SafeSAC": ("SafeSAC", "continuous", 5000, _ON),
    "TD3": ("TD3", "continuous", 5000, _ON),
    "REDQ": ("REDQ", "continuous", 2500, _ON | {"num_critic_updates": 2}),
    "ASAC": ("ASAC", "asymmetric", 10_000, _ON),
    "SAC-running": ("SAC", "continuous", 5000, {"normalize_obs_running": True}),
}


def _p1_read(model: Any, cell: str, n: R.Nets) -> dict:
    """One seed: the inputs the networks see, and a value at s0 minus its
    oracle (DQN and PQN: Q-values; APO: none)."""
    name, chain = P1_CELLS[cell][:2]
    r, (k0, k1) = P1_REWARD[chain], jax.random.split(R.KEY)
    pi0, pi1 = R.pi(n, S0), R.pi(n, S1)
    out = {"input error": R.input_error(n, *n.extra[:2])}
    info = getattr(n.actor, "obs_norm_info", None)
    for label, x in (("input(s0)", S0), ("input(s1)", S1)):
        seen = R.inp(n, x) if info is None else apply_obs_norm(R.inp(n, x), info)
        out[label] = seen.reshape(())
    if name in ("DQN", "PQN"):
        return out | {"Q(s0,0)": R.q_values(n, S0)[0], "Q(s1,2)": R.q_values(n, S1)[2]}
    if name == "APO":
        return out
    if name == "PPO" and chain == "discrete":
        a = jnp.arange(3)
        oracle = pi0.probs.reshape(-1) @ r(0, a) + G * (pi1.probs.reshape(-1) @ r(1, a))
        return out | {"V(s0) - oracle": R.critic(n, S0) - oracle}
    if name == "PPO":
        a0 = pi0.sample(seed=k0, sample_shape=(4096,)).reshape(-1)
        a1 = pi1.sample(seed=k1, sample_shape=(4096,)).reshape(-1)
        oracle = jnp.mean(r(0, a0)) + G * jnp.mean(r(1, a1))
        return out | {"V(s0) - oracle": R.critic(n, S0) - oracle}
    if name == "ASAC":
        # ASAC charges a learned penalty p on terminating rewards and neither
        # discounts nor masks: Q(s0, .) = r0 - c + V1, Q(s1, .) = r1 - p - c + V0,
        # V the soft means, so V1 - V0 = (w1 - p - w0) / 2 whatever the gauge c.
        p, alpha = n.extra[2].reshape(()), R.alpha(n)
        w0 = R.soft_value(pi0, alpha, lambda a: r(0, a), k0)
        w1 = R.soft_value(pi1, alpha, lambda a: r(1, a), k1)
        oracle = r(0, -0.5) - (r(1, 0.5) - p) + 0.5 * (w1 - p - w0)
        gap = R.critic(n, S0, -0.5) - R.critic(n, S1, 0.5)
        return out | {"Q(s0,-0.5) - Q(s1,0.5) - oracle": gap - oracle}
    if name == "TD3":
        c = model.agent_config
        noise = c.target_policy_noise * jax.random.normal(R.KEY, (4096,))
        noise = jnp.clip(noise, -c.target_noise_clip, c.target_noise_clip)
        boot = jnp.mean(r(1, pi1.mean().reshape(()) + noise))
    else:
        boot = R.soft_value(pi1, R.alpha(n), lambda a: r(1, a), R.KEY)
    q = R.critic(n, S0, -0.5)
    return out | {"Q(s0,-0.5) - oracle": q - (r(0, -0.5) + G * boot)}


@functools.cache
def p1_readings(cell: str, seeds: tuple, budget: int) -> dict[str, np.ndarray]:
    """Train ``cell`` logging an evaluation at half and full budget."""
    agent, chain, _, kw = P1_CELLS[cell]
    preset = None if agent == "AVG" else "probe"
    model = agents.make(agent, *P1_CHAIN[chain].make(), preset=preset, **kw)
    run = runs.train(model, seeds, budget, log_every=budget // 2)
    if (run.timesteps().max(1) < 0.75 * budget).any():
        raise RuntimeError(f"{cell}: no evaluation logged in the last quarter")
    s = run.state
    raw = np.where(R.field(s, "s") == 0, S0, S1)[..., None].astype(np.float32)
    penalty = getattr(s, "episode_termination_penalty", None)
    n = R.nets(s, (raw, s.collector_state.last_obs, penalty), R.env_stats(run))
    out = R.per_seed(functools.partial(_p1_read, model, cell), n)
    return out | {"eval": run.logged("Eval/episodic mean reward")}


_A = "(a) every stored next_obs is raw: gymnax writes the raw pre-reset obs in info['final_observation'] on every step, the normaliser passes it through (src/ajax/wrappers.py:443-447) and get_final_obs prefers it (src/ajax/environments/interaction.py:113-125, 874)"
_B = "(b) PPO normalises twice in evaluation: the env normaliser does not apply in training (src/ajax/agents/PPO/PPO.py:166) but setup_environment rebuilds it applying (src/ajax/evaluate.py:150-158) and get_pi normalises again (interaction.py:347-350)"
_C = "(c) ClipAction(-1, 1) wraps every normalised env whatever its action space (src/ajax/environments/create.py:375-388) after the index is cast to int32 (interaction.py:204-205): action 2 reaches the env as 1.0"
CLIP = "(c) training clip sends action 2 as 1"
RAW = "(a) raw next obs: s0 bootstraps on s0"
_D, _ALONE = (
    "(d) evaluation without the normaliser",
    "(b) alone: s0 plays s1's action 2",
)
_TWICE = "(b) normalised twice / (d) raw, at the optimum: at most"
EVAL_D = Query("eval", 2.0, {CLIP: 1.3, _ALONE: 1.0, _D: 1.3})
EVAL_C = Query("eval", 2.0, {_TWICE: 1.0, "untrained (actions near 0)": 1.5})
EVAL_A = Query("eval", 1.0, {f"{_D}: at most": 0.0, "untrained": 0.5})
_GREEDY = {RAW: 1 / (1 - G), "(c) training clip: V(s1) = 0.3": 1 + G * 0.3}
Q_GREEDY = Query("Q(s0,0)", 1 + G, _GREEDY | {"untrained": 0.0})
Q_A2 = Query("Q(s1,2)", 1.0, {CLIP: 0.0})
# Under (a), gamma^2 w / (1 - gamma) above the oracle, w ~ 0.95 a soft value
# at the optimum; discrete PPO's oracle today is 1.186 (it avoids action 2).
Q_SOFT = Query("Q(s0,-0.5) - oracle", 0.0, {RAW: 0.95, "untrained": -1.58})
V_CONTINUOUS = Query("V(s0) - oracle", 0.0, {RAW: 1.0, "untrained": -1.2})
V_DISCRETE = Query("V(s0) - oracle", 0.0, {RAW: 1.45, "untrained": -1.19})
# Under (a) s1 also bootstraps on the untrained input 5: measured (seeds 0-7).
_MEASURED = {"(a) raw next obs (planted, measured)": -0.9, "untrained": 1.15}
ASAC_GAP = Query("Q(s0,-0.5) - Q(s1,0.5) - oracle", 0.0, _MEASURED)
P1_CASES = {  # cell, queries, tolerances (none for a live defect: half the gap)
    "SAC": ("SAC", (EVAL_C, Q_SOFT), (0.036, 0.073)),
    "SafeSAC": ("SafeSAC", (EVAL_C, Q_SOFT), (0.036, 0.073)),
    "TD3": ("TD3", (EVAL_C, Q_SOFT), (0.02, 0.051)),
    "REDQ": ("REDQ", (EVAL_C, Q_SOFT), (0.051, 0.085)),
    "SAC-running": ("SAC-running", (EVAL_C, Q_SOFT), (0.037, 0.075)),
    "ASAC": ("ASAC", (EVAL_A, ASAC_GAP), (0.02, 0.056)),
    "AVG": ("AVG", (EVAL_C,), (0.084,)),
    "PPO-continuous": ("PPO-continuous", (V_CONTINUOUS,), (0.044,)),
    "PPO-discrete": ("PPO-discrete", (V_DISCRETE,), (0.046,)),
    "DQN-c": ("DQN", (EVAL_D, Q_GREEDY, Q_A2), ()),
    "PQN-c": ("PQN", (EVAL_D, Q_A2), ()),
    "PQN-a": ("PQN", (Q_GREEDY,), ()),
    "PPO-continuous-b": ("PPO-continuous", (EVAL_C,), ()),
    "PPO-discrete-bc": ("PPO-discrete", (EVAL_D,), ()),
    "APO-discrete-c": ("APO-discrete", (EVAL_D,), ()),
    "AVG-a": ("AVG", (Q_SOFT,), ()),
}
P1_LIVE = {
    "DQN-c": f"{_C}; right answer eval 2.0, Q(s0,0) 1.62, Q(s1,2) 1.0; today 1.3, 1.186, 0.00",
    "PQN-c": f"{_C}; right answer eval 2.0, Q(s1,2) 1.0; today 1.3, 0.00",
    "PQN-a": f"{_A} (bootstrap at src/ajax/agents/PQN/train_PQN.py:162-166); right answer Q(s0,0) 1.62, today 2.631 (1.186 once (a) alone is fixed)",
    "PPO-continuous-b": f"{_B}; right answer eval 2.0, today 0.63-1.06",
    "PPO-discrete-bc": f"{_B} and {_C}; right answer eval 2.0, today 1.3",
    "APO-discrete-c": f"{_C}; right answer eval 2.0, today 1.3",
    "AVG-a": f"{_A} (target at src/ajax/agents/AVG/train_AVG.py:792 -> 216, 227); right answer Q(s0,-0.5) - oracle 0, today +0.93 to +1.02",
}
for label, (cell, p1_queries, p1_tols) in P1_CASES.items():
    reads, steps = functools.partial(p1_readings, cell), P1_CELLS[cell][2]
    why = P1_LIVE.get(label, "")
    CASES[f"p1-{label}"] = Case(f"p1-{label}", p1_queries, reads, steps, p1_tols, why)


@pytest.mark.parametrize("cell", list(P1_CELLS))
def test_p1_readout_feeds_what_training_does(cell: str) -> None:
    """The readout's input equals the last observation training gave the
    agent, and the networks see s0 at +1 and s1 at -1 (to 0.01)."""
    r = p1_readings(cell, STAGE_1, P1_CELLS[cell][2])
    assert np.max(r["input error"]) < 1e-5, r["input error"]
    np.testing.assert_allclose(r["input(s0)"], 1.0, atol=0.01)
    np.testing.assert_allclose(r["input(s1)"], -1.0, atol=0.01)


# --- P8: the soft bootstrap with alpha pinned; TD3's target smoothing --------
# s0 (obs 0) -> s1 (obs 1) -> end, gamma 0.62. SoftChain-M, actions in
# [-1, 1]^2: r(s0, a) = -20 (a1 - 0.5)^2 - 5 (a2 + 0.5)^2, r(s1) = 1. At
# alpha = 1 the critic's Q(s0, a) - r(s0, a) is gamma (1 + 2 H*) = 1.4677,
# H* the max-entropy tanh-Gaussian's entropy (actor, soft target with summed
# log-probs and the Jacobian, target networks and alpha at once), read on
# policy samples at s0, where the critic was fitted. SoftChain-T: r(s1, a) =
# 1 - 4 a^2, TD3's target noise N(0, 2) clipped to 0.8 (non-default, as its
# exploration noise 0.4): Q(s0) = -0.6348. Every cell 32/32 of 3000-3031.


def _soft_reward(s: Any, a: jax.Array) -> jax.Array:
    r0 = -20.0 * (a[..., 0] - 0.5) ** 2 - 5.0 * (a[..., 1] + 0.5) ** 2
    return jnp.where(s == 0, r0, 1.0)


def _soft_chain(dim: int, reward: Callable) -> envs.Spec:
    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        a = jnp.clip(a.reshape(dim), -1.0, 1.0)
        return st.replace(s=st.s + 1), reward(st.s, a), st.s >= 1

    return envs.Spec(f"soft{dim}", transition, obs_box=(0.0, 2.0), box=(-1.0, 1.0, dim))


SOFT_M = _soft_chain(2, _soft_reward)
SOFT_T = _soft_chain(1, lambda s, a: jnp.where(s == 0, 0.0, 1.0 - 4.0 * a[0] ** 2))
H = oracles.tanh_gaussian_entropy(oracles.max_entropy_sigma())


def td3_q(noise: float, clip: float) -> float:
    """SoftChain-T's Q(s0) with target noise N(0, noise) clipped to ``clip``."""
    return G * (1.0 - 4.0 * oracles.clipped_square_mean(noise, min(clip, 1.0)))


# The same quadrature: sigma_u 1.2847 a dimension with the Jacobian left out
# of the target, at the clip e^2 with it dropped everywhere; the batch mean
# averages s1's and s0's log-probs (summed entropy -0.4105 at s0); untrained,
# Q ~ 0 leaves -E[r(s0, a)]. A free alpha (442336a) breaks the pin, a
# precondition of every cell but TD3's.
SOFT_VALUE = Query(
    "Q(s0, a) - r(s0, a)",
    G * (1.0 + 2.0 * H),
    {
        "entropy term dropped": G,
        "entropy sign flipped": G * (1.0 - 2.0 * H),
        "log-prob averaged over dimensions (S07)": G * (1.0 + H),
        "pre-tanh log-prob in the target": G * (1.0 + 2.0 * 1.2847),
        "Jacobian dropped everywhere": 4.8595,
        "batch-mean log-prob (b7d30ab)": G * (1.0 + (2.0 * H - 0.4105) / 2.0),
        "entropy term outside gamma": G + 2.0 * H,
        "untrained": 8.958,
    },
)
# The alpha = 1 optimum at s0 per dimension, (sigma_u, tanh(mu)), and the
# spread of its actions. The untrained sigma_u e^-1 sits 0.02 from sigma_u2.
NO_BONUS, NO_JACOBIAN = "no entropy bonus in the actor", "Jacobian dropped everywhere"
P8_POLICY = (
    Query("sigma_u1(s0)", 0.2076, {NO_BONUS: 0.0, "untrained": math.exp(-1.0)}),
    Query("sigma_u2(s0)", 0.3886, {NO_BONUS: 0.0, NO_JACOBIAN: 7.389}),
    Query("tanh mu1(s0)", 0.5106, {"untrained": 0.0}),
    Query("tanh mu2(s0)", -0.4990, {"untrained": 0.0, NO_JACOBIAN: -1.0}),
)
P8_SPREAD = tuple(
    Query(f"action std {i} at s0", s, {"collector acts with the mean": 0.0})
    for i, s in ((1, 0.1532), (2, 0.2843))
)
_T05 = {"no exploration noise (T05)": 0.0, "exploration noise not forwarded (0.1)": 0.1}
P8_TD3 = (
    Query(
        "Q(s0)",
        td3_q(2.0, 0.8),
        {
            "inner noise clip missing (T02)": td3_q(2.0, 1.0),
            "no target smoothing": G,
            "noise settings not forwarded (0.2 / 0.5)": td3_q(0.2, 0.5),
            "only the noise clip forwarded": td3_q(0.2, 0.8),
            "only the noise scale forwarded": td3_q(2.0, 0.5),
            "noise and clip swapped": td3_q(0.8, 2.0),
            "untrained": 0.0,  # rejected by the NaN spread of an empty buffer
        },
    ),
    Query("action std at s1", math.sqrt(oracles.clipped_square_mean(0.4, 1.0)), _T05),
)
# A zero learning rate pins alpha (Adam's step scales by it). REDQ: 5 critic
# updates a step, not 20, keep its cost near SAC's. AVG never updates alpha
# (train_AVG.py:662-683); at its own learning rates its policy still wanders
# by 0.09 at 40,000 steps.
_LR0 = {"alpha_learning_rate": 0.0, "alpha_init": 1.0}
_SMOOTH = {"target_policy_noise": 2.0, "target_noise_clip": 0.8}
P8_AGENTS: dict[str, dict[str, Any]] = {  # a 50,000-row buffer but for AVG
    "SAC": {"fixed_alpha": True, "alpha_init": 1.0},
    "SafeSAC": {"fixed_alpha": True, "alpha_init": 1.0},
    "REDQ": _LR0 | {"num_critic_updates": 5},
    "AVG": _LR0 | {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4},
    "TD3": _SMOOTH | {"exploration_noise": 0.4},
}


def _p8_read(name: str, n: R.Nets) -> dict:
    """One seed; Q through the target critic (the Polyak average the agent
    bootstraps from, steadier than the online one), ensemble mean."""

    def q(a: jax.Array) -> jax.Array:
        x = jnp.concatenate([jnp.broadcast_to(R.inp(n, 0.0), (len(a), 1)), a], -1)
        q = predict_value(n.critic, n.critic.target_params, x)
        return q.reshape(-1, len(a)).mean(0)

    def spread(i: int, at_s1: bool) -> jax.Array:  # over the replay window
        obs, act = n.extra
        mask = (obs[:, 0] > 0.5 if at_s1 else obs[:, 0] < 0.5).astype(jnp.float32)
        mean = (act[:, i] * mask).sum() / mask.sum()
        return jnp.sqrt((((act[:, i] - mean) * mask) ** 2).sum() / mask.sum())

    p = R.pi(n, 0.0)
    if name == "TD3":
        a = p.mean().reshape(()) + 0.4 * jax.random.normal(R.KEY, (512, 1))
        q0 = q(jnp.clip(a, -1.0, 1.0)).mean()
        return {"Q(s0)": q0, "action std at s1": spread(0, True)}
    sigma, mean = p.distribution.scale.reshape(-1), p.mean().reshape(-1)
    out = {f"sigma_u{i + 1}(s0)": sigma[i] for i in range(2)}
    out |= {f"tanh mu{i + 1}(s0)": mean[i] for i in range(2)}
    out["log_alpha"] = n.alpha.params["log_alpha"]
    if name == "AVG":  # its online critic misses Q(s0) by up to 0.2: unread
        return out
    a = p.sample(seed=R.KEY, sample_shape=(512,)).reshape(512, 2)
    out["Q(s0, a) - r(s0, a)"] = (q(a) - _soft_reward(0, a)).mean()
    return out | {f"action std {i + 1} at s0": spread(i, False) for i in range(2)}


def _p8_readings(name: str) -> Callable:
    def build() -> Any:
        kw = P8_AGENTS[name] | ({} if name == "AVG" else {"buffer_size": 50_000})
        return agents.make(name, *(SOFT_T if name == "TD3" else SOFT_M).make(), **kw)

    def read(run: runs.Run) -> dict:
        window = None  # the last 2000 rows of env 0, newest first
        if name != "AVG":
            rows = R.replay_rows(run.state, ("obs", "action"))
            window = tuple(rows[k][:, 0, ::-1][:, :2000] for k in ("obs", "action"))
        n = R.nets(run.state, window, R.env_stats(run))
        out = R.per_seed(functools.partial(_p8_read, name), n)
        if name != "TD3" and (out["log_alpha"] != 0.0).any():
            raise RuntimeError(f"alpha moved off its pin: log alpha {out['log_alpha']}")
        return out

    return runs.readings(build, read)


P8_SOFT = (SOFT_VALUE, *P8_POLICY, *P8_SPREAD)
_SAC8 = (0.02, 0.035, 0.044, 0.059, 0.077, 0.02, 0.032)
P8_CAL = {  # queries, budget, tolerances, note
    "SAC": (P8_SOFT, 10_000, _SAC8, "5000 failed (tanh mu2 0.050)"),
    "SafeSAC": (P8_SOFT, 10_000, _SAC8, "SAC's readings, bit for bit"),
    "REDQ": (P8_SOFT, 2500, (0.087, 0.022, 0.071, 0.087, 0.091, 0.022, 0.029), ""),
    "AVG": (P8_POLICY, 20_000, (0.066, 0.1, 0.074, 0.077), "10,000 failed: 0.196"),
    "TD3": (P8_TD3, 2500, (0.046, 0.051), ""),
}
for name, (p8_queries, steps, p8_tols, note) in P8_CAL.items():
    reads, slow = _p8_readings(name), name != "SAC"
    case = Case(f"p8-{name}", p8_queries, reads, steps, p8_tols, slow=slow, note=note)
    CASES[case.id] = case


# --- Q6: the fixed entropy bonus (ent_coef) of PPO and APO -------------------
# One-step bandits at the constant observation 1 (at 0, PPO's LayerNorm
# amplifies the first updates): the policy settles where E[r] + c H peaks.
# a: Discrete(2) paying 1 for action 0, c = 1: pi(0) = sigmoid(1 / c). b: r =
# -(a - 0.5)^2, unclipped, c = 2: mu 0.5, sigma sqrt(c / 2) (b2: per dimension
# of the joint action). c: r = 0, c = 0.1, squashed: sigma_u 0.8744, the
# max-entropy tanh-Gaussian (PPO starts at e^0.5: no fault crosses it). Off:
# advantage normalisation and gradient clipping (both move the stationary
# point); APO's gae_lambda 0 (advantage r - rho). 8 envs; learning rates set
# the spread, not the answer. Measured wrong answers: medians, 1000-1031.


def _bandit(reward: Callable, actions: int = 0, dim: int = 1) -> envs.Spec:
    def step(st: envs.State, a: jax.Array, key: Any) -> tuple:
        return st.replace(s=st.s + 1), reward(a), True

    one, box = (lambda st: jnp.ones(1)), (-1.0, 1.0, dim)
    return envs.Spec("bandit", step, one, obs_box=(0.0, 2.0), actions=actions, box=box)


def _quadratic(a: jax.Array) -> jax.Array:
    return -jnp.sum((a.reshape(-1) - 0.5) ** 2)


BANDITS = {
    "a": _bandit(lambda a: jnp.where(a == 0, 1.0, 0.0), actions=2),
    "b": _bandit(_quadratic),
    "b2": _bandit(_quadratic, dim=2),
    "c": _bandit(lambda a: jnp.float32(0.0)),
}


def _q6_agent(agent: str, bandit: str, c: float, **kw: Any) -> Any:
    lr = {"PPO": 1e-5, "APO": 3e-4}[agent]
    kw |= {"gae_lambda": 0.0} if agent == "APO" else {}
    kw |= {"normalize_advantage": False, "max_grad_norm": None, "ent_coef": c}
    kw |= {"n_envs": 8, "actor_learning_rate": lr, "critic_learning_rate": lr}
    return agents.make(agent, *BANDITS[bandit].make(), **kw)


def _q6_read(n: R.Nets) -> dict:
    """The policy at the constant observation."""
    p = R.pi(n, 1.0)
    if isinstance(p, distrax.Categorical):
        return {"pi(best)": p.probs.reshape(-1)[0]}
    if hasattr(p, "unsquashed_stddev"):  # the tanh-Gaussian's latent
        mu_u, sigma_u = p.unsquashed_mean(), p.unsquashed_stddev()
        return {"mu_u": mu_u.reshape(()), "sigma_u": sigma_u.reshape(())}
    mu, sigma = p.mean().reshape(-1), p.stddev().reshape(-1)
    if mu.shape[0] == 1:
        return {"mu": mu[0], "sigma": sigma[0]}
    return {"mu_1": mu[0], "mu_2": mu[1], "sigma_1": sigma[0], "sigma_2": sigma[1]}


def apo_latent_sigma(budget: int) -> float:
    """APO's sigma_u today: on the flat bandit its advantage is exactly 0, so
    Adam raises the log-std by the learning rate at each of 4 steps per 8 x 32
    rollout until the clip at 2."""
    return math.exp(min(2.0, 3e-4 * 4 * (budget // (8 * 32) + 1)))


DROPPED, P07 = "bonus dropped or ent_coef not forwarded", "bonus sign flipped (P07)"
PI_BEST, SIGMA_C = oracles.softmax_optimum(1.0), oracles.max_entropy_sigma()
# Untrained: equal logits; at obs 1 they are not, one update reads 0.10-0.96.
_PI_A = {"untrained": 0.5, DROPPED: 0.976, f"{P07}, better arm": 1.0}
_PI_A[f"{P07}, worse arm"] = 0.0  # a flipped bonus collapses each seed on an arm
_APO_A = {"untrained": 0.5, DROPPED: 0.990, "bonus sign flipped": 0.997}
_AVERAGED = {"entropy averaged over the dimensions (today)": math.sqrt(0.5)}
_CLIP = "sign flipped (P07) or latent-Gaussian entropy (train_PPO.py:343-344): the log-std clip"
_APO_C = {DROPPED: 1.0, "latent-Gaussian entropy (today)": apo_latent_sigma(80_000)}
_APO_C["latent-Gaussian entropy, sign flipped"] = 1 / apo_latent_sigma(80_000)
Q6_QUERIES = {  # a dropped bonus keeps PPO-a climbing to 1 (0.959-0.997)
    "PPO-a": (Query("pi(best)", PI_BEST, _PI_A),),
    "APO-a": (Query("pi(best)", PI_BEST, _APO_A),),
    # Both faults still shrink sigma at 160,000 steps; mu is a sanity check.
    "PPO-b": (
        Query("sigma", 1.0, {"untrained": math.exp(-1.0), DROPPED: 0.130, P07: 0.067}),
        Query("mu", 0.5, {"untrained": 0.0}),
    ),
    "PPO-b2": tuple(
        Query(f"sigma_{i}", 1.0, _AVERAGED | {"untrained": math.exp(-1.0)})
        for i in (1, 2)
    ),
    "PPO-c": (
        Query("sigma_u", SIGMA_C, {DROPPED: math.exp(0.5), _CLIP: math.exp(2.0)}),
    ),
    "APO-c": (Query("sigma_u", SIGMA_C, _APO_C),),
}
Q6_CELLS: dict[str, tuple[str, str, float, int, tuple, dict]] = {
    # agent, bandit, c, budget, tolerances (32/32 of 3000-3031), overrides
    "PPO-a": ("PPO", "a", 1.0, 80_000, (0.02,), {}),  # 40,000 failed: 0.106
    "APO-a": ("APO", "a", 1.0, 20_000, (0.023,), {}),
    "PPO-b": ("PPO", "b", 2.0, 160_000, (0.054, 0.048), {}),  # 80,000: 0.251
    "PPO-b2": ("PPO", "b2", 2.0, 160_000, (), {}),
    "PPO-c": (
        "PPO",
        "c",
        0.1,
        160_000,
        (0.023,),
        {"squash": True, "log_std_init": 0.5},
    ),
    "APO-c": ("APO", "c", 0.1, 80_000, (0.062,), {}),  # half the gap to 1.0
}
Q6_LIVE = {
    "PPO-b2": "(convention, pending the owner's choice) PPO's unsquashed bonus averages the entropy over the action dimensions (pi.entropy().mean(), train_PPO.py:346), so ent_coef acts as c / d, while its squashed path sums them (SAC/utils.py:62, 68); right answer sigma_i 1.0 (joint entropy), today 0.707",
    "APO-c": "APO's squashed bonus maximises the latent Gaussian's entropy (pi.unsquashed_entropy(), train_APO.py:102-105), which grows without bound in sigma, not the executed action's entropy as PPO does (train_PPO.py:341-342); right answer sigma_u 0.874, today exp(min(2, lr x Adam steps)) = 1.456 at 80,000 steps",
}
for cell, (agent, bandit, c, steps, q6_tols, kw) in Q6_CELLS.items():
    build = functools.partial(_q6_agent, agent, bandit, c, **kw)
    reads = runs.readings(build, lambda run: R.per_seed(_q6_read, R.nets(run.state)))
    why, slow = Q6_LIVE.get(cell, ""), cell != "PPO-a"
    case = Case(f"q6-{cell}", Q6_QUERIES[cell], reads, steps, q6_tols, why, slow=slow)
    CASES[case.id] = case


# --- every judged case --------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
