"""Memory: a recurrent agent must act on what an episode showed it and
restart at the episode's end (P0b memory cue), and its critic must value the
actions actually taken (Q2 delayed echo). Ids name the memory kind: CI
shards select by it.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.networks.memory import MEMORY_KINDS, MemoryConfig

from . import agents, envs, oracles, runs
from . import readouts as R
from .verdict import Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "80b1fbb6d47b"  # verdict.digest(CASES): every answer, pinned


# --- P0b: act on a cue seen two steps back, forget it at the episode's end ----
# A 3-step episode shows a cue c in {-1, 0, +1} at step 0 (its sign one-hot,
# blanks after) and pays c clip(a, -1, 1) at step 2. Read at the last step of
# a second episode after one with a known cue, from a fresh carry (the actor
# stepped, the critic over the sequence at its actions): actions +-1, V = 1
# (margins); after a cue-free episode V = 0 and an action gap of exactly 0
# (the carry restarts). Without memory the cue queries read about 0; a carry
# never restarted acts on the old cue (a gap up to 2). Not built (seeds
# 1000-2031): PPO with other kinds and TD3 leave seeds acting alike for both
# cues; SAC and REDQ with other kinds are uncalibrated; ASAC's values are
# differential; APG needs the env's gradients.


def _cue(st: envs.State, a: jax.Array, key: Any) -> tuple:
    paid = st.c * jnp.clip(a.reshape(()), -1.0, 1.0)
    return st.replace(s=st.s + 1), jnp.where(st.s == 2, paid, 0.0), st.s + 1 >= 3


def _draw(key: jax.Array, p: envs.Params) -> tuple:
    return 0, jax.random.randint(key, (), -1, 2).astype(jnp.float32)


def _sign(c: Any) -> jax.Array:
    return jnp.stack([c > 0, c < 0], -1).astype(jnp.float32)


def _shown(st: envs.State) -> jax.Array:
    return jnp.where(st.s == 0, st.c, 0.0)


CUE = envs.Spec("memory_cue", _cue, lambda st: _sign(_shown(st)), _draw, obs_dim=2)
SCALAR_CUE = dataclasses.replace(
    CUE, obs=lambda st: _shown(st).reshape(1), obs_dim=1, obs_box=(-1.0, 1.0)
)
# Each read: its (first, second) episode cues, read as +1, -1, none | +1, none | -1.
PAIRS = jnp.array([(-1.0, 1.0), (1.0, -1.0), (1.0, 0.0), (-1.0, 0.0)])
STARTS = jnp.broadcast_to((jnp.arange(6) % 3 == 0)[:, None], (6, 4))
CUE_OBS = _sign(jnp.repeat(PAIRS.T, 3, 0) * STARTS)  # (T, B, 2)
NO_MEMORY = "no memory (the cue forgotten by the last step)"
LEAK = "carry not restarted (the previous episode's cue read as this one's)"
P0B = (
    Query("a(+1)", 1.0, {NO_MEMORY: 0.0, "reversed": -1.0}, margin=True),
    Query("a(-1)", -1.0, {NO_MEMORY: 0.0, "reversed": 1.0}, margin=True),
    *(Query(f"V({c})", 1.0, {NO_MEMORY: 0.0}, margin=True) for c in ("+1", "-1")),
    *(Query(f"V(none | {c})", 0.0, {LEAK: 1.0}) for c in ("+1", "-1")),
    Query("a(none | +1) - a(none | -1)", 0.0, {LEAK: 2.0}),
)


def _p0b_read(agent: str) -> Callable:
    def read(n: R.Nets) -> dict:
        a = jnp.clip(R.step_actor(n, CUE_OBS, STARTS), -1.0, 1.0)
        x = CUE_OBS if agent == "PPO" else jnp.concatenate([CUE_OBS, a], -1)
        a, v = a[-1, :, 0], R.critic_sequence(n, x, STARTS)[-1]
        out = dict(zip([q.name for q in P0B[:2]], a[:2]))
        out |= dict(zip([q.name for q in P0B[2:6]], v))
        return out | {P0B[6].name: a[2] - a[3]}

    return lambda run: R.per_seed(read, R.nets(run.state))


def _p0b_agent(agent: str, spec: envs.Spec = CUE, **kw: Any) -> Any:
    """A GRU; replay batches of 4 windows of 25 steps (64 trained positions)."""
    kw = ({} if agent == "PPO" else {"batch_size": 4}) | kw
    if agent == "SAC":  # actor and temperature updates start with the critic's
        kw |= {"policy_update_start": 100, "alpha_update_start": 100}
    if agent == "REDQ":
        kw |= {"num_critic_updates": 1, "num_critics": 4, "subset_size": 2}
    return agents.make(agent, *spec.make(), memory=MemoryConfig("gru", 32), **kw)


# PPO's budget is set by the cue-free value, SAC's and REDQ's by 1250 failing;
# all certified 32/32. The action gap reads exactly 0: it takes the 0.02 floor.
P0B_CELLS: dict[str, tuple[int, tuple[float, ...]]] = {
    "PPO": (10_000, (0.5, 0.5, 0.478, 0.495, 0.091, 0.091, 0.02)),
    "SAC": (2500, (0.294, 0.304, 0.285, 0.304, 0.086, 0.086, 0.02)),
    "REDQ": (2500, (0.319, 0.312, 0.312, 0.303, 0.069, 0.069, 0.02)),
}
for agent, (budget, tol) in P0B_CELLS.items():
    cid, build = f"p0b-{agent}-gru", functools.partial(_p0b_agent, agent)
    reads = runs.readings(build, _p0b_read(agent))
    CASES[cid] = Case(cid, P0B, reads, budget, tol, slow=agent != "PPO")


def test_p0b_env_pays_the_cue_at_the_last_step() -> None:
    """Resets draw every cue; it shows at step 0 only; steps 0 and 1 pay 0,
    step 2 pays cue * clip(a, -1, 1) and terminates."""
    env, params = CUE.make()
    cues = {float(env.reset(jax.random.PRNGKey(k), params)[1].c) for k in range(64)}
    assert cues == {-1.0, 0.0, 1.0}
    for cue in cues:
        st = CUE.fresh(0, cue)
        steps = [(env.get_obs(st), 0.0, False)]
        for _ in range(3):
            obs, st, r, done, _ = env.step_env(R.KEY, st, jnp.array([3.0]), params)
            steps.append((obs, float(r), bool(done)))
        seen, paid, ends = zip(*steps)
        assert paid[1:] == (0.0, 0.0, cue) and ends[1:] == (False, False, True)
        want = _sign(jnp.array([cue, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(jnp.stack(seen), want)


def test_p0b_recurrent_sac_trains_on_a_one_feature_observation() -> None:
    runs.train(_p0b_agent("SAC", SCALAR_CUE, buffer_size=1000), (0,), 200)


# --- Q2: a recurrent critic values the actions actually taken -----------------
# A 2-step episode observes a one-hot of its phase; its first action a0 pays
# nothing but is echoed, terminally, at step 1: 2 a0 (TD3: -10 (a0 - 0.5)^2,
# as on the linear echo its deterministic actor saturates at +1). Read from a
# fresh carry flagged as an episode start, as windows present o0: the slope
# Q(o0, 0.5) - Q(o0, -0.5) = 2 gamma (ASAC 2: undiscounted; TD3's curvature
# gamma 10 0.3^2) and, at a pinned alpha of 1, the max-entropy actor's
# tanh(mu(o0)) = 0.515. Windows of L = 16 (the default) and 2 pin two live
# defects; L = 1 removes both and is the control. Every cell trains 64
# positions per update (64 // L windows); the actor is judged for SAC and
# REDQ only: ASAC's double count (2/16 + 2 15/16) equals its right slope, 2,
# and both defects only scale TD3's quadratic, so its a0 is 0.5 either way.
# The controls break the rule's 0.1 ceiling, pending the maintainer: the
# critic's per-seed spread is a few % of its answer whatever the budget,
# learning rate or kind. Each tolerance is twice the worst error on seeds
# 1000-1031 and 2000-2031, within half the gap to a blind critic (0).
# Budgets: SAC's worst errors are smallest at 1500 (1000: 0.118 / 0.083,
# 3000: 0.088 / 0.077); TD3 at 1500 has not settled, and reads about 8% low
# at L = 1 whatever the steps (12000: 0.513) or noise (0.5: 0.503).

ALPHA, L = 1.0, 16
BUDGET = {"SAC": 1500, "REDQ": 1500, "ASAC": 1500, "TD3": 6000}
O0 = jnp.array([1.0, 0.0])
SLOPE = "Q(o0, 0.5) - Q(o0, -0.5)"
CURVATURE = "Q(o0, 0.5) - (Q(o0, 0.2) + Q(o0, 0.8)) / 2"
A0 = "tanh(mu(o0))"
BLIND = "critic blind to the first action (no action in its memory, step-aligned resets, target carry burned short)"


def _echo(payoff: Callable) -> envs.Spec:
    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        first, a = st.s == 0, jnp.clip(a.reshape(()), -1.0, 1.0)
        reward = jnp.where(first, 0.0, payoff(st.c))
        st = st.replace(s=jnp.where(first, 1, 2), c=jnp.where(first, a, st.c))
        return st, reward, ~first

    def obs(st: envs.State) -> jax.Array:
        return envs.one_hot(jnp.minimum(st.s, 1), 2)

    return envs.Spec("delayed_echo", transition, obs, obs_dim=2)


ECHO = _echo(lambda e: 2.0 * e)
QUADRATIC_ECHO = _echo(lambda e: -10.0 * (e - 0.5) ** 2)


def _act(slope: float) -> float:
    return oracles.maxent_mean_action(slope, ALPHA)


def _truth(agent: str) -> float:
    return {"TD3": agents.GAMMA * 10 * 0.3**2, "ASAC": 2.0}.get(agent, 2 * agents.GAMMA)


def _critic(agent: str, length: int) -> Query:
    """The slope (TD3: curvature) in a0; today 1/L of it."""
    wrong = {BLIND: 0.0}
    if length > 1:
        today = "target history counterfactual after the first position (today)"
        wrong[today] = _truth(agent) / length
    return Query(CURVATURE if agent == "TD3" else SLOPE, _truth(agent), wrong)


def _actor(length: int) -> Query:
    """Where the actor's slope in a0 puts tanh(mu(o0)): the critic's, plus
    2 (L - 1) / L when its loss credits a0 again through the next Q."""
    right, today = 2 * agents.GAMMA, 2 * agents.GAMMA / length
    twice, wrong = 2 * (length - 1) / length, {BLIND: _act(0.0)}
    if length > 1:
        both = _act(today + twice)
        wrong["today: counterfactual target and double-counting actor"] = both
        wrong["target fixed, actor still double counts"] = _act(right + twice)
        wrong["actor fixed, target still counterfactual"] = _act(today)
    return Query(A0, _act(right), wrong)


def _q0(n: R.Nets, acts: jax.Array) -> jax.Array:
    """Q(o0, a) for each action, each from a fresh carry flagged as a start."""
    x = jnp.concatenate([jnp.broadcast_to(O0, (len(acts), 2)), acts[:, None]], -1)
    return R.critic_sequence(n, x[None], jnp.ones((1, len(acts)), bool))[0]


def _a0(n: R.Nets) -> jax.Array:
    p = R.actor_sequence(n, O0[None, None], jnp.ones((1, 1), bool))
    return jnp.clip(p.mean().reshape(()), -1.0, 1.0)


def _q2_read(agent: str) -> Callable:
    def read(n: R.Nets) -> dict:
        if agent == "TD3":
            q = _q0(n, jnp.array([0.2, 0.5, 0.8]))
            return {CURVATURE: q[1] - 0.5 * (q[0] + q[2]), A0: _a0(n)}
        q = _q0(n, jnp.array([-0.5, 0.5]))
        return {SLOPE: q[1] - q[0], A0: _a0(n), "alpha": R.alpha(n)}

    def read_run(run: runs.Run) -> dict:
        out = R.per_seed(read, R.nets(run.state))
        if np.abs(out.get("alpha", ALPHA) - ALPHA).max() > 1e-6:
            raise RuntimeError(f"{agent}: alpha left {ALPHA}: {out['alpha']}")
        return out

    return read_run


def _q2_agent(agent: str, kind: str, length: int, batch: int = 0) -> Any:
    """TD3 collects with noise 0.3, so that 0.2 and 0.8 lie in its data."""
    kw: dict[str, Any] = {"alpha_init": ALPHA, "alpha_learning_rate": 0.0}
    if agent == "SAC":  # the actor updates from learning_starts, as the others'
        kw = {"alpha_init": ALPHA, "fixed_alpha": True, "policy_update_start": 100}
    if agent == "REDQ":
        kw |= {"num_critic_updates": 1, "num_critics": 4, "subset_size": 2}
    if agent == "TD3":
        kw = {"exploration_noise": 0.3}
    kw |= {"sequence_length": length, "batch_size": batch or 64 // length}
    spec = QUADRATIC_ECHO if agent == "TD3" else ECHO
    memory = MemoryConfig(kind, 32)
    return agents.make(agent, *spec.make(), memory=memory, buffer_size=20_000, **kw)


TARGET = "the recurrent target critic sees the action taken only at the window's first training position and policy samples after it (recurrent.py:140-146 with SAC/core.py:146-154, recurrent.q_values with train_TD3.py:182-189, train_REDQ.py:204-211, train_ASAC.py:160-167); right 2*gamma = 1.24 (ASAC 2.0, TD3 curvature 0.558), today 1/L of it (L = 16: 0.078, ASAC 0.125, TD3 0.035; L = 2: 0.62, 1.0, 0.279). Calibrate with the fix: at the controls' spread, fixed TD3 and ASAC cells would pass 0.1 only about 73% and 54% of the time"
ACTOR = "the actor loss runs the critic over its own fresh actions as one sequence, so each first action is credited again by the next step's Q through the critic's memory (SAC/train_SAC.py:563-575, REDQ/train_REDQ.py:268-271), on top of the target defect; right tanh(mu(o0)) = 0.515, today 0.670 (L = 16; 0.609 at L = 2). Calibrate with the fix"
CONTROL = {  # tolerances (critic, actor); certification on seeds 3000-3031
    ("SAC", "gru"): ((0.183, 0.124), "cert 32/32, worst 0.089 / 0.062"),
    ("SAC", "lstm"): ((0.224, 0.13), "cert 32/32, worst 0.076 / 0.057"),
    ("SAC", "transformer"): ((0.211, 0.192), "cert 32/32, worst 0.121 / 0.123"),
    ("SAC", "mamba"): ((0.14, 0.18), "cert 32/32, worst 0.068 / 0.115"),
    ("REDQ", "gru"): ((0.186, 0.159), "cert 32/32, worst 0.097 / 0.073"),
    ("TD3", "gru"): ((0.269,), "cert 32/32, worst 0.263 (it reads 8% low)"),
    ("ASAC", "gru"): ((0.323,), "cert 32/32, worst 0.127"),
}


@functools.cache
def _q2_readings(agent: str, kind: str, length: int) -> Callable:
    """One training per window length, shared by its target and actor parts."""
    build = functools.partial(_q2_agent, agent, kind, length)
    return runs.readings(build, _q2_read(agent))


def _q2_case(
    part: str, agent: str, kind: str, length: int, *qs: Query, **kw: Any
) -> None:
    cell = f"q2-{part}-{agent}-{kind}" + "-L2" * (length == 2)
    reads, slow = _q2_readings(agent, kind, length), (agent, kind) != ("SAC", "gru")
    CASES[cell] = Case(cell, qs, reads, BUDGET[agent], slow=slow, **kw)


for agent in BUDGET:
    for kind, length in [(k, L) for k in MEMORY_KINDS] + [("gru", 2)]:
        _q2_case("target", agent, kind, length, _critic(agent, length), defect=TARGET)
        if agent in ("SAC", "REDQ"):
            _q2_case("actor", agent, kind, length, _actor(length), defect=ACTOR)
for (agent, kind), (tol, note) in CONTROL.items():
    queries = (_critic(agent, 1), _actor(1))[: len(tol)]
    _q2_case("control", agent, kind, 1, *queries, tol=tol, ceiling=0.5, note=note)


def test_q2_oracle_reproduces_the_verified_answers() -> None:
    """At alpha 1 and L = 16: 0.515 at the right slope 1.24, blind 0, today
    0.670 (1.24 / 16 + 2 15/16 = 1.9525), the target alone fixed 0.794, the
    actor alone 0.039; critic answers 1.24, 1.24, 2.0, 0.558."""
    actor = _actor(L)
    got = [actor.truth, *actor.wrong.values()]
    np.testing.assert_allclose(got, [0.5153, 0.0, 0.670, 0.794, 0.039], atol=5e-4)
    assert [_truth(a) for a in BUDGET] == pytest.approx([1.24, 1.24, 2.0, 0.558])


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_q2_readout_matches_an_episode_start_inside_a_window(kind: str) -> None:
    """Inside a window with obs-aligned resets an episode's first step reads
    what the fresh-carry readouts read; a second step, whose carry holds a0,
    does not (untrained SAC, seed 0)."""
    run = runs.train(_q2_agent("SAC", kind, L, 16), (0,), 0)
    n = jax.tree.map(lambda x: x[0], R.nets(run.state))
    obs, acts = (
        jnp.tile(jnp.eye(2), (3, 1)),
        jnp.array([0.7, -0.3, -0.9, 0.4, 0.2, 0.8]),
    )
    resets, starts = jnp.array([0, 0, 1, 0, 1, 0], bool)[:, None], np.array([0, 2, 4])
    x = jnp.concatenate([obs, acts[:, None]], -1)[:, None]
    q = R.critic_sequence(n, x, resets)[:, 0]
    np.testing.assert_allclose(q[starts], _q0(n, acts[starts]), atol=1e-5)
    fresh = R.critic_sequence(n, x[1:2], jnp.ones((1, 1), bool))[0, 0]
    assert abs(q[1] - fresh) > 1e-4
    a = jnp.clip(R.actor_sequence(n, obs[:, None], resets).mean().reshape(-1), -1, 1)
    np.testing.assert_allclose(a[starts], _a0(n), atol=1e-5)


# --- every judged case ---------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
