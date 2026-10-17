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

from ajax.agents.recurrent import previous_actions
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig
from ajax.networks.networks import action_value_input, policy_input

from . import agents, envs, oracles, runs
from . import readouts as R
from .verdict import Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "fb306b91cd2f"  # verdict.digest(CASES): every answer, pinned


# --- P0b: act on a cue seen two steps back, forget it at the episode's end ----
# A 3-step episode shows a cue c in {-1, 0, +1} at step 0 (its sign one-hot,
# blanks after) and pays 1 - (1 - 2 c a)^2 at step 2, a clipped to [-1, 1]:
# 1 at a = c / 2, inside the box (the action-range probes own its bounds),
# and 0 at any action after no cue. Read at the last step of a second episode
# after one with a known cue, from a fresh carry (the actor stepped, the
# critic over the sequence at its actions): a(+1) - a(-1) = 1 and V = 1,
# margins: PPO's V is 1 - 4 sigma^2 at its best mean, and a max-entropy
# actor's tanh(mu) sits within 0.013 of 1/2 at any temperature from 0.001 to
# 1 (oracles' quadrature), as PPO's mean does while sigma <= 0.2. After a
# cue-free episode V = 0 and an action gap of exactly 0 (the carry restarts).
# Without memory the gap and V read 0 (a cue-blind action a pays -8 a^2 / 3);
# reversed, the gap reads -1; a carry never restarted acts on the old cue
# (V 1, gap 1). SAC, REDQ and TD3 with other kinds are uncalibrated; ASAC's
# values are differential; APG needs the env's gradients.


def _cue(st: envs.State, a: jax.Array, key: Any) -> tuple:
    paid = 1.0 - (1.0 - 2.0 * st.c * jnp.clip(a.reshape(()), -1.0, 1.0)) ** 2
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
    Query("a(+1) - a(-1)", 1.0, {NO_MEMORY: 0.0, "reversed": -1.0}, margin=True),
    *(Query(f"V({c})", 1.0, {NO_MEMORY: 0.0}, margin=True) for c in ("+1", "-1")),
    *(Query(f"V(none | {c})", 0.0, {LEAK: 1.0}) for c in ("+1", "-1")),
    Query("a(none | +1) - a(none | -1)", 0.0, {LEAK: 1.0}),
)


def _q_input(obs: jax.Array, a: jax.Array, starts: jax.Array) -> jax.Array:
    """A recurrent Q-critic's input over (T, B) steps that took ``a``."""
    return action_value_input(obs, a, previous_actions(a, starts))


def _p0b_read(agent: str) -> Callable:
    def read(n: R.Nets) -> dict:
        a = jnp.clip(R.step_actor(n, CUE_OBS, STARTS, int(agent != "PPO")), -1, 1)
        x = CUE_OBS if agent == "PPO" else _q_input(CUE_OBS, a, STARTS)
        a, v = a[-1, :, 0], R.critic_sequence(n, x, STARTS)[-1]
        out = dict(zip([q.name for q in P0B[1:5]], v, strict=True))
        return out | {P0B[0].name: a[0] - a[1], P0B[5].name: a[2] - a[3]}

    return lambda run: R.per_seed(read, R.nets(run.state))


def _p0b_agent(agent: str, spec: envs.Spec = CUE, kind: str = "gru", **kw: Any) -> Any:
    """32 units; replay batches of 4 windows of 25 steps (64 trained positions)."""
    kw = ({} if agent == "PPO" else {"batch_size": 4}) | kw
    if agent == "SAC":  # actor and temperature updates start with the critic's
        kw |= {"policy_update_start": 100, "alpha_update_start": 100}
    if agent == "REDQ":
        kw |= {"num_critic_updates": 1, "num_critics": 4, "subset_size": 2}
    return agents.make(agent, *spec.make(), memory=MemoryConfig(kind, 32), **kw)


# Budgets: the smallest qualifying on seeds 1000-2031 of 1250-3750 steps
# (TD3 also 5000 and 7500; PPO 10,000, 15,000, 20,000, 30,000 and 40,000);
# certified 32/32 on 3000-3031 (PPO-gru and PPO-lstm 31/32). The critic's V
# after a cue-free episode settles last: it sets SAC's, REDQ's and TD3's
# budgets, 3750, 2500 and 5000 with Ni et al.'s layout. TD3's single seeds
# spike: at 3750 seed 1006 reads it -0.23, at 7500 seed 2026 reads V(-1)
# 0.31. PPO's log-std, one parameter from 0 (std 1), falls steadily: median
# std 0.38-0.41 at 10,000 steps, where V's margin needs sigma under 0.35 and
# 52 to 64 of 64 seeds fail; 0.09-0.11 at 30,000, where none fails; 0.05-0.07
# at 40,000 (one mamba seed fails). With the Dense head PPO had before (std
# 0.37 at the start, about 0.02 by 5000 steps) failures returned at longer
# budgets and its transformer and mamba qualified at no budget of 2000-20,000.
# The action gap reads exactly 0: it takes the 0.02 floor.
P0B_CELLS: dict[tuple[str, str], tuple[int, tuple[float, ...]]] = {
    ("PPO", "gru"): (20_000, (0.4, 0.334, 0.327, 0.048, 0.048, 0.02)),
    ("PPO", "lstm"): (15_000, (0.385, 0.273, 0.259, 0.069, 0.069, 0.02)),
    ("PPO", "transformer"): (30_000, (0.431, 0.354, 0.356, 0.081, 0.081, 0.02)),
    ("PPO", "mamba"): (30_000, (0.429, 0.441, 0.428, 0.089, 0.089, 0.02)),
    ("SAC", "gru"): (3750, (0.469, 0.476, 0.468, 0.09, 0.09, 0.02)),
    ("REDQ", "gru"): (2500, (0.448, 0.461, 0.459, 0.073, 0.073, 0.02)),
    ("TD3", "gru"): (5000, (0.407, 0.453, 0.393, 0.072, 0.072, 0.02)),
}
for (agent, kind), (budget, tol) in P0B_CELLS.items():
    cid = f"p0b-{agent}-{kind}"
    build = functools.partial(_p0b_agent, agent, kind=kind)
    reads = runs.readings(build, _p0b_read(agent))
    CASES[cid] = Case(cid, P0B, reads, budget, tol, slow=agent != "PPO")


def test_p0b_env_pays_the_cue_at_the_last_step() -> None:
    """Resets draw every cue; it shows at step 0 only; steps 0 and 1 pay 0,
    step 2 pays 1 - (1 - 2 cue clip(a, -1, 1))^2 and terminates: 1 at
    a = cue / 2, 0 at any action after no cue."""
    env, params = CUE.make()
    cues = {float(env.reset(jax.random.PRNGKey(k), params)[1].c) for k in range(64)}
    assert cues == {-1.0, 0.0, 1.0}
    acts = (0.5, -0.5, 3.0)  # 3 is clipped to 1
    pays = {1.0: (1.0, -3.0, 0.0), -1.0: (-3.0, 1.0, -8.0), 0.0: (0.0, 0.0, 0.0)}
    for cue in cues:
        for a, last in zip(acts, pays[cue], strict=True):
            st = CUE.fresh(0, cue)
            steps = [(env.get_obs(st), 0.0, False)]
            for _ in range(3):
                obs, st, r, done, _ = env.step_env(R.KEY, st, jnp.array([a]), params)
                steps.append((obs, float(r), bool(done)))
            seen, paid, ends = zip(*steps, strict=False)
            assert paid[1:] == (0.0, 0.0, last) and ends[1:] == (False, False, True)
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
# tanh(mu(o0)) = 0.515. Windows of L = 16 (the default) and 2 catch two
# faults: a target critic whose carry is fed the policy's samples after the
# window's first row reads 1/L of the slope, and an actor loss feeding its
# own fresh actions to the critic's carry credits a0 again through the next
# Q; L = 1 hides both and is the control. Every cell trains 64 positions per
# update (64 // L windows); the actor is judged for SAC and REDQ only:
# ASAC's double count (2/16 + 2 15/16) equals its right slope, 2, and both
# faults only scale TD3's quadratic, so its a0 is 0.5 either way.
# The controls judge each seed as the mean of 4 (Case.mean_of): alone, a
# seed's actor (SAC, REDQ) or critic (ASAC, TD3) spreads too far for twice
# the worst of 64 seeds to fit the 0.1 ceiling at any budget laddered. Each
# tolerance is twice the worst of 64 means, seeds 4000-4127 and 8000-8127,
# laddered at 1000 and 1500 steps (TD3 1500 and 3000). The other cells judge
# single seeds, laddered on 1000-2031 at 1000 and 1500 (TD3 3000 and 6000):
# SAC's and REDQ's critics at L = 16 and TD3's at L = 2 qualify (TARGET);
# the actors, ASAC's critics, TD3's at L = 16 and SAC's and REDQ's at L = 2
# qualify at none (twice the worst error 0.10-0.33) and keep the cap. On
# 3000-3031 all read 32/32 but REDQ's actors (gru 31, lstm 30, transformer
# 27, mamba 29; L = 2 21) and SAC's at L = 2 (28); their seeds 0-15 pass
# (REDQ's L = 2 7 of 8); means of 4 qualify none of those under 30 by 1500.

ALPHA, L = 1.0, 16
BUDGET = {"SAC": 1500, "REDQ": 1500, "ASAC": 1500, "TD3": 6000}
O0 = jnp.array([1.0, 0.0])
ACTS_ON_O0 = policy_input(O0, jnp.zeros(1))  # and on no action before it
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
    """The slope (TD3: curvature) in a0; 1/L of it when the target carry
    is fed the policy's samples."""
    wrong = {BLIND: 0.0}
    if length > 1:
        fed = "target history counterfactual after the first position"
        wrong[fed] = _truth(agent) / length
    return Query(CURVATURE if agent == "TD3" else SLOPE, _truth(agent), wrong)


def _actor(length: int) -> Query:
    """Where the actor's slope in a0 puts tanh(mu(o0)): the critic's, plus
    2 (L - 1) / L when its loss credits a0 again through the next Q."""
    right, fed = 2 * agents.GAMMA, 2 * agents.GAMMA / length
    twice, wrong = 2 * (length - 1) / length, {BLIND: _act(0.0)}
    if length > 1:
        wrong["counterfactual target and double-counting actor"] = _act(fed + twice)
        wrong["actor double counts"] = _act(right + twice)
        wrong["target counterfactual"] = _act(fed)
    return Query(A0, _act(right), wrong)


def _q0(n: R.Nets, acts: jax.Array) -> jax.Array:
    """Q(o0, a) for each action, each from a fresh carry flagged as a start."""
    starts = jnp.ones((1, len(acts)), bool)
    obs = jnp.broadcast_to(O0, (1, len(acts), 2))
    return R.critic_sequence(n, _q_input(obs, acts[None, :, None], starts), starts)[0]


def _a0(n: R.Nets) -> jax.Array:
    p = R.actor_sequence(n, ACTS_ON_O0[None, None], jnp.ones((1, 1), bool))
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


CONTROLS = [("SAC", k) for k in MEMORY_KINDS] + [("REDQ", "gru"), ("TD3", "gru")]
CONTROLS += [("ASAC", "gru")]
CAL = {  # the calibrated cells: budget, tolerances, each seed the mean of, note
    "q2-target-SAC-gru": (1500, (0.095,), 1, "cert 32/32, worst 0.032"),
    "q2-target-SAC-lstm": (1500, (0.096,), 1, "cert 32/32, worst 0.037"),
    "q2-target-SAC-transformer": (1000, (0.081,), 1, "cert 32/32, worst 0.041"),
    "q2-target-SAC-mamba": (1000, (0.091,), 1, "cert 32/32, worst 0.038"),
    "q2-target-REDQ-gru": (1500, (0.085,), 1, "cert 32/32, worst 0.042"),
    "q2-target-REDQ-lstm": (1500, (0.099,), 1, "cert 32/32, worst 0.036"),
    "q2-target-REDQ-transformer": (1500, (0.097,), 1, "cert 32/32, worst 0.037"),
    "q2-target-REDQ-mamba": (1500, (0.087,), 1, "cert 32/32, worst 0.040"),
    "q2-target-TD3-gru-L2": (3000, (0.067,), 1, "cert 32/32, worst 0.033"),
    "q2-control-SAC-gru": (1500, (0.07, 0.08), 4, "cert 32/32, worst 0.021 / 0.040"),
    "q2-control-SAC-lstm": (1500, (0.074, 0.079), 4, "cert 32/32, worst 0.025 / 0.038"),
    "q2-control-SAC-transformer": (
        1000,
        (0.04, 0.09),
        4,
        "cert 32/32, worst 0.023 / 0.052",
    ),
    "q2-control-SAC-mamba": (
        1000,
        (0.039, 0.094),
        4,
        "cert 32/32, worst 0.025 / 0.049",
    ),
    "q2-control-REDQ-gru": (1500, (0.049, 0.089), 4, "cert 32/32, worst 0.036 / 0.032"),
    "q2-control-TD3-gru": (3000, (0.05,), 4, "cert 32/32, worst 0.021"),
    "q2-control-ASAC-gru": (1500, (0.078,), 4, "cert 32/32, worst 0.055"),
}


@functools.cache
def _q2_readings(agent: str, kind: str, length: int) -> Callable:
    """One training per window length, shared by its target and actor parts."""
    build = functools.partial(_q2_agent, agent, kind, length)
    return runs.readings(build, _q2_read(agent))


def _q2_case(part: str, agent: str, kind: str, length: int, *qs: Query) -> None:
    """A cell CAL does not list judges single seeds at BUDGET, to the cap."""
    cell = f"q2-{part}-{agent}-{kind}" + "-L2" * (length == 2)
    budget, tol, m, note = CAL.get(cell, (BUDGET[agent], (), 1, ""))
    reads, slow = _q2_readings(agent, kind, length), (agent, kind) != ("SAC", "gru")
    CASES[cell] = Case(cell, qs, reads, budget, tol, slow=slow, mean_of=m, note=note)


for agent in BUDGET:
    for kind, length in [(k, L) for k in MEMORY_KINDS] + [("gru", 2)]:
        _q2_case("target", agent, kind, length, _critic(agent, length))
        if agent in ("SAC", "REDQ"):
            _q2_case("actor", agent, kind, length, _actor(length))
for agent, kind in CONTROLS:
    actor = (_actor(1),) if agent in ("SAC", "REDQ") else ()
    _q2_case("control", agent, kind, 1, _critic(agent, 1), *actor)


def test_q2_oracle_reproduces_the_verified_answers() -> None:
    """At alpha 1 and L = 16: 0.515 at the right slope 1.24, blind 0, both
    faults 0.670 (1.24 / 16 + 2 15/16 = 1.9525), the actor's alone 0.794,
    the target's alone 0.039; critic answers 1.24, 1.24, 2.0, 0.558."""
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
    x = _q_input(obs[:, None], acts[:, None, None], resets)
    q = R.critic_sequence(n, x, resets)[:, 0]
    np.testing.assert_allclose(q[starts], _q0(n, acts[starts]), atol=1e-5)
    one = jnp.ones((1, 1), bool)
    fresh = R.critic_sequence(
        n, _q_input(obs[1:2, None], acts[1:2, None, None], one), one
    )
    assert abs(q[1] - fresh[0, 0]) > 1e-4
    x = policy_input(obs[:, None], previous_actions(acts[:, None, None], resets))
    a = jnp.clip(R.actor_sequence(n, x, resets).mean().reshape(-1), -1, 1)
    np.testing.assert_allclose(a[starts], _a0(n), atol=1e-5)


# --- every judged case ---------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
