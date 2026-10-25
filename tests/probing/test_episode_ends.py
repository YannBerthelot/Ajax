"""Episode ends: how agents treat terminations, time limits, rollout ends,
parallel envs and reward scale (P6 time-limit twin, P5 reward scale and
average reward, P9 desynchronised envs, Q1 reset cycles). Values come from
each agent's own estimator (``oracles``) or closed forms; the bookkeeping
around every end is read exactly from the same training. The last test
judges every case.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import probing_environments.gymnax_envs as discrete
import pytest
from probing_environments.gymnax_envs import continuous_actions as continuous

from ajax.agents.APO.APO import compute_value_loss
from ajax.environments.create import build_env_from_id
from ajax.environments.interaction import reset, step
from ajax.evaluate import evaluate
from ajax.extensions.base import Extension
from ajax.networks.networks import predict_value

from . import agents, envs, oracles, runs
from . import readouts as R
from .oracles import RIGHT, Rule
from .verdict import STAGE_1, Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "75e5b6c3c7d6"  # verdict.digest(CASES): every answer, pinned


# --- P6: the time-limit twin --------------------------------------------------
# T = 5; obs one-hot {0, A, B}: the type, drawn at reset, shows from step 1;
# reward 0 then 1. B terminates on step T, A only hits the limit T; gymnax
# flags B's last step terminated and truncated, brax truncated = 1 - done.
# gamma 0.5. PPO and PQN run 4-step rollouts (coprime with T), so rollout
# ends fall on every phase; the time limit doubles as the eval horizon.

T, G6 = 5, 0.5


def _twin(st: envs.State, a: jax.Array, key: Any) -> tuple:
    b_ends = (st.c == 1) & (st.s + 1 >= T)
    return st.replace(s=st.s + 1), jnp.where(st.s == 0, 0.0, 1.0), b_ends


def _twin_obs(st: envs.State) -> jax.Array:
    return envs.one_hot(jnp.where(st.s == 0, 0, 1 + st.c.astype(jnp.int32)), 3)


def _twin_reset(key: jax.Array, p: envs.Params) -> tuple:
    return 0, jax.random.bernoulli(key)


TWIN = {
    n: envs.Spec("twin", _twin, _twin_obs, _twin_reset, 3, actions=n, limit=T)
    for n in (0, 2)
}
BRAX_TWIN = envs.register_brax(TWIN[0], "ajax_probe_time_limit_twin")
LEARNERS = {"1-step": (0.0, 1), "PPO": (0.95, 4), "PQN": (0.65, 4)}
FAULTS = {
    "1-step": {
        "OR flags (today's rule)": Rule(mask="or"),
        "truncation precedence": Rule(mask="precedence"),
        "split flags + reset-obs bootstrap": Rule(next="reset"),
        "precedence + reset-obs bootstrap": Rule(mask="precedence", next="reset"),
        "flags swapped": Rule(swap=True),
    },
    "PPO": {
        "OR flags": Rule(mask="or"),
        "flags swapped (B03)": Rule(swap=True),
        "truncation precedence": Rule(mask="precedence"),
        "reset-obs bootstrap": Rule(next="reset"),
        "rollout end treated as terminal": Rule(rollout_end="terminal"),
        "last bootstrap of the rollout zeroed": Rule(rollout_end="zero"),
        "bootstrap from the transition's own obs (P09)": Rule(next="self"),
    },
    "PQN": {  # its right rule is purejaxql's (LEARNER_RULE)
        "time limits bootstrapped": Rule(off_by_one=True),
        "ends ignored (Q03)": Rule(mask="none", cut=False, off_by_one=True),
        "rollout end treated as terminal": Rule(
            mask="or", rollout_end="terminal", off_by_one=True
        ),
    },
}
# PQN's estimator is purejaxql's: a time limit ends an episode as a
# termination does (gymnax's done flag), so V(A) = V(B), and each rollout's
# step T - 2 reads step T - 1's bootstrap.
LEARNER_RULE = {"PQN": Rule(mask="or", off_by_one=True)}


def twin_values(learner: str, encoding: str, rule: Rule = RIGHT) -> np.ndarray:
    """Exact (V(0), V(A), V(B)) of the learner's estimator on one backend's flags."""
    episodes = []
    for kind in (0, 1):  # A, B
        rows = []
        for j in range(T):
            last, x = j == T - 1, 1 + kind
            term = int(last and kind == 1)
            trunc = int(last and (encoding == "gymnax" or not term))
            rows.append((x if j else 0, float(j > 0), x, 0 if last else x, term, trunc))
        episodes.append((0.5, rows))
    lam, n_steps = LEARNERS[learner]
    return oracles.fixed_point(oracles.rollouts(episodes, n_steps), 3, G6, lam, rule)


def twin_queries(learner: str, encoding: str) -> tuple[Query, ...]:
    """A value gates only the faults that move it by at least 0.15: below
    that, seed noise would set a tolerance no budget here reaches (V(0) is
    gated only for the on-policy rollout-end faults and P09)."""
    truth, out = twin_values(learner, encoding, LEARNER_RULE.get(learner, RIGHT)), []
    faulty = {k: twin_values(learner, encoding, r) for k, r in FAULTS[learner].items()}
    for i, name in enumerate(("V(0)", "V(A)", "V(B)")):
        wrong = {k: v[i] for k, v in faulty.items() if abs(v[i] - truth[i]) >= 0.15}
        if wrong:
            out.append(Query(name, truth[i], wrong))
    return tuple(out)


def _row_counts(obs: np.ndarray, term: np.ndarray, trunc: np.ndarray) -> dict:
    """Per seed, the rows learned from, by observation and flag."""
    seeds = obs.shape[0]
    kind = obs.reshape(seeds, -1, 3).argmax(-1)
    term, trunc = (x.reshape(seeds, -1) > 0.5 for x in (term, trunc))
    end = term | trunc
    out = {"rows": np.full(seeds, kind.shape[1]), "terminated": term.sum(1)}
    out |= {"truncated": trunc.sum(1), "ends": end.sum(1)}
    for i, x in enumerate("0AB"):
        for flag, mask in (("ends", end), ("terminated", term), ("truncated", trunc)):
            out[f"{flag} {x}"] = (mask & (kind == i)).sum(1)
    return out


def _twin_agent(name: str, backend: str) -> Any:
    lam, n_steps = LEARNERS.get(name, (0.0, 1))
    kw: dict[str, Any] = {"gamma": G6}
    if name == "PQN":
        kw |= {"n_envs": 1, "n_steps": n_steps, "q_lambda": lam}
        kw["expose_recent_rollout"] = True
    elif name == "PPO":
        # One 4-row minibatch per epoch; no global-norm clipping, which would
        # weigh each batch by its own norm and move the critic's fixed point.
        kw |= {"n_steps": n_steps, "batch_size": n_steps, "gae_lambda": lam}
        kw |= {"max_grad_norm": None, "expose_recent_rollout": True}
    elif name not in ("TD3", "DQN"):
        # The rewards ignore the action: from 1e-3 the soft term alpha H
        # stays at a few thousandths.
        kw["alpha_init"] = 1e-3
    if name == "AVG":  # at its own rates the last updates pull V(B) 0.05 low
        kw |= {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3}
    if backend == "gymnax":
        return agents.make(name, *TWIN[2 if name in ("DQN", "PQN") else 0].make(), **kw)
    if name == "PPO":
        return agents.make(name, BRAX_TWIN, None, episode_length=T, **kw)
    env = build_env_from_id(BRAX_TWIN, n_envs=1, episode_length=T)[0]
    return agents.make(name, env, **kw)


def _twin_read(name: str, run: runs.Run) -> dict:
    def one(n: R.Nets) -> dict:
        return {
            f"V({x})": R.value(name, n, envs.one_hot(i, 3)) for i, x in enumerate("0AB")
        }

    return R.per_seed(one, R.nets(run.state, stats=R.env_stats(run)))


def _twin_run(name: str, backend: str) -> Any:
    """Values, rows learned from, train return and the agent's own
    evaluation, from one training. PPO and PQN resume 3 T - 1 times by one
    rollout and average the values over the phases they land on: every
    episode lasts T, so seeds share the recency bias of their last updates."""

    def read(run: runs.Run) -> dict:
        cycle, values, rollouts, phases = name in ("PPO", "PQN"), [], [], []
        rows: dict = {}  # AVG keeps none
        for k in range(3 * T if cycle else 1):
            run = runs.resume(run, LEARNERS[name][1]) if k else run
            values.append(_twin_read(name, run))
            if cycle:
                flags = R.rollout_rows(run.state, ("obs", "terminated", "truncated"))
                rollouts.append(list(flags.values()))
                steps = np.asarray(run.state.collector_state.timestep).reshape(-1)
                if (steps != steps[0]).any():
                    raise RuntimeError(f"seeds at different timesteps {steps}")
                phases.append(int(steps[0]) % T)
        if cycle:
            rows = _row_counts(
                *(np.concatenate(z, 1) for z in zip(*rollouts, strict=False))
            )
        elif name != "AVG":
            replay = R.replay_rows(run.state, ("obs", "terminated", "truncated"))
            rows = _row_counts(*replay.values())
        args = run.agent.env_args

        def own_eval(actor: Any, key: jax.Array) -> tuple:
            kw = {"actor_state": actor, "num_episodes": 10, "rng": key}
            out = evaluate(args.env, env_params=args.env_params, **kw)
            return out[0], out[4]

        ret, length = jax.vmap(own_eval)(run.state.actor_state, run.state.eval_rng)
        out = {k: np.mean([v[k] for v in values], 0) for k in values[0]}
        out |= {f"rows {k}": v for k, v in rows.items()} | {"phases": np.array(phases)}
        out["train return"] = run.state.collector_state.episodic_mean_return
        return out | {"eval return": ret, "eval length": length}

    return runs.readings(lambda: _twin_agent(name, backend), read)


ONE_STEP = {"SAC": 5000, "REDQ": 1000, "TD3": 5000, "DQN": 5000}
# Calibrated on seeds 1000-1031 and 2000-2031 (5000 fails for each: worst
# 0.057 to 0.075); certified on 3000-3031, 32/32 for each.
TWIN_CAL = {
    ("gymnax", "PPO"): (10_000, (0.071, 0.056, 0.099)),
    ("gymnax", "PQN"): (20_000, (0.02, 0.034, 0.042)),  # 10,000: V(B) 0.055 off
    ("brax", "PPO"): (5000, (0.068, 0.087, 0.089)),
    ("gymnax", "AVG"): (2500, (0.051, 0.055)),  # 1250: 0.19 off; cert 32/32
    ("brax", "REDQ"): (8000, (0.052, 0.095)),  # 4000: 0.072 off; cert 32/32
}
P6_CELLS = [("gymnax", n) for n in (*ONE_STEP, "PPO", "PQN", "AVG")]
P6_CELLS += [("brax", n) for n in ("SAC", "PPO", "REDQ", "TD3")]
P6: dict[str, Case] = {}
for backend, name in P6_CELLS:
    budget, tol = TWIN_CAL.get((backend, name), (ONE_STEP.get(name, 0), ()))
    slow = name == "REDQ" or (backend == "brax" and name not in ("SAC", "PPO"))
    queries = twin_queries(name if name in ("PPO", "PQN") else "1-step", backend)
    case = Case(
        f"p6-{name}-{backend}",
        queries,
        _twin_run(name, backend),
        budget,
        tol,
        slow=slow,
    )
    P6[case.id] = case
CASES |= P6
P6_ROWS = {k: c for k, c in P6.items() if "AVG" not in k}  # AVG keeps no rows


def test_p6_oracle_reproduces_the_closed_forms() -> None:
    """1-step TD: V(A) = 1 / (1 - gamma), V(B) = 1 / (1 - gamma (T - 2) /
    (T - 1)), V(0) = gamma (V(A) + V(B)) / 2, on both backends' flags."""
    v_a, v_b = 1 / (1 - G6), 1 / (1 - G6 * (T - 2) / (T - 1))
    for encoding in ("gymnax", "brax"):
        want = (G6 * (v_a + v_b) / 2, v_a, v_b)
        np.testing.assert_allclose(twin_values("1-step", encoding), want, atol=1e-9)


@pytest.mark.parametrize("backend", ["gymnax", "brax"])
def test_p6_twin_ends_every_episode_on_step_T_with_its_backends_flags(
    backend: str,
) -> None:
    """Through Ajax's own reset and step: observation 0 first, then the
    type; the end on step T only; B terminated there; truncated on every end
    (gymnax) or on A ends only (brax); both types occur."""
    n, key = 64, jax.random.PRNGKey(0)
    if backend == "gymnax":
        env, params = TWIN[2].make()
        obs, state = reset(jax.random.split(key, n), env, "gymnax", params)
    else:
        env, params = build_env_from_id(BRAX_TWIN, n_envs=n, episode_length=T)[0], None
        obs, state = reset(key, env, "brax", params)
    rows = []
    for _ in range(2 * T):
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, n) if backend == "gymnax" else sub
        a = jnp.zeros((n,), jnp.int32) if backend == "gymnax" else jnp.zeros((n, 1))
        nxt, state, _, term_t, trunc_t, _ = step(keys, state, a, env, backend, params)
        kind_t = np.asarray(obs).reshape(n, 3).argmax(-1)
        rows.append((kind_t, np.asarray(term_t) > 0.5, np.asarray(trunc_t) > 0.5))
        obs = nxt
    kind, term, trunc = (
        np.stack(z).reshape(2 * T, n) for z in zip(*rows, strict=False)
    )
    first, last = (np.arange(2 * T) % T == i for i in (0, T - 1))
    assert (kind[first] == 0).all() and (kind[~first] > 0).all()
    assert (term | trunc)[last].all() and not (term | trunc)[~last].any()
    is_b = kind[last] == 2
    assert 0 < is_b.mean() < 1 and (term[last] == is_b).all()
    assert (trunc[last] == (True if backend == "gymnax" else ~is_b)).all()


@pytest.mark.parametrize("case", params(P6_ROWS, lambda c: ""))
def test_p6_time_limit_fires_and_flags_every_end(case: Case) -> None:
    """On the value run: train return 4; the ends among the rows learned
    from (replay; for PPO and PQN the 15 rollouts read, 3 at each phase)
    number rows / 5 (or 12), none on observation 0, terminated exactly on B
    ends, truncated on every end (gymnax) or exactly on A ends (brax)."""
    r = case.readings(STAGE_1, case.budget)
    rows = {k.removeprefix("rows "): v for k, v in r.items() if k.startswith("rows ")}
    np.testing.assert_array_equal(r["train return"].reshape(-1), T - 1)
    if len(r["phases"]):
        assert sorted(r["phases"]) == sorted(list(range(T)) * 3), r["phases"]
    ends = 12 if len(r["phases"]) else rows["rows"] // T  # 3 phase cycles of rollouts
    np.testing.assert_array_equal(rows["ends"], ends)
    np.testing.assert_array_equal(rows["ends 0"], 0)
    np.testing.assert_array_equal(rows["terminated"], rows["ends B"])
    np.testing.assert_array_equal(rows["terminated B"], rows["ends B"])
    want = rows["ends"] if case.id.endswith("gymnax") else rows["ends A"]
    np.testing.assert_array_equal(rows["truncated"], want)


@pytest.mark.parametrize("case", params(P6, lambda c: ""))
def test_p6_evaluation_runs_the_training_time_limit(case: Case) -> None:
    """The agent's own evaluation (training env rebuilt, its eval key, 10
    episodes) reads return 4 and length 5 on every seed."""
    r = case.readings(STAGE_1, case.budget)
    np.testing.assert_array_equal(r["eval return"].reshape(-1), T - 1)
    np.testing.assert_array_equal(r["eval length"].reshape(-1), T)


# --- P5: reward scale and the average-reward estimators -----------------------
# Every cell at reward_scale s = 2. (i) The 1-step value probe, observed at
# 1 (envs.package): Q(1) = s, no bootstrap. (ii) Every on_target batch
# carrying rewards states the scale. (iii) TwoStepCoin: obs 0, two-step
# episodes ending by termination, each step paying 2 Bernoulli(1/2),
# actions ignored. A moving average of independent draws steps by
# ``ema_step_rms``; a trace of the last updates reads each step as log2 of
# its ratio to that (a swapped weight: +3.6 or more; never updated: -inf),
# which the converged levels cannot see.

S5, G5 = 2.0, agents.GAMMA


@dataclasses.dataclass(frozen=True)
class IdentityTarget(Extension):
    """REDQ with an extension: its target once took a separate path, which
    scaled the reward at another line; one path now, kept checked."""

    name = "identity_target"

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target


def _scaled(cell: str, extensions: tuple = ()) -> Any:
    """The probe preset at reward_scale 2 on the package's value probe."""
    name = cell.removesuffix("+on_target")
    kw: dict[str, Any] = {"reward_scale": S5}
    if name == "REDQ":
        kw["num_critic_updates"] = 1
    if name == "AVG":
        kw |= {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3}
    if cell.endswith("+on_target"):
        extensions = (IdentityTarget(),)
    env = (discrete if name in ("DQN", "PQN") else continuous).ValueLossOrOptimizerEnv
    return agents.make(name, *envs.package(env), extensions=extensions, **kw)


VALUE_Q = Query(
    "Q(1)",
    S5,
    {
        "reward_scale ignored (wrapper default 1.0)": 1.0,
        "reward scaled twice (S13)": S5**2,
        "a 5.0 default leaks (SACConfig, REDQConfig, AVG internals)": 5.0,
        "done mask ignored": S5 / (1.0 - G5),
        "done mask and reward_scale ignored": 1.0 / (1.0 - G5),
    },
)
# Calibrated on 1000-1031 + 2000-2031, worst error 0.0059 or less (AVG, at
# its reference initialisation, 0.0091 at 2500); 3000-3031 within 0.009.
P5_VALUE = {"SAC": 625, "REDQ": 625, "REDQ+on_target": 625, "TD3": 625}
P5_VALUE |= {"AVG": 2500, "DQN": 312, "PQN": 80_000}  # DQN: tol 0.063


def _p5_read(cell: str, run: runs.Run) -> dict:
    """Q(1) through the env-side normaliser (AVG's): at observation 0 the
    raw and the normalised input coincided."""
    read = functools.partial(R.value, cell.removesuffix("+on_target"))
    n = R.nets(run.state, stats=R.env_stats(run))
    return R.per_seed(lambda n: {"Q(1)": read(n, 1.0)}, n)


for cell, budget in P5_VALUE.items():
    reads = runs.readings(
        functools.partial(_scaled, cell), functools.partial(_p5_read, cell)
    )
    tol5 = (0.063 if cell == "DQN" else 0.02,)
    CASES[f"p5-value-{cell}"] = Case(
        f"p5-value-{cell}", (VALUE_Q,), reads, budget, tol5
    )

SEEN: dict[str, list] = {}


@dataclasses.dataclass(frozen=True)
class RecordTargetBatch(Extension):
    """Records each on_target batch's keys and stated scale at trace time."""

    label: str
    name = "record_target_batch"

    def on_target(self, agent_state, ext_state, batch, target, context):
        SEEN.setdefault(self.label, []).append(
            (frozenset(batch), batch.get("reward_scale"))
        )
        return target


@pytest.mark.parametrize(
    "cell",
    ["SAC", "DQN", "PQN", "TD3", "REDQ", "AVG", "ASAC"],
)
def test_p5_on_target_batch_states_its_reward_scale(cell: str) -> None:
    """Rewards reach every on_target batch unscaled, so the batch states the
    scale the target applies (2); traced on one seed, never run. PPO and APO
    have no scale."""
    label = f"{cell}-{len(SEEN)}"
    ext = (RecordTargetBatch(label),)
    agent = _asac(ext) if cell == "ASAC" else _scaled(cell, ext)
    jax.eval_shape(lambda: agent.train(seed=[0], n_timesteps=256, logging_config=None))
    scales = [s for keys, s in SEEN.pop(label, []) if "rewards" in keys]
    if not scales:
        raise RuntimeError(f"{cell} folded no on_target batch carrying rewards")
    assert all(s == S5 for s in scales), f"{cell}: reward_scale {scales}, want 2.0"


def _coin(st: envs.State, a: jax.Array, key: Any) -> tuple:
    return st, 2.0 * jax.random.bernoulli(key), st.time >= 1


COIN = {
    n: envs.Spec("coin", _coin, lambda st: jnp.zeros(1), obs_box=(-1.0, 1.0), actions=n)
    for n in (0, 2)
}
TRACED: dict[str, Callable] = {
    "theta": lambda s: s.theta,
    "penalty": lambda s: s.episode_termination_penalty,
    "Q(0,0)": lambda s: R.critic(R.nets(s), 0.0, 0.0),  # ASAC's origin shift
    "average_reward": lambda s: s.average_reward,
}


@dataclasses.dataclass(frozen=True)
class Trace(Extension):
    """The last ``length`` values of ``TRACED`` fields after every update (a
    ring buffer): ASAC folds post_update per update, APO per rollout."""

    fields: tuple[str, ...]
    length: int
    name = "p5_trace"

    def init_state(self, agent_state, rng):
        values = jnp.zeros((len(self.fields), self.length), jnp.float32)
        return {"values": values, "count": jnp.zeros((), jnp.int32)}

    def post_update(self, agent_state, ext_state, context):
        now = [jnp.asarray(TRACED[f](agent_state), jnp.float32) for f in self.fields]
        slot = ext_state["count"] % self.length
        values = (
            ext_state["values"].at[:, slot].set(jnp.stack([x.reshape(()) for x in now]))
        )
        return agent_state, {"values": values, "count": ext_state["count"] + 1}

    def unroll(self, ext_state: Any) -> dict[str, np.ndarray]:
        """Each field's trace per seed, oldest first: (seeds, length)."""
        values, counts = np.asarray(ext_state["values"]), np.asarray(ext_state["count"])
        if (counts < self.length).any():
            raise RuntimeError(
                f"the trace saw {counts.min()} updates, not {self.length}"
            )
        rolled = np.stack(
            [
                np.roll(v, -(c % self.length), -1)
                for v, c in zip(values, counts, strict=False)
            ]
        )
        return {f: rolled[:, i] for i, f in enumerate(self.fields)}


def _step_query(what: str, right: float, swapped: float, per: str = "update") -> Query:
    wrong = {f"{what} moving-average weights swapped": math.log2(swapped / right)}
    wrong[f"{what} never updated"] = -math.inf
    return Query(f"log2(rms one-{per} change of {what} / {right:.5f})", 0.0, wrong)


def _steps(trace: np.ndarray, right: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log2(np.sqrt(np.mean(np.diff(trace, axis=-1) ** 2, -1)) / right)


# ASAC (p_0 20, tau 0.005, batch 256), half its rows terminating: the penalty
# settles at P = p_0 E[max(r_0, r_1)] = 30, the coins of the batch's rows 0
# and 1 (darer/ASAC.py:187 reads them); theta at what update_theta averages,
# the reward without its penalty (:191), s E[r] + alpha H = 2 + alpha H in
# the target's units. A termination drops the next value and is charged P,
# the origin shift outside the mask: (1 + 1/2) Q(0, .) = -(s P + alpha H) / 2,
# so Q(0, 0) + (alpha H + s P) / 3 reads 0, and theta off by d in the target
# moves it by -2 d / 3 (THETA owns small d).
P0, TAU, PEN = 20.0, 0.005, 30.0
R_PEN = 1.0 - PEN / 2  # mean penalised reward per row, before the scale
THETA = Query(
    "theta - alpha*H",
    S5,
    {
        "theta averages unscaled rewards": 1.0,
        "theta averages the penalised rewards": S5 * R_PEN,
        "theta never updated": 0.0,
    },
)
PENALTY_WRONG = {"p_0 Config default (10) leaks": 0.75, "penalty never updated": 0.0}
PENALTY_WRONG |= {"the batch's mean non-terminal reward": 0.5, "B06: row 0's": 1.0}
PENALTY = Query("penalty / p_0", 1.5, PENALTY_WRONG)
SP = S5 * PEN  # the penalty in the target's units
CRITIC = Query(
    "Q(0,0) + (alpha*H + s*penalty)/3",
    0.0,
    {
        "theta sign flipped in the target (A01)": 4 * S5 / 3,
        "theta penalised, or the penalty never charged": SP / 3,
        "penalty sign flipped": 2 * SP / 3,
        "terminated read as truncated (B06)": S5 * P0 / 3,
        "no done mask, or the origin shift inside it": -SP / 6,
        "reward scaled twice in the target (S13)": (S5**2 - S5) * (2 - PEN) / 3,
    },
)
PEN_SD = P0 * math.sqrt(0.75)  # one batch's p_0 max(r_0, r_1)
THETA_RMS = oracles.ema_step_rms(TAU, S5 / 16)  # one batch's s mean(r): sd 1 / 16
PEN_RMS = oracles.ema_step_rms(TAU, PEN_SD)
THETA_STEP = _step_query("theta", THETA_RMS, oracles.ema_step_rms(1 - TAU, S5 / 16))
PEN_STEP = _step_query("the penalty", PEN_RMS, oracles.ema_step_rms(1 - TAU, PEN_SD))
ASAC_TRACE = Trace(("theta", "penalty", "Q(0,0)"), 1500)


def _asac(extensions: tuple) -> Any:
    """ASAC's own batch, tau and p_0, learning from step 1000 (not 1e4)."""
    kw: dict[str, Any] = {"learning_starts": 1000, "buffer_size": 100_000}
    kw |= {"batch_size": 256, "reward_scale": S5, "extensions": extensions}
    return agents.make("ASAC", *COIN[0].make(), **kw)


def _entropy0(actor: Any, key: jax.Array, n: int = 4096) -> jax.Array:
    """-E[sum log pi(a | 0)] over ``n`` actions the policy samples at 0."""
    log_p = actor.apply_fn(actor.params, jnp.zeros((n, 1))).sample_and_log_prob(
        seed=key
    )[1]
    return -log_p.reshape(n, -1).sum(-1).mean()


def _asac_read(run: runs.Run) -> dict:
    """theta - alpha*H at the end; Q(0, 0) and the penalty it reads over the
    last 1000 updates (the critic wanders by 0.4 per update); the penalty's
    level over the trace."""
    keys = jax.random.split(R.KEY, len(run.seeds))
    nets = R.nets(run.state, keys)
    a_h = R.per_seed(lambda n: {"": R.alpha(n) * _entropy0(n.actor, n.extra)}, nets)[""]
    tr = ASAC_TRACE.unroll(run.state.ext_state[0])
    q00, pen = (tr[k][:, -1000:].mean(-1) for k in ("Q(0,0)", "penalty"))
    out = {THETA.name: np.asarray(run.state.theta) - a_h}
    out |= {CRITIC.name: q00 + (a_h + S5 * pen) / 3}
    out |= {PENALTY.name: tr["penalty"].mean(-1) / P0}
    return (
        out
        | {THETA_STEP.name: _steps(tr["theta"], THETA_RMS)}
        | {PEN_STEP.name: _steps(tr["penalty"], PEN_RMS)}
    )


# Calibrated on 1000-1031 + 2000-2031 at 3000 steps, certified 32/32 on
# 3000-3031 (worst errors 0.19; 0.075 and 0.26; 0.41 and 0.064). Not 0.1:
# theta carries the buffer's coin mean and the critic the two-row penalty's
# noise, which a penalty read over the critic's own window halves.
ASAC_RUN = runs.readings(lambda: _asac((ASAC_TRACE,)), _asac_read)
for part, queries, tols in (
    ("theta", (THETA,), (0.463,)),
    ("levels", (PENALTY, CRITIC), (0.195, 0.695)),
    ("steps", (THETA_STEP, PEN_STEP), (0.989, 0.152)),
):
    CASES[f"p5-ASAC-{part}"] = Case(
        f"p5-ASAC-{part}", queries, ASAC_RUN, 3000, tols, ceiling=math.inf
    )

# APO (alpha 0.1, 32-step rollouts): average_reward 1, stepping by 0.0181
# per rollout. Calibrated at 8000 steps (250 rollouts): worst errors 0.030
# and 0.184, certified 0.034 and 0.296; both action spaces read alike.
APO_TRACE, APO_SD = Trace(("average_reward",), 200), 1.0 / math.sqrt(32)
APO_RMS = oracles.ema_step_rms(0.1, APO_SD)
LEVEL = Query(
    "average_reward (mean over the last 200 rollouts)", 1.0, {"never updated": 0.0}
)
APO_STEP = _step_query(
    "average_reward", APO_RMS, oracles.ema_step_rms(0.9, APO_SD), "rollout"
)


def _apo_read(run: runs.Run) -> dict:
    trace = APO_TRACE.unroll(run.state.ext_state[0])["average_reward"]
    return {LEVEL.name: trace.mean(-1), APO_STEP.name: _steps(trace, APO_RMS)}


for cid, n in (("p5-APO-box", 0), ("p5-APO-discrete", 2)):
    build = functools.partial(
        agents.make, "APO", *COIN[n].make(), extensions=(APO_TRACE,)
    )
    reads = runs.readings(build, _apo_read)
    CASES[cid] = Case(
        cid, (LEVEL, APO_STEP), reads, 6400, (0.09, 0.504), ceiling=math.inf
    )


# --- P9: the desynchronised signed hazard chain -------------------------------
# 4 envs (no other axis here is 4 long), each drawing c = +-1 at reset from
# its own key: s0 pays 0, s1 pays c and ends with probability 1/2 (its own
# step key); gamma 0.8, so V(s1) = c / 0.6 and V(s0) = 0.8 V(s1). The envs
# run out of step, so an input taken from env e - 1's column moves the odd
# parts (V(s, +1) - V(s, -1)) / 2, where entropy, ensemble or offset biases
# cancel. APO, undiscounted, reads c paid for 2 steps on average: (2, 2).
# ASAC did not calibrate: its odd parts wander 0.06 (pooled worst at 80,000
# and 120,000 steps, with a whole-run buffer and tau 0.001).

G9, N9 = 0.8, 4
TRUTH9 = (G9 / 0.6, 1.0 / 0.6)


def _hazard(st: envs.State, a: jax.Array, key: Any) -> tuple:
    at_s1 = st.s == 1
    end = at_s1 & (jax.random.uniform(key) < 0.5)
    return st.replace(s=jnp.ones_like(st.s)), jnp.where(at_s1, st.c, 0.0), end


def _sign(key: jax.Array, p: envs.Params) -> tuple:
    return 0, jnp.where(jax.random.bernoulli(key), 1.0, -1.0)


def _hazard_obs(st: envs.State) -> jax.Array:
    return jnp.stack([st.s.astype(jnp.float32), st.c])


HAZARD = {
    n: envs.Spec("hazard", _hazard, _hazard_obs, _sign, 2, (-1.0, 1.0), n)
    for n in (0, 2)
}
FAULTS9 = {
    "A": "next obs and done from another env",
    "B": "env 0's done applied to every env",
    "C": "done from another env",
    "D": "reward from another env",
    "P10": "next value read at the current obs",
    "untrained": "untrained networks",
}
D0, UNTRAINED = {"D": (0.0, 0.0)}, {"untrained": (0.0, 0.0)}
TD9 = {"A": (0.0, 1.0), "B": (0.854, 1.427), "C": (0.723, 1.362)} | D0
# Estimator, its oracle readings under each fault (seed 0, so within about
# 0.005 of exact), budget, tolerances and overrides. Calibrated on 1000-1031
# + 2000-2031, certified 32/32 on 3000-3031 each. The spread must stay near
# 0.015: large buffers and rollouts, and the target networks read (the same
# fixed point, the SGD jitter averaged out).
_P9_PPO = {"n_steps": 4096, "num_minibatches": 2, "gae_lambda": 0.5, "batch_size": 64}
P9_CELLS: dict[str, tuple] = {
    "SAC": (("td", 1, 0.0), TD9, 20_000, (0.046, 0.062), {"buffer_size": 16384}),
    "DQN": (
        ("td", 1, 0.0),
        TD9,
        10_000,
        (0.069, 0.075),
        {"buffer_size": 8192, "tau": 0.005, "target_update_interval": 1},
    ),
    "PPO": (  # 2 minibatches split the env axis, GAE recomputed per minibatch
        ("gae", 4096, 0.5),
        {"A": (0.0, 1.0), "B": (1.102, 1.940), "C": (1.028, 2.066), "P10": (0.0, 1.663)}
        | D0,
        320_000,  # 160,000 failed: 0.128
        (0.045, 0.06),
        _P9_PPO | {"unroll_length": 4096},  # brax's time geometry, opted into
    ),
    "PQN": (
        ("q_lambda", 512, 0.65),
        {"A": (0.394, 1.198), "B": (0.902, 1.562), "C": (0.766, 1.527)} | D0,
        320_000,  # with RAdam; 80,000 and 160,000 do not qualify
        (0.064, 0.078),
        {"learning_rate": 2.5e-4, "n_steps": 512, "num_minibatches": 2, "n_epochs": 3},
    ),
}
# The cells below normalise observations (AVG always does). REDQ makes 2
# critic updates a step, not 20; AVG learns from the final obs; APO runs
# PPO's rollouts, with its own global-norm clipping (0.5).
_ON9 = {"normalize_observations": True, "buffer_size": 16384}
_REDQ9 = _ON9 | {"num_critic_updates": 2}
P9_CELLS |= {
    "TD3": (("td", 1, 0.0), TD9, 30_000, (0.058, 0.086), _ON9),
    "REDQ": (("td", 1, 0.0), TD9, 10_000, (0.069, 0.093), _REDQ9),
    "AVG": (
        ("gae", 1, 0.0),
        {"A": (0.0, 1.0), "B": (1.195, 1.997), "C": (1.139, 2.144)} | D0,
        # C planted: (1.080, 2.045) at 80,000 steps, still climbing; C by 320,000
        80_000,  # critic rate 2e-4: worst 0.04 on 8 seeds
        (0.075, 0.082),
        {"actor_learning_rate": 1e-3, "critic_learning_rate": 5e-5},
    ),
    "APO": (
        ("apo", 4096, 0.5),
        {"A": (0.0, 1.0), "B": (1.758, 2.507), "C": (1.698, 2.777)} | D0,
        240_000,  # 160,000 failed: 0.137
        (0.073, 0.078),
        _P9_PPO | {"normalize_observations": True},
    ),
}


def _p9_step(state: tuple) -> tuple:
    rng, s, c = state
    obs, reward = (2 * s + (c > 0)).astype(int), np.where(s == 1, c, 0.0)
    done = (s == 1) & (rng.random(N9) < 0.5)
    c_next = np.where(done, rng.choice([-1.0, 1.0], N9), c)
    return (rng, np.where(done, 0.0, 1.0), c_next), obs, reward, done, 2 + (c > 0)


@functools.cache
def _p9_data() -> dict:
    rng = np.random.default_rng(0)
    start = (rng, np.zeros(N9), rng.choice([-1.0, 1.0], N9))
    return oracles.simulate(_p9_step, start, 40_000)


def p9_oracle(kind: str, rollout: int, lam: float, fault: str = "") -> np.ndarray:
    """The odd parts at (s0, s1) of the estimator's fixed point; a replay
    buffer's next row is the reset obs after an end."""
    d, other = dict(_p9_data()), functools.partial(np.roll, shift=1, axis=1)
    d["final"] = d["reset"] if kind == "td" else d["final"]
    if fault == "A":
        d["final"], d["terminated"] = other(d["final"]), other(d["terminated"])
    if fault == "B":
        d["terminated"] = np.repeat(d["terminated"][:, :1], N9, axis=1)
    if fault in ("C", "D"):
        key = "terminated" if fault == "C" else "reward"
        d[key] = other(d[key])
    rule = Rule(next="self" if fault == "P10" else "final", peng=kind == "q_lambda")
    gamma = 1.0 if kind == "apo" else G9  # APO: undiscounted GAE
    v = oracles.fixed_point(oracles.blocks(d, rollout), 4, gamma, lam, rule)
    return np.array([v[1] - v[0], v[3] - v[2]]) / 2


def _sign_changes(state: Any) -> np.ndarray:
    """Per seed, the smallest per-env share of replayed episodes whose sign
    differs from the previous one's: 1/2 when every reset draws afresh."""
    buffer = state.collector_state.buffer_state
    obs, index = np.asarray(buffer.experience["obs"]), np.asarray(buffer.current_index)
    out = np.zeros(obs.shape[:2])
    for i, full in enumerate(np.asarray(buffer.is_full)):
        rows = np.roll(obs[i], -index[i], 1) if full else obs[i, :, : index[i]]
        for e, r in enumerate(rows):
            signs = r[r[:, 0] == 0.0, 1]
            out[i, e] = np.mean(signs[1:] != signs[:-1])
    return out.min(1)


def _p9_run(agent: str, kw: dict) -> Callable:
    """Odd parts (SAC, DQN: target networks); raises if the envs end in
    lockstep on most seeds (0.026 a seed with per-env keys) or a replay
    env's resets repeat their sign."""

    def value(n: R.Nets, x: list) -> jax.Array:
        if agent == "DQN":
            return n.actor.apply_fn(n.actor.target_params, R.inp(n, x)).q_values.max()
        if agent in ("SAC", "TD3", "REDQ"):
            return R.critic(n, x, R.pi(n, x).mean(), n.critic.target_params)
        return R.value(agent, n, x)

    def read(run: runs.Run) -> dict:
        def one(n: R.Nets) -> dict:
            return {
                f"odd V(s{s})": (value(n, [s, 1.0]) - value(n, [s, -1.0])) / 2
                for s in (0, 1)
            }

        out = R.per_seed(one, R.nets(run.state, stats=R.env_stats(run)))
        last = np.stack([R.field(run.state, f) for f in "sc"], -1)  # raw (s, c)
        lockstep = (last == last[:, :1]).all(axis=(1, 2))
        if lockstep.sum() > len(lockstep) // 2:
            raise RuntimeError(f"4 envs in lockstep on {lockstep.sum()} seeds")
        if agent in ("SAC", "DQN"):
            out["sign change"] = _sign_changes(run.state)
            if not ((out["sign change"] > 0.3) & (out["sign change"] < 0.7)).all():
                raise RuntimeError(f"resets repeat their sign: {out['sign change']}")
        return out | {"lockstep": lockstep * 1.0}  # the ladder rounds readings

    spec = HAZARD[2 if agent in ("DQN", "PQN") else 0]
    kw = kw if agent == "APO" else {"gamma": G9, **kw}
    build = functools.partial(agents.make, agent, *spec.make(), n_envs=N9, **kw)
    return runs.readings(build, read)


for agent, ((kind, *_), table, budget, tol9, kw) in P9_CELLS.items():
    wrong = {f"{k}: {FAULTS9[k]}": w for k, w in (table | UNTRAINED).items()}
    truth = (2.0, 2.0) if kind == "apo" else TRUTH9
    queries = tuple(
        Query(
            f"odd V(s{i})",
            truth[i],
            {k: w[i] for k, w in wrong.items() if abs(w[i] - truth[i]) >= 0.2},
        )
        for i in (0, 1)
    )  # a fault gates a reading it moves by at least 0.2: twice the ceiling
    CASES[f"p9-{agent}"] = Case(
        f"p9-{agent}", queries, _p9_run(agent, kw), budget, tol9
    )


def test_p9_oracle_reproduces_the_analytic_values_and_the_fault_table() -> None:
    """The shared oracle on 40 000 simulated steps of the 4 envs: within
    0.02 of the analytic values and of each tabled fault reading; every fault
    0.2 or more from the truth on some reading, so some query gates it."""
    for agent, ((kind, rollout, lam), table, *_) in P9_CELLS.items():
        truth = (2.0, 2.0) if kind == "apo" else TRUTH9
        np.testing.assert_allclose(p9_oracle(kind, rollout, lam), truth, atol=0.02)
        for fault, want in table.items():
            got = p9_oracle(kind, rollout, lam, fault)
            np.testing.assert_allclose(got, want, atol=0.02, err_msg=f"{agent} {fault}")
            assert np.abs(np.subtract(want, truth)).max() >= 0.2, (agent, fault)


# --- Q1: reset cycles for the average-reward agents ---------------------------
# Position i shows obs[i] and pays r[i] whatever the action; from the last
# one the episode terminates (its final obs kept) or the walk wraps and only
# the time limit ends it. ASAC and APO learn differential values, whose
# differences have closed forms set by the boundary convention alone.


def cycle(obs: tuple, r: tuple, terminal: bool, **kw: Any) -> envs.Spec:
    n = len(obs)

    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        nxt = jnp.minimum(st.s + 1, n - 1) if terminal else (st.s + 1) % n
        return st.replace(s=nxt), jnp.asarray(r)[st.s], terminal & (st.s == n - 1)

    def show(st: envs.State) -> jax.Array:
        return jnp.asarray(obs, jnp.float32)[st.s][None]

    return envs.Spec("cycle", transition, show, obs_box=(-1.0, 1.0), **kw)


# ASAC: R -> A -> B -> terminated (obs 1, 0, -1; r 0, 1, 3; p_0 1/6). The reset
# obs R differs from the anchor A (obs 0), so a SAC-style done mask shows.
ASAC_TERM = cycle((1.0, 0.0, -1.0), (0.0, 1.0, 3.0), True)
ASAC_TRUNC = cycle((0.0, 1.0), (0.0, 3.0), False, limit=3)  # A <-> B, limit 3
APO_TERM = cycle((0.0, 1.0), (0.0, 1.0), True, actions=2)  # A -> B: 16 per rollout
APO_CONST = cycle((1.0,), (1.0,), False, actions=2)  # no boundary at all


def asac_values(rule: Rule, penalty: float) -> np.ndarray:
    """Q(s) - Q(A) on the chain from ASAC's target (train_ASAC.py:130-185):
    r - P term - theta + (1 - term) Q'(s') - Q'(0, 0), theta the mean reward
    without the penalty, alpha log pi dropped; one buffer row per position."""
    t, raw = np.array([0, 0, 1]), np.array([0.0, 1.0, 3.0])
    r = raw - penalty * t - raw.mean()
    rows = [(1.0, [(i, r[i], min(i + 1, 2), (i + 1) % 3, t[i], 0)]) for i in range(3)]
    q = oracles.fixed_point(oracles.stack(rows), 3, 1.0, 0.0, rule)
    return q - q[1]


P_TERM = 1.0 / 3  # p_0 E[max(r_0, r_1)], two rows' rewards: 2 p_0
MASK = Rule(anchor=1, shift_outside=True)
Q1_RIGHT = asac_values(MASK, P_TERM)
Q1_WRONG = {
    "reset obs bootstrapped": asac_values(Rule("none", "reset", anchor=1), P_TERM),
    "final obs bootstrapped": asac_values(Rule(mask="none", anchor=1), P_TERM),
    "done mask, shift inside": asac_values(Rule(anchor=1), P_TERM),
    "penalty dropped (P5 owns it)": asac_values(MASK, 0.0),
    "penalty sign flipped (P5 owns it)": asac_values(MASK, -P_TERM),
}


def test_q1_oracle_matches_the_hand_derivation() -> None:
    """Penalty 1/3, theta 4/3: (Q(R) - Q(A), Q(B) - Q(A)) = (-5/3, 2/3); the
    reset obs bootstrapped (-11/9, 2/9), the final obs (-8/3, 5/3); the shift
    inside the mask (-11/6, 5/6); penalty dropped (-16/9, 7/9), flipped
    (-17/9, 8/9). theta from the penalised rewards reads (-44/27, 17/27),
    1/27 off: P5 owns it (a gap of 10 there)."""
    w = [(-5, 2, 3), (-11, 2, 9), (-8, 5, 3), (-11, 5, 6), (-16, 7, 9), (-17, 8, 9)]
    for v, (r_a, b_a, d) in zip([Q1_RIGHT, *Q1_WRONG.values()], w, strict=True):
        np.testing.assert_allclose(v, [r_a / d, 0.0, b_a / d], atol=1e-9)


def _q1_asac(spec: envs.Spec, at: dict[str, float]) -> Callable:
    """ASAC with alpha fixed at 1e-3 (alpha log pi negligible), p_0 1/6:
    Q(x, mean action) - Q(A, mean action), averaged over the ensemble."""

    def one(n: R.Nets) -> dict:
        base = R.value("ASAC", n, 0.0)
        return {k: R.value("ASAC", n, x) - base for k, x in at.items()}

    kw: dict[str, Any] = {"alpha_init": 1e-3, "alpha_learning_rate": 0.0, "p_0": 1 / 6}
    build = functools.partial(agents.make, "ASAC", *spec.make(), **kw)
    return runs.readings(build, lambda run: R.per_seed(one, R.nets(run.state)))


# Calibrated on 1000-1031 + 2000-2031 at 1250 steps (worst 0.019, 0.019),
# certified 32/32 on 3000-3031 (worst 0.014, 0.012).
TERM_Q = tuple(
    Query(name, Q1_RIGHT[i], {k: v[i] for k, v in Q1_WRONG.items()})
    for name, i in (("Q(R) - Q(A)", 0), ("Q(B) - Q(A)", 2))
)
term_run = _q1_asac(ASAC_TERM, {"Q(R) - Q(A)": 1.0, "Q(B) - Q(A)": -1.0})
CASES["q1-ASAC-termination"] = Case(
    "q1-ASAC-termination", TERM_Q, term_run, 1250, (0.022, 0.021)
)
# A <-> B (obs 0, 1; r 0, 3): the limit fires on a step from A, whose true
# next obs is B; bootstrapping there, the cycle reads as if it never ended.
CASES["q1-ASAC-truncation"] = Case(
    "q1-ASAC-truncation",
    (Query("Q(B) - Q(A)", 1.5, {"truncation bootstraps the reset observation": 2.0}),),
    _q1_asac(ASAC_TRUNC, {"Q(B) - Q(A)": 1.0}),
    625,  # worst 0.034 (1250: 0.008); certified 32/32
    (0.067,),
)


def _q1_apo(spec: envs.Spec, at: dict[str, float]) -> Callable:
    """APO's own lambda, alpha and nu, no entropy bonus; V and b."""

    def read(run: runs.Run) -> dict:
        ns, b = R.nets(run.state), {"b": run.state.b}
        return R.per_seed(lambda n: {k: R.critic(n, x) for k, x in at.items()}, ns) | b

    build = functools.partial(agents.make, "APO", *spec.make(), ent_coef=0.0)
    return runs.readings(build, read)


def test_q1_apo_value_loss_pulls_values_towards_zero() -> None:
    """With targets equal to the predictions and b = 1 > 0 only the
    centring force is left: a small gradient step must lower mean V."""
    agent = agents.make("APO", *APO_CONST.make(), ent_coef=0.0)
    critic = agent.init(R.KEY, agent.make_context(1)[1]).critic_state
    obs, nu = jnp.linspace(-1.0, 1.0, 9)[:, None], agent.nu
    values = predict_value(critic, critic.params, obs).squeeze(0)

    def loss(p: Any) -> jax.Array:
        return compute_value_loss(p, critic, obs, values, nu, 1.0)[0]

    grads = jax.grad(loss)(critic.params)
    new = jax.tree.map(lambda p, g: p - 1e-3 * g, critic.params, grads)
    before, after = float(values.mean()), float(predict_value(critic, new, obs).mean())
    assert after < before, f"mean V went from {before:.6f} to {after:.6f}"


# With the sign fixed on a scratch copy, seeds 1000-1031 read |b|, |V(1)|
# <= 0.005 on the constant cycle; on the termination cycle, with a reset
# bootstrap at the rollout end, |b| < 1e-4, V(A) -0.250, V(B) 0.250. APO's
# GAE now drops V(s') at a termination and cuts the lambda-carry at every
# end (xtma/apo's generalized_advantage_estimation): rho 1/2, V(B) = 1/2 -
# nu b, V(A) = -(2 - lambda) nu b, b = 1/2 / (2 + nu (3 - lambda)) (nu 0.1,
# lambda 0.95). Seeds 0-7 read medians b 0.2268, V(A) -0.0238, V(B) 0.4773
# (before APO's reference hidden layers; unchanged at 10,000 on 1000-1031).
GROWS = {"centring sign reversed: grows without bound": math.inf}
CASES["q1-APO-constant"] = Case(
    "q1-APO-constant",
    (Query("b", 0.0, GROWS), Query("V(1)", 0.0, GROWS)),
    _q1_apo(APO_CONST, {"V(1)": 1.0}),
    2500,  # 10,000 fails (V(1): twice the worst error 0.133); cert 32/32
    (0.021, 0.02),
)
FIXED = "reset obs inside a rollout, final obs at its end (the old GAE), sign fixed"
RESET_OBS = "the reset observation bootstrapped at every end, sign fixed"
B_MASK = 0.5 / (2 + 0.1 * (3 - 0.95))  # b at the done mask's fixed point
B_WRONG = {FIXED: 2.553, RESET_OBS: 0.0, "final obs everywhere": 35.78}
B_WRONG["final obs everywhere, lambda carry cut"] = 5.0
CASES["q1-APO-termination"] = Case(
    "q1-APO-termination",
    (
        Query("b", B_MASK, B_WRONG),
        Query("V(A)", -(2 - 0.95) * 0.1 * B_MASK, {FIXED: 2.296, RESET_OBS: -0.25}),
        Query("V(B)", 0.5 - 0.1 * B_MASK, {FIXED: 2.809, RESET_OBS: 0.25}),
    ),
    _q1_apo(APO_TERM, {"V(A)": 0.0, "V(B)": 1.0}),
    10_000,
    (0.1, 0.1, 0.1),
)


# A <-> B (obs -1, 1; r 0, 1), limit 3: the limit fires on a step from A,
# whose true next obs is B. As xtma/apo, the limit ends the episode (no
# bootstrap) and its row is not fitted (bootstrap_timelimit), while rho and
# b average every row; b settles where APO's fixed point puts it (32-step
# rollouts at every phase). With A at obs 0, one seed of 32 read a flat V.
APO_TRUNC = cycle((-1.0, 1.0), (0.0, 1.0), False, actions=2, limit=3)
TRUNC_ROWS = [(0, 0.0, 1, 1, 0, 0), (1, 1.0, 0, 0, 0, 0), (0, 0.0, 1, 0, 0, 1)]


def apo_values(rows: list, rule: Rule = RIGHT, phases: bool = True) -> np.ndarray:
    """(b, V(A), V(B)) at APO's fixed point (lambda 0.95, nu 0.1) on one
    episode's rows, rho their mean reward; 32-step rollouts at every phase,
    or (``phases`` False) aligned with the episodes."""
    aligned = [(1.0, rows * (32 // len(rows)))]
    ro = oracles.rollouts([(1.0, rows)], 32) if phases else oracles.stack(aligned)
    ro["reward"] = ro["reward"] - np.mean([r[1] for r in rows])
    v = oracles.fixed_point(ro, 2, 1.0, 0.95, rule, bias=0.1)
    visits = np.bincount([r[0] for r in rows], minlength=2) / len(rows)
    return np.array([visits @ v, *v])


def test_q1_apo_oracle_reproduces_the_closed_forms() -> None:
    """On aligned rollouts, b, V(A), V(B) as derived: q1-APO-termination's,
    and the truncation cycle's bootstrapped at the limit (the oracle's
    default rule), where b weights A's two visits; a plain state
    mean in the fixed point moves b to 3.164 (3.111 on the cell's phases)."""
    rows = [(0, 0.0, 1, 1, 0, 0), (1, 1.0, 1, 0, 1, 0)]
    want = (B_MASK, -(2 - 0.95) * 0.1 * B_MASK, 0.5 - 0.1 * B_MASK)
    np.testing.assert_allclose(apo_values(rows, phases=False), want, atol=1e-9)
    lam, nu = 0.95, 0.1  # A, B, A: each state's mean advantage is nu b
    b = (1 / 3) / (nu * (1 + (1 - lam) * (2 - lam) / 2))
    gap = 1 / 3 + nu * b * (2 - lam) / 2  # V(B) - V(A)
    want = (b, b - gap / 3, b + 2 * gap / 3)  # b = (2 V(A) + V(B)) / 3
    np.testing.assert_allclose(apo_values(TRUNC_ROWS, phases=False), want, atol=1e-9)


TRUNC_RIGHT = apo_values(TRUNC_ROWS, Rule(mask="or", drop_truncated=True))
TRUNC_WRONG = {  # each reading gated by the faults that move it 0.05 or more
    "final obs bootstrapped and fitted": apo_values(TRUNC_ROWS),
    "time-limited row fitted": apo_values(TRUNC_ROWS, Rule(mask="or")),
    "reset obs bootstrapped": apo_values(TRUNC_ROWS, Rule("terminated", "reset")),
}
GATED = [
    {k: v[i] for k, v in TRUNC_WRONG.items() if abs(v - TRUNC_RIGHT)[i] >= 0.05}
    for i in range(3)
]
CASES["q1-APO-truncation"] = Case(
    "q1-APO-truncation",
    tuple(
        Query(n, TRUNC_RIGHT[i], GATED[i]) for i, n in enumerate(("b", "V(A)", "V(B)"))
    ),
    _q1_apo(APO_TRUNC, {"V(A)": -1.0, "V(B)": 1.0}),
    2500,
    (0.02, 0.056, 0.063),
    note="worst errors 0.004, 0.028, 0.031 (5000: 0.003, 0.021, 0.018); cert 32/32",
)


def test_q1_apo_termination_cycle_keeps_b_bounded() -> None:
    """Past |b| = 100 b feeds itself (1373 to 2225 with the value bias's
    sign reversed, before the done mask); under any convention |b| <= 35.8."""
    case = CASES["q1-APO-termination"]
    b = case.readings(STAGE_1, case.budget)["b"]
    assert (np.abs(b) < 100.0).all(), f"b reads {b}"


# --- every judged case --------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
