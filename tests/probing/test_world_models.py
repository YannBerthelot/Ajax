"""World models: tiny MDPs answered only through a learned model (Q10 task
twins for multi-task TD-MPC2, Q12 payback past TD-MPC2's horizon, Q13
risky-safe for DreamerV3's imagination); the last test judges every case.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3 import networks as dreamer
from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.dataset import TaskEpisodes, pool_tasks
from ajax.agents.TDMPC2.networks import make_policy_prior, make_world_model
from ajax.agents.TDMPC2.planner import estimate_value
from ajax.agents.TDMPC2.train_TDMPC2 import evaluate_tdmpc2

from . import agents, envs, runs
from .verdict import STAGE_1, Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "6956060745f9"  # verdict.digest(CASES): every answer, pinned
KEY, RET = jax.random.PRNGKey(0), "episodic mean reward"


# --- Q10: multi-task TD-MPC2 routes by task id --------------------------------
# Two tasks show the agent [1, 0] (task 0's [1] padded) and differ only in
# their id: task 0 pays +a for 10 steps (discount 0.5), task 1 pays -a for 25
# (0.8, at discount_denom 5 and discount_min 0). Trained offline on an equal
# pool without done flags, Q_i(a) = r_i(a) + g_i V_i: the odd part
# (Q_i(+1) - Q_i(-1)) / 2 is +-1 and the implied discount (Q_i(+1) + Q_i(-1))
# / (2 V_i) is g_i whatever the policy; the policy prior and the planner must
# each play the task's rewarding action. Lost routing pools E[r | a] to 0.

SIGNS, LENGTHS, DISCOUNTS = (1.0, -1.0), (10, 25), (0.5, 0.8)


def _twin(sign: float, length: int, dim: int) -> envs.Spec:
    def step(st: envs.State, a: jax.Array, key: Any) -> tuple:
        return st, sign * jnp.clip(a.reshape(-1)[0], -1.0, 1.0), False

    def obs(st: envs.State) -> jax.Array:
        return envs.one_hot(0, dim)

    box = (-np.inf, np.inf)
    return envs.Spec("twin", step, obs, obs_dim=dim, obs_box=box, limit=length)


TWINS = [_twin(s, t, d) for s, t, d in zip(SIGNS, LENGTHS, (1, 2))]


def _episodes(task: int, rng: np.random.Generator) -> TaskEpisodes:
    """16 episodes of 11 obs-aligned rows: uniform actions, r_k = s a_{k-1}."""
    obs = np.zeros((16, 11, task + 1), np.float32)
    obs[..., 0] = 1.0
    action = rng.uniform(-1.0, 1.0, (16, 11, 1)).astype(np.float32)
    action[:, -1] = 0.0
    reward = np.zeros((16, 11), np.float32)
    reward[:, 1:] = SIGNS[task] * action[:, :-1, 0]
    return TaskEpisodes(obs, action, reward, LENGTHS[task], f"task{task}")


def _q10_agent() -> Any:
    rng, twins = np.random.default_rng(0), [spec.make()[0] for spec in TWINS]
    data = pool_tasks([_episodes(i, rng) for i in range(2)])
    kw: dict[str, Any] = {"dataset": data, "eval_envs": twins, "discount_denom": 5}
    return agents.make("TDMPC2MultiTask", None, preset="tiny", discount_min=0.0, **kw)


def _q10_values(agent: Any, wm: Any, target_q: Any, pi: Any, key: Any) -> dict:
    """One seed's Q_i(+1), Q_i(-1), the TD target's V_i = E_{a ~ pi_i}[min_k
    Qbar_k(a)] and the policy's E[s_i a] (256 draws), at each task's context."""
    config, tasks, out = agent.agent_config, agent.tasks, []
    apply, decode = make_world_model(config).apply, config.two_hot.decode
    pi_apply = make_policy_prior(config, tasks.action_dim).apply
    obs, ends = envs.one_hot(0, tasks.obs_dim), jnp.array([[1.0], [-1.0]])
    for task in range(tasks.num_tasks):
        ctx = tasks.context(task)
        params = core.renorm_task_embedding(wm, ctx.ids)
        emb = core.task_embedding(params, ctx)
        z = apply({"params": params}, obs, emb, method="encode")
        z2, zs = (jnp.broadcast_to(z, (n, z.shape[-1])) for n in (2, 256))
        q = decode(core.q_logits(apply, params, z2, ends, task_emb=emb)).mean(0)
        eps = jax.random.normal(jax.random.fold_in(key, task), (256, 1))
        a = core.policy_sample(pi_apply, pi, zs, eps, config, emb, ctx.mask).action
        kw = {"q_params": target_q, "task_emb": emb}
        v = decode(core.q_logits(apply, params, zs, a, **kw))
        out.append((q[0], q[1], v.min(0).mean(), (SIGNS[task] * a[:, 0]).mean()))
    return {k: jnp.stack(x) for k, x in zip(("+", "-", "v", "m"), zip(*out))}


def _q10_read(run: runs.Run) -> dict:
    s, wm, agent = run.state, run.state.world_model_state, run.agent
    keys = jax.random.split(KEY, len(run.seeds))
    fn = jax.jit(jax.vmap(functools.partial(_q10_values, agent)))
    v = fn(wm.params, wm.target_params, s.actor_state.params, keys)
    v, ev, out = {k: np.asarray(x) for k, x in v.items()}, agent.evaluate(s, 2), {}
    for i, name in enumerate(agent.tasks.names):
        plus, minus = v["+"][:, i], v["-"][:, i]
        n, ret = (ev[f"Eval/{name}/{k}"] for k in ("mean episodic length", RET))
        out[f"odd part, task {i}"] = (plus - minus) / 2
        out[f"implied discount, task {i}"] = (plus + minus) / (2 * v["v"][:, i])
        out[f"policy action, task {i}"] = v["m"][:, i]
        out[f"eval reward per step, task {i}"] = ret / n
        out[f"eval length, task {i}"] = n
    return out


def _q10_queries(i: int) -> tuple[Query, ...]:
    s, odd = SIGNS[i], {"task routing lost (constant embedding)": 0.0}
    other = {f"discounts swapped, or task {1 - i}'s for both": DISCOUNTS[1 - i]}
    other["discount_min not forwarded (0.95 for both)"] = 0.95
    pi = {"policy prior trained on the other task's Q": -1.0}
    pi["no preferred action (Q flat in the action)"] = 0.0
    plan = {"evaluation context or env swapped": -1.0}
    plan["task routing lost (arbitrary planner)"] = 0.0
    return (
        Query(f"odd part, task {i}", s, odd | {"training labels swapped": -s}, True),
        Query(f"implied discount, task {i}", DISCOUNTS[i], other),
        Query(f"policy action, task {i}", 1.0, pi, True),
        Query(f"eval reward per step, task {i}", 1.0, plan, True),
    )


CASES["q10-TDMPC2MultiTask"] = Case(
    "q10-TDMPC2MultiTask",
    tuple(q for pair in zip(_q10_queries(0), _q10_queries(1)) for q in pair),
    runs.readings(_q10_agent, _q10_read),
    3000,
    (0.459, 0.438, 0.075, 0.061, 0.497, 0.483, 0.498, 0.496),
    slow=True,
    note="at 2000 task 1's discount read 0.759 and one policy -1; cert 32/32",
)


@pytest.mark.slow
def test_q10_each_task_is_evaluated_in_its_own_env() -> None:
    """Task i's planner runs in eval_envs[i]: episodes of exactly 10 and 25
    steps on every seed (25 and 10 with the pairing swapped)."""
    case = CASES["q10-TDMPC2MultiTask"]
    r = case.readings(STAGE_1, case.budget)
    lengths = [r[f"eval length, task {i}"].tolist() for i in (0, 1)]
    assert lengths == [[float(t)] * len(STAGE_1) for t in LENGTHS], lengths


# --- Q12: TD-MPC2 plans past its horizon ---------------------------------------
# From s0 = [1, 0] an action a > 0 pays -2 and moves to s1 = [0, 1], which pays
# 1 a step; 10 steps, none terminal, gamma 0.8, horizon 3. Q(s0, +1) = 2 and
# Q(s0, -1) = 1.6; the planner values investing at t = 0, 1, 2 or never at 2.0,
# 1.6, 1.28, 1.024: the payback lies past the horizon, in the terminal Q at the
# latent the learned dynamics reach. All but the last 500 steps are the seed
# phase, the only visits to the wait branch once the planner invests.

G12, COST, PLANNER_STEPS = 0.8, 2.0, 500
PLANS = ("invest at 0", "invest at 1", "invest at 2", "never")
FAULTS = ("terminal Q dropped", "Q read at z_0", "latent not rolled")


def _payback(st: envs.State, a: jax.Array, key: Any) -> tuple:
    invest = (st.s == 0) & (a.reshape(()) > 0.0)
    reward = jnp.where(st.s == 1, 1.0, jnp.where(invest, -COST, 0.0))
    return st.replace(s=st.s | invest), reward, False


PAYBACK = envs.Spec(
    "payback", _payback, lambda st: envs.one_hot(st.s, 2), obs_dim=2, limit=10
)


def plan_sequences(horizon: int) -> list[tuple[str, tuple[float, ...]]]:
    """Each plan: -1 until it invests (+1), then every completion."""
    out = []
    for k in range(horizon):
        tails = itertools.product((-1.0, 1.0), repeat=horizon - 1 - k)
        out += [(f"invest at {k}", (-1.0,) * k + (1.0, *t)) for t in tails]
    return [*out, ("never", (-1.0,) * horizon)]


def plan_values(gamma: float = G12, horizon: int = 3, fault: str = "") -> dict:
    """Each plan's exact planner value (its best completion), with the model
    and Q learned exactly and an optional planner fault of FAULTS."""
    v1 = 1.0 / (1.0 - G12)
    q0 = -COST + G12 * v1  # Q(s0, pi(s0)): the prior invests
    values: dict[str, float] = {}
    for name, actions in plan_sequences(horizon):
        value, latent = 0.0, 0
        for t, a in enumerate(actions):
            invest = latent == 0 and a > 0
            value += gamma**t * (1.0 if latent else -COST if invest else 0.0)
            latent = latent if fault == FAULTS[2] else int(latent or invest)
        end = {FAULTS[0]: 0.0, FAULTS[1]: q0}.get(fault, v1 if latent else q0)
        values[name] = max(values.get(name, -np.inf), value + gamma**horizon * end)
    return values


def _gap(v: dict) -> float:
    return v[PLANS[0]] - max(x for k, x in v.items() if k != PLANS[0])


def _best(v: dict) -> str:
    return max(v, key=lambda k: v[k])


def _q12_read(agent: Any, s: Any, key: jax.Array) -> dict:
    """At s0: the plan gap of planner.estimate_value (no noise, no dropout),
    the Q, reward and rolled-reward gaps of +1 over -1, both returns."""
    config, wm, pi = agent.agent_config, s.world_model_state, s.actor_state.params

    def apply(*args: Any, **kw: Any) -> jax.Array:
        return wm.apply_fn({"params": wm.params}, *args, **kw)

    def q(z: jax.Array, a: jax.Array) -> jax.Array:
        logits = core.q_logits(wm.apply_fn, wm.params, z, a)
        return config.two_hot.decode(logits).mean(0)[0]

    def r(z: jax.Array, a: jax.Array) -> jax.Array:
        return config.two_hot.decode(apply(z, a, method="reward_logits"))[0]

    z0, up = apply(jnp.eye(2), method="encode")[:1], jnp.ones((1, 1))
    out = {"Q gap": q(z0, up) - q(z0, -up), "R gap": r(z0, up) - r(z0, -up)}
    rolled = [r(apply(z0, a, method="next"), up) for a in (up, -up)]
    seqs = plan_sequences(config.horizon)
    actions = jnp.asarray([a for _, a in seqs], jnp.float32).T[:, :, None]
    n, cfg = actions.shape[1], config.replace(dropout=0.0)
    z = jnp.broadcast_to(z0, (n, z0.shape[-1]))
    args = (z, actions, jnp.zeros((n, 1)), jnp.arange(2), KEY)
    v = estimate_value(wm.params, pi, *args, config=cfg, gamma=agent.gamma)
    best = jnp.stack(
        [jnp.max(v[np.flatnonzero([k == p for k, _ in seqs])]) for p in PLANS]
    )
    kw = {"env_args": agent.env_args, "config": config, "gamma": agent.gamma}
    ev = evaluate_tdmpc2(s, jax.random.split(key)[1], num_episodes=8, **kw)
    out |= {"rolled R gap": rolled[0] - rolled[1], "plan gap": best[0] - best[1:].max()}
    out["eval return"] = ev[f"Eval/{RET}"]
    return out | {"train return": s.collector_state.episodic_mean_return}


@functools.cache
def _q12_run(seeds: tuple[int, ...], budget: int) -> dict[str, np.ndarray]:
    steps: dict[str, Any] = {"n_envs": 2, "seed_steps": budget - PLANNER_STEPS}
    agent = agents.make("TDMPC2", PAYBACK.make()[0], preset="tiny", gamma=G12, **steps)
    keys = jax.random.split(jax.random.PRNGKey(12), len(seeds))
    read = jax.jit(jax.vmap(functools.partial(_q12_read, agent)))
    out = read(runs.train(agent, seeds, budget).state, keys)
    return {k: np.asarray(v) for k, v in out.items()}


NEVER = {"the planner never invests (any planner-value fault)": 0.0}
Q12 = {  # every reading's wrong answers lie at or across zero: one-sided
    "plan gap": (_gap(plan_values()), {f: _gap(plan_values(fault=f)) for f in FAULTS}),
    "Q gap": (0.4, {"the critic ignores the action at s0": 0.0}),
    "R gap": (-COST, {"the reward model ignores the action at s0": 0.0}),
    "rolled R gap": (1.0 + COST, {"the dynamics ignore the action at s0": 0.0}),
    "eval return": (7.0, NEVER),
    "train return": (7.0, NEVER),
}
CASES["q12-TDMPC2"] = Case(
    "q12-TDMPC2",
    tuple(Query(name, truth, wrong, True) for name, (truth, wrong) in Q12.items()),
    _q12_run,
    3000,
    (0.16, 0.179, 0.96, 1.413, 2.938, 2.475),
    slow=True,
    note="1500-2500 the eval planner lands on the blurred a = 0 step; cert 31/32",
)


def test_q12_exact_plan_values_rank_investing_now_first_unless_faulted() -> None:
    """Plan values 2.0, 1.6, 1.28, 1.024; each planner fault or a discount of
    0.5 ranks "never" first; 0.95 to 1 and horizons 1 to 5 invest now, at 2."""
    expected = {
        "": (2.0, 1.6, 1.28, 1.024),
        FAULTS[0]: (-0.56, -0.96, -1.28, 0.0),
        FAULTS[1]: (0.464, 0.064, -0.256, 1.024),
        FAULTS[2]: (-0.976, -0.576, -0.256, 1.024),
    }
    for fault, right in expected.items():
        got = plan_values(fault=fault)
        assert tuple(got[p] for p in PLANS) == pytest.approx(right), fault
        assert _best(got) == (PLANS[3] if fault else PLANS[0]), fault
    assert _best(plan_values(gamma=0.5)) == PLANS[3]
    assert {_best(plan_values(gamma=g)) for g in (0.95, 0.99, 1.0)} == {PLANS[0]}
    for horizon in range(1, 6):
        got = plan_values(horizon=horizon)
        assert _best(got) == PLANS[0] and got[PLANS[0]] == pytest.approx(2.0)


def test_q12_env_pays_back_only_after_investing() -> None:
    """Investing (any a > 0) first returns -2 + 9 = 7 over the 10 steps, a
    step later 6, never (a <= 0) 0; no step terminates."""
    env, p = PAYBACK.make()
    w, never = [-1.0] * 9, ([0.0] * 10, [-1.0] * 10)
    plans = {7: ([1.0, *w], [1e-3, *w], [1.0] * 10), 6: ([-1.0, 0.5, *w[1:]],)}
    for want, plan in [(k, x) for k, v in (plans | {0: never}).items() for x in v]:
        st, total = env.reset_env(KEY, p)[1], 0.0
        for a in plan:
            _, st, reward, done, _ = env.step_env(KEY, st, jnp.array([a]), p)
            assert not done
            total += float(reward)
        assert total == want, plan
    assert p.max_steps_in_episode == 10


# --- Q13: DreamerV3 imagines sampled outcomes, not their mode -------------------
# From observation 0, safe (action 0) pays 0.75 (obs 1), risky 1 w.p. 0.6
# (obs 2) else 0 (obs -2), then terminal. P(safe) tends to the actor's unimix
# cap 0.995; imagining the prior's mode makes every risky step a win (0.005),
# an action cut from the dynamics reads 0.5. At free_nats 0 the prior matches
# the outcome frequencies. At 2000 to 6000 rows the mode fault is not yet at
# 0.005 (seed 1005 strays to 0.18 at 4000); certified 32/32 at 8000.

CAP = 0.99 + 0.01 / 2
WRONG13 = {"no prior sampling in imagination (the mode wins)": 1 - CAP}
WRONG13["untrained, or the action not reaching the dynamics"] = 0.5
SAFE = Query("P(safe | s0)", CAP, WRONG13)


def _risky(st: envs.State, a: jax.Array, key: Any) -> tuple:
    win, risky = jax.random.bernoulli(key, 0.6), a == 1
    x = jnp.where(risky, jnp.where(win, 2.0, -2.0), 1.0).astype(jnp.float32)
    return st.replace(c=x), jnp.where(risky, win.astype(jnp.float32), 0.75), True


RISKY = envs.Spec(
    "risky_safe", _risky, lambda st: st.c.reshape(1), obs_box=(-2.0, 2.0), actions=2
)


def _p_safe(run: runs.Run) -> dict:
    """P(safe): the actor at the filtered posterior's mode of observation 0
    (one observe step from a fresh carry, a zero previous action)."""
    c, s = run.agent.dreamer_config, run.state
    model, rssm = dreamer.WorldModel(c, 1), dreamer.RSSM(c)

    def one(wm: dict, pi: dict) -> jax.Array:
        x = model.apply({"params": wm}, jnp.zeros((1, 1)), method="encode")
        h0, a0 = dreamer.initial_state(c, (1,)), jnp.zeros((1, 2))
        step = (h0, x, a0, jnp.ones(1, bool), jnp.zeros((1, c.stoch, c.classes)))
        h = rssm.apply({"params": wm["rssm"]}, *step, method="observe_step")[0]
        z = dreamer.features(h.deter, h.stoch)
        return jnp.exp(dreamer.Actor(c, 2, True).apply({"params": pi}, z).logits[0, 0])

    return {SAFE.name: jax.vmap(one)(s.world_model_state.params, s.actor_state.params)}


def _q13_agent() -> Any:
    kw: dict[str, Any] = {"free_nats": 0.0, "return_horizon": 1 / (1 - agents.GAMMA)}
    return agents.make("DreamerV3", *RISKY.make(), preset="tiny", **kw)


Q13_READ = runs.readings(_q13_agent, _p_safe)
CASES["q13-DreamerV3"] = Case(
    "q13-DreamerV3", (SAFE,), Q13_READ, 8000, (0.052,), slow=True, note="cert 32/32"
)


# --- every judged case ---------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
