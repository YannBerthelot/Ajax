"""PQN against its original, purejaxql (vendored at 47af6d7), one training iteration at a time.

Each check maps purejaxql's state at the start of an iteration into Ajax's, gives Ajax the draws
purejaxql made in that iteration (by role and order), runs the iteration on both sides, and compares
its sub-steps: the rollout, the Q(lambda) targets, the update. Where the codes differ, by an Ajax
choice, a convention or a defect of the original (refcmp_specs.PUREJAXQL_DEVIATIONS), the test
asserts the predicted difference instead of hiding it.
"""

import contextlib
import dataclasses
import re
from typing import Any, Iterator

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import refcmp_ajax
import refcmp_jax
from refcmp_specs import COMPARISONS, PUREJAXQL_DEVIATIONS

import ajax
import ajax.agents.PQN.train_PQN as train_PQN
from ajax.agents.DQN.networks import predict_q
from ajax.agents.DQN.train_DQN import linear_epsilon
from ajax.agents.PQN.utils import compute_q_lambda_targets
from tests.conformance import kit, submaps

SPEC = COMPARISONS["PQN"]
N_ENVS, N_STEPS = SPEC.matched["n_envs"], SPEC.matched["n_steps"]
#: Targets re-derived in float64 against float32 code: a 64-step recursion rounds to about 4e-6.
TARGET_TOL = 1e-5


@dataclasses.dataclass
class Sides:
    ref_step: Any
    rs0: Any
    config: dict
    ajax: kit.Step
    s0: Any


@contextlib.contextmanager
def time_limit(steps: int | None) -> Iterator[None]:
    """Both sides build their CartPole through gymnax.make: patch it to cut episodes at ``steps``."""
    original = gymnax.make

    def make(name: str, **kw: Any) -> Any:
        env, params = original(name, **kw)
        return env, params.replace(max_steps_in_episode=steps)

    if steps is not None:
        gymnax.make = make
    try:
        yield
    finally:
        gymnax.make = original


def build(constant_epsilon: bool, limit: int | None = None) -> Sides:
    eps = 0.2
    ref_over = {"EPS_START": eps, "EPS_FINISH": eps} if constant_epsilon else {}
    ajax_over = {"epsilon_start": eps, "epsilon_end": eps} if constant_epsilon else {}
    with time_limit(limit):
        # Evaluation only, not part of the step.
        config = (
            dict(SPEC.reference_config, NUM_SEEDS=1, TEST_DURING_TRAINING=False)
            | ref_over
        )
        train = refcmp_jax.load_purejaxql().make_train(config)
        ref_step, rs0 = kit.capture_scan(
            "_update_step", lambda: train(jax.random.PRNGKey(0))
        )
        agent = ajax.PQN(
            SPEC.env_id, **(refcmp_ajax.agent_kwargs(SPEC, "matched") | ajax_over)
        )
        step = kit.capture_step(agent, SPEC.n_timesteps)
    return Sides(ref_step, rs0, config, step, step.fresh(jax.random.PRNGKey(0)))


def norm(path: Any) -> str:
    return jax.tree_util.keystr(path).replace("['params']", "")


def ref_path(p: str) -> str | None:
    """Ajax's state path -> purejaxql's runner_state path; None: not mapped (PQN's unused critic and
    target copies, bookkeeping)."""
    if p.startswith(".actor_state.params"):
        return "[0].params" + p[len(".actor_state.params") :]
    if p.startswith(".actor_state.opt_state"):
        return "[0].opt_state" + p[len(".actor_state.opt_state") :]
    if m := re.fullmatch(r"\.collector_state\._env_state\.(\w+)", p):
        return f"[1][1].env_state.{m.group(1)}"
    return {
        ".actor_state.step": "[0].step",
        ".collector_state.last_obs": "[1][0]",
        ".collector_state.timestep": "[0].timesteps",
        ".n_updates": "[0].n_updates",
    }.get(p)


def leaves(tree: Any) -> dict[str, Any]:
    return {norm(p): x for p, x in jax.tree_util.tree_leaves_with_path(tree)}


def phi(sides: Sides, rs: Any, prev_done: np.ndarray) -> Any:
    """purejaxql's runner_state at an iteration's start, as Ajax's state."""
    ref = leaves(rs)

    def put(path: Any, x: Any) -> Any:
        p = norm(path)
        if p == ".collector_state.last_terminated":
            return jnp.asarray(prev_done, x.dtype)
        r = ref_path(p)
        return (
            x
            if r is None or r not in ref
            else jnp.asarray(ref[r], x.dtype).reshape(x.shape)
        )

    return jax.tree_util.tree_map_with_path(put, sides.s0)


def role(draws: dict, sampler: str, site: str) -> kit.Role:
    [r] = [r for r in draws if r[0] == sampler and site in r[1]]
    return r


def psi(ref: dict, aj: dict) -> dict:
    """Ajax's replay table from purejaxql's draws: epsilon-greedy per step (32 per-env draws there,
    one (32,) draw here), CartPole resets in order, one minibatch permutation per epoch (purejaxql
    applies one key to its 7 leaves)."""
    values = {}
    for sampler in ("randint", "uniform"):
        a, r = (
            role(aj, sampler, "pipeline"),
            role(ref, sampler, "eps_greedy_exploration"),
        )
        flat = [v for _, v in ref[r]]
        values[a] = [
            np.stack(flat[t * N_ENVS : (t + 1) * N_ENVS]) for t in range(len(aj[a]))
        ]
    a, r = role(aj, "uniform", "reset_env"), role(ref, "uniform", "reset_env")
    values[a] = [v for _, v in ref[r]]
    a, r = (
        role(aj, "permutation", "get_minibatches_from_batch"),
        role(ref, "permutation", "preprocess_transition"),
    )
    perms, per = [v for _, v in ref[r]], len(ref[r]) // len(aj[a])
    assert all(np.array_equal(perms[k - k % per], perms[k]) for k in range(len(perms)))
    values[a] = perms[::per]
    assert values.keys() == aj.keys(), f"unmapped Ajax draws: {set(aj) - set(values)}"
    return kit.key_table(aj, values)


def taped(fn: Any, x: Any, context: Any) -> tuple[Any, kit.Session]:
    with context as session:
        out = jax.jit(fn)(x)
        jax.block_until_ready(out)
    return out, session


@dataclasses.dataclass
class Iteration:
    rs: Any
    rs1: Any
    ref: dict  # stashed: "_step_env" (transitions, infos), "_get_target"
    state: Any
    out: Any
    roll: Any
    own_table: dict
    table: dict
    ref_draws: dict
    collected: dict  # each side's collection scan: (carry, stacked), by scan name


def iterate(sides: Sides, n: int) -> list[Iteration]:
    def ref_fn(rs: Any) -> Any:
        with kit.stashing("_step_env", "_get_target") as stash:
            out = sides.ref_step(rs, None)[0]
        return out, dict(stash)

    def ajax_fn(s: Any) -> Any:
        with kit.stashing("collect_experience") as stash:
            out = sides.ajax.iteration(s, 0, 0)[0]
        return out, stash["collect_experience"]

    rs, prev_done, its = sides.rs0, np.zeros(N_ENVS, np.float32), []
    for _ in range(n):
        _, rec = taped(ref_fn, rs, kit.recording())
        ref_draws = kit.by_role(rec.log)
        own = kit.key_table(
            ref_draws, {r: [v for _, v in d] for r, d in ref_draws.items()}
        )
        (rs1, stash), _ = taped(ref_fn, rs, kit.replaying(own))
        state = phi(sides, rs, prev_done)
        _, arec = taped(ajax_fn, state, kit.recording())
        table = psi(ref_draws, kit.by_role(arec.log))
        (out, collected), rep = taped(ajax_fn, state, kit.replaying(table))
        assert not rep.misses
        ref = {name: ys for name, (_, ys) in stash.items()}
        both = {"_step_env": stash["_step_env"], "collect_experience": collected}
        its.append(
            Iteration(
                rs, rs1, ref, state, out, collected[1], own, table, ref_draws, both
            )
        )
        prev_done, rs = np.asarray(ref["_step_env"][0].done[-1], np.float32), rs1
    return its


@pytest.fixture(scope="module")
def plain() -> tuple[Sides, list[Iteration]]:
    """Epsilon held at 0.2 on both sides (the per-step epsilon cannot act) and no time-limit cut
    within two iterations (128 steps per env, against 500)."""
    sides = build(constant_epsilon=True)
    return sides, iterate(sides, 2)


def rollouts(it: Iteration, rew_scale: float) -> tuple[dict, dict]:
    a, t = it.roll, it.ref["_step_env"][0]
    done = np.asarray(a.terminated[..., 0] > 0) | np.asarray(a.truncated[..., 0] > 0)
    ajax_view = {
        "obs": a.obs,
        "action": np.asarray(a.action[..., 0]).astype(np.int32),
        "reward": np.asarray(a.reward[..., 0] * rew_scale),
        "done": done,
    }
    ref_view = {
        "obs": t.obs,
        "action": np.asarray(t.action, np.int32),
        "reward": t.reward,
        "done": np.asarray(t.done, bool),
    }
    return ajax_view, ref_view


def test_collection_matches_bit_for_bit(plain):
    sides, its = plain
    for it in its:
        a, r = rollouts(it, sides.config["REW_SCALE"])
        for k in r:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(r[k]), err_msg=k)


def q_lambda(
    r: np.ndarray,
    term: np.ndarray,
    done: np.ndarray,
    v_next: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Q(lambda) as defined, in float64: G_t = r_t + gamma ((1 - lambda) v_t + lambda G_{t+1}), the
    trace cut at an episode's end, v_t = V(next_obs_t), zero after a termination."""
    v = v_next * (1 - term)
    g = np.empty_like(r)
    g[-1] = r[-1] + gamma * v[-1]
    for t in range(len(r) - 2, -1, -1):
        g[t] = r[t] + gamma * (
            (1 - lam) * v[t] + lam * ((1 - done[t]) * g[t + 1] + done[t] * v[t])
        )
    return g


def purejaxql_as_written(
    r: np.ndarray,
    done: np.ndarray,
    v_next: np.ndarray,
    v_obs: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """pqn_gymnax.py:231-254 in float64: the backward pass over transitions 0-62 starts with
    next_q = V(next_obs_63) (the last step's bootstrap), then takes next_q = V(obs_t)."""
    last_q = v_next[-1] * (1 - done[-1])
    lam_ret, next_q, out = r[-1] + gamma * last_q, last_q, []
    for t in range(len(r) - 2, -1, -1):
        bootstrap = r[t] + gamma * (1 - done[t]) * next_q
        lam_ret = bootstrap + gamma * lam * (lam_ret - next_q)
        lam_ret = (1 - done[t]) * lam_ret + done[t] * r[t]
        next_q = v_obs[t]
        out.append(lam_ret)
    return np.stack(out[::-1])


def re_derived(it: Iteration, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The textbook targets (64 steps) and purejaxql's as written (63), from the iteration's rollout and
    the network before the update."""
    roll, m = it.roll, SPEC.matched
    r = np.asarray(roll.reward[..., 0], np.float64) * scale
    term = np.asarray(roll.terminated[..., 0], np.float64)
    done = np.maximum(term, np.asarray(roll.truncated[..., 0], np.float64))
    v_next = np.asarray(
        predict_q(it.state.actor_state, it.state.actor_state.params, roll.next_obs)
    ).max(-1)
    v_obs = np.asarray(it.ref["_step_env"][0].q_val).max(-1)
    return q_lambda(
        r, term, done, v_next, m["gamma"], m["q_lambda"]
    ), purejaxql_as_written(r, done, v_next, v_obs, m["gamma"], m["q_lambda"])


def ajax_targets(it: Iteration) -> np.ndarray:
    q = predict_q(it.state.actor_state, it.state.actor_state.params, it.roll.next_obs)
    t = compute_q_lambda_targets(
        it.roll.reward * SPEC.matched["reward_scale"],
        jnp.max(q, -1, keepdims=True),
        it.roll.terminated,
        it.roll.truncated,
        SPEC.matched["gamma"],
        SPEC.matched["q_lambda"],
    )
    return np.asarray(t[..., 0])


def test_targets_differ_only_by_purejaxqls_off_by_one(plain):
    sides, its = plain
    for it in its:
        textbook, as_written = re_derived(it, sides.config["REW_SCALE"])
        ours, theirs = ajax_targets(it), np.asarray(it.ref["_get_target"])
        tol = TARGET_TOL * np.abs(textbook).max()
        np.testing.assert_allclose(ours, textbook, rtol=0, atol=tol)
        np.testing.assert_allclose(theirs, as_written, rtol=0, atol=tol)
        assert (
            np.abs(ours[N_STEPS - 2] - theirs[N_STEPS - 2]).max() > tol
        )  # the defect shows


def returning(value: np.ndarray) -> Any:
    return lambda *args, **kwargs: jnp.asarray(value)


def widened(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x, np.float64)
        if jnp.issubdtype(np.asarray(x).dtype, jnp.floating)
        else x,
        tree,
    )


def in_float64(fn: Any, x: Any, scan: str, it: Iteration, table: dict) -> Any:
    """``fn(x)`` in float64, its collection ``scan`` given the float32 run's: only the arithmetic of
    the targets and the update changes, never the collected data."""
    with jax.enable_x64(True), kit.substituting({scan: widened(it.collected[scan])}):
        out, session = taped(fn, widened(x), kit.replaying(table))
    assert not session.misses
    return out


def as_reference(state: Any, ref_leaves: dict) -> dict:
    """Ajax's state leaves under the reference paths they map to."""
    return {
        ref_path(p): x for p, x in leaves(state).items() if ref_path(p) in ref_leaves
    }


def first_true(mask: np.ndarray) -> list:
    """Per env (column), the first step where ``mask`` holds, or None."""
    return [
        int(np.argmax(mask[:, e])) if mask[:, e].any() else None
        for e in range(mask.shape[1])
    ]


def test_the_update_matches_when_both_get_the_same_targets(plain, monkeypatch):
    """Tolerance: each side's float32 rounding, its run against the same run in float64 from the
    same collection (submaps.rounding_null)."""
    sides, its = plain
    step = lambda s: sides.ajax.iteration(s, 0, 0)[0]  # noqa: E731
    ref = lambda rs: sides.ref_step(rs, None)[0]  # noqa: E731
    for i, it in enumerate(its):
        same = np.concatenate(
            [np.asarray(it.ref["_get_target"]), ajax_targets(it)[-1:]]
        )[..., None]
        monkeypatch.setattr(train_PQN, "compute_q_lambda_targets", returning(same))
        a32, _ = taped(step, it.state, kit.replaying(it.table))
        same64 = returning(same.astype(np.float64))
        monkeypatch.setattr(train_PQN, "compute_q_lambda_targets", same64)
        a64 = in_float64(step, it.state, "collect_experience", it, it.table)
        r64 = leaves(in_float64(ref, it.rs, "_step_env", it, it.own_table))
        ref32 = leaves(it.rs1)
        A, A64, start = (as_reference(s, ref32) for s in (a32, a64, it.state))
        R, R64 = {k: ref32[k] for k in A}, {k: r64[k] for k in A}
        nulls = submaps.rounding_null((R, R64), (A, A64), old=start)
        report = submaps.compare(A, R, nulls, old=start, context=f"iteration {i}")
        assert not report.failures, str(report)


def test_exploration_differs_only_by_the_per_step_epsilon():
    sides = build(constant_epsilon=False)
    cfg, m = sides.config, SPEC.matched
    events = 0
    for it in iterate(sides, 2):
        u = np.stack(
            [
                v
                for _, v in it.ref_draws[
                    role(it.ref_draws, "uniform", "eps_greedy_exploration")
                ]
            ]
        ).reshape(N_STEPS, N_ENVS)
        rand = np.stack(
            [
                v
                for _, v in it.ref_draws[
                    role(it.ref_draws, "randint", "eps_greedy_exploration")
                ]
            ]
        ).reshape(N_STEPS, N_ENVS)
        greedy = np.asarray(it.ref["_step_env"][0].q_val).argmax(-1)
        n_upd, ts = (
            int(leaves(it.rs)["[0].n_updates"]),
            int(leaves(it.rs)["[0].timesteps"]),
        )
        eps_ref = cfg["EPS_START"] + min(
            n_upd / (cfg["EPS_DECAY"] * cfg["NUM_UPDATES_DECAY"]), 1.0
        ) * (cfg["EPS_FINISH"] - cfg["EPS_START"])
        eps_ajax = np.array(
            [
                float(
                    linear_epsilon(
                        ts + N_ENVS * t,
                        SPEC.n_timesteps,
                        m["epsilon_start"],
                        m["epsilon_end"],
                        m["epsilon_decay_frac"],
                    )
                )
                for t in range(N_STEPS)
            ]
        )
        # purejaxql explores and Ajax, its epsilon already lower, does not; the random action differs.
        event = (u >= eps_ajax[:, None]) & (u < eps_ref) & (rand != greedy)
        a, r = rollouts(it, cfg["REW_SCALE"])
        mismatch = a["action"] != r["action"]
        assert first_true(mismatch) == first_true(event)
        events += int(event.sum())
    assert events > 0


def test_time_limit_cuts_change_only_their_episodes_targets():
    sides = build(constant_epsilon=True, limit=16)
    [it] = iterate(sides, 1)
    a, r = rollouts(it, sides.config["REW_SCALE"])
    for k in r:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(r[k]), err_msg=k)
    ours, theirs = ajax_targets(it)[:-1], np.asarray(it.ref["_get_target"])
    trunc = np.asarray(it.roll.truncated[..., 0] > 0)[:-1]
    done = r["done"][:-1]
    cut, chain = (
        np.zeros_like(trunc),
        np.zeros_like(trunc),
    )  # episodes ending in a cut; the off-by-one's reach
    for e in range(N_ENVS):
        start = 0
        for t in np.nonzero(done[:, e])[0]:
            cut[start : t + 1, e] = trunc[t, e]
            start = t + 1
        chain[start:, e] = True
    beyond = np.abs(ours - theirs) > TARGET_TOL * np.abs(theirs).max()
    assert not (beyond & ~cut & ~chain).any()
    assert (beyond & cut & ~chain).any()


def test_every_purejaxql_deviation_is_pinned_here():
    for d in PUREJAXQL_DEVIATIONS:
        path, name = d.check.split("::")
        assert path.endswith(__file__.split("/tests/")[-1]) and name in globals(), (
            d.name
        )
