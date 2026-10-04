"""Self-test of the CartPole port against gymnax (PROTOCOL.md 2.2 "Port self-test").

Named ``check_*`` (PROTOCOL.md calls it ``test_cartpole_env.py``) so that pytest
does not collect it: it is a script, run by hand. Run with the Ajax venv (it has
gymnax 1.0.0 and numpy; the port imports without ``embodied``):

    JAX_PLATFORMS=cpu $APY docs/world_models/reference_comparison/check_cartpole_port.py

Checks
1. single step, 400 random float32 states inside the thresholds (incl. near
   the thresholds) x both actions: obs within 1e-6 abs, terminated /
   truncated / reward exactly equal (gymnax ``CartPole().step``, jitted);
2. lockstep rollout of 4000 random-action steps: before every step the port is
   put in gymnax's current state (so both step from the same state), then
   both step with the same action; obs / flags / reward compared as in 1;
   gymnax's auto-reset state is copied into the port at episode ends;
3. row structure of a port rollout with resets (600 rows): L + 1 rows per
   episode, reset rows reward 0 / is_first, is_terminal only for terminated,
   truncation at time 500 forced from time 499 gives is_terminal False and
   the recorder logs it with terminal False;
4. reset distribution: 20000 port draws vs 20000 gymnax draws per variable:
   range [-0.05, 0.05), mean and std match (|diff| < 5 standard errors), and
   two-sample KS statistic below the 0.1 % critical value.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cartpole_env as ce  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gymnax.environments.classic_control.cartpole import (  # noqa: E402
    CartPole,
    EnvState,
)

ENV = CartPole()
PARAMS = ENV.default_params
STEP = jax.jit(lambda k, s, a: ENV.step(k, s, a, PARAMS))
TOL = 1e-6


def gx_state(vec, time):
    vec = np.asarray(vec, np.float32)
    return EnvState(
        x=jnp.float32(vec[0]),
        x_dot=jnp.float32(vec[1]),
        theta=jnp.float32(vec[2]),
        theta_dot=jnp.float32(vec[3]),
        time=jnp.int32(time),
    )


def gx_vec(s):
    return np.array([s.x, s.x_dot, s.theta, s.theta_dot], np.float32)


def compare_step(port, key, vec, time, action, stats):
    """Step gymnax and the port from (vec, time) with action; compare."""
    s = gx_state(vec, time)
    _, s_new, r, term, trunc, info = STEP(key, s, jnp.int32(action))
    obs_gx = np.asarray(info["final_observation"], np.float32)
    port.set_state(vec, time)
    o = port.step({"action": np.int32(action), "reset": False})
    err = np.abs(o["vector"] - obs_gx).max()
    stats["max_err"] = max(stats["max_err"], float(err))
    stats["n"] += 1
    assert err <= TOL, (vec, action, o["vector"], obs_gx)
    assert o["is_terminal"] == bool(term), (vec, action, o, term)
    assert o["is_last"] == bool(term or trunc), (vec, action, o, term, trunc)
    assert np.float32(o["reward"]) == np.float32(r), (o["reward"], r)
    return s_new, bool(term), bool(trunc), obs_gx


def test_single_steps(rng, stats):
    lim = np.array([2.4, 3.0, ce.THETA_THRESHOLD, 3.0], np.float32)
    for i in range(400):
        vec = rng.uniform(-1, 1, 4).astype(np.float32) * lim
        if i % 4 == 0:  # near the thresholds: termination decided on the edge
            vec[0] = np.float32(np.sign(vec[0]) * rng.uniform(2.3, 2.4))
            vec[2] = np.float32(np.sign(vec[2]) * rng.uniform(0.19, 0.2094))
        time = 499 if i % 10 == 1 else int(rng.integers(0, 500))  # 499: truncation
        for a in (0, 1):
            port = ce.CartPolePort([0, 0])
            port._row = 0
            _, term, trunc, _ = compare_step(
                port, jax.random.PRNGKey(i), vec, time, a, stats
            )
            stats["trunc_only"] += trunc and not term


def test_lockstep(rng, stats, n_steps=4000):
    key = jax.random.PRNGKey(123)
    key, k0 = jax.random.split(key)
    _, s = ENV.reset(k0, PARAMS)
    port = ce.CartPolePort([0, 1])
    port._row = 0
    n_term = n_trunc = 0
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        a = int(rng.integers(0, 2))
        s_new, term, trunc, _ = compare_step(port, k, gx_vec(s), int(s.time), a, stats)
        n_term += term
        n_trunc += trunc
        s = s_new  # gymnax auto-resets on done: the next step starts from its reset
    assert n_term > 50, n_term
    return n_term, n_trunc


def test_rows():
    rec = []
    port = ce.CartPolePort([5, 3], index=3, recorder=rec)
    rng = np.random.default_rng(7)
    rows, act = [], {"action": np.int32(0), "reset": True}
    forced = False
    for t in range(600):
        if t >= 300 and not forced and not act["reset"]:
            # force a truncation: the next step is the 500th of the episode
            forced = True
            port._time = 499
            port._state = np.zeros(4, np.float32)
        o = port.step(act)
        rows.append(o)
        act = {"action": np.int32(rng.integers(0, 2)), "reset": bool(o["is_last"])}
    # structure: a reset row starts every episode; L + 1 rows per episode
    assert rows[0]["is_first"] and rows[0]["reward"] == 0
    starts = [i for i, o in enumerate(rows) if o["is_first"]]
    lasts = [i for i, o in enumerate(rows) if o["is_last"]]
    for st, la in zip(starts, lasts):
        assert la > st
        body = rows[st + 1 : la + 1]
        assert all(o["reward"] == 1.0 and not o["is_first"] for o in body)
        assert all(not o["is_last"] for o in body[:-1])
        assert rows[st]["reward"] == 0.0 and not rows[st]["is_last"]
    for la, nxt in zip(lasts, starts[1:]):
        assert nxt == la + 1  # the row after is_last is the next reset row
    assert len(rec) == len(lasts)
    trunc = [r for r in rec if not r["terminal"]]
    assert len(trunc) == 1, trunc
    tr = trunc[0]
    assert not rows[tr["row_last"]]["is_terminal"] and rows[tr["row_last"]]["is_last"]
    for r, la, st in zip(rec, lasts, starts):
        assert (
            r["row_last"] == la and r["length"] == la - st and r["score"] == r["length"]
        )
        assert r["worker"] == 3
        assert rows[la]["is_terminal"] == r["terminal"]
    return len(rec)


def test_reset_distribution(n=20000):
    port = ce.CartPolePort([11, 0])
    a = np.stack([port.reset_state() for _ in range(n)])
    keys = jax.random.split(jax.random.PRNGKey(99), n)
    b = np.asarray(jax.vmap(lambda k: gx_vec_j(ENV.reset_env(k, PARAMS)[1]))(keys))
    assert a.dtype == np.float32 and b.dtype == np.float32
    out = []
    crit = 1.95 * np.sqrt(2 / n)  # two-sample KS, alpha = 0.001
    for j in range(4):
        x, y = np.sort(a[:, j]), np.sort(b[:, j])
        assert (
            x.min() >= -0.05 and x.max() < 0.05 and y.min() >= -0.05 and y.max() <= 0.05
        )
        se = np.sqrt(x.var() / n + y.var() / n)
        assert abs(x.mean() - y.mean()) < 5 * se, (j, x.mean(), y.mean())
        assert abs(x.std() - y.std()) < 5 * x.std() / np.sqrt(n), (j, x.std(), y.std())
        grid = np.concatenate([x, y])
        ks = np.max(
            np.abs(
                np.searchsorted(x, grid, "right") / n
                - np.searchsorted(y, grid, "right") / n
            )
        )
        assert ks < crit, (j, ks, crit)
        out.append(
            (
                float(x.mean()),
                float(y.mean()),
                float(x.std()),
                float(y.std()),
                float(ks),
            )
        )
    return out, crit


def gx_vec_j(s):
    return jnp.stack([s.x, s.x_dot, s.theta, s.theta_dot])


def main():
    rng = np.random.default_rng(0)
    stats = {"max_err": 0.0, "n": 0, "trunc_only": 0}
    test_single_steps(rng, stats)
    assert stats["trunc_only"] > 20, stats
    print(
        f"[1] single steps: {stats['n']} compared ({stats['trunc_only']} time-limit ends"
        f" with is_terminal False), max |obs diff| {stats['max_err']:.3g}"
    )
    n0 = stats["n"]
    n_term, n_trunc = test_lockstep(rng, stats)
    print(
        f"[2] lockstep rollout: {stats['n'] - n0} steps, {n_term} terminations,"
        f" {n_trunc} truncations, max |obs diff| so far {stats['max_err']:.3g}"
    )
    n_ep = test_rows()
    print(f"[3] row structure: 600 rows, {n_ep} episodes, forced truncation ok")
    dist, crit = test_reset_distribution()
    for j, (mx, my, sx, sy, ks) in enumerate(dist):
        print(
            f"[4] reset var {j}: mean port {mx:+.5f} gymnax {my:+.5f};"
            f" std port {sx:.5f} gymnax {sy:.5f} (uniform 0.02887); KS {ks:.4f} < {crit:.4f}"
        )
    print("ALL PORT TESTS PASSED")


if __name__ == "__main__":
    main()
