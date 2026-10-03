"""Playground auto-reset: every episode must start from a fresh initial state.

``FreshAutoResetWrapper`` (the default; ``build_env_from_id(...,
fresh_reset=True)``) draws a new initial state per episode, as dm_control
does. Playground's ``BraxAutoResetWrapper`` (``full_reset=False``), kept
behind ``fresh_reset=False`` to reproduce earlier results, restarts every
episode of env i from the same cached first state, so a run sees only
``n_envs`` initial conditions. Rollouts are vmapped over seeds because that
is how every Ajax agent runs (``ajax.agents.base``), and it is the case the
wrapper's gating is designed for.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.environments.create import build_env_from_id
from ajax.wrappers import FreshAutoResetWrapper, _call_if_any


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)

N_ENVS, EPISODE_LENGTH, N_STEPS, N_SEEDS = 2, 3, 9, 2  # 4 episode starts each


def _rollout(env, seeds):
    """Zero-action rollout of ``env`` vmapped over ``seeds``. Returns the reset
    observation and the per-step state fields the tests inspect, each with
    leading (seed, time, env) axes."""

    def run(seed):
        state = env.reset(jax.random.PRNGKey(seed))
        action = jnp.zeros((N_ENVS, env.action_size))

        def body(state, _):
            prev_rng = state.info["rng"]
            state = env.step(state, action)
            return state, {
                "obs": state.obs,
                "done": state.done,
                "truncation": state.info["truncation"],
                "final_obs": state.info["final_obs"],
                "episode_done": state.info["episode_done"],
                "episode_length": state.info["episode_metrics"]["length"],
                "rng_changed": jnp.any(state.info["rng"] != prev_rng, axis=-1),
            }

        _, out = jax.lax.scan(body, state, None, length=N_STEPS)
        return state.obs, out

    first_obs, out = jax.jit(jax.vmap(run))(seeds)
    return np.asarray(first_obs), jax.tree.map(np.asarray, out)


def _episode_starts(first_obs, out, seed, env_idx):
    """Initial observation of every episode of one env, in order."""
    done = out["done"][seed, :, env_idx] > 0
    later = [out["obs"][seed, t, env_idx] for t in np.flatnonzero(done)]
    return np.stack([first_obs[seed, env_idx], *later])


def _n_distinct(rows):
    return len(np.unique(np.round(rows, 6), axis=0))


@requires_playground
def test_opting_out_restores_the_cached_reset():
    """``fresh_reset=False`` keeps playground's cached semantics, the setting
    playground results produced before fresh resets became the default were
    trained with: every episode of an env restarts from the same
    observation."""
    from mujoco_playground._src.wrapper import BraxAutoResetWrapper

    env, _ = build_env_from_id(
        "CartpoleBalance",
        n_envs=N_ENVS,
        episode_length=EPISODE_LENGTH,
        fresh_reset=False,
    )
    assert isinstance(env.env, BraxAutoResetWrapper)
    first_obs, out = _rollout(env, jnp.arange(N_SEEDS))
    starts = _episode_starts(first_obs, out, seed=0, env_idx=0)
    assert len(starts) == 4
    assert _n_distinct(starts) == 1


@requires_playground
def test_default_starts_every_episode_from_a_new_state():
    env, _ = build_env_from_id(
        "CartpoleBalance", n_envs=N_ENVS, episode_length=EPISODE_LENGTH
    )
    assert isinstance(env.env, FreshAutoResetWrapper)
    first_obs, out = _rollout(env, jnp.arange(N_SEEDS))
    for seed in range(N_SEEDS):
        for env_idx in range(N_ENVS):
            starts = _episode_starts(first_obs, out, seed, env_idx)
            assert len(starts) == 4
            assert _n_distinct(starts) == 4, (seed, env_idx, starts)


@requires_playground
def test_fresh_reset_keeps_the_transition_bookkeeping():
    """On a done step the new episode's state comes from the reset, but what
    describes the finished transition must survive it: ``truncation`` (else
    the time limit reads as a termination), ``final_obs`` (else V(s_T) is
    bootstrapped on the reset observation) and EpisodeWrapper's episode
    summary. The base env's own per-episode info (its ``rng``) is renewed."""
    env, _ = build_env_from_id(
        "CartpoleBalance", n_envs=N_ENVS, episode_length=EPISODE_LENGTH
    )
    _, out = _rollout(env, jnp.arange(N_SEEDS))
    done = out["done"] > 0
    assert done.sum() == N_SEEDS * N_ENVS * (N_STEPS // EPISODE_LENGTH)
    assert np.all(out["truncation"][done] == 1)
    assert np.all(np.any(out["final_obs"][done] != out["obs"][done], axis=-1))
    assert np.all(out["final_obs"][~done] == out["obs"][~done])
    assert np.all(out["episode_done"][done] == 1)
    assert np.all(out["episode_length"][done] == EPISODE_LENGTH)
    np.testing.assert_array_equal(out["rng_changed"], done)


@requires_playground
def test_fresh_reset_with_differentiable_reset_gives_the_same_rollout():
    """``differentiable_reset=True`` evaluates the reset on every step (for
    reverse mode through it, see ``build_env_from_id``) with the same keys,
    so the rollout is the same: exactly in its discrete parts, and up to
    float rounding (the reset compiles differently outside the gate's
    while_loop) in its states."""
    rollouts = {}
    for differentiable_reset in (False, True):
        env, _ = build_env_from_id(
            "CartpoleBalance",
            n_envs=N_ENVS,
            episode_length=EPISODE_LENGTH,
            fresh_reset=True,
            differentiable_reset=differentiable_reset,
        )
        assert env.env.differentiable_reset is differentiable_reset
        rollouts[differentiable_reset] = _rollout(env, jnp.arange(N_SEEDS))
    (first_gated, gated), (first_every, every) = rollouts[False], rollouts[True]
    np.testing.assert_allclose(first_gated, first_every, rtol=1e-5, atol=1e-5)
    for name in ("done", "truncation", "rng_changed"):
        np.testing.assert_array_equal(gated[name], every[name])
    for name in ("obs", "final_obs"):
        np.testing.assert_allclose(gated[name], every[name], rtol=1e-5, atol=1e-5)


@requires_playground
def test_eval_rebuild_keeps_the_reset_semantics():
    from ajax.evaluate import setup_environment

    for fresh_reset in (False, True):
        env, _ = build_env_from_id("CartpoleBalance", n_envs=1, fresh_reset=fresh_reset)
        eval_env, _, _ = setup_environment(
            env, None, num_episodes=2, norm_info=None, gamma=0.99
        )
        assert eval_env._ajax_fresh_reset is fresh_reset


def _fn(keys):
    """Deterministic in ``keys``; returns a pytree to exercise the zeros."""
    return {"a": keys.astype(jnp.float32).sum(-1), "b": jnp.ones(keys.shape[0])}


def test_call_if_any_returns_fn_or_zeros_per_vmapped_element():
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    batched_keys = jnp.stack([keys, keys])
    pred = jnp.array([[False, True, False], [False, False, False]])
    out = jax.jit(jax.vmap(lambda p, k: _call_if_any(p, _fn, k)))(pred, batched_keys)
    expected = _fn(jax.vmap(jax.random.split)(keys)[:, 1])
    for name in ("a", "b"):
        np.testing.assert_array_equal(out[name][0], expected[name])
        np.testing.assert_array_equal(out[name][1], 0.0)


def test_call_if_any_passes_fn_the_keys_advance_returns():
    """``advance(keys)`` returns ``(next_keys, subkeys)`` and ``fn`` gets
    ``subkeys``; ``AutoResetWrapper`` uses this to key its reset with the
    next value of its seed stream."""

    def advance(key):
        key = jax.random.split(key)[0]
        return key, key

    key = jax.random.PRNGKey(0)
    pred = jnp.array([[False, True], [False, False]])
    out = jax.jit(
        jax.vmap(lambda p, k: _call_if_any(p, lambda s: s, k, advance=advance))
    )(pred, jnp.stack([key, key]))
    np.testing.assert_array_equal(out[0], jax.random.split(key)[0])
    np.testing.assert_array_equal(out[1], 0)


def test_call_if_any_does_not_evaluate_fn_when_nothing_is_set():
    """The point of the helper is cost: with an all-false (batched) predicate
    an expensive ``fn`` must not run. This catches both regressions measured
    while designing it: a ``lax.cond`` (lowered to ``select`` under vmap) and
    closing ``fn`` over constant keys (hoisted out of the loop by XLA). The
    real gap is orders of magnitude; the bound leaves wide margin."""

    def expensive(keys):
        x = jax.vmap(lambda k: jax.random.normal(k, (256,)))(keys)
        w = jax.random.normal(jax.random.PRNGKey(1), (256, 256)) / 16.0

        def layer(_, x):
            return jnp.tanh(x @ w)

        return jax.lax.fori_loop(0, 500, layer, x)

    keys = jnp.stack([jax.random.split(jax.random.PRNGKey(s), 64) for s in (0, 1)])
    nothing_set = jnp.zeros((2, 64), dtype=bool)
    gated = jax.jit(jax.vmap(lambda p, k: _call_if_any(p, expensive, k)))
    always = jax.jit(jax.vmap(expensive))

    def best_time(f, *args):
        jax.block_until_ready(f(*args))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            jax.block_until_ready(f(*args))
            times.append(time.perf_counter() - start)
        return min(times)

    assert best_time(gated, nothing_set, keys) < 0.1 * best_time(always, keys)
