"""Brax auto-reset: the reset is evaluated only on steps where an env is done.

``AutoResetWrapper`` (the auto-reset of Ajax's brax stack, built by
``_build_brax_env``) used to evaluate a full batched reset on every step and
discard it unless some env was done. It now gates the reset with
``_call_if_any``. These tests pin down that the gating changes the cost and
nothing else: the seed stream, the reset states and every transition match
the ungated implementation, kept verbatim below as the reference (bit for
bit on a toy env, up to float rounding on brax physics, see the last test),
and that ``differentiable_reset=True`` restores that implementation for
reverse-mode gradients through the reset itself. Rollouts are vmapped over
seeds because that is how every Ajax agent runs (``ajax.agents.base``), and
it is the case the gating is designed for.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State
from brax.envs.wrappers import training as brax_training

from ajax.environments.create import build_env_from_id
from ajax.wrappers import AutoResetWrapper, FinalObsWrapper, split

N_SEEDS = 2


class _UngatedAutoResetWrapper(AutoResetWrapper):
    """``AutoResetWrapper.step`` before gating: resets on every step."""

    def step(self, state, action):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        rng = state.info["rng"][0]
        new_rng = jax.lax.cond(state.done.any(), split, lambda x: x, rng)
        new_init_state = self.reset(new_rng)
        state.info["rng"] = (
            new_rng.reshape(1, -1)
            if self.single_env
            else jnp.tile(new_rng, (self.n_envs, 1))
        )

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done,
            new_init_state.info["first_pipeline_state"],
            state.pipeline_state,
        )
        obs = where_done(new_init_state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs, info=state.info)


class _DriftEnv(Env):
    """Toy brax env whose episode boundaries depend on the reset keys.

    The reset draws ``x`` uniformly in [0, start_scale)^3; every step adds
    ``0.25 + 0.1 * action`` and the episode terminates once ``x[0] > 2``, so
    (at the default scale) episodes last 5 to 8 steps and envs fall out of
    step with each other.
    """

    def __init__(self, start_scale=1.0):
        self.start_scale = start_scale

    def reset(self, rng):
        x = self._start(rng)
        zero = jnp.zeros(())
        return State(pipeline_state=x, obs=x, reward=zero, done=zero)

    def _start(self, rng):
        return self.start_scale * jax.random.uniform(rng, (3,))

    def step(self, state, action):
        x = state.pipeline_state + 0.25 + 0.1 * action
        done = (x[0] > 2.0).astype(jnp.float32)
        return state.replace(pipeline_state=x, obs=x, reward=x.sum(), done=done)

    @property
    def observation_size(self):
        return 3

    @property
    def action_size(self):
        return 3

    @property
    def backend(self):
        return "toy"


class _ExpensiveResetEnv(_DriftEnv):
    """``_DriftEnv`` whose reset costs far more than a step and whose
    episodes start ~400 steps away from terminating."""

    def _start(self, rng):
        w = jax.random.normal(jax.random.PRNGKey(1), (128, 128)) / 12.0
        x = jax.lax.fori_loop(
            0, 200, lambda _, x: jnp.tanh(w @ x), jax.random.normal(rng, (128,))
        )
        return x[:3] - 100.0


def _stack(env, n_envs, episode_length, auto_reset_wrapper=AutoResetWrapper):
    """The wrapper stack of ``_build_brax_env``."""
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=1)
    env = brax_training.VmapWrapper(env, batch_size=n_envs)
    return auto_reset_wrapper(FinalObsWrapper(env))


def _rollout_fn(env, n_steps):
    """Jitted rollout over ``N_SEEDS`` seeds: ``actions`` (time, env, act)
    -> the state after every step, with leading (seed, time) axes."""

    def run(key, actions):
        def body(state, action):
            state = env.step(state, action)
            return state, state

        return jax.lax.scan(body, env.reset(key), actions, length=n_steps)[1]

    return jax.jit(jax.vmap(run, in_axes=(0, None)))


def _seeds():
    return jax.random.split(jax.random.PRNGKey(0), N_SEEDS)


def _assert_trees_equal(actual, expected):
    paths = jax.tree_util.tree_flatten_with_path(expected)[0]
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for (path, e), a in zip(paths, jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a, e, err_msg=jax.tree_util.keystr(path))


def test_gated_reset_reproduces_the_ungated_rollout_exactly():
    """Same seed stream, same reset states, same transitions: bit for bit,
    under jit and the seed vmap. The rollout covers steps where no env is
    done, steps where only one seed has a done env (the batched while_loop
    runs its body there, and the other seed must keep its own state), and
    both terminations and truncations."""
    n_envs, episode_length, n_steps = 4, 7, 40
    actions = jnp.zeros((n_steps, n_envs, 3))
    gated = _rollout_fn(_stack(_DriftEnv(), n_envs, episode_length), n_steps)
    ungated = _rollout_fn(
        _stack(_DriftEnv(), n_envs, episode_length, _UngatedAutoResetWrapper),
        n_steps,
    )
    states = gated(_seeds(), actions)
    _assert_trees_equal(states, ungated(_seeds(), actions))

    any_done = np.asarray(states.done).any(axis=-1)  # (seed, time)
    assert not any_done.all(axis=0).all()
    assert (any_done[0] != any_done[1]).any()
    truncation = np.asarray(states.info["truncation"])
    done = np.asarray(states.done) > 0
    assert (truncation[done] == 1).any() and (truncation[done] == 0).any()


def test_gated_reset_draws_a_fresh_start_for_every_episode():
    n_envs, episode_length, n_steps = 4, 7, 40
    actions = jnp.zeros((n_steps, n_envs, 3))
    states = _rollout_fn(_stack(_DriftEnv(), n_envs, episode_length), n_steps)(
        _seeds(), actions
    )
    done = np.asarray(states.done) > 0
    starts = np.asarray(states.obs)[done]
    assert len(starts) > 2 * N_SEEDS * n_envs
    assert len(np.unique(starts, axis=0)) == len(starts)


def test_gradients_flow_through_the_gated_step():
    """Brax is differentiable; the gate's while_loop must not get in the
    way (reverse mode cannot cross a while_loop that carries a tangent, and
    nothing here may)."""
    n_envs, episode_length, n_steps = 4, 7, 20
    gated = _rollout_fn(_stack(_DriftEnv(), n_envs, episode_length), n_steps)
    ungated = _rollout_fn(
        _stack(_DriftEnv(), n_envs, episode_length, _UngatedAutoResetWrapper),
        n_steps,
    )

    def grad(rollout):
        return jax.grad(lambda a: rollout(_seeds(), a).reward.sum())(
            jnp.full((n_steps, n_envs, 3), 0.1)
        )

    g = grad(gated)
    assert np.abs(np.asarray(g)).sum() > 0
    np.testing.assert_array_equal(g, grad(ungated))


_DIFFERENTIABLE = partial(AutoResetWrapper, differentiable_reset=True)


def test_differentiable_reset_reproduces_the_ungated_rollout_exactly():
    n_envs, episode_length, n_steps = 4, 7, 40
    actions = jnp.zeros((n_steps, n_envs, 3))
    states = _rollout_fn(
        _stack(_DriftEnv(), n_envs, episode_length, _DIFFERENTIABLE), n_steps
    )(_seeds(), actions)
    expected = _rollout_fn(
        _stack(_DriftEnv(), n_envs, episode_length, _UngatedAutoResetWrapper),
        n_steps,
    )(_seeds(), actions)
    _assert_trees_equal(states, expected)


def _reward_through_resets(start_scale, auto_reset_wrapper):
    """Total reward of a seed-vmapped rollout, as a function of a parameter
    that only the reset depends on."""
    n_envs, n_steps = 4, 20
    env = _stack(_DriftEnv(start_scale), n_envs, 7, auto_reset_wrapper)
    rewards = _rollout_fn(env, n_steps)(_seeds(), jnp.zeros((n_steps, n_envs, 3)))
    return rewards.reward.sum()


def test_reverse_mode_through_the_reset_needs_differentiable_reset():
    """What the gate costs in differentiability, and the option that gives
    it back. Reverse mode cannot cross the gate's while_loop when the reset
    depends on what is differentiated: that fails loudly, at trace time,
    and ``differentiable_reset=True`` makes it work, with the gradient of
    the ungated implementation. Forward mode works either way."""
    with pytest.raises(ValueError, match="Reverse-mode differentiation does not"):
        jax.grad(_reward_through_resets)(1.0, AutoResetWrapper)

    grad = jax.grad(_reward_through_resets)(1.0, _DIFFERENTIABLE)
    expected = jax.grad(_reward_through_resets)(1.0, _UngatedAutoResetWrapper)
    assert grad != 0
    np.testing.assert_array_equal(grad, expected)

    _, tangent = jax.jvp(
        lambda s: _reward_through_resets(s, AutoResetWrapper), (1.0,), (1.0,)
    )
    np.testing.assert_allclose(tangent, expected, rtol=1e-5)


def test_build_env_from_id_passes_differentiable_reset():
    env, _ = build_env_from_id("inverted_pendulum", n_envs=2)
    assert type(env) is AutoResetWrapper and not env.differentiable_reset
    env, _ = build_env_from_id("inverted_pendulum", n_envs=2, differentiable_reset=True)
    assert env.differentiable_reset


def test_reset_is_skipped_on_steps_where_no_env_is_done():
    """The point of the gate is cost: with no env done the reset must not
    run. This catches the regressions measured while designing
    ``_call_if_any`` (a ``lax.cond``, lowered to ``select`` under the seed
    vmap; a reset hoisted out of the loop by XLA). The real gap is orders of
    magnitude; the bound leaves a wide margin."""
    n_envs, n_steps = 4, 50
    actions = jnp.zeros((n_steps, n_envs, 3))
    gated = _rollout_fn(_stack(_ExpensiveResetEnv(), n_envs, 1000), n_steps)
    ungated = _rollout_fn(
        _stack(_ExpensiveResetEnv(), n_envs, 1000, _UngatedAutoResetWrapper),
        n_steps,
    )
    assert not np.asarray(gated(_seeds(), actions).done).any()

    def best_time(f):
        jax.block_until_ready(f(_seeds(), actions))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            jax.block_until_ready(f(_seeds(), actions))
            times.append(time.perf_counter() - start)
        return min(times)

    assert best_time(gated) < 0.1 * best_time(ungated)


@pytest.mark.slow
def test_gated_reset_matches_the_ungated_rollout_on_brax_physics():
    """Same check on a real brax env, built the way agents build it. The
    seed stream, the done flags and every reset observation match bit for
    bit, but the states only to float32 rounding: with the reset inside the
    gate's while_loop, XLA compiles the fields that brax's ``pipeline.init``
    derives from positions (centre of mass, inertia, ...) differently,
    which moves them by about one ulp, and the next steps carry that on."""
    n_envs, n_steps = 4, 80
    env, _ = build_env_from_id("inverted_pendulum", n_envs=n_envs, episode_length=30)
    assert type(env) is AutoResetWrapper
    actions = jnp.zeros((n_steps, n_envs, env.action_size))
    states = _rollout_fn(env, n_steps)(_seeds(), actions)
    expected = _rollout_fn(_UngatedAutoResetWrapper(env.env), n_steps)(
        _seeds(), actions
    )
    # differentiable_reset=True is the ungated implementation: bit for bit.
    _assert_trees_equal(
        _rollout_fn(_DIFFERENTIABLE(env.env), n_steps)(_seeds(), actions), expected
    )

    done = np.asarray(states.done) > 0
    assert done.any() and not done.any(axis=-1).all()
    for name in ("rng", "steps", "truncation"):
        np.testing.assert_array_equal(states.info[name], expected.info[name])
    np.testing.assert_array_equal(states.done, expected.done)
    np.testing.assert_array_equal(states.obs[done], expected.obs[done])
    np.testing.assert_allclose(states.obs, expected.obs, rtol=1e-5, atol=1e-5)
    for actual, ref in zip(
        jax.tree.leaves(states.pipeline_state),
        jax.tree.leaves(expected.pipeline_state),
    ):
        np.testing.assert_allclose(actual, ref, rtol=1e-5, atol=1e-5)
