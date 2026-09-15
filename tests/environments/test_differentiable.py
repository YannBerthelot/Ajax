"""Tests for differentiable closed-loop rollouts."""

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.differentiable import (
    Rollout,
    closed_loop_rollout,
    with_transition_gradients,
)
from ajax.environments.system_class import UniformPerturbation
from ajax.networks.memory import MemoryCell, MemoryConfig, init_carry
from ajax.wrappers import ClipAction

B, T = 4, 10


@pytest.fixture
def pendulum():
    env, params = make_gymnax_env("Pendulum-v1")
    return env, params


def linear_policy(gain):
    """Memoryless proportional controller u = -gain * theta_dot."""

    def policy_step(carry, obs, resets):
        del resets
        return (-gain * obs[:, 2:3]), carry

    return policy_step


def total_return(rollout: Rollout) -> jax.Array:
    return rollout.reward.sum()


# ------------------------------------------------- with_transition_gradients


def test_with_transition_gradients_rejects_unsupported_env():
    env, _ = make_gymnax_env("CartPole-v1")
    with pytest.raises(ValueError, match="transition gradients"):
        with_transition_gradients(env)


def test_with_transition_gradients_walks_wrappers(pendulum):
    env, _ = pendulum
    wrapped = ClipAction(env)
    enabled = with_transition_gradients(wrapped)
    assert isinstance(enabled, ClipAction)
    assert enabled._env.transition_gradients_enabled
    assert not env.transition_gradients_enabled  # original untouched
    # idempotent
    assert with_transition_gradients(enabled)._env.transition_gradients_enabled


# --------------------------------------------------------- closed_loop_rollout


def test_rollout_shapes_and_reset_flags(pendulum):
    env, params = pendulum
    rollout, carry = closed_loop_rollout(
        linear_policy(0.1), None, jax.random.PRNGKey(0), env, params, T, n_envs=B
    )
    assert carry is None
    assert rollout.obs.shape == (T, B, 3)
    assert rollout.next_obs.shape == (T, B, 3)
    assert rollout.action.shape == (T, B, 1)
    assert rollout.reward.shape == (T, B)
    assert rollout.done.shape == (T, B) and rollout.done.dtype == bool
    assert bool(rollout.resets[0].all()) and not bool(rollout.resets[1:].any())
    # next_obs feeds the next step's obs
    assert jnp.array_equal(rollout.next_obs[:-1], rollout.obs[1:])


def test_rollout_requires_n_envs_for_unbatched_params(pendulum):
    env, params = pendulum
    with pytest.raises(ValueError, match="n_envs is required"):
        closed_loop_rollout(
            linear_policy(0.1), None, jax.random.PRNGKey(0), env, params, T
        )


def test_rollout_infers_batch_from_batched_params_and_checks_n_envs(pendulum):
    env, params = pendulum
    batched = UniformPerturbation(params, fields=("m",), scale=0.3).sample(
        jax.random.PRNGKey(1), B
    )
    rollout, _ = closed_loop_rollout(
        linear_policy(0.1), None, jax.random.PRNGKey(0), env, batched, T
    )
    assert rollout.reward.shape == (T, B)
    with pytest.raises(ValueError, match="batch of"):
        closed_loop_rollout(
            linear_policy(0.1),
            None,
            jax.random.PRNGKey(0),
            env,
            batched,
            T,
            n_envs=B + 1,
        )


def test_each_env_runs_its_own_system(pendulum):
    """Same keys, same policy, different masses -> different trajectories."""
    env, params = pendulum
    sc = UniformPerturbation(params, fields=("m", "l"), scale=0.5)
    batched = sc.sample(jax.random.PRNGKey(1), B)
    # identical initial conditions across envs
    same_key = jax.random.PRNGKey(0)

    def run(p):
        r, _ = closed_loop_rollout(linear_policy(0.5), None, same_key, env, p, T)
        return r

    rollout = run(batched)
    # different systems (and per-env reset keys) -> different returns,
    # and the batched run is deterministic under its key
    assert rollout.reward.sum(0).std() > 0
    assert jnp.array_equal(run(batched).reward, rollout.reward)


def test_gradient_flows_through_env_when_enabled(pendulum):
    env, params = pendulum
    env_grad = with_transition_gradients(env)

    def objective(gain, e):
        rollout, _ = closed_loop_rollout(
            linear_policy(gain), None, jax.random.PRNGKey(0), e, params, T, n_envs=B
        )
        return total_return(rollout)

    g_enabled = jax.grad(objective)(jnp.asarray(0.3), env_grad)
    assert jnp.isfinite(g_enabled) and g_enabled != 0
    # With gymnax's default detached transitions only the action-cost
    # term of the CURRENT step contributes, so the value differs: the
    # opt-in genuinely changes the gradient.
    g_detached = jax.grad(objective)(jnp.asarray(0.3), env)
    assert not jnp.allclose(g_enabled, g_detached)


def test_gradient_flows_through_recurrent_policy_carry(pendulum):
    """BPTT: gradient reaches params through the memory carry across steps."""
    env, params = pendulum
    env = with_transition_gradients(env)
    config = MemoryConfig(kind="gru", hidden_size=8)
    cell = MemoryCell(config)
    carry0 = init_carry(config, jax.random.PRNGKey(0), B)
    x0 = jnp.zeros((1, B, 3))
    cell_params = cell.init(jax.random.PRNGKey(1), carry0, x0, jnp.zeros((1, B), bool))
    readout = jnp.full((8, 1), 0.1)

    def make_policy(p):
        def policy_step(carry, obs, resets):
            new_carry, h = cell.apply(p["cell"], carry, obs[None], resets[None])
            return h[0] @ p["readout"], new_carry

        return policy_step

    def objective(p):
        rollout, _ = closed_loop_rollout(
            make_policy(p), carry0, jax.random.PRNGKey(0), env, params, T, n_envs=B
        )
        return total_return(rollout)

    grads = jax.grad(objective)({"cell": cell_params, "readout": readout})
    leaves = jax.tree.leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert any(jnp.any(g != 0) for g in jax.tree.leaves(grads["cell"]))


def test_rollout_is_jittable(pendulum):
    env, params = pendulum
    env = with_transition_gradients(env)

    @jax.jit
    def run(gain, key):
        rollout, _ = closed_loop_rollout(
            linear_policy(gain), None, key, env, params, T, n_envs=B
        )
        return total_return(rollout)

    assert jnp.isfinite(run(jnp.asarray(0.2), jax.random.PRNGKey(0)))
