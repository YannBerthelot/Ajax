"""Tests for system classes (distributions over EnvParams)."""

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.interaction import reset, step
from ajax.environments.system_class import (
    FixedSystem,
    UniformPerturbation,
    broadcast_env_params,
    env_params_is_batched,
    select_env_params,
)

N = 6


@pytest.fixture
def pendulum():
    return make_gymnax_env("Pendulum-v1")


def test_unbatched_params_are_detected_as_such(pendulum):
    _, params = pendulum
    assert not env_params_is_batched(params)
    assert not env_params_is_batched(None)


def test_broadcast_and_select_round_trip(pendulum):
    _, params = pendulum
    batched = broadcast_env_params(params, N)
    assert env_params_is_batched(batched)
    for leaf in jax.tree.leaves(batched):
        assert leaf.shape == (N,)
    # broadcasting an already-batched params is a no-op
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.array_equal(a, b),
            broadcast_env_params(batched, N),
            batched,
        )
    )
    single = select_env_params(batched, 2)
    assert not env_params_is_batched(single)
    assert float(single.m) == float(params.m)


def test_fixed_system_replicates_nominal(pendulum):
    _, params = pendulum
    batched = FixedSystem(params).sample(jax.random.PRNGKey(0), N)
    assert jnp.all(batched.m == params.m)
    assert jnp.all(batched.l == params.l)


def test_uniform_perturbation_rejects_unknown_field_and_negative_scale(pendulum):
    _, params = pendulum
    with pytest.raises(ValueError, match="Unknown EnvParams fields"):
        UniformPerturbation(params, fields=("mass",))
    with pytest.raises(ValueError, match="non-negative"):
        UniformPerturbation(params, fields=("m",), scale=-0.1)


def test_uniform_perturbation_only_touches_listed_fields_within_scale(pendulum):
    _, params = pendulum
    sc = UniformPerturbation(params, fields=("m", "l"), scale=0.1)
    batched = sc.sample(jax.random.PRNGKey(3), 512)
    for name in ("m", "l"):
        values = getattr(batched, name)
        nominal = float(getattr(params, name))
        assert values.shape == (512,)
        assert jnp.all(values >= nominal * 0.9) and jnp.all(values <= nominal * 1.1)
        assert values.std() > 0  # actually perturbed
    # untouched fields stay exactly nominal
    assert jnp.all(batched.g == params.g)
    assert jnp.all(batched.dt == params.dt)


def test_uniform_perturbation_is_deterministic_under_key(pendulum):
    _, params = pendulum
    sc = UniformPerturbation(params, fields=("m",), scale=0.05)
    a = sc.sample(jax.random.PRNGKey(7), N)
    b = sc.sample(jax.random.PRNGKey(7), N)
    c = sc.sample(jax.random.PRNGKey(8), N)
    assert jnp.array_equal(a.m, b.m)
    assert not jnp.array_equal(a.m, c.m)


def test_reset_and_step_accept_batched_params(pendulum):
    """Batched params vmap through reset/step: each env sees its own system."""
    env, params = pendulum
    sc = UniformPerturbation(params, fields=("g",), scale=0.5)
    batched = sc.sample(jax.random.PRNGKey(0), N)
    keys = jax.random.split(jax.random.PRNGKey(1), N)
    obs, state = reset(keys, env, "gymnax", batched)
    assert obs.shape == (N, 3)
    # Same state and same (zero) torque in every env: the next angular
    # velocity differs only through each env's own gravity.
    state = jax.tree.map(lambda x: jnp.broadcast_to(x[0], x.shape), state)
    action = jnp.zeros((N, 1))
    _, new_state, reward, term, trunc, _ = step(
        keys, state, action, env, "gymnax", batched
    )
    assert reward.shape == (N,) and term.shape == (N,) and trunc.shape == (N,)
    expected = (
        state.theta_dot
        + 3 * batched.g / (2 * batched.l) * jnp.sin(state.theta) * batched.dt
    )
    assert jnp.allclose(new_state.theta_dot, expected, atol=1e-5)
    assert new_state.theta_dot.std() > 0


def test_step_gradient_flows_through_batched_params(pendulum):
    """The gymnax step is differentiable w.r.t. action and params.

    gymnax >= 1.0 detaches transitions unless the env is configured with
    ``with_transition_gradients()``; the batched-params path must keep
    those gradients intact.
    """
    env, params = pendulum
    env = env.with_transition_gradients()
    batched = broadcast_env_params(params, N)
    keys = jax.random.split(jax.random.PRNGKey(1), N)
    _, state = reset(keys, env, "gymnax", batched)

    def next_theta_dot(action, mass):
        # differentiate w.r.t. a float field only (max_steps_in_episode is int)
        _, s, *_ = step(keys, state, action, env, "gymnax", batched.replace(m=mass))
        return s.theta_dot.sum()

    g_action, g_mass = jax.grad(next_theta_dot, argnums=(0, 1))(
        jnp.full((N, 1), 0.5), batched.m
    )
    assert jnp.all(jnp.isfinite(g_action)) and jnp.any(g_action != 0)
    assert jnp.all(jnp.isfinite(g_mass)) and jnp.any(g_mass != 0)
