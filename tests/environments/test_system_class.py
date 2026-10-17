"""Tests for system classes (distributions over EnvParams)."""

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

import ajax
from ajax.agents.base import ActorCritic
from ajax.environments.interaction import reset, step
from ajax.environments.system_class import (
    FixedSystem,
    UniformPerturbation,
    broadcast_env_params,
    env_params_is_batched,
    n_systems,
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
    assert n_systems(batched) == N
    single = jax.tree.map(lambda leaf: leaf[2], batched)
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


def test_matrix_valued_unbatched_params_are_not_mistaken_for_a_batch(pendulum):
    """A plant whose params hold vectors/matrices (a state-space model) is
    still one system; only a leading axis shared by every leaf is a batch."""
    from flax import struct
    from gymnax.environments.environment import EnvParams

    @struct.dataclass
    class ModelParams(EnvParams):
        max_steps_in_episode: int = 10
        A: jax.Array = struct.field(default_factory=lambda: jnp.eye(3))
        b: jax.Array = struct.field(default_factory=lambda: jnp.ones(3))

    single = ModelParams()
    assert not env_params_is_batched(single)
    batched = broadcast_env_params(single, 4)
    assert env_params_is_batched(batched)
    assert batched.A.shape == (4, 3, 3) and batched.max_steps_in_episode.shape == (4,)
    assert n_systems(batched) == 4
    assert not env_params_is_batched(jax.tree.map(lambda leaf: leaf[1], batched))


@pytest.mark.parametrize("n", [3, 4])
def test_uniform_perturbation_scales_vector_fields_per_system(n):
    """One relative factor per system scales a whole vector field; a field
    whose length equals ``n`` must not be mistaken for the batch axis."""
    from flax import struct
    from gymnax.environments.environment import EnvParams

    @struct.dataclass
    class ModelParams(EnvParams):
        max_steps_in_episode: int = 10
        k: float = 2.0
        b: jax.Array = struct.field(
            default_factory=lambda: jnp.array([1.0, 2.0, 3.0, 4.0])
        )

    nominal = ModelParams()
    batched = UniformPerturbation(nominal, fields=("k", "b"), scale=0.1).sample(
        jax.random.PRNGKey(0), n
    )
    assert env_params_is_batched(batched)
    assert batched.b.shape == (n, 4)
    factors = batched.b / nominal.b
    assert jnp.allclose(factors, factors[:, :1])  # one factor per system
    assert jnp.all(jnp.abs(factors - 1.0) <= 0.1 + 1e-6)
    assert batched.k.shape == (n,)


#: The exported agents built on ActorCritic, by name.
AGENTS = {
    name: cls
    for name, cls in vars(ajax).items()
    if isinstance(cls, type) and issubclass(cls, ActorCritic)
}


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_an_agent_takes_per_copy_params_only_if_it_traces_them(name):
    """A system class's sample carries one system per env copy. APG and the
    world models step with traced env params and take it; the shared
    loop's agents compile them as a static argument, which jax cannot
    hash, so they refuse it when built, naming the alternatives."""
    cls = AGENTS[name]
    env_id = "CartPole-v1" if name in ("DQN", "PQN", "UDRL") else "Pendulum-v1"
    env, params = make_gymnax_env(env_id)
    batched = broadcast_env_params(params, 2)
    traced = name in ("APG", "DreamerV3", "TDMPC2")
    assert cls.supports_batched_env_params == traced
    if traced:
        cls(env, env_params=batched, n_envs=2)
        return
    with pytest.raises(ValueError, match="one system per copy.*APG"):
        cls(env, env_params=batched, n_envs=2)
