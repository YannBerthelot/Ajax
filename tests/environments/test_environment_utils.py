import gymnax
import pytest
from brax.envs import create as create_brax_env

from ajax.environments.utils import (
    check_env_is_brax,
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
    get_action_dim,
    get_env_type,
    get_raw_env,
    get_state_action_shapes,
)
from ajax.wrappers import ClipAction


def test_check_continuous_actions_under_trace():
    """The check must survive being called inside a jit trace.

    Regression: envs that build Box bounds with ``jnp.array`` (e.g.
    target_gym Plane) hand a tracer to gymnax>=1.0's ``Box.__repr__``,
    which raised when the check was repr-based.
    """
    import jax
    import jax.numpy as jnp
    from gymnax.environments import spaces

    class TracedBoxEnv:
        def action_space(self, params=None):
            return spaces.Box(
                low=jnp.array([-1.0, -1.0]),
                high=jnp.array([1.0, 1.0]),
                shape=(2,),
                dtype=jnp.float32,
            )

    env = TracedBoxEnv()

    @jax.jit
    def probe(x):
        assert check_if_environment_has_continuous_actions(env)
        assert get_action_dim(env) == 2
        return x

    probe(jnp.zeros(()))


#: ``(build, is_brax, is_gymnax, env type, continuous, action dim, obs
#: shape, action shape)``; built lazily (in the test).
ENVS = {
    "CartPole-v1": (
        lambda: gymnax.make("CartPole-v1")[0],
        False,
        True,
        "gymnax",
        False,
        2,
        (4,),
        (1,),
    ),
    "Pendulum-v1": (
        lambda: gymnax.make("Pendulum-v1")[0],
        False,
        True,
        "gymnax",
        True,
        1,
        (3,),
        (1,),
    ),
    "fast": (
        lambda: create_brax_env("fast"),
        True,
        False,
        "brax",
        True,
        1,
        (2,),
        (1,),
    ),
}


@pytest.mark.parametrize("env_id", list(ENVS))
def test_env_predicates_and_shapes(env_id):
    build, is_brax, is_gymnax, env_type, continuous, action_dim, obs, action = ENVS[
        env_id
    ]
    env = build()
    assert check_env_is_brax(env) == is_brax
    assert check_env_is_gymnax(env) == is_gymnax
    assert get_env_type(env) == env_type
    assert check_if_environment_has_continuous_actions(env) == continuous
    assert get_action_dim(env, getattr(env, "default_params", None)) == action_dim
    assert get_state_action_shapes(env) == (obs, action)


def test_get_env_type_rejects_unknown_envs():
    with pytest.raises(ValueError):
        get_env_type("unsupported_env")


def test_get_raw_env():
    env, _ = gymnax.make("CartPole-v1")
    assert get_raw_env(env) is env
    assert get_raw_env(ClipAction(env)) is env
