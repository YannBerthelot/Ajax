import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.create import build_env_from_id, prepare_env
from ajax.wrappers import NormalizeVecObservationBrax, NormalizeVecObservationGymnax


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


@pytest.mark.parametrize(
    "env_id,expected_params",
    [
        ("CartPole-v1", True),  # Gymnax discrete environment
        ("Pendulum-v1", True),  # Gymnax continuous environment
        ("fast", False),  # Brax environment
    ],
)
def test_build_env_from_id(env_id, expected_params):
    env, env_params = build_env_from_id(env_id)
    assert env is not None
    assert (env_params is not None) == expected_params

    if env_id == "fast":
        with pytest.raises(ValueError):
            build_env_from_id("unknown_env")


@pytest.mark.parametrize(
    "env_input,normalize_obs,normalize_reward,expected_continuous,mode",
    [
        ("CartPole-v1", True, False, False, "gymnax"),  # Gymnax discrete environment
        ("Pendulum-v1", True, True, True, "gymnax"),  # Gymnax continuous environment
        ("fast", False, True, True, "brax"),  # Brax environment
    ],
)
def test_prepare_env(
    env_input, normalize_obs, normalize_reward, expected_continuous, mode
):
    gamma = 0.99  # Example gamma value for reward normalization
    env, env_params, env_id_out, continuous = prepare_env(
        env_input,
        normalize_obs=normalize_obs,
        normalize_reward=normalize_reward,
        gamma=gamma if normalize_reward else None,
    )
    assert env is not None
    if isinstance(env_input, str):
        assert env_id_out == env_input
        assert continuous == expected_continuous

    # Test Gymnax environments
    if env_input in ["CartPole-v1", "Pendulum-v1"]:
        _, real_env_params = make_gymnax_env(env_input)
        assert env_params == real_env_params
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    # Test Brax environments
    if env_input == "fast":
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    # Check if normalization wrappers are applied
    if normalize_obs or normalize_reward:
        if mode == "gymnax":
            assert isinstance(env, NormalizeVecObservationGymnax) or isinstance(
                env._env, NormalizeVecObservationGymnax
            )

        else:
            assert isinstance(env, NormalizeVecObservationBrax) or isinstance(
                env.env, NormalizeVecObservationBrax
            )


@requires_playground
def test_build_playground_env_preserves_final_obs():
    """Playground path must expose info['final_obs'] so PPO/SAC bootstrap on
    V(s_final), not V(s_reset), at truncation. Uses a tiny episode length so
    truncation fires quickly."""
    env, _ = build_env_from_id("HopperHop", n_envs=2, episode_length=5)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert "final_obs" in state.info
    step_fn = jax.jit(env.step)
    for _ in range(5):
        state = step_fn(state, jnp.zeros((2, env.action_size)))
    assert bool(jnp.all(state.done == 1.0))
    assert bool(jnp.all(state.info["truncation"] == 1.0))
    # At truncation, obs is reset obs but final_obs is the real pre-reset obs.
    diff = jnp.abs(state.obs - state.info["final_obs"]).sum()
    assert float(diff) > 1e-3


def test_build_brax_env_preserves_final_obs():
    env, _ = build_env_from_id("inverted_pendulum", n_envs=2, episode_length=5)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert "final_obs" in state.info
    step_fn = jax.jit(env.step)
    for _ in range(5):
        state = step_fn(state, jnp.zeros((2, env.action_size)))
    assert "truncation" in state.info


@requires_playground
def test_playground_unknown_env_raises():
    with pytest.raises(ValueError):
        build_env_from_id("NotARealPlaygroundEnv")


def test_env_params_given_with_a_string_id_are_kept():
    """gymnax params passed next to an env id are the env's params (they
    were silently replaced by the defaults); an agent built from the id
    trains on them, and TD-MPC2 derives its episode length from them.
    brax / playground ids take no params: passing some raises."""
    from ajax import SAC, TDMPC2

    _, default = make_gymnax_env("Pendulum-v1")
    params = default.replace(max_steps_in_episode=10, max_torque=1.0)
    _, kept, _, _ = prepare_env("Pendulum-v1", env_params=params)
    assert kept == params
    _, none, _, _ = prepare_env("Pendulum-v1")
    assert none == default
    assert SAC("Pendulum-v1", env_params=params).env_args.env_params == params
    assert TDMPC2("Pendulum-v1", env_params=params).agent_episode_length == 10
    with pytest.raises(ValueError, match="env_params"):
        prepare_env("fast", env_params=params)
