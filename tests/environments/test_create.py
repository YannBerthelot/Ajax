import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.create import (
    add_ajax_wrappers,
    build_env_from_id,
    prepare_env,
    strip_ajax_wrappers,
)
from ajax.environments.utils import wrapper_chain
from ajax.wrappers import (
    ClipAction,
    InitialStateWrapper,
    NormalizeVecObservationBrax,
    NormalizeVecObservationGymnax,
)


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


def test_strip_ajax_wrappers_reads_back_the_layers_prepare_env_added():
    """The evaluation rebuild's two halves: the task under prepare_env's
    layers (a user's wrapper kept) and the keywords that rebuild them."""
    env, _ = make_gymnax_env("Pendulum-v1")
    task = InitialStateWrapper(env, lambda key, state, params: state)
    trained, *_ = prepare_env(task, normalize_obs=True, apply_obs_normalization=False)
    stripped, layers = strip_ajax_wrappers(trained)
    assert stripped is task and strip_ajax_wrappers(task) == (task, {})
    assert layers == {
        "normalize_obs": True,
        "normalize_reward": False,
        "gamma": None,
        "apply_obs_normalization": False,
    }
    rebuilt = add_ajax_wrappers(stripped, **layers)
    assert list(map(type, wrapper_chain(rebuilt))) == list(
        map(type, wrapper_chain(trained))
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_prepare_env_clips_continuous_actions_to_their_space(normalize):
    """Whatever the normalisation flags: a discrete env is never clipped, a
    continuous one always, to its own bounds (Pendulum's torque is 2)."""
    env, *_ = prepare_env("CartPole-v1", normalize_obs=normalize)
    assert not any(isinstance(layer, ClipAction) for layer in wrapper_chain(env))
    env, params, *_ = prepare_env("Pendulum-v1", normalize_obs=normalize)
    assert isinstance(env, ClipAction)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)

    def reward(a: float) -> float:
        return float(env.step(key, state, jnp.array([a]), params)[2])

    assert reward(3.0) == reward(2.0) != reward(1.0)
