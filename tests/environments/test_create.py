import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.create import build_env_from_id, prepare_env
from ajax.wrappers import NormalizeVecObservationBrax, NormalizeVecObservationGymnax


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
