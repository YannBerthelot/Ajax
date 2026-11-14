"""A bare mujoco_playground env object given to an agent instead of an id.

Contract (ajax.environments.create.prepare_env, ajax.evaluate.setup_environment):

- the object gets the wrapper stack an env built from an id gets, with the
  env's own time limit (its ``_config.episode_length``) unless the caller
  gives one, and the caller's action repeat;
- evaluation wraps the same object again, keeping the time limit, since
  the object has no id to be rebuilt from.

The object here is gymnasium-mjx's Reacher-v5: its 50-step time limit tells
the env's own length apart from the 1000-step fallback.
"""

import pytest

from ajax.environments.create import prepare_env
from ajax.environments.utils import env_action_repeat, get_raw_env
from ajax.evaluate import setup_environment

pytest.importorskip("mujoco_playground")
gymnasium_mjx = pytest.importorskip("gymnasium_mjx")
playground = pytest.importorskip("gymnasium_mjx.playground")


def reacher():
    return playground.PlaygroundTask(gymnasium_mjx.make("Reacher-v5"))


def test_a_bare_object_is_wrapped_with_its_own_time_limit():
    raw = reacher()

    env, env_params, _, continuous = prepare_env(raw, n_envs=2)

    assert get_raw_env(env) is raw
    assert env.episode_length == 50
    assert env_params is None
    assert continuous


def test_a_given_time_limit_and_action_repeat_apply():
    env, *_ = prepare_env(reacher(), episode_length=20, n_envs=2, action_repeat=2)

    assert env.episode_length == 20
    assert env_action_repeat(env) == 2


def test_evaluation_wraps_the_same_object_again():
    raw = reacher()
    env, env_params, *_ = prepare_env(raw, n_envs=2)

    evaluation_env, _, _ = setup_environment(
        env, env_params, num_episodes=3, norm_info=None, gamma=0.99
    )

    assert get_raw_env(evaluation_env) is raw
    assert evaluation_env.episode_length == 50
