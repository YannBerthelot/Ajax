"""What a newcomer's mistakes are answered with, and the observation wrapper.

Contract (docs/guide/environments.md, deploy.md, networks.md, training.md;
the CHANGELOG's "Clearer errors and checks"):
  - ``ObservationWrapper`` applies its ``observation`` to what reset and
    step return and to the finished episode's last observation in the info;
  - an unknown environment id is refused naming the closest known ids, an
    unknown prefix naming the two prefixes, and an unknown agent argument
    naming the arguments it resembles;
  - an agent on the wrong kind of actions names the agents acting in them;
  - ``predict`` refuses an observation of another shape than the
    environment's, ``policy`` and ``predict`` a seed the state does not
    hold, ``evaluate`` train's evaluations, and ``policy.act`` an
    observation without a row per environment of its carry;
  - on a gymnax image environment, an agent without ``cnn_image_shape``
    warns, naming the shape, and one with another shape is refused;
  - a budget below one rollout warns that no evaluation is recorded;
  - a Gymnasium environment whose package Ajax's extra does not hold is
    not answered with that extra, and a ``mujoco:`` id without MuJoCo names
    the playground extra.
"""

import importlib.util
import sys
import warnings

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, PQN, SAC
from ajax.environments.create import build_env_from_id
from ajax.logging.progress import Evaluations
from ajax.wrappers import ObservationWrapper

POSITIONS_ONLY = np.array([1.0, 0.0, 1.0, 0.0], np.float32)


class HideVelocities(ObservationWrapper):
    def observation(self, obs):
        return obs * jnp.asarray(POSITIONS_ONLY)


def test_an_observation_wrapper_changes_every_observation_the_agent_reads():
    """The same keys and actions drive CartPole bare and wrapped: every
    wrapped observation is the bare one with its velocities at zero, the
    finished episode's last one included. Pushing left alone drops the pole
    within the 30 steps, so an episode ends."""
    bare, params = gymnax.make("CartPole-v1")
    wrapped = HideVelocities(bare)
    key = jax.random.PRNGKey(0)
    bare_obs, bare_state = bare.reset(key, params)
    obs, state = wrapped.reset(key, params)
    np.testing.assert_array_equal(obs, np.asarray(bare_obs) * POSITIONS_ONLY)
    ended = 0
    for index in range(30):
        step_key = jax.random.fold_in(key, index)
        bare_obs, bare_state, _, bare_terminated, _, bare_info = bare.step(
            step_key, bare_state, 0, params
        )
        obs, state, _, terminated, _, info = wrapped.step(step_key, state, 0, params)
        assert bool(terminated) == bool(bare_terminated)
        ended += int(terminated)
        np.testing.assert_array_equal(obs, np.asarray(bare_obs) * POSITIONS_ONLY)
        np.testing.assert_array_equal(
            info["final_observation"],
            np.asarray(bare_info["final_observation"]) * POSITIONS_ONLY,
        )
    assert ended >= 1


def test_an_unknown_environment_id_names_the_closest():
    with pytest.raises(ValueError, match=r"Did you mean .*'CartPole-v1'"):
        PPO("CartPole-v2")


def test_an_unknown_prefix_names_the_two_prefixes():
    with pytest.raises(ValueError, match=r"'gymnasium:'.*'mujoco:'"):
        PPO("gym:Zzz")


@pytest.mark.parametrize(
    ("argument", "meant"),
    [("learning_rate", "actor_learning_rate"), ("n_step", "n_steps")],
)
def test_an_unknown_agent_argument_names_the_ones_it_resembles(argument, meant):
    with pytest.raises(TypeError, match=f"no argument '{argument}'.*{meant}"):
        PPO("CartPole-v1", **{argument: 1})


def test_an_agent_on_the_wrong_kind_of_actions_names_those_that_fit():
    """CartPole's actions are discrete: PPO and PQN act in them, SAC and TD3
    in continuous ones only."""
    with pytest.raises(ValueError, match="only supports continuous") as refusal:
        SAC("CartPole-v1")
    fitting = str(refusal.value).split("acting in them:")[1]
    assert "PPO" in fitting and "PQN" in fitting
    assert "SAC" not in fitting and "TD3" not in fitting


@pytest.fixture(scope="module")
def trained():
    """PPO on CartPole (observations of shape (4,)), two seeds."""
    agent = PPO("CartPole-v1", n_envs=2, n_steps=16)
    state, _ = agent.train(seed=[0, 1], n_timesteps=64, logging_config=None)
    return agent, state


def test_predict_takes_one_observation_or_a_batch(trained):
    agent, state = trained
    agent.predict(state, np.zeros(4, np.float32))
    assert agent.predict(state, np.zeros((3, 4), np.float32)).shape[0] == 3


def test_predict_refuses_an_observation_of_another_shape(trained):
    agent, state = trained
    with pytest.raises(ValueError, match=r"shape \(3,\).*shape \(4,\)"):
        agent.predict(state, np.zeros(3, np.float32))


def test_a_seed_the_state_does_not_hold_is_refused(trained):
    agent, state = trained
    with pytest.raises(IndexError, match="holds 2 seed"):
        agent.predict(state, np.zeros(4, np.float32), seed_index=5)
    with pytest.raises(IndexError, match="holds 2 seed"):
        agent.policy(state, seed_index=2)


def test_the_evaluations_are_refused_where_the_state_goes(trained):
    agent, _ = trained
    with pytest.raises(TypeError, match="the state"):
        agent.evaluate(Evaluations())


def test_act_refuses_an_observation_without_a_row_per_environment(trained):
    agent, state = trained
    policy = agent.policy(state)
    carry = policy.reset(1)
    with pytest.raises(ValueError, match=r"obs\[None\]"):
        policy.act(np.zeros(4, np.float32), carry)
    actions, _ = policy.act(np.zeros((1, 4), np.float32), carry)
    assert actions.shape[0] == 1


def test_images_read_by_an_mlp_warn_naming_the_shape_to_pass():
    """MinAtar's Breakout observes 10 x 10 images of 4 channels."""
    with pytest.warns(UserWarning, match=r"cnn_image_shape=\(10, 10, 4\)"):
        PQN("Breakout-MinAtar")


def test_an_image_shape_in_another_order_is_refused():
    with pytest.raises(ValueError, match=r"pass cnn_image_shape=\(10, 10, 4\)"):
        PQN("Breakout-MinAtar", cnn_image_shape=(4, 10, 10))


def test_the_environments_image_shape_and_plain_vectors_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PQN("Breakout-MinAtar", cnn_image_shape=(10, 10, 4))
        PQN("CartPole-v1")


def test_a_budget_below_one_rollout_warns_that_no_evaluation_comes():
    """One rollout is 2 x 64 = 128 steps, more than the 100 asked."""
    agent = PPO("CartPole-v1", n_envs=2, n_steps=64)
    with pytest.warns(UserWarning, match="no evaluation is recorded"):
        _, evaluations = agent.train(seed=0, n_timesteps=100)
    assert evaluations.returns.shape[1] == 0


def test_a_gymnasium_environment_outside_the_extra_is_not_answered_with_it():
    """LunarLander needs Box2D, which Ajax's gymnasium extra does not hold."""
    pytest.importorskip("gymnasium")
    if importlib.util.find_spec("Box2D") is not None:
        pytest.skip("Box2D is installed")
    from ajax.environments.gymnasium import GymnasiumEnv

    with pytest.raises(ImportError, match="does not install") as refusal:
        GymnasiumEnv("LunarLander-v3")
    assert "pip install 'ajax-rl[gymnasium]'" not in str(refusal.value)


def test_a_mujoco_id_without_mujoco_names_the_playground_extra(monkeypatch):
    # None in sys.modules makes the import fail as a missing package does.
    monkeypatch.setitem(sys.modules, "ajax.environments.mujoco.tasks", None)
    with pytest.raises(ImportError, match=r"ajax-rl\[playground\]"):
        build_env_from_id("mujoco:Reacher-v5")
