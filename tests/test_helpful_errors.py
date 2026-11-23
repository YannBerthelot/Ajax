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
    the playground extra;
  - (docs/guide/expert-guidance.md, "The expert", convention 3) an expert
    keeps its action's last axis: one that drops it is refused, naming the
    expert, where Ajax first calls it;
  - (docs/guide/training.md) ``record`` takes a trained agent's ``predict``;
  - ``LoggingConfig`` refuses a backend passed without its tuple, and
    ``train`` a keyword it does not take, naming the one meant;
  - a dataset whose rows do not fit the agent's environment is refused
    saying so;
  - (docs/guide/environments.md, "Episode keys") a gymnax environment with
    a step of its own is refused naming ``episode_keys=False``, and builds
    that way;
  - (docs/guide/checkpoints.md) a resumed run continues the same whatever
    seeds the resuming call names.
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
from ajax.buffers.dataset import TransitionDataset, record
from ajax.environments.create import build_env_from_id
from ajax.logging import LoggingConfig
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


def torque_of_one(obs):
    """An expert on Pendulum: a torque of 1 whatever the observation."""
    return jnp.ones_like(obs[..., :1])


def torque_without_its_axis(obs):
    return jnp.ones_like(obs[..., 0])


@pytest.mark.parametrize("n_envs", [1, 4])
def test_an_expert_dropping_the_actions_axis_is_refused_by_name(n_envs):
    agent = SAC("Pendulum-v1", n_envs=n_envs)
    with pytest.raises(ValueError, match="expert policy returned") as refusal:
        record(torque_without_its_axis, agent.env_args, jax.random.PRNGKey(0), 200)
    assert "[..., None]" in str(refusal.value)


def test_an_expert_keeping_the_actions_axis_is_recorded_in_agent_units():
    """Pendulum's torque spans [-2, 2], so a torque of 1 is 0.5 in agent
    units; its time limit of 200 steps makes 200 steps one whole episode."""
    agent = SAC("Pendulum-v1", n_envs=1)
    data = record(torque_of_one, agent.env_args, jax.random.PRNGKey(0), 200)
    np.testing.assert_array_equal(data.action, np.full((200, 1), 0.5, np.float32))


def test_record_takes_a_trained_agents_predict():
    """The recorded actions are predict's at the recorded observations:
    predict returns env units, the dataset holds agent units, half of them
    on Pendulum's [-2, 2]."""
    agent = SAC("Pendulum-v1", n_envs=1, learning_starts=10)
    state, _ = agent.train(seed=[0], n_timesteps=20, logging_config=None)
    data = record(
        lambda obs: agent.predict(state, obs),
        agent.env_args,
        jax.random.PRNGKey(0),
        200,
    )
    assert data.action.shape == (200, 1)
    # float32: one affine map apart, a few ulp.
    np.testing.assert_allclose(
        2.0 * data.action, agent.predict(state, data.obs), rtol=1e-5, atol=1e-6
    )


class ListBackend:
    """A logging backend by its three methods (LoggingBackend)."""

    def open(self, run):
        pass

    def log(self, metrics, step):
        pass

    def close(self):
        pass


def test_a_backend_without_its_tuple_is_refused_showing_the_tuple():
    with pytest.raises(TypeError, match=r"backends=\(ListBackend\(\.\.\.\),\)"):
        LoggingConfig(backends=ListBackend())
    assert LoggingConfig(backends=(ListBackend(),)).backends is not None


@pytest.mark.parametrize("agent", [PPO("CartPole-v1"), SAC("Pendulum-v1")])
def test_train_refuses_a_keyword_it_does_not_take_naming_the_one_meant(agent):
    with pytest.raises(
        TypeError, match=r"\.train has no argument 'logging'"
    ) as refusal:
        agent.train(seed=0, n_timesteps=64, logging=None)
    assert "logging_config" in str(refusal.value)


# render and train_offline name the agent, not the class that defines the
# method, and say which arguments they take. Both refuse before using what
# they were given, so None stands for the state and the dataset.
def test_render_refuses_a_keyword_it_does_not_take_naming_the_one_meant():
    agent = PPO("CartPole-v1")
    with pytest.raises(
        TypeError, match=r"PPO\.render has no argument 'max_step'"
    ) as refusal:
        agent.render(None, "episode.gif", max_step=10)
    assert "max_steps" in str(refusal.value)


def test_render_without_a_state_says_what_it_needs():
    agent = PPO("CartPole-v1")
    with pytest.raises(TypeError, match=r"PPO\.render needs agent_state, path"):
        agent.render()
    with pytest.raises(TypeError, match=r"PPO\.render needs agent_state:"):
        agent.render(path="episode.gif")


def test_evaluate_refuses_a_keyword_it_does_not_take_naming_the_one_meant():
    """SB3's evaluate_policy takes n_eval_episodes; Ajax's evaluate, episodes."""
    agent = PPO("CartPole-v1")
    with pytest.raises(
        TypeError, match=r"PPO\.evaluate has no argument 'n_eval_episodes'"
    ) as refusal:
        agent.evaluate(None, n_eval_episodes=5)
    assert "episodes" in str(refusal.value)
    with pytest.raises(TypeError, match=r"PPO\.evaluate needs agent_state"):
        agent.evaluate(episodes=5)


# An architecture names an activation after each width. Widths alone (SB3's
# net_arch=[64, 64]) build layers that compose to one linear map, so the
# agent warns and shows the tuple to write.
@pytest.mark.parametrize("widths", [[64, 64], ("64", "64")])
def test_widths_without_activations_warn_showing_the_tuple_to_write(widths):
    with pytest.warns(UserWarning, match="no activation") as caught:
        PPO("CartPole-v1", actor_architecture=widths)
    assert 'actor_architecture=("64", "tanh", "64", "tanh")' in str(caught[0].message)


def test_an_architecture_with_activations_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        PPO("CartPole-v1", actor_architecture=("32", "relu", "32", "relu"))
        PPO("CartPole-v1", actor_architecture=("32",))  # one layer: nothing between


def test_train_offline_refuses_a_keyword_it_does_not_take_naming_the_one_meant():
    agent = SAC("Pendulum-v1")
    with pytest.raises(
        TypeError, match=r"SAC\.train_offline has no argument 'n_update'"
    ) as refusal:
        agent.train_offline(None, n_update=5)
    assert "n_updates" in str(refusal.value)


def test_a_dataset_from_another_environment_is_refused_saying_so():
    """Four rows of two-column observations, for Pendulum's three."""
    two_columns = np.zeros((4, 2), np.float32)
    data = TransitionDataset(
        obs=two_columns,
        action=np.zeros((4, 1), np.float32),
        reward=np.zeros(4, np.float32),
        terminated=np.zeros(4, bool),
        truncated=np.array([False, False, False, True]),
        next_obs=two_columns,
    )
    with pytest.raises(ValueError, match="another one"):
        SAC("Pendulum-v1").train_offline(data, n_updates=4, seed=[0])


def test_an_environment_with_its_own_step_names_the_way_to_build_it():
    with pytest.raises(ValueError, match="episode_keys=False"):
        PPO("DeepSea-bsuite")
    env, env_params = build_env_from_id("DeepSea-bsuite", n_envs=2, episode_keys=False)
    agent = PPO(env, env_params=env_params, n_envs=2)
    assert agent.env_args.n_envs == 2


def test_a_resumed_run_continues_the_same_whatever_seeds_the_call_names():
    """Two identical runs, each resumed for one more rollout, one under
    other seeds: the same parameters, bit for bit (the same program on the
    same inputs)."""

    def resumed(seeds):
        agent = PPO("CartPole-v1", n_envs=2, n_steps=16)
        state, _ = agent.train(seed=[0, 1], n_timesteps=32, logging_config=None)
        state, _ = agent.train(
            seed=seeds, n_timesteps=32, initial_state=state, logging_config=None
        )
        return jax.tree.leaves(state.actor_state.params)

    same, others = resumed([0, 1]), resumed([5, 6])
    assert len(same) == len(others)
    for kept, renamed in zip(same, others, strict=True):
        np.testing.assert_array_equal(kept, renamed)
