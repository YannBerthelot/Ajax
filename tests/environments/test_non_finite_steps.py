"""A simulator can diverge; NonFiniteStepWrapper keeps that from an agent.

The contract (the fix chosen for APO on HumanoidWalk, 2026-11-04):

* A step whose observation or reward holds a NaN or an infinity ends the
  episode as a termination, on a time limit's step too: reward 0, the
  observation its action was taken from as the final observation, and its
  metrics 0, which brax's EpisodeWrapper would otherwise keep summing.
* It counts one more non-finite step for that env, for the whole run, past
  the reset; a run logs the sum as Train/non-finite env steps.
* Any other step passes unchanged, and so do the batch's other envs.
* Nothing non-finite reaches an agent: its parameters and its evaluation stay
  finite through such a step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State
from mujoco_playground._src import mjx_env

from ajax import APO, PPO, SAC, UDRL
from ajax.environments.create import build_env_from_id
from ajax.environments.interaction import get_final_obs, reset, step
from ajax.logging.wandb_logging import LoggingConfig
from ajax.wrappers import NON_FINITE_STEPS_KEY, NonFiniteStepWrapper
from tests.environments.breaking_env import ALWAYS, BREAKS_AT, ON_POSITIVE_ACTIONS

NET = ("4", "tanh")


class Scripted(Env):
    """Every step gives this observation and reward."""

    def __init__(self, obs: float, reward: float):
        self.obs, self.reward = obs, reward

    def reset(self, rng):
        return State(
            jnp.zeros(1), jnp.full(1, 7.0), jnp.zeros(()), jnp.zeros(()), {}, {}
        )

    def step(self, state, action):
        return state.replace(obs=jnp.full(1, self.obs), reward=jnp.asarray(self.reward))

    observation_size = 1
    action_size = 1
    backend = "scripted"


KEY = jax.random.PRNGKey(0)


@pytest.mark.parametrize(
    ("obs", "reward"),
    [
        (jnp.nan, 1.0),
        (jnp.inf, 1.0),
        (-jnp.inf, 1.0),
        (1.0, jnp.nan),
        (1.0, jnp.inf),
        (1.0, -jnp.inf),
    ],
)
def test_a_non_finite_step_ends_the_episode_and_counts(obs, reward):
    env = NonFiniteStepWrapper(Scripted(obs, reward))
    state = env.reset(KEY)
    stepped = env.step(state, jnp.zeros(1))
    assert stepped.done == 1
    assert stepped.reward == 0
    np.testing.assert_array_equal(stepped.obs, state.obs)  # 7, its last finite one
    assert stepped.info[NON_FINITE_STEPS_KEY] == 1


def test_a_finite_step_passes_unchanged():
    inner = Scripted(2.0, 3.0)
    env = NonFiniteStepWrapper(inner)
    theirs = inner.step(inner.reset(KEY), jnp.zeros(1))
    ours = env.step(env.reset(KEY), jnp.zeros(1))
    assert ours.reward == theirs.reward == 3
    assert ours.done == theirs.done == 0
    np.testing.assert_array_equal(ours.obs, theirs.obs)
    assert ours.info[NON_FINITE_STEPS_KEY] == 0


@pytest.mark.parametrize("episode_keys", [True, False], ids=["fresh", "seeded"])
def test_the_stack_ends_only_the_env_that_broke(episode_keys):
    env, _ = build_env_from_id(ON_POSITIVE_ACTIONS, n_envs=3, episode_keys=episode_keys)
    _, state = reset(KEY, env, "brax")
    actions = jnp.array([[1.0], [-1.0], [-1.0]])
    outcomes = []
    for _ in range(BREAKS_AT):
        outcomes.append(step(KEY, state, actions, env, "brax"))
        state = outcomes[-1][1]
    obs, state, reward, terminated, truncated, info = outcomes[-1]
    np.testing.assert_array_equal(terminated, [1, 0, 0])
    np.testing.assert_array_equal(truncated, [0, 0, 0])
    np.testing.assert_array_equal(reward, [0, 1, 1])
    np.testing.assert_array_equal(obs, [[0], [3], [3]])  # the reset's, and on
    np.testing.assert_array_equal(get_final_obs(info, None), [[2], [3], [3]])
    assert np.isfinite(state.pipeline_state).all()
    np.testing.assert_array_equal(state.info[NON_FINITE_STEPS_KEY], [1, 0, 0])
    np.testing.assert_array_equal(state.info["episode_metrics"]["m"], [3, 6, 6])

    _, state, *_ = step(KEY, state, -jnp.ones((3, 1)), env, "brax")
    np.testing.assert_array_equal(state.info[NON_FINITE_STEPS_KEY], [1, 0, 0])
    assert np.isfinite(state.info["episode_metrics"]["m"]).all()


def test_a_step_on_the_time_limit_that_breaks_is_a_termination():
    """A cut episode bootstraps on its final observation: this one has none."""
    env, _ = build_env_from_id(ON_POSITIVE_ACTIONS, n_envs=2, episode_length=BREAKS_AT)
    _, state = reset(KEY, env, "brax")
    actions = jnp.array([[1.0], [-1.0]])
    for _ in range(BREAKS_AT):
        _, state, _, terminated, truncated, _ = step(KEY, state, actions, env, "brax")
    np.testing.assert_array_equal(terminated, [1, 0])
    np.testing.assert_array_equal(truncated, [0, 1])


def test_a_playground_env_restarts_its_physics(monkeypatch):
    """The real Playground stack, its state diverging where the first env
    pushes: Playground's own done is not what ends this episode."""
    real_step = mjx_env.step

    def diverging(model, data, action, n_substeps=1):
        data = real_step(model, data, action, n_substeps)
        return data.replace(qvel=jnp.where(action[0] > 0, jnp.nan, data.qvel))

    monkeypatch.setattr(mjx_env, "step", diverging)
    env, _ = build_env_from_id("PendulumSwingup", n_envs=2)
    _, state = reset(KEY, env, "brax")
    actions = jnp.array([[1.0], [-1.0]])
    obs, state, reward, terminated, _, info = step(KEY, state, actions, env, "brax")
    np.testing.assert_array_equal(terminated, [1, 0])
    assert reward[0] == 0
    assert np.isfinite(obs).all()
    assert np.isfinite(get_final_obs(info, None)).all()
    assert np.isfinite(state.data.qvel).all()
    np.testing.assert_array_equal(state.info[NON_FINITE_STEPS_KEY], [1, 0])


@pytest.mark.parametrize(
    ("agent_class", "options"),
    [
        (APO, {"n_steps": 8, "num_minibatches": 1, "n_epochs": 1}),
        (PPO, {"n_steps": 8, "batch_size": 8, "n_epochs": 1}),
        (SAC, {"learning_starts": 2, "batch_size": 4, "buffer_size": 40}),
    ],
    ids=["APO", "PPO", "SAC"],
)
def test_an_agent_trains_and_evaluates_through_a_non_finite_step(agent_class, options):
    """8 steps on each of 2 envs break at steps 3 and 6: 4 non-finite steps.
    An evaluation episode earns 1 + 1 + 0, the guard's reward where it broke."""
    agent = agent_class(
        env_id=ALWAYS,
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        **options,
    )
    state, evaluations = agent.train(
        seed=0,
        n_timesteps=16,
        logging_config=LoggingConfig(log_frequency=16, backends=()),
    )
    params = (state.actor_state.params, state.critic_state.params)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(params))
    assert np.asarray(evaluations["Eval/episodic mean reward"]).item() == 2
    assert np.asarray(evaluations["Train/non-finite env steps"]).item() == 4


def test_an_agent_with_its_own_evaluation_logs_the_count():
    """UDRL evaluates on its own schedule (TrainLoop.evaluate_every): the same
    4 non-finite steps as above, logged on that path."""
    agent = UDRL(
        env_id=ALWAYS,
        n_envs=2,
        n_steps=8,
        batch_size=4,
        buffer_capacity=8,
        n_updates_per_iter=1,
        actor_architecture=NET,
    )
    _, evaluations = agent.train(
        seed=0,
        n_timesteps=16,
        logging_config=LoggingConfig(log_frequency=16, backends=()),
    )
    assert np.asarray(evaluations["Train/non-finite env steps"]).reshape(-1)[-1] == 4
