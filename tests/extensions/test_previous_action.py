"""PreviousAction: the action executed just before each observation.

Contract (owner's design, 2026-11-13; ajax.extensions.observe.PreviousAction,
ajax.controller.previous_action_input):

- At every observation it appends the action the environment executed just
  before it, in agent units: a continuous action clipped to [-1, 1] (what
  the environment and the replay take), a discrete action one-hot over the
  env's action count.
- At an episode's first observation, where no action came before, it is
  zero; so at a reset, and on the row after an episode ended. The
  transition's next observation (the episode's final one where it ended)
  carries the action that led to it.
- It stays raw where the agent normalises its observations.
- Collection stores it with each observation, and evaluation and
  agent.policy append it as collection does: the policy reproduces the
  evaluation's return (tests/test_policy.py's oracle) with it in the stack.

Oracles: the stored actions and episode ends of the same rows (a property of
every collected row), and hand-computed columns for previous_action_input.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import ASAC, DQN, PPO, SAC
from ajax.controller import (
    ObserveInputs,
    observation_layout,
    observe_features,
    observers_of,
    previous_action_input,
    restart_observers,
)
from ajax.environments.create import build_env_from_id
from ajax.extensions.observe import PreviousAction, TrainingProgressInput
from ajax.networks.memory import MemoryConfig

SMALL = ("16", "relu")


def _columns(env_id: str, action, n_envs: int = 2):
    env, params = build_env_from_id(env_id)
    observers = (PreviousAction(),)
    return previous_action_input(observers, action, n_envs, env, params)


def test_a_continuous_action_enters_clipped_to_the_agents_bounds():
    columns = _columns("Pendulum-v1", jnp.array([[1.5], [-0.25]]))
    np.testing.assert_array_equal(columns, [[1.0], [-0.25]])


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_a_discrete_action_enters_one_hot(shape):
    """CartPole's two actions: index 1 then 0, as a vector or a column."""
    columns = _columns("CartPole-v1", jnp.array([1, 0]).reshape(shape))
    np.testing.assert_array_equal(columns, [[0.0, 1.0], [1.0, 0.0]])


def test_no_action_yet_enters_as_zeros():
    np.testing.assert_array_equal(_columns("CartPole-v1", None), np.zeros((2, 2)))


def test_nothing_is_computed_when_no_feature_reads_it():
    env, params = build_env_from_id("Pendulum-v1")
    observers = (TrainingProgressInput(),)
    assert previous_action_input(observers, jnp.ones((2, 1)), 2, env, params) is None


def test_an_episodes_first_observation_reads_zero_and_the_next_the_action():
    """Two envs; env 0's episode starts again at the third observation."""
    observers = (PreviousAction(),)
    obs = jnp.zeros((2, 3))
    states = (PreviousAction().init_observer(2),)
    seen = []
    for action, starts in [
        (jnp.zeros((2, 1)), jnp.array([True, True])),  # a reset
        (jnp.array([[0.5], [-0.5]]), jnp.array([False, False])),
        (jnp.array([[0.25], [0.75]]), jnp.array([True, False])),
    ]:
        states, _ = restart_observers(starts, observers, states, None, None, 2)
        inputs = ObserveInputs(obs, previous_action=action)
        observed, states = observe_features(obs, inputs, observers, states)
        seen.append(np.asarray(observed[:, -1]))
    np.testing.assert_array_equal(seen, [[0.0, 0.0], [0.5, -0.5], [0.0, 0.75]])


def test_a_caller_that_does_not_pass_the_previous_action_is_refused():
    feature = PreviousAction()
    with pytest.raises(ValueError, match="previous_action"):
        feature.observe(ObserveInputs(jnp.zeros((2, 3))), feature.init_observer(2))


def test_its_width_is_the_actions():
    for env_id, width in [("Pendulum-v1", 1), ("CartPole-v1", 2)]:
        env, params = build_env_from_id(env_id)
        env_args = SimpleNamespace(env=env, env_params=params)
        assert PreviousAction().observation_width(env_args) == width


# -- Collection ----------------------------------------------------------------


def _expected_columns(action: np.ndarray, continuous: bool) -> np.ndarray:
    if continuous:
        return np.clip(action, -1.0, 1.0)
    return np.eye(2)[action[..., 0].astype(int)]


def _check_rows(obs, action, next_obs, terminated, truncated, width, continuous):
    """Per env and stored row t (time on axis 1): obs[t + 1] carries action[t]
    unless row t ended its episode, zero then; next_obs[t] carries it too,
    unless row t terminated, where the replay keeps the reset observation
    (interaction.bootstrap_obs), an episode's first: zero. The first row is
    an episode's first observation."""
    taken = _expected_columns(action, continuous)
    ended = np.logical_or(terminated, truncated)
    following = np.where(ended[:, :-1, None], 0.0, taken[:, :-1])
    np.testing.assert_allclose(obs[:, 1:, -width:], following, atol=1e-6)
    after = np.where(terminated[..., None], 0.0, taken)
    np.testing.assert_allclose(next_obs[..., -width:], after, atol=1e-6)
    np.testing.assert_array_equal(obs[:, 0, -width:], 0.0)


def _replay(state, rows: int) -> dict:
    """The first seed's replay rows, (n_envs, rows, ...), in the order written."""
    experience = state.collector_state.buffer_state.experience
    return {key: np.asarray(value[0, :, :rows]) for key, value in experience.items()}


def test_dqn_stores_the_one_hot_previous_action_and_zero_after_an_episode_end():
    """Random CartPole episodes end within a few dozen steps, so 48 steps
    an env see episode ends (asserted)."""
    agent = DQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=SMALL,
        learning_starts=96,
        batch_size=8,
        buffer_size=256,
        extensions=(PreviousAction(),),
    )
    state, _ = agent.train(seed=0, n_timesteps=96)
    rows = _replay(state, 48)
    terminated, truncated = (rows[k][..., 0] > 0 for k in ("terminated", "truncated"))
    assert terminated.any()
    _check_rows(
        rows["obs"], rows["action"], rows["next_obs"], terminated, truncated, 2, False
    )


def test_sac_stores_the_previous_action_across_time_limits():
    """Pendulum cut every 5 steps: the final observation carries the last
    action, the next episode's first observation zero."""
    env, params = build_env_from_id("Pendulum-v1")
    agent = SAC(
        env_id=env,
        env_params=params.replace(max_steps_in_episode=5),
        n_envs=2,
        actor_architecture=SMALL,
        critic_architecture=SMALL,
        learning_starts=64,
        batch_size=8,
        buffer_size=128,
        extensions=(PreviousAction(),),
    )
    state, _ = agent.train(seed=0, n_timesteps=64)
    rows = _replay(state, 32)
    terminated, truncated = (rows[k][..., 0] > 0 for k in ("terminated", "truncated"))
    assert truncated.sum() >= 2 * 5
    _check_rows(
        rows["obs"], rows["action"], rows["next_obs"], terminated, truncated, 1, True
    )


def test_the_normaliser_leaves_it_raw():
    """Pendulum's 3 columns are normalised, the feature's 1 is not
    (ajax.controller.ObservationLayout), whatever the stack order."""
    env, params = build_env_from_id("Pendulum-v1")
    env_args = SimpleNamespace(env=env, env_params=params)
    stack = (PreviousAction(), TrainingProgressInput())
    layout = observation_layout(3, observers_of(stack), env_args)
    assert layout.normalised_width == 3 + 1  # the env's and the progress
    assert layout.slice("previous_action") == slice(4, 5)


# -- Evaluation and agent.policy -------------------------------------------------


def test_an_episode_start_zeroes_what_the_policy_carries_on():
    """Where done says obs starts an episode, the policy acts as from a
    reset, whatever action it executed before."""
    agent = PPO(
        "Pendulum-v1",
        n_envs=2,
        n_steps=8,
        memory=MemoryConfig(kind="gru", hidden_size=4),
        actor_architecture=SMALL,
        critic_architecture=SMALL,
        extensions=(PreviousAction(),),
    )
    state, _ = agent.train(seed=0, n_timesteps=0)
    policy = agent.policy(state)
    observations = jax.random.normal(jax.random.PRNGKey(0), (4, 2, 3))
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    carry = policy.reset(2)
    for obs, key in zip(observations[:3], keys[:3], strict=True):
        _, carry = policy.act(obs, carry, key=key)
    fresh = policy.act(observations[3], policy.reset(2), key=keys[3])

    started = policy.act(observations[3], carry, jnp.ones(2, bool), key=keys[3])

    leaves = zip(jax.tree.leaves(started), jax.tree.leaves(fresh), strict=True)
    for got, expected in leaves:
        np.testing.assert_array_equal(got, expected)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["PPO-gru", "SAC-gru", "APG", "DreamerV3-features"])
def test_the_policy_appends_it_as_the_evaluation_does(name):
    """tests/test_policy.py's oracle, with the feature in the stack: a
    recurrent actor (PPO, SAC), APG's closed-loop rollout and a world
    model's row collector."""
    from tests.exact_pins import CASES
    from tests.test_policy import RTOL, evaluated_run, evaluation_return

    cls, kwargs, steps = CASES[name]
    extensions = (*kwargs.get("extensions", ()), PreviousAction())
    agent = cls(**{**kwargs, "extensions": extensions})
    state, logged = evaluated_run(agent, steps)

    acted = evaluation_return(agent, agent.policy(state))

    np.testing.assert_allclose(float(acted), logged, rtol=RTOL)


# -- A recurrent Q-critic --------------------------------------------------------


def _recurrent_sac(extensions: tuple) -> Any:
    return SAC(
        "Pendulum-v1",
        n_envs=2,
        memory=MemoryConfig(kind="gru", hidden_size=4),
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=64,
        extensions=extensions,
    )


def _embedded_widths(params: Any) -> dict[str, int]:
    """The width each part of a network's input embedding reads (its
    kernel's input axis, after the seed axis)."""
    encoder = params["params"]["encoder"]
    return {part: int(layer["kernel"].shape[-2]) for part, layer in encoder.items()}


@pytest.mark.parametrize("with_feature", [False, True])
def test_a_recurrent_q_critic_reads_the_previous_action_once(with_feature):
    """Pendulum: 3 observation columns and 1 action. A run with memory
    observes the previous action whether or not it was given the feature (its
    networks read it, as Ni et al.'s do; owner, 2026-11-13): the critic and
    the actor embed it once, from the observation, which the replay does
    not append to."""
    agent = _recurrent_sac((PreviousAction(),) if with_feature else ())
    state, _ = agent.train(seed=0, n_timesteps=0)
    assert state.critic_state.previous_action_in_observation
    for params in (state.critic_state.params, state.actor_state.params):
        widths = _embedded_widths(params)
        assert widths == {"previous_action": 1, "previous_reward": 1, "observation": 3}


def test_asac_trains_with_memory_reading_the_observations_previous_action():
    """ASAC's target also asks its critic about the origin, (0, 0) at an
    episode start: with the feature, the zero observation carries the zero
    previous action."""
    agent = ASAC(
        "Pendulum-v1",
        n_envs=2,
        memory=MemoryConfig(kind="gru", hidden_size=4),
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=64,
        batch_size=4,
        extensions=(PreviousAction(),),
    )
    start, _ = agent.train(seed=0, n_timesteps=0)
    state, _ = agent.train(seed=0, n_timesteps=128)
    before = jax.tree.leaves(start.critic_state.params)
    after = jax.tree.leaves(state.critic_state.params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in after)
    assert any(not np.array_equal(a, b) for a, b in zip(after, before, strict=True))
