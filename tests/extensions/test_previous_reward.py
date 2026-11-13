"""PreviousReward: the reward paid for the action executed before each observation.

Contract (owner's design, 2026-11-13: Ni et al. 2022's recurrent networks read
the previous reward, as an observation feature beside PreviousAction;
ajax.extensions.observe.PreviousReward, ajax.controller.previous_reward_input):

- At every observation it appends one column, the reward the environment paid
  for the action executed just before it, raw.
- At an episode's first observation, where no action came before, it is zero:
  Ni et al.'s r[0] = 0 (pomdp-baselines e7c19c3, policies/models/
  policy_rnn.py:126-131). The transition's next observation (the episode's
  final one where a time limit cut it) carries the reward that led to it.
- It stays raw where the agent normalises its observations.
- agent.policy reads the reward the last action earned (it refuses an act
  with done and no reward), and evaluation appends the column as collection
  does.
- APG's closed-loop rollout passes no reward: APG refuses it when built.

Oracles: the stored rewards and episode ends of the same rows (a property of
every collected row), and hand-computed columns for previous_reward_input.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APG, SAC
from ajax.controller import (
    ObserveInputs,
    observation_layout,
    observe_features,
    observers_of,
    previous_reward_input,
    restart_observers,
)
from ajax.environments.create import build_env_from_id
from ajax.extensions.capabilities import NotWired
from ajax.extensions.observe import (
    PreviousAction,
    PreviousReward,
    TrainingProgressInput,
)

SMALL = ("16", "relu")


def test_a_reward_enters_as_one_raw_column():
    columns = previous_reward_input((PreviousReward(),), jnp.array([-3.5, 0.25]), 2)
    np.testing.assert_array_equal(columns, [[-3.5], [0.25]])


def test_no_reward_yet_enters_as_zero():
    np.testing.assert_array_equal(
        previous_reward_input((PreviousReward(),), None, 2), np.zeros((2, 1))
    )


def test_nothing_is_computed_when_no_feature_reads_it():
    observers = (PreviousAction(),)
    assert previous_reward_input(observers, jnp.ones(2), 2) is None


def test_an_episodes_first_observation_reads_zero_and_the_next_the_reward():
    """Two envs; env 0's episode starts again at the third observation."""
    observers = (PreviousReward(),)
    obs = jnp.zeros((2, 3))
    states = (PreviousReward().init_observer(2),)
    seen = []
    for reward, starts in [
        (jnp.zeros(2), jnp.array([True, True])),  # a reset
        (jnp.array([1.5, -2.0]), jnp.array([False, False])),
        (jnp.array([4.0, 0.5]), jnp.array([True, False])),
    ]:
        states, _ = restart_observers(starts, observers, states, None, None, 2)
        inputs = ObserveInputs(
            obs, previous_reward=previous_reward_input(observers, reward, 2)
        )
        observed, states = observe_features(obs, inputs, observers, states)
        seen.append(np.asarray(observed[:, -1]))
    np.testing.assert_array_equal(seen, [[0.0, 0.0], [1.5, -2.0], [0.0, 0.5]])


def test_a_caller_that_does_not_pass_the_reward_is_refused():
    feature = PreviousReward()
    with pytest.raises(ValueError, match="previous_reward"):
        feature.observe(ObserveInputs(jnp.zeros((2, 3))), feature.init_observer(2))


def test_the_normaliser_leaves_it_raw():
    """Pendulum's 3 columns are normalised, the feature's 1 is not
    (ajax.controller.ObservationLayout), after the previous action."""
    env, params = build_env_from_id("Pendulum-v1")
    env_args = SimpleNamespace(env=env, env_params=params)
    stack = (PreviousReward(), TrainingProgressInput(), PreviousAction())
    layout = observation_layout(3, observers_of(stack), env_args)
    assert layout.normalised_width == 3 + 1  # the env's and the progress
    assert layout.slice("previous_reward") == slice(4, 5)


def _pendulum_sac(**kwargs) -> SAC:
    env, params = build_env_from_id("Pendulum-v1")
    assert params is not None
    return SAC(
        env_id=env,
        env_params=params.replace(max_steps_in_episode=5),
        n_envs=2,
        actor_architecture=SMALL,
        critic_architecture=SMALL,
        learning_starts=64,
        batch_size=8,
        buffer_size=128,
        extensions=(PreviousReward(),),
        **kwargs,
    )


def test_sac_stores_the_previous_reward_across_time_limits():
    """Pendulum cut every 5 steps, rewards negative and varying: per env and
    stored row t, obs[t + 1] carries reward[t] unless row t ended its
    episode (zero then), and next_obs[t], the final observation where a time
    limit cut it, carries reward[t]. The first row starts an episode."""
    state, _ = _pendulum_sac().train(seed=0, n_timesteps=64)
    experience = state.collector_state.buffer_state.experience
    rows = {key: np.asarray(value[0, :, :32]) for key, value in experience.items()}
    reward, ended = rows["reward"][..., 0], rows["truncated"][..., 0] > 0
    assert ended.sum() >= 2 * 5 and np.ptp(reward) > 0.1

    following = np.where(ended[:, :-1], 0.0, reward[:, :-1])
    np.testing.assert_allclose(rows["obs"][:, 1:, -1], following, atol=1e-6)
    np.testing.assert_allclose(rows["next_obs"][..., -1], reward, atol=1e-6)
    np.testing.assert_array_equal(rows["obs"][:, 0, -1], 0.0)


def test_the_policy_refuses_an_act_with_done_and_no_reward():
    agent = _pendulum_sac()
    state, _ = agent.train(seed=0, n_timesteps=0)
    policy = agent.policy(state)
    carry = policy.reset(2)
    _, carry = policy.act(jnp.zeros((2, 3)), carry)
    with pytest.raises(ValueError, match="reward"):
        policy.act(jnp.zeros((2, 3)), carry, jnp.zeros(2, bool))


def test_the_policy_reads_the_reward_it_is_given():
    """Two acts from one carry, one key, differing only in the reward: the
    actor reads it through the feature, so the actions differ."""
    agent = _pendulum_sac()
    state, _ = agent.train(seed=0, n_timesteps=0)
    policy = agent.policy(state)
    obs, key, no_start = jnp.ones((2, 3)), jax.random.PRNGKey(0), jnp.zeros(2, bool)
    _, carry = policy.act(obs, policy.reset(2), key=key)
    low, _ = policy.act(obs, carry, no_start, jnp.full(2, -5.0), key=key)
    high, _ = policy.act(obs, carry, no_start, jnp.full(2, 5.0), key=key)
    assert not np.allclose(low, high)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["SAC-gru", "DreamerV3-features"])
def test_the_policy_appends_it_as_the_evaluation_does(name):
    """tests/test_policy.py's oracle, with the feature in the stack: a
    recurrent actor and a world model's row collector."""
    from tests.exact_pins import CASES
    from tests.test_policy import RTOL, evaluated_run, evaluation_return

    cls, kwargs, steps = CASES[name]
    extensions = (*kwargs.get("extensions", ()), PreviousReward())
    agent = cls(**{**kwargs, "extensions": extensions})
    state, logged = evaluated_run(agent, steps)

    acted = evaluation_return(agent, agent.policy(state))

    np.testing.assert_allclose(float(acted), logged, rtol=RTOL)


def test_apg_refuses_it():
    with pytest.raises(NotWired, match="PreviousReward"):
        APG("Pendulum-v1", extensions=(PreviousReward(),))
