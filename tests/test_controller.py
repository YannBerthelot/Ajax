"""The acting state a step acts on (``ajax.controller``)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ajax.controller import (
    ActingState,
    Controller,
    ObserveInputs,
    acting_state,
    executed_actions,
    executed_fn,
    feature_width,
    init_observers,
    make_run_controller,
    make_scored_actions,
    observation_layout,
    observe_features,
    observers_of,
    restart_observers,
)
from ajax.environments.obs_norm import init_agent_obs_norm, update_obs_norm
from ajax.extensions.base import Extension
from ajax.extensions.expert import ExpertGains, ResidualPolicy


def test_a_collecting_agents_acting_state_is_its_collectors_and_actors():
    """The observation, where the last step ended an episode (terminated or
    truncated), the timestep, the expert's state, the actor's carry, the
    step in the episode and the observation statistics."""
    collector = SimpleNamespace(
        last_obs=jnp.ones((3, 2)),
        last_terminated=jnp.array([True, False, False]),
        last_truncated=jnp.array([False, True, False]),
        timestep=jnp.asarray(12),
        expert_state=jnp.zeros((3, 1)),
        step_in_episode=jnp.arange(3),
        obs_norm_info="stats",
    )
    agent = SimpleNamespace(
        collector_state=collector, actor_state=SimpleNamespace(hidden_state="carry")
    )
    acting = acting_state(agent)
    assert isinstance(acting, ActingState)
    np.testing.assert_array_equal(acting.done, [True, True, False])
    assert acting.obs is collector.last_obs and acting.timestep is collector.timestep
    assert acting.expert_state is collector.expert_state and acting.carry == "carry"
    assert acting.step_in_episode is collector.step_in_episode
    assert acting.obs_norm == "stats"


class _GainExpert:
    """A stateful gain expert: its action is ``-gain * obs``, its state
    counts the calls."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([2.0])

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state, obs, gains):
        return -gains[0] * obs[..., :1], state + 1.0


def _acting(obs):
    return ActingState(
        obs=obs, done=jnp.zeros(2, bool), timestep=None, expert_state=jnp.zeros((2, 1))
    )


def test_the_actors_input_is_normalised_when_the_agent_stores_it_so():
    """With the statistics of three observations 0, 2 and 4 (mean 2,
    variance 8/3) the observation 2 + sqrt(8/3) normalises to 1, the
    leading raw column left as it is."""
    stats = update_obs_norm(
        jnp.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]]), init_agent_obs_norm(2)
    )[1]
    obs = jnp.array([[7.0, 2.0 + np.sqrt(8 / 3)]])
    acting = ActingState(
        obs=obs, done=jnp.zeros(1, bool), timestep=None, obs_norm=stats
    )
    normalising = Controller(normalize_obs=True, raw_columns=1)
    actor_obs = normalising.observe(acting)
    np.testing.assert_allclose(actor_obs, [[7.0, 1.0]], rtol=1e-4)


class _Mix(Extension):
    """A transform executing the actor's action plus the expert's."""

    name = "mix"

    def transform_action(self, actions, inputs):
        return actions + inputs.expert_action


def test_the_executed_action_is_the_actors_the_transforms_or_the_gains():
    obs, expert, actions = jnp.ones((2, 1)), _GainExpert(), jnp.full((2, 1), 0.5)
    acting = _acting(obs)
    own, _ = Controller().act(actions, acting, expert, obs)
    np.testing.assert_array_equal(own, actions)
    mixed, state = Controller(transforms=(_Mix(),)).act(actions, acting, expert, obs)
    np.testing.assert_array_equal(mixed, 0.5 - 2.0)  # the expert called for it
    np.testing.assert_array_equal(state, 1.0)
    gains = Controller(transforms=(ExpertGains(),))
    gained, _ = gains.act(actions, acting, expert, obs)
    np.testing.assert_allclose(gained, -2.0 * 10**0.5, rtol=1e-6)  # anchor * 10^a
    assert gains.actuates and not Controller(transforms=(_Mix(),)).actuates


def test_the_critics_score_the_residual_transforms_executed_actions():
    """executed_actions: the run's residual applied to the policy's actions
    at the stored expert actions; gain mode's critics score the gains."""
    actions, a_expert = jnp.full((2, 1), 0.5), jnp.full((2, 1), 0.75)
    residual = SimpleNamespace(transforms=(ResidualPolicy(scale=0.5),))
    np.testing.assert_allclose(
        executed_actions(residual, actions, a_expert), 0.75 + 0.25
    )
    for transforms in ((), (ExpertGains(),)):
        env_args = SimpleNamespace(transforms=transforms)
        np.testing.assert_array_equal(
            executed_actions(env_args, actions, a_expert), actions
        )


def test_the_scored_actions_are_the_executed_ones_under_a_residual_policy_only():
    """make_scored_actions: at s, actions -> executed(actions, a_expert) at
    the rows' stored expert actions, at s' at their next ones; the actions
    themselves without a residual policy."""
    actions = jnp.full((2, 1), 0.5)
    batch = SimpleNamespace(a_expert=jnp.full((2, 1), 0.75), next_a_expert=-actions)
    unchanged = make_scored_actions(None, batch)
    assert unchanged.map_actions(actions) is actions
    assert unchanged.map_next_actions(actions) is actions
    residual = SimpleNamespace(transforms=(ResidualPolicy(scale=0.5),))
    scored = make_scored_actions(executed_fn(residual), batch)
    np.testing.assert_allclose(scored.map_actions(actions), 0.75 + 0.25)
    np.testing.assert_allclose(scored.map_next_actions(actions), -0.5 + 0.25)


class _Count(Extension):
    """A feature counting the observations of its episode (normalised)."""

    name = "count"

    def observation_width(self, env_args):
        return 1

    def init_observer(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def observe(self, inputs, state):
        state = state + 1.0
        return state, state


class _Sign(Extension):
    """A raw feature: the sign of the observation's first entry."""

    name = "sign"
    observes_normalised = False

    def observation_width(self, env_args):
        return 1

    def observe(self, inputs, state):
        return jnp.sign(inputs.raw_obs[..., :1]), state


def test_the_raw_features_follow_the_normalised_ones_whatever_the_stack_order():
    sign, count = _Sign(), _Count()
    observers = observers_of([sign, Extension(), count])
    assert observers == (count, sign)
    layout = observation_layout(3, observers, None)
    assert (layout.width, layout.normalised_width, layout.feature_width) == (5, 4, 2)
    assert layout.slice("count") == slice(3, 4) and layout.slice("sign") == slice(4, 5)
    args = SimpleNamespace(observers=observers)
    assert (feature_width(args), feature_width(args, normalised=False)) == (2, 1)


def test_features_join_the_observation_and_carry_their_state():
    observers = (_Count(), _Sign())
    raw = jnp.array([[-2.0, 1.0], [3.0, 1.0]])
    states = init_observers(observers, 2)
    obs, states = observe_features(raw, ObserveInputs(raw), observers, states)
    np.testing.assert_array_equal(obs, [[-2.0, 1.0, 1.0, -1.0], [3.0, 1.0, 1.0, 1.0]])
    obs, _ = observe_features(raw, ObserveInputs(raw), observers, states)
    np.testing.assert_array_equal(obs[:, 2], [2.0, 2.0])
    same, _ = observe_features(raw, ObserveInputs(raw), (), ())
    assert same is raw  # no features: the observation itself


def test_the_expert_steps_every_step_its_state_feeds_the_features():
    """With observation features the expert is called each step, as
    collection calls it, so the next observation's expert state is current;
    without, its state is left as it is."""
    obs, expert, actions = jnp.ones((2, 1)), _GainExpert(), jnp.full((2, 1), 0.5)
    acting = _acting(obs)
    own, still = Controller().act(actions, acting, expert, obs)
    observed, stepped = Controller(observers=(_Count(),)).act(
        actions, acting, expert, obs
    )
    np.testing.assert_array_equal(observed, own)  # the actor's action either way
    np.testing.assert_array_equal(still, 0.0)
    np.testing.assert_array_equal(stepped, 1.0)


def test_observers_and_a_stateful_expert_restart_where_an_episode_starts():
    states, expert_state = (jnp.array([[3.0], [5.0]]),), jnp.array([[7.0], [9.0]])
    started, count, expert = jnp.array([True, False]), (_Count(),), _GainExpert()
    restarted = restart_observers(started, count, states, expert, expert_state, 2)
    np.testing.assert_array_equal(restarted[0][0], [[0.0], [5.0]])
    np.testing.assert_array_equal(restarted[1], [[0.0], [9.0]])
    # A stateless expert's state is left as it is.
    kept = restart_observers(started, count, states, None, expert_state, 2)[1]
    assert kept is expert_state


def test_the_run_controller_carries_the_runs_features_and_transforms():
    run = SimpleNamespace(store_normalized_obs=False, observers=(_Count(),))
    run.transforms = (_Mix(),)
    expected = Controller(observers=run.observers, transforms=run.transforms)
    assert make_run_controller(run) == expected
    run.observers, run.transforms = (), ()
    assert make_run_controller(run) == Controller()
