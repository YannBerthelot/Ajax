"""The acting state a step acts on (``ajax.controller``)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ajax.controller import ActingState, Controller, acting_state


def test_a_collecting_agents_acting_state_is_its_collectors_and_actors():
    """The observation, where the last step ended an episode (terminated or
    truncated), the timestep, the expert's state, the actor's carry and the
    step in the episode."""
    collector = SimpleNamespace(
        last_obs=jnp.ones((3, 2)),
        last_terminated=jnp.array([True, False, False]),
        last_truncated=jnp.array([False, True, False]),
        timestep=jnp.asarray(12),
        expert_state=jnp.zeros((3, 1)),
        step_in_episode=jnp.arange(3),
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


def test_the_actors_input_is_the_observation_then_the_experts_state_and_action():
    obs, expert = jnp.ones((2, 3)), _GainExpert()
    plain, untouched = Controller().observe(_acting(obs), expert, obs)
    np.testing.assert_array_equal(plain, obs)
    assert untouched is None  # the input does not read the expert
    both = Controller(observe_expert_action=True, observe_expert_state=True)
    actor_obs, state = both.observe(_acting(obs), expert, obs)
    np.testing.assert_array_equal(actor_obs[:, :3], obs)
    np.testing.assert_array_equal(actor_obs[:, 3:], [[0.0, -2.0], [0.0, -2.0]])
    np.testing.assert_array_equal(state, 1.0)  # after the call made for it


def test_the_executed_action_is_the_actors_the_transforms_or_the_gains():
    obs, expert, actions = jnp.ones((2, 1)), _GainExpert(), jnp.full((2, 1), 0.5)
    acting = _acting(obs)
    own, _ = Controller().act(actions, acting, expert, obs, None, None)
    np.testing.assert_array_equal(own, actions)
    mixed, state = Controller(transform=lambda a, e, _obs, _state: a + e).act(
        actions, acting, expert, obs, None, None
    )
    np.testing.assert_array_equal(mixed, 0.5 - 2.0)  # the expert called for it
    np.testing.assert_array_equal(state, 1.0)
    gained, _ = Controller(gains=True).act(actions, acting, expert, obs, None, None)
    np.testing.assert_allclose(gained, -2.0 * 10**0.5, rtol=1e-6)  # anchor * 10^a
