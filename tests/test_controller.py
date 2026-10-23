"""The acting state a step acts on (``ajax.controller``)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ajax.controller import ActingState, acting_state


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
