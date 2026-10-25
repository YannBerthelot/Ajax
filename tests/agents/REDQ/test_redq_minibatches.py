"""REDQ's update loop: a fresh replay minibatch per critic update, and the
actor step on the last one, reading the critics from before that update.

Chen et al. (2021), Algorithm 1, samples a minibatch inside the loop of
``num_critic_updates`` critic steps and updates the actor on the last one.
REDQ's official code computes that actor loss (watchernyu/REDQ@7b5d1bf,
``redq/algos/redq_sac.py:220``) before the last critic update's optimizers
step (``:242``). The replay sampler is replaced by one whose batch depends
on its key, and the critic and actor updates record what they receive.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import ajax.agents.REDQ.REDQ as redq_module
from ajax.agents.REDQ.REDQ import REDQ, REDQState
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.state import Transition

UPDATES = 3
BATCH = 4

update_values = REDQ.update_critic
update_policy = REDQ.update_actor


def record(seen: list[np.ndarray], value: jax.Array) -> None:
    jax.debug.callback(lambda v: seen.append(np.asarray(v)), value, ordered=True)


def one_update(monkeypatch: pytest.MonkeyPatch) -> REDQState:
    """One update of a tiny REDQ on keyed random batches, after the spies
    the caller patched in."""
    agent = REDQ(
        "Pendulum-v1",
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        num_critic_updates=UPDATES,
    )
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)

    def keyed_batch(_state: Any, _buffer: Any, key: jax.Array, *_: Any) -> Any:
        obs = jax.random.normal(key, (BATCH, *obs_shape))
        column = jnp.zeros((BATCH, 1))
        return Transition(obs, column, column, column, column, obs, obs), None

    monkeypatch.setattr(redq_module, "sample_replay", keyed_batch)
    context = agent.make_context(1_000)[1]
    context = dataclasses.replace(context, extensions=ExtensionStack(()))
    state = agent.init(jax.random.PRNGKey(0), context)
    state, _ = agent.update(state, context)
    jax.effects_barrier()
    return state


def test_each_critic_update_gets_a_fresh_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    critic_obs: list[np.ndarray] = []
    actor_obs: list[np.ndarray] = []

    def spy_values(agent: Any, agent_state: Any, batch: Any, *args: Any) -> Any:
        record(critic_obs, batch.obs)
        return update_values(agent, agent_state, batch, *args)

    def spy_policy(agent: Any, agent_state: Any, batch: Any, *args: Any) -> Any:
        record(actor_obs, batch.obs)
        return update_policy(agent, agent_state, batch, *args)

    monkeypatch.setattr(REDQ, "update_critic", spy_values)
    monkeypatch.setattr(REDQ, "update_actor", spy_policy)
    one_update(monkeypatch)

    assert len(critic_obs) == UPDATES and len(actor_obs) == 1
    for i in range(UPDATES):
        for j in range(i):
            assert not np.allclose(critic_obs[i], critic_obs[j]), (i, j)
    np.testing.assert_array_equal(actor_obs[0], critic_obs[-1])


def test_the_actor_reads_the_critics_before_their_last_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critics_in: list[np.ndarray] = []
    critics_out: list[np.ndarray] = []
    actor_reads: list[np.ndarray] = []

    def flat_critics(agent_state: Any) -> jax.Array:
        return ravel_pytree(agent_state.critic_state.params)[0]

    def spy_values(agent: Any, agent_state: Any, *args: Any) -> Any:
        record(critics_in, flat_critics(agent_state))
        agent_state, *out = update_values(agent, agent_state, *args)
        record(critics_out, flat_critics(agent_state))
        return agent_state, *out

    def spy_policy(agent: Any, agent_state: Any, *args: Any) -> Any:
        record(actor_reads, flat_critics(agent_state))
        return update_policy(agent, agent_state, *args)

    monkeypatch.setattr(REDQ, "update_critic", spy_values)
    monkeypatch.setattr(REDQ, "update_actor", spy_policy)
    state = one_update(monkeypatch)

    assert len(critics_in) == UPDATES and len(actor_reads) == 1
    assert not np.allclose(critics_in[-1], critics_out[-1])
    np.testing.assert_array_equal(actor_reads[0], critics_in[-1])
    # The last critic update still lands in the returned state.
    np.testing.assert_array_equal(np.asarray(flat_critics(state)), critics_out[-1])
