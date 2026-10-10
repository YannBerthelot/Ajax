"""REDQ draws a fresh replay minibatch for every critic update.

Chen et al. (2021), Algorithm 1, samples a minibatch inside the loop of
``num_critic_updates`` critic steps and updates the actor on the last one.
The replay sampler is replaced by one whose batch depends on its key, and
the critic and actor updates record the observations they receive.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.REDQ import train_REDQ
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.REDQ.state import REDQConfig
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.state import Transition

UPDATES = 3
BATCH = 4


def test_each_critic_update_gets_a_fresh_batch(monkeypatch: pytest.MonkeyPatch) -> None:
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

    critic_obs: list[np.ndarray] = []
    actor_obs: list[np.ndarray] = []
    update_values = train_REDQ.update_value_functions
    update_policy = train_REDQ.update_policy

    def record(seen: list[np.ndarray], obs: jax.Array) -> None:
        jax.debug.callback(lambda o: seen.append(np.asarray(o)), obs, ordered=True)

    def spy_values(agent_state: Any, batch: Any, *args: Any) -> Any:
        record(critic_obs, batch.obs)
        return update_values(agent_state, batch, *args)

    def spy_policy(agent_state: Any, obs: jax.Array, *args: Any) -> Any:
        record(actor_obs, obs)
        return update_policy(agent_state, obs, *args)

    monkeypatch.setattr(train_REDQ, "sample_replay", keyed_batch)
    monkeypatch.setattr(train_REDQ, "update_value_functions", spy_values)
    monkeypatch.setattr(train_REDQ, "update_policy", spy_policy)

    state = train_REDQ.init_REDQ(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
        number_of_critics=agent.agent_config.num_critics,
    )
    train_REDQ.update_agent(
        state,
        agent.buffer,
        False,
        cast(REDQConfig, agent.agent_config),
        ExtensionStack(()),
        total_timesteps=1_000,
    )
    jax.effects_barrier()

    assert len(critic_obs) == UPDATES and len(actor_obs) == 1
    for i in range(UPDATES):
        for j in range(i):
            assert not np.allclose(critic_obs[i], critic_obs[j]), (i, j)
    np.testing.assert_array_equal(actor_obs[0], critic_obs[-1])
