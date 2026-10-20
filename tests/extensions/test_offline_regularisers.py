"""ConservativeQ (CQL) and CalQL (Cal-QL): the floor on the policy actions'
values, the stored return to go, and the refusals. Whether they make a
critic conservative is the probes' (tests/probing/test_offline)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.buffers.dataset import TransitionDataset, fill_replay
from ajax.extensions.base import ExtensionContext
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.offline import CalQL, ConservativeQ

KEY = jax.random.PRNGKey(0)
CTX = ExtensionContext(step=jnp.asarray(0), rng=KEY)
ROWS = 8


def sac_batch(return_to_go: float) -> dict[str, Any]:
    """A SAC critic-loss batch at its initial networks, on Pendulum."""
    agent = SAC(
        "Pendulum-v1",
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    obs = jax.random.normal(KEY, (ROWS, 3))
    return {
        "observations": obs,
        "next_observations": obs[::-1],
        "actions": jnp.zeros((ROWS, 1)),
        "critic_params": state.critic_state.params,
        "critic_state": state.critic_state,
        "actor_state": state.actor_state,
        "return_to_go": jnp.full((ROWS, 1), return_to_go),
    }


def test_without_a_floor_calql_is_cql_and_a_floor_above_every_value_binds_all() -> None:
    """A -inf return to go floors nothing: CalQL's term is CQL's on the same
    draws. One above every value raises every policy action's."""
    batch = sac_batch(-jnp.inf)
    cql, _ = ConservativeQ(alpha=2.0, n_actions=4).critic_loss(None, (), batch, CTX)
    calql, metrics = CalQL(alpha=2.0, n_actions=4).critic_loss(None, (), batch, CTX)
    np.testing.assert_allclose(float(calql), float(cql), rtol=1e-6)
    assert float(metrics["calql/bound_frac"]) == 0.0
    high, metrics = CalQL(alpha=2.0, n_actions=4).critic_loss(
        None, (), sac_batch(1e3), CTX
    )
    assert float(metrics["calql/bound_frac"]) == 1.0 and float(high) > float(cql)


def test_the_replay_keeps_a_datasets_return_and_minus_inf_for_collected_rows() -> None:
    agent = SAC("Pendulum-v1", buffer_size=64, batch_size=8, extensions=(CalQL(),))
    state, _ = agent.train(seed=0, n_timesteps=4)
    replay = jax.tree.map(lambda x: x[0], state.collector_state.buffer_state)
    np.testing.assert_array_equal(replay.experience["return_to_go"][0, :4], -np.inf)
    obs, ends = np.zeros((ROWS, 3)), np.ones(ROWS, bool)
    rows = TransitionDataset(
        obs, np.zeros((ROWS, 1)), np.arange(ROWS), ends, ~ends, obs
    )
    empty = agent.train(seed=0, n_timesteps=0)[0].collector_state.buffer_state
    filled = fill_replay(agent.buffer, jax.tree.map(lambda x: x[0], empty), rows)
    np.testing.assert_array_equal(
        filled.experience["return_to_go"][0, :ROWS, 0], rows.return_to_go
    )


def test_expert_prefill_rows_keep_minus_inf() -> None:
    agent = SAC(
        "Pendulum-v1",
        buffer_size=64,
        batch_size=8,
        extensions=(CalQL(),),
        expert_policy=lambda obs: jnp.zeros((*obs.shape[:-1], 1)),
        expert_buffer_n_steps=4,
    )
    state, _ = agent.train(seed=0, n_timesteps=0)
    replay = jax.tree.map(lambda x: x[0], state.collector_state.buffer_state)
    np.testing.assert_array_equal(replay.experience["return_to_go"][0, :4], -np.inf)


def test_the_terms_refuse_bad_settings_and_agents_without_their_inputs() -> None:
    for bad in ({"alpha": -1.0}, {"n_actions": 0}):
        with pytest.raises(ValueError, match="alpha >= 0 and n_actions >= 1"):
            ConservativeQ(**bad)
    from ajax import ASAC, AVG

    with pytest.raises(MissingCapability, match="discounted"):
        ASAC("Pendulum-v1", extensions=(CalQL(),))  # an average-reward critic
    with pytest.raises(MissingCapability, match="replay"):
        AVG("Pendulum-v1", extensions=(CalQL(),))  # no replay to hold a dataset
