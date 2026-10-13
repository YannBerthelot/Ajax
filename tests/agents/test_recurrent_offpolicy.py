"""The recurrent off-policy agents (SAC, ASAC, REDQ, TD3 with
``memory=MemoryConfig(...)``) on tiny Pendulum runs: the sequence-replay
plumbing (trajectory buffer, burn-in carries, sequence losses, R2D2 stored
state), not performance.

Their recurrent code never branches on the memory kind; the kinds' own
contract is pinned cell by cell in ``tests/networks/test_memory.py``. So SAC,
the lineage's reference, trains every kind, and ASAC, REDQ and TD3 each train
two kinds, one plain and one with stored state: each agent meets two carry
structures, and each kind runs once through the stored-state path. TD3's
own path (a deterministic actor, a target actor's carry) meets a lone array
(gru) and a windowed float-and-bool carry (transformer).
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.base import Extension
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig

AGENTS: dict[str, tuple[type, dict[str, Any]]] = {
    "SAC": (SAC, {}),
    "ASAC": (ASAC, {}),
    "REDQ": (REDQ, {"num_critics": 3, "subset_size": 2, "num_critic_updates": 2}),
    "TD3": (
        TD3,
        {"actor_architecture": ("32", "relu"), "critic_architecture": ("32", "relu")},
    ),
}
CRITICS = {"ASAC": 2, "REDQ": 3}  # an ensemble's carry keeps its critics' axis
TRAINS = [("SAC", kind) for kind in MEMORY_KINDS] + [
    ("ASAC", "lstm"),
    ("REDQ", "mamba"),
    ("TD3", "transformer"),
]
STORED = [("SAC", "transformer"), ("ASAC", "mamba"), ("REDQ", "lstm"), ("TD3", "gru")]


def _agent(name: str, **kwargs: Any) -> Any:
    cls, extra = AGENTS[name]
    return cls(env_id="Pendulum-v1", n_envs=1, **extra, **kwargs)


def _train(name: str, kind: str, **kwargs: Any) -> Any:
    agent = _agent(
        name,
        batch_size=8,
        buffer_size=1000,
        learning_starts=200,
        burn_in=4,
        sequence_length=8,
        memory=MemoryConfig(kind=kind, hidden_size=8, window=8),
        **kwargs,
    )
    state, _ = agent.train(seed=0, n_timesteps=260)
    return state


def _assert_finite(name: str, state: Any) -> None:
    trees = [
        state.actor_state.params,
        state.critic_state.params,
        state.critic_state.target_params,
    ]
    if name == "TD3":
        trees.append(state.actor_state.target_params)
    if name == "ASAC":
        trees.append(state.theta)
    for leaf in jax.tree.leaves(trees):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize("name, kind", TRAINS)
def test_a_recurrent_agent_trains_without_nans(name: str, kind: str) -> None:
    state = _train(name, kind)
    _assert_finite(name, state)
    # The actor's live carry advanced during collection (TD3's goes through
    # its action pipeline, which must hand the carry back).
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )
    # A critic ensemble's carry keeps its (num, batch) leading axes, after
    # the seed axis train vmaps over.
    for leaf in jax.tree.leaves(state.critic_state.hidden_state):
        assert name not in CRITICS or leaf.shape[1] == CRITICS[name]


@pytest.mark.parametrize("name, kind", STORED)
def test_stored_state_replays_the_carries_collection_recorded(
    name: str, kind: str
) -> None:
    """R2D2 stored-state replay: the actor's carries recorded at collection
    are written to the buffer (non-zero once the actor acted) and start the
    replayed sequences; TD3's target actor starts from them too."""
    state = _train(name, kind, stored_state=True)
    _assert_finite(name, state)
    buffer = state.collector_state.buffer_state.experience
    assert "actor_carry" in buffer
    assert jnp.any(buffer["actor_carry"] != 0)


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_learning_starts_must_cover_a_burn_in_and_a_sequence(name: str) -> None:
    with pytest.raises(ValueError, match="learning_starts"):
        _agent(
            name,
            learning_starts=10,
            burn_in=8,
            sequence_length=16,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_stored_state_requires_memory(name: str) -> None:
    with pytest.raises(ValueError, match="stored_state"):
        _agent(name, stored_state=True)


def _no_action(obs: jax.Array) -> jax.Array:
    return jnp.zeros(obs.shape[:-1] + (1,))


@pytest.mark.parametrize(
    "name, option",
    [("SAC", {"expert_policy": _no_action}), ("ASAC", {"extensions": (Extension(),)})],
)
def test_a_recurrent_agent_rejects_what_it_does_not_support(
    name: str, option: dict
) -> None:
    """At make_train time, before anything compiles."""
    agent = _agent(
        name,
        learning_starts=200,
        memory=MemoryConfig(kind="gru", hidden_size=8),
        **option,
    )
    with pytest.raises(NotImplementedError, match=f"Recurrent {name}"):
        agent.train(seed=0, n_timesteps=10)


def test_an_agent_without_memory_support_refuses_a_memory_config() -> None:
    no_memory = type("NoMemoryASAC", (ASAC,), {"supports_memory": False})
    with pytest.raises(NotImplementedError, match="does not support recurrent"):
        no_memory(env_id="Pendulum-v1", memory=MemoryConfig(kind="gru"))


def test_an_agent_acting_across_updates_refuses_cached_keys_and_values() -> None:
    """Its acting carry outlives an update, so the cached keys would be stale."""
    memory = MemoryConfig(kind="transformer", hidden_size=8, kv_cache=True)
    with pytest.raises(ValueError, match="kv_cache=False"):
        SAC(env_id="Pendulum-v1", memory=memory)
