"""Integration tests: memory blocks inside Actor / Critic / MultiCritic (and
TD3's DeterministicActor, Ni et al.'s layout) and the train-state
initialization plumbing."""

from typing import Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from ajax.agents.TD3.networks import DeterministicActor
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig, init_carry
from ajax.networks.networks import (
    Actor,
    Critic,
    MultiCritic,
    init_network_carry,
    init_network_state,
    predict_value_sequence,
)
from ajax.networks.utils import get_adam_tx

T, B, OBS_DIM, ACTION_DIM = 5, 3, 4, 2
ARCH = ("16", "relu")
KINDS = MEMORY_KINDS


def _memory(kind):
    return MemoryConfig(kind=kind, hidden_size=8, window=4)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("continuous", [True, False])
def test_recurrent_actor_returns_dist_and_carry(kind, continuous):
    memory = _memory(kind)
    actor = Actor(
        input_architecture=ARCH,
        action_dim=ACTION_DIM,
        continuous=continuous,
        squash=continuous,
        memory=memory,
    )
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (T, B, OBS_DIM))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_carry(memory, rng, B)
    params = actor.init(rng, obs, hidden_state=carry, done=resets)

    pi, new_carry = actor.apply(params, obs, hidden_state=carry, done=resets)
    assert isinstance(pi, distrax.Distribution) or hasattr(pi, "sample_and_log_prob")
    action = pi.sample(seed=rng)
    expected = (T, B, ACTION_DIM) if continuous else (T, B)
    assert action.shape == expected
    assert jax.tree.structure(new_carry) == jax.tree.structure(carry)


@pytest.mark.parametrize("kind", KINDS)
def test_recurrent_actor_requires_carry_and_done(kind):
    actor = Actor(
        input_architecture=ARCH,
        action_dim=ACTION_DIM,
        continuous=True,
        memory=_memory(kind),
    )
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((T, B, OBS_DIM))
    with pytest.raises(ValueError, match="hidden_state and done"):
        actor.init(rng, obs)


@pytest.mark.parametrize("kind", KINDS)
def test_recurrent_critic(kind):
    memory = _memory(kind)
    critic = Critic(input_architecture=ARCH, memory=memory)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (T, B, OBS_DIM + ACTION_DIM))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_carry(memory, rng, B)
    params = critic.init(rng, x, hidden_state=carry, done=resets)
    values, new_carry = critic.apply(params, x, hidden_state=carry, done=resets)
    assert values.shape == (T, B, 1)
    assert jax.tree.structure(new_carry) == jax.tree.structure(carry)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num", [1, 2])
def test_recurrent_multicritic_carry_per_member(kind, num):
    memory = _memory(kind)
    critic = MultiCritic(input_architecture=ARCH, num=num, memory=memory)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (T, B, OBS_DIM + ACTION_DIM))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_network_carry(critic, memory, rng, B)
    # ensemble carry: leading (num,) axis on every leaf
    for leaf in jax.tree.leaves(carry):
        assert leaf.shape[:2] == (num, B)
    params = critic.init(rng, x, hidden_state=carry, done=resets)
    values, new_carry = critic.apply(params, x, hidden_state=carry, done=resets)
    assert values.shape == (num, T, B, 1)
    for leaf in jax.tree.leaves(new_carry):
        assert leaf.shape[:2] == (num, B)
    # ensemble members have different params, so their memories must diverge
    if num > 1:
        assert not jnp.allclose(values[0], values[1])


def test_multicritic_memory_none_signature_unchanged():
    critic = MultiCritic(input_architecture=ARCH, num=2)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (B, OBS_DIM))
    params = critic.init(rng, x)
    values = critic.apply(params, x)
    assert values.shape == (2, B, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_init_network_state_builds_carry_and_flags(kind):
    memory = _memory(kind)
    actor = Actor(
        input_architecture=ARCH,
        action_dim=ACTION_DIM,
        continuous=True,
        memory=memory,
    )
    state = init_network_state(
        init_x=jnp.zeros((B, OBS_DIM)),
        network=actor,
        key=jax.random.PRNGKey(0),
        tx=get_adam_tx(),
        memory=memory,
        n_envs=B,
    )
    assert state.recurrent
    assert state.hidden_state is not None


def test_init_network_state_feedforward_unchanged():
    actor = Actor(input_architecture=ARCH, action_dim=ACTION_DIM, continuous=True)
    state = init_network_state(
        init_x=jnp.zeros((B, OBS_DIM)),
        network=actor,
        key=jax.random.PRNGKey(0),
        tx=get_adam_tx(),
        n_envs=B,
    )
    assert not state.recurrent
    assert state.hidden_state is None


@pytest.mark.parametrize("kind", KINDS)
def test_predict_value_sequence(kind):
    memory = _memory(kind)
    critic = MultiCritic(input_architecture=ARCH, num=2, memory=memory)
    state = init_network_state(
        init_x=jnp.zeros((B, OBS_DIM + ACTION_DIM)),
        network=critic,
        key=jax.random.PRNGKey(0),
        tx=get_adam_tx(),
        memory=memory,
        n_envs=B,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, OBS_DIM + ACTION_DIM))
    resets = jnp.zeros((T, B), dtype=bool)
    values, carry = predict_value_sequence(
        state, state.params, x, resets, state.hidden_state
    )
    assert values.shape == (2, T, B, 1)
    assert jax.tree.structure(carry) == jax.tree.structure(state.hidden_state)


@pytest.mark.parametrize("kind", KINDS)
def test_a_recurrent_q_critic_queries_its_action_after_the_memory(kind):
    """With ``query_dim`` the input's last features are the action asked
    about: they move that step's value only, never the carry."""
    memory = _memory(kind)
    critic = MultiCritic(
        input_architecture=ARCH, num=2, memory=memory, query_dim=ACTION_DIM
    )
    state = init_network_state(
        init_x=jnp.zeros((B, OBS_DIM + 2 * ACTION_DIM)),
        network=critic,
        key=jax.random.PRNGKey(0),
        tx=get_adam_tx(),
        memory=memory,
        n_envs=B,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, OBS_DIM + 2 * ACTION_DIM))
    resets = jnp.zeros((T, B), dtype=bool)
    values, carry = predict_value_sequence(
        state, state.params, x, resets, state.hidden_state
    )
    asked = x.at[2, :, -ACTION_DIM:].add(1.0)
    moved, moved_carry = predict_value_sequence(
        state, state.params, asked, resets, state.hidden_state
    )
    same = jnp.arange(T) != 2
    assert jnp.array_equal(values[:, same], moved[:, same])
    assert not jnp.allclose(values[:, 2], moved[:, 2])
    for leaf, moved_leaf in zip(
        jax.tree.leaves(carry), jax.tree.leaves(moved_carry), strict=False
    ):
        assert jnp.array_equal(leaf, moved_leaf)


def _kernels(params: Any) -> dict[str, tuple]:
    """Each Dense kernel's (inputs, outputs), by its path; an ensemble's
    leading member axis and name dropped."""
    flat = flatten_dict(params["params"], sep="/")
    kernels = {k: v.shape[-2:] for k, v in flat.items() if k.endswith("kernel")}
    return {k.removeprefix("ensemble/"): v for k, v in kernels.items()}


def _ni_networks(hidden: int) -> dict[str, dict[str, tuple]]:
    """The three recurrent off-policy networks, initialised on (obs,
    previous action[, current action]) for a GRU of ``hidden`` units."""
    memory, rng = MemoryConfig("gru", hidden), jax.random.PRNGKey(0)
    actor = Actor(ARCH, ACTION_DIM, True, memory=memory, previous_action_dim=ACTION_DIM)
    td3 = DeterministicActor(ARCH, ACTION_DIM, memory=memory)
    critic = MultiCritic(ARCH, num=2, memory=memory, query_dim=ACTION_DIM)
    acting = jnp.zeros((1, B, OBS_DIM + ACTION_DIM))
    out = {}
    for name, net, x in [
        ("actor", actor, acting),
        ("td3", td3, acting),
        ("critic", critic, jnp.zeros((1, B, OBS_DIM + 2 * ACTION_DIM))),
    ]:
        carry = init_network_carry(net, memory, rng, B)
        out[name] = _kernels(net.init(rng, x, carry, jnp.zeros((1, B), bool)))
    return out


@pytest.mark.parametrize("hidden", [8, 128])
def test_the_recurrent_off_policy_networks_take_ni_et_als_layout(hidden: int) -> None:
    """Ni et al. 2022: fixed-width ReLU embedders, the observation in 32 and
    the previous action in 16, feed the memory (48 inputs); the MLP after it
    reads its output and the current observation embedded in 32 (actors)
    or the current observation and action in 48 (the critic), whatever the
    memory's width."""
    nets, enc = _ni_networks(hidden), "encoder/network/layers_0/kernel"
    for name, net in nets.items():
        assert net["observation_embedder/network/layers_0/kernel"] == (OBS_DIM, 32)
        assert net["action_embedder/network/layers_0/kernel"] == (ACTION_DIM, 16)
        assert net["memory_cell/core/gru_0/ir/kernel"] == (48, hidden), name
    shortcut = "shortcut_embedder/network/layers_0/kernel"
    for name in ("actor", "td3"):
        assert nets[name][shortcut] == (OBS_DIM, 32)
    assert nets["actor"][enc] == (hidden + 32, 16)
    assert nets["td3"]["hidden/layers_0/kernel"] == (hidden + 32, 16)
    assert nets["critic"][shortcut] == (OBS_DIM + ACTION_DIM, 48)
    assert nets["critic"][enc] == (hidden + 48, 16)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("td3", [False, True])
def test_a_recurrent_off_policy_actor_remembers_its_previous_action(
    kind: str, td3: bool
) -> None:
    """The input's last features, the previous action, feed the carry: a
    change at step 2 moves the actions from step 2 on, and the carry."""
    memory, rng = _memory(kind), jax.random.PRNGKey(0)
    actor = (
        DeterministicActor(ARCH, ACTION_DIM, memory=memory)
        if td3
        else Actor(ARCH, ACTION_DIM, True, memory=memory, previous_action_dim=2)
    )
    x = jax.random.normal(rng, (T, B, OBS_DIM + ACTION_DIM))
    carry, resets = init_carry(memory, rng, B), jnp.zeros((T, B), bool)
    params = actor.init(rng, x, hidden_state=carry, done=resets)
    pi, last = actor.apply(params, x, hidden_state=carry, done=resets)
    moved_x = x.at[2, :, -ACTION_DIM:].add(1.0)
    moved, moved_last = actor.apply(params, moved_x, hidden_state=carry, done=resets)
    np.testing.assert_array_equal(pi.mean()[:2], moved.mean()[:2])
    for t in range(2, T):
        assert not np.allclose(pi.mean()[t], moved.mean()[t])
    assert not all(
        np.allclose(a, b)
        for a, b in zip(jax.tree.leaves(last), jax.tree.leaves(moved_last), strict=True)
    )


@pytest.mark.parametrize("memory, cnn", [(None, None), (_memory("gru"), (2, 2, 1))])
def test_an_actor_reads_its_previous_action_only_with_memory_on_vectors(
    memory: Any, cnn: Any
) -> None:
    actor = Actor(
        ARCH,
        ACTION_DIM,
        True,
        memory=memory,
        previous_action_dim=ACTION_DIM,
        cnn_image_shape=cnn,
    )
    x, carry = jnp.zeros((1, B, OBS_DIM + ACTION_DIM)), None
    if memory is not None:
        carry = init_carry(memory, jax.random.PRNGKey(0), B)
    with pytest.raises(ValueError, match="only with memory, on vector"):
        actor.init(jax.random.PRNGKey(0), x, carry, jnp.zeros((1, B), bool))
