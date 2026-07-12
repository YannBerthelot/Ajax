"""Integration tests: memory blocks inside Actor / Critic / MultiCritic and
the train-state initialization plumbing."""

import distrax
import jax
import jax.numpy as jnp
import pytest

from ajax.networks.memory import MemoryConfig, init_carry
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
KINDS = ["gru", "lstm", "transformer", "mamba"]


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
    # legacy GRU path (recurrent + lstm_hidden_size) must still work
    legacy_actor = Actor(
        input_architecture=ARCH,
        action_dim=ACTION_DIM,
        continuous=True,
        memory=MemoryConfig(kind="gru", hidden_size=8),
    )
    legacy = init_network_state(
        init_x=jnp.zeros((B, OBS_DIM)),
        network=legacy_actor,
        key=jax.random.PRNGKey(0),
        tx=get_adam_tx(),
        recurrent=True,
        lstm_hidden_size=8,
        n_envs=B,
    )
    assert legacy.recurrent


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
