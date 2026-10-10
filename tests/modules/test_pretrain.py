"""Tests for ajax.modules.pretrain — composable pretraining helpers.

These are integration-style tests: they build real (tiny) flax TrainStates
and drive the pretrain helpers end-to-end. Heavy helpers
(``pretrain_critic_mc``, ``refresh_phi_star``) that require real env rollouts
or buffers with expert flags are out of scope here — they are exercised
indirectly through the SAC probing tests.
"""

from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import struct

from ajax.modules.pretrain import MCPretrainAux, pretrain_critic_online_light
from ajax.state import LoadedTrainState

OBS_DIM = 4
ACTION_DIM = 2
BATCH_SIZE = 8
N_BATCHES = 3


class TinyCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(16)(x)
        h = nn.relu(h)
        q = nn.Dense(1)(h)
        # Ensemble dimension expected by ``predict_value``.
        return q[None, ...]


@struct.dataclass
class FakePretrainState:
    critic_state: LoadedTrainState
    expert_critic_params: Optional[Any] = None


def _make_critic_state(rng):
    critic = TinyCritic()
    dummy_in = jnp.zeros((1, OBS_DIM + ACTION_DIM))
    params = critic.init(rng, dummy_in)
    return LoadedTrainState.create(
        apply_fn=critic.apply,
        params=params,
        tx=optax.sgd(1e-2),
        target_params=params,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def pretrain_batches(rng):
    obs_key, act_key = jax.random.split(rng)
    obs = jax.random.normal(obs_key, (N_BATCHES, BATCH_SIZE, OBS_DIM))
    actions = jax.random.normal(act_key, (N_BATCHES, BATCH_SIZE, ACTION_DIM))
    return obs, actions


# ---------------------------------------------------------------------------
# Auxiliary dataclass smoke tests
# ---------------------------------------------------------------------------


def test_mc_pretrain_aux_fields_round_trip():
    aux = MCPretrainAux(
        initial_loss=jnp.asarray(1.0),
        final_loss=jnp.asarray(0.5),
        q_expert_mean=jnp.asarray(2.0),
        q_expert_min=jnp.asarray(-1.0),
        q_expert_max=jnp.asarray(5.0),
        v_min=jnp.asarray(-1.0),
        v_max=jnp.asarray(5.0),
    )
    # Field access plus dataclass-style replace should both work.
    replaced = aux.replace(final_loss=jnp.asarray(0.25))
    assert float(replaced.final_loss) == pytest.approx(0.25)
    assert float(replaced.initial_loss) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pretrain_critic_online_light
# ---------------------------------------------------------------------------


def test_pretrain_critic_online_light_returns_agent_state(rng, pretrain_batches):
    critic_state = _make_critic_state(rng)
    # Frozen "expert" params = different params so the nudge has signal.
    expert_params = _make_critic_state(jax.random.fold_in(rng, 1)).params
    state = FakePretrainState(
        critic_state=critic_state,
        expert_critic_params=expert_params,
    )
    obs, actions = pretrain_batches
    out = pretrain_critic_online_light(state, obs, actions, n_steps=2, lr_scale=0.1)
    assert isinstance(out, FakePretrainState)


def test_pretrain_critic_online_light_changes_params(rng, pretrain_batches):
    critic_state = _make_critic_state(rng)
    expert_params = _make_critic_state(jax.random.fold_in(rng, 1)).params
    state = FakePretrainState(
        critic_state=critic_state,
        expert_critic_params=expert_params,
    )
    obs, actions = pretrain_batches
    out = pretrain_critic_online_light(state, obs, actions, n_steps=2, lr_scale=1.0)
    # At least one leaf in the critic params should change after updates.
    orig_leaves = jax.tree_util.tree_leaves(critic_state.params)
    new_leaves = jax.tree_util.tree_leaves(out.critic_state.params)
    diffs = [float(jnp.abs(a - b).sum()) for a, b in zip(orig_leaves, new_leaves)]
    assert max(diffs) > 0.0


def test_pretrain_critic_online_light_zero_lr_is_no_op(rng, pretrain_batches):
    critic_state = _make_critic_state(rng)
    expert_params = critic_state.params  # same as online — zero loss anyway
    state = FakePretrainState(
        critic_state=critic_state,
        expert_critic_params=expert_params,
    )
    obs, actions = pretrain_batches
    out = pretrain_critic_online_light(state, obs, actions, n_steps=2, lr_scale=0.0)
    for a, b in zip(
        jax.tree_util.tree_leaves(critic_state.params),
        jax.tree_util.tree_leaves(out.critic_state.params),
    ):
        assert jnp.allclose(a, b)
