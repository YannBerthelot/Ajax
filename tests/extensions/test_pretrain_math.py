"""The pretraining helpers and extensions of :mod:`ajax.extensions.pretrain`.

These are integration-style tests: they build real (tiny) flax TrainStates
and drive the pretrain helpers end-to-end. The helpers that need real env
rollouts or buffers with expert flags (``pretrain_critic_mc``,
``refresh_phi_star``) run inside SAC: the MCPretrain and PhiRefresh goldens
of test_sac_extensions_equivalence.py, test_expert_observations.py,
test_expert_extensions_raw_units.py and the probing tests.
"""

from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import struct

from ajax.agents.SAC.SAC import SAC
from ajax.extensions.base import ExtensionContext
from ajax.extensions.pretrain import (
    BellmanPretrain,
    CloningPretrain,
    PhiRefresh,
    pretrain_critic_online_light,
)
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
# pretrain_critic_online_light
# ---------------------------------------------------------------------------


def test_pretrain_critic_online_light_changes_params(rng, pretrain_batches):
    critic_state = _make_critic_state(rng)
    expert_params = _make_critic_state(jax.random.fold_in(rng, 1)).params
    state = FakePretrainState(
        critic_state=critic_state,
        expert_critic_params=expert_params,
    )
    obs, actions = pretrain_batches
    out = pretrain_critic_online_light(state, obs, actions, n_steps=2, lr_scale=1.0)
    assert isinstance(out, FakePretrainState)
    # At least one leaf in the critic params should change after updates.
    orig_leaves = jax.tree_util.tree_leaves(critic_state.params)
    new_leaves = jax.tree_util.tree_leaves(out.critic_state.params)
    diffs = [
        float(jnp.abs(a - b).sum())
        for a, b in zip(orig_leaves, new_leaves, strict=False)
    ]
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
        strict=False,
    ):
        assert jnp.allclose(a, b)


def test_bellman_pretrain_binds_the_agent_context():
    """SAC's binding hands BellmanPretrain its buffer, discount, reward scale
    and critic updates; an extension that is already bound keeps its own."""
    ext = BellmanPretrain(n_steps=3)
    context = {
        "buffer": "buffer",
        "gamma": 0.9,
        "reward_scale": 2.0,
        "recurrent": False,
        "update_value_fn": len,
        "update_target_fn": abs,
        "expert_policy": abs,
    }
    bound = ext.bind_to_agent(**context)
    assert (bound.buffer, bound.gamma, bound.reward_scale) == ("buffer", 0.9, 2.0)
    assert (bound.update_value_fn, bound.update_target_fn) == (len, abs)
    assert bound.bind_to_agent(**context | {"gamma": 0.5}) is bound
    assert ext.bind_to_agent(env_args=None) is ext


def test_bellman_pretrain_needs_the_agent_context_and_expert():
    """Unbound, its pretrain phase raises; bound by an agent without an
    expert (so no expert prefill to pre-train on), SAC's training raises."""
    ctx = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        BellmanPretrain().pretrain(None, (), ctx)
    agent = SAC("Pendulum-v1", extensions=(BellmanPretrain(n_steps=1),))
    with pytest.raises(ValueError, match="expert_policy"):
        agent.train(seed=0, n_timesteps=10)


def test_cloning_pretrain_binds_the_agent_context():
    """An agent hands CloningPretrain its env, actor optimizer and
    observation layout; an extension already bound keeps its own, and an
    agent without an actor optimizer leaves it unbound."""
    ext = CloningPretrain(expert_policy=abs, n_steps=8)
    context = {
        "env_args": "env",
        "actor_optimizer_args": "adam",
        "expert_policy": abs,
        "augment_obs_with_expert_action": True,
        "use_train_frac": True,
    }
    bound = ext.bind_to_agent(**context)
    assert (bound.env_args, bound.actor_optimizer_args) == ("env", "adam")
    assert bound.augment_obs_with_expert_action and bound.use_train_frac
    assert not bound.augment_obs_with_expert_state
    assert bound.bind_to_agent(**context | {"env_args": "other"}) is bound
    assert ext.bind_to_agent(env_args="env") is ext


@pytest.mark.parametrize(
    "layout", ["augment_obs_with_expert_action", "augment_obs_with_expert_state"]
)
def test_cloning_pretrain_clones_the_expert_the_observations_carry(layout):
    """Observations carrying the agent's expert's action or state would hold
    another expert's in the cloning dataset: binding raises."""
    ext = CloningPretrain(expert_policy=abs)
    context = {"env_args": "env", "actor_optimizer_args": "adam", layout: True}
    with pytest.raises(ValueError, match="agent's expert_policy"):
        ext.bind_to_agent(**context, expert_policy=len)
    assert ext.bind_to_agent(**context, expert_policy=abs).env_args == "env"


def test_cloning_pretrain_comes_before_the_other_pretraining():
    """Cloning seeds the statistics the critic reads: a critic pretrained
    before it would be queried on other inputs, so binding raises."""
    ext, other = CloningPretrain(expert_policy=abs), BellmanPretrain()
    context = {"env_args": "env", "actor_optimizer_args": "adam"}
    assert ext.bind_to_agent(**context, extensions=(ext, other)).env_args == "env"
    with pytest.raises(ValueError, match="before the other pretraining"):
        ext.bind_to_agent(**context, extensions=(other, ext))


def test_cloning_pretrain_needs_the_agent_context():
    ctx = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        CloningPretrain(expert_policy=abs).pretrain(None, (), ctx)


def test_phi_refresh_needs_the_agent_context():
    """Unbound (no agent handed it its buffer), its post_update raises
    rather than never refreshing phi*."""
    ctx = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        PhiRefresh(expert_policy=abs).post_update(None, (), ctx)
