"""The pretraining helpers and extensions of :mod:`ajax.extensions.pretrain`.

These are integration-style tests: they build real (tiny) flax TrainStates
and drive the pretrain helpers end-to-end. The helpers that need real env
rollouts or buffers with expert flags (``pretrain_critic_mc``,
``refresh_phi_star``) run inside SAC: the MCPretrain and PhiRefresh goldens
of test_sac_extensions_equivalence.py, test_expert_observations.py,
test_expert_extensions_raw_units.py and the probing tests.
"""

from types import SimpleNamespace
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct

from ajax.agents.SAC.SAC import SAC
from ajax.extensions.base import ExtensionContext
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)
from ajax.extensions.pretrain import (
    BellmanPretrain,
    CloningPretrain,
    MCPretrain,
    PhiRefresh,
    pretrain_critic_mc,
    pretrain_critic_online_light,
)
from ajax.state import LoadedTrainState
from tests.environments import paid_action

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
    assert bound.bind_to_agent(**{**context, "gamma": 0.5}) is bound
    assert ext.bind_to_agent(env_args=None) is ext


def test_mc_pretrain_binds_the_agents_discount_and_reward_scale():
    ext, env = MCPretrain(expert_policy=abs), SimpleNamespace(observers=())
    context = {"env_args": env, "network_args": "net", "gamma": 0.9}
    context.update({"critic_optimizer_args": "opt", "reward_scale": 2.0})
    bound = ext.bind_to_agent(**context, expert_policy=abs)
    assert (bound.env_args, bound.gamma, bound.reward_scale) == (env, 0.9, 2.0)


def test_mc_returns_carry_the_reward_scale():
    """phi* fits G_t = sum_k gamma^k c r_{t+k}, the rewards scaled by the
    agent's reward_scale c as its critic's targets are. On one-step episodes
    paying the expert's 0.5, from a critic at 0 everywhere, the first loss
    is the mean of G^2 = (0.5 c)^2."""
    env, params = paid_action.make()
    net = ("8", "relu")
    agent = SAC(env, env_params=params, actor_architecture=net, critic_architecture=net)
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    zero = jax.tree.map(jnp.zeros_like, state.critic_state.params)
    critic = state.critic_state.replace(params=zero)

    def first_loss(reward_scale: float) -> float:
        out = pretrain_critic_mc(
            state,
            critic,
            lambda obs: jnp.full(obs.shape[:-1] + (1,), 0.5),
            agent.env_args.mode,
            agent.env_args,
            gamma=0.9,
            reward_scale=reward_scale,
            n_mc_steps=8,
            n_mc_episodes=1,
            n_steps=1,
            batch_size=4,
        )
        return float(out[4].initial_loss)

    np.testing.assert_allclose(
        [first_loss(1.0), first_loss(2.0)], [0.25, 1.0], rtol=1e-5
    )


def test_bellman_pretrain_needs_the_agent_context_and_a_seeded_replay():
    """Unbound, its pretrain phase raises; without a seeder (ExpertPrefill
    or PriorData) the replay is empty when it runs, so SAC refuses it when
    built, with an expert or without one."""
    context = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        BellmanPretrain().pretrain(None, (), context)
    bellman = (BellmanPretrain(n_steps=1),)
    with pytest.raises(MissingCapability, match="add ExpertPrefill or PriorData"):
        SAC("Pendulum-v1", extensions=bellman)
    with pytest.raises(MissingCapability, match="add ExpertPrefill or PriorData"):
        SAC(
            "Pendulum-v1",
            expert_policy=lambda obs: obs[..., :1],
            expert_buffer_n_steps=0,
            extensions=bellman,
        )


def test_cloning_pretrain_binds_the_agent_context():
    """An agent hands CloningPretrain its env (its observation features
    with it) and actor optimizer; an extension already bound keeps its own,
    and an agent without an actor optimizer leaves it unbound."""
    ext = CloningPretrain(expert_policy=abs, n_steps=8)
    env = SimpleNamespace(observers=(TrainingProgressInput(),))
    context = {"env_args": env, "actor_optimizer_args": "adam", "expert_policy": abs}
    bound = ext.bind_to_agent(**context)
    assert (bound.env_args, bound.actor_optimizer_args) == (env, "adam")
    assert bound.bind_to_agent(**{**context, "env_args": "other"}) is bound
    assert ext.bind_to_agent(env_args=env) is ext


@pytest.mark.parametrize(
    "feature", [ExpertActionInput(), ExpertStateInput(width=1)], ids=["action", "state"]
)
def test_cloning_pretrain_clones_the_expert_the_observations_carry(feature):
    """Observations whose features read the agent's expert would hold another
    expert's in the cloning dataset: binding raises."""
    ext, env = CloningPretrain(expert_policy=abs), SimpleNamespace(observers=(feature,))
    context = {"env_args": env, "actor_optimizer_args": "adam"}
    with pytest.raises(ValueError, match="agent's expert_policy"):
        ext.bind_to_agent(**context, expert_policy=len)
    assert ext.bind_to_agent(**context, expert_policy=abs).env_args is env


def test_cloning_pretrain_comes_before_the_other_pretraining():
    """Cloning seeds the statistics the critic reads: a critic pretrained
    before it would be queried on other inputs, so binding raises."""
    ext, other = CloningPretrain(expert_policy=abs), BellmanPretrain()
    env = SimpleNamespace(observers=())
    context = {"env_args": env, "actor_optimizer_args": "adam"}
    assert ext.bind_to_agent(**context, extensions=(ext, other)).env_args is env
    with pytest.raises(ValueError, match="before the other pretraining"):
        ext.bind_to_agent(**context, extensions=(other, ext))


def test_cloning_pretrain_needs_the_agent_context():
    context = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        CloningPretrain(expert_policy=abs).pretrain(None, (), context)


def test_phi_refresh_needs_the_agent_context():
    """Unbound (no agent handed it its buffer), its post_update raises
    rather than never refreshing phi*."""
    context = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="did not bind"):
        PhiRefresh(expert_policy=abs).post_update(None, (), context)
