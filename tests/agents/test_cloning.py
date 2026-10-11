import importlib
import warnings
from dataclasses import dataclass

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from ajax.agents.cloning import pre_train
from ajax.extensions.base import Extension
from ajax.extensions.pretrain import CloningPretrain
from ajax.state import Transition


# -----------------------------
# Dummy networks and dataset
# -----------------------------
class DummyPolicy(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return distrax.Normal(x, 1)


# -----------------------------
# Fixtures
# -----------------------------
def create_dummy_actor_state(obs_dim=4, action_dim=2):
    rng = jax.random.PRNGKey(0)
    model = DummyPolicy(action_dim=action_dim)
    params = model.init(rng, jnp.ones((1, obs_dim)))
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def create_dummy_dataset():
    """Two transitions of one env (leading axes: step, env)."""

    def rows(*values):
        return jnp.array(values)[:, None]

    return Transition(
        obs=rows([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]),
        action=rows([0.5, 0.1], [0.2, 0.3]),
        reward=rows([1.0], [0.5]),
        terminated=rows([0.0], [1.0]),
        truncated=rows([0.0], [0.0]),
        next_obs=rows([1.1, 2.1, 3.1, 4.1], [2.1, 3.1, 4.1, 5.1]),
        log_prob=None,
    )


def expand_dataset(dataset, repeat=20):
    return jax.tree.map(lambda x: jnp.tile(x, (repeat, 1, 1)), dataset)


# -----------------------------
# Tests for pre_train
# -----------------------------
def test_pre_train_actor_converges():
    actor_state = create_dummy_actor_state()
    dataset = create_dummy_dataset()
    key = jax.random.PRNGKey(42)
    trained_actor, metrics, obs_mean, obs_std = pre_train(
        key,
        actor_state,
        dataset,
        actor_lr=5e-2,
        actor_epochs=100,
        actor_batch_size=2,
    )

    actor_losses = jnp.array(metrics["actor_loss"])
    assert actor_losses[-1] <= actor_losses[0]
    for i in range(2):
        # pre_train standardises obs internally; apply the same transform
        # at eval so the actor sees the input distribution it was trained on.
        std_obs = (dataset.obs[i] - obs_mean) / obs_std
        pred_action = trained_actor.apply_fn(trained_actor.params, std_obs).mean()
        assert jnp.allclose(pred_action, dataset.action[i], atol=0.3)


def test_pre_train_leaves_raw_columns_unstandardised():
    """The env's declared raw columns keep mean 0 / std 1: the clone and
    the statistics it seeds read them raw, the rest standardised."""
    dataset = create_dummy_dataset()
    _, _, obs_mean, obs_std = pre_train(
        jax.random.PRNGKey(0),
        create_dummy_actor_state(),
        dataset,
        actor_epochs=1,
        actor_batch_size=2,
        skip_actor=True,
        raw_columns=1,
    )
    flat = dataset.obs.reshape(-1, dataset.obs.shape[-1])
    assert obs_mean[0] == 0.0 and obs_std[0] == 1.0
    assert jnp.allclose(obs_mean[1:], flat[:, 1:].mean(0))


def test_pre_train_vmap_compatible():
    n = 3
    rngs = jax.random.split(jax.random.PRNGKey(0), n)
    actor_states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * n), create_dummy_actor_state()
    )

    dataset = expand_dataset(create_dummy_dataset(), repeat=10)

    batched_pre_train = jax.vmap(
        lambda rng, actor: pre_train(
            rng,
            actor,
            dataset,
            actor_lr=1e-3,
            actor_epochs=2,
            actor_batch_size=2,
        ),
        in_axes=(0, 0),
    )

    trained_actors, metrics, _obs_mean, _obs_std = batched_pre_train(rngs, actor_states)

    assert metrics["actor_loss"].shape[0] == n

    for a_old, a_new in zip(
        jax.tree_util.tree_leaves(create_dummy_actor_state().params),
        jax.tree_util.tree_leaves(trained_actors.params),
        strict=False,
    ):
        assert not jnp.allclose(a_old, a_new)


class _HalfTorqueExpert:
    """A stateful expert (``init_state``): SAC's evaluation steps it with a
    state."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0), state


def test_cloning_pretraining_leaves_jax_caches_alone():
    """Behaviour cloning runs while the agent's train function is traced;
    it must not clear JAX's caches there, which would make every function
    compiled before compile again."""
    from ajax import SAC

    double = jax.jit(lambda x: 2 * x)
    double(jnp.ones(2))
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        expert_policy=_HalfTorqueExpert(),
        pre_train_n_steps=64,
        actor_cloning_epochs=1,
        expert_buffer_n_steps=0,
        learning_starts=8,
        batch_size=8,
    )
    agent.train(seed=0, n_timesteps=16)
    assert double._cache_size() == 1


@pytest.mark.parametrize("agent_name", ["APO", "REDQ", "SAC", "TD3"])
def test_actor_cloning_lr_is_deprecated(agent_name):
    """Cloning trains at the actor's learning rate and never read
    ``actor_cloning_lr``: setting it warns instead of being silently
    ignored, and leaving it out stays quiet."""
    module = importlib.import_module(f"ajax.agents.{agent_name}.{agent_name}")
    agent_cls = getattr(module, agent_name)
    with pytest.warns(FutureWarning, match="actor_cloning_lr"):
        agent_cls("Pendulum-v1", actor_cloning_lr=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        agent_cls("Pendulum-v1")


def _agent_class(agent_name):
    module = importlib.import_module(f"ajax.agents.{agent_name}.{agent_name}")
    return getattr(module, agent_name)


def _half_torque(obs):
    return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0)


def test_the_cloning_keywords_build_a_cloning_pretrain():
    """The cloning keywords set the fields of the CloningPretrain the agent
    runs; 0 steps clone nothing; without an expert, or with a
    CloningPretrain extension (cloning twice), they raise."""
    from ajax import SAC

    keywords = {
        "pre_train_n_steps": 5,
        "actor_cloning_epochs": 3,
        "actor_cloning_batch_size": 7,
        "skip_actor_pretrain": True,
        "reset_log_std_after_bc": True,
        "reset_actor_head_after_bc": True,
    }
    agent = SAC("Pendulum-v1", expert_policy=_half_torque, **keywords)
    fields = {"n_steps": 5, "epochs": 3, "batch_size": 7, "skip_actor": True}
    fields |= {"reset_log_std": True, "reset_actor_head": True}
    assert agent.cloning == CloningPretrain(agent.expert_policy, **fields)
    assert SAC("Pendulum-v1", expert_policy=_half_torque).cloning is None
    with pytest.raises(ValueError, match="expert_policy"):
        SAC("Pendulum-v1", pre_train_n_steps=5)
    with pytest.raises(ValueError, match="extension only"):
        SAC(
            "Pendulum-v1",
            expert_policy=_half_torque,
            pre_train_n_steps=5,
            extensions=(CloningPretrain(_half_torque),),
        )


@dataclass(frozen=True)
class _OnTheKeywordsKey(CloningPretrain):
    """Clones on the key the cloning keywords clone on: the pretrain key the
    loop splits from the seed's, of which the pretrain phase gets a split."""

    seed: int = 0

    def pretrain(self, agent_state, ext_state, ctx):
        del ctx
        key = jax.random.split(jax.random.PRNGKey(self.seed))[1]
        return self.clone(agent_state, key), ext_state


@dataclass(frozen=True)
class _Inert(Extension):
    """No phase: a stack as long as the extension's, so that both agents
    draw the same keys after init (APO updates once even at 0 steps)."""


# SAC clones with its observation layout: the expert's action and the
# training fraction appended (its expert prefill off, for speed).
_LAYOUT = {
    "SAC": {
        "augment_obs_with_expert_action": True,
        "use_train_frac": True,
        "expert_buffer_n_steps": 0,
    },
    "APO": {"n_steps": 8, "batch_size": 8},
}


@pytest.mark.parametrize("agent_name", ["APO", "REDQ", "SAC", "TD3"])
def test_cloning_pretrain_clones_as_the_cloning_keywords(agent_name):
    """The cloning keywords clone at init on the pretrain key, a
    CloningPretrain extension in the pretrain phase: on the same key they
    give the same state (its observation statistics seeded), up to the
    rounding of the two places of the compiled program (an ulp on SAC)."""
    tiny = ("8", "relu")
    kw = {"n_envs": 2, "actor_architecture": tiny, "critic_architecture": tiny}
    kw |= {"expert_policy": _half_torque, **_LAYOUT.get(agent_name, {})}
    agent_cls = _agent_class(agent_name)
    by_keywords = agent_cls(
        "Pendulum-v1",
        pre_train_n_steps=16,
        actor_cloning_epochs=2,
        actor_cloning_batch_size=8,
        extensions=(_Inert(),),
        **kw,
    )
    cloning = _OnTheKeywordsKey(_half_torque, n_steps=16, epochs=2, batch_size=8)
    by_extension = agent_cls("Pendulum-v1", extensions=(cloning,), **kw)
    states = [
        agent.train(seed=0, n_timesteps=0)[0] for agent in (by_keywords, by_extension)
    ]
    assert states[1].actor_state.obs_norm_info is not None
    leaves = [jax.tree.leaves(state) for state in states]
    for a, b in zip(*leaves, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)
