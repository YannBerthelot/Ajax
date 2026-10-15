import functools
import importlib

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from ajax.agents.cloning import pre_train
from ajax.extensions.pretrain import CloningPretrain
from ajax.modules.pid_actor import PIDActor, PIDPolicy
from ajax.networks.networks import Actor
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
    expert = _HalfTorqueExpert()
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        expert_policy=expert,
        extensions=(CloningPretrain(expert, n_steps=64, epochs=1),),
        expert_buffer_n_steps=0,
        learning_starts=8,
        batch_size=8,
    )
    agent.train(seed=0, n_timesteps=16)
    assert double._cache_size() == 1


@pytest.mark.parametrize("agent_name", ["APO", "REDQ", "SAC", "TD3"])
def test_actor_cloning_lr_is_removed(agent_name):
    """``actor_cloning_lr`` was never read (cloning trains at the actor's
    learning rate) and warned from 5affdaa; it is now refused."""
    module = importlib.import_module(f"ajax.agents.{agent_name}.{agent_name}")
    with pytest.raises(TypeError, match="actor_cloning_lr"):
        getattr(module, agent_name)("Pendulum-v1", actor_cloning_lr=1e-3)


def _half_torque(obs):
    return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0)


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
def test_cloning_pretrain_seeds_the_agents_statistics(agent_name):
    """Each agent binding CloningPretrain clones in the pretrain phase, with
    its observation layout, and starts its observation statistics from the
    dataset's."""
    tiny = ("8", "relu")
    kw = {"n_envs": 2, "actor_architecture": tiny, "critic_architecture": tiny}
    kw |= {"expert_policy": _half_torque, **_LAYOUT.get(agent_name, {})}
    module = importlib.import_module(f"ajax.agents.{agent_name}.{agent_name}")
    cloning = CloningPretrain(_half_torque, n_steps=16, epochs=2, batch_size=8)
    agent = getattr(module, agent_name)("Pendulum-v1", extensions=(cloning,), **kw)
    state = agent.train(seed=0, n_timesteps=0)[0]
    assert state.actor_state.obs_norm_info is not None


def _sac_actor(*extensions):
    """SAC's actor parameters after its pretrain phase (a run of 0 steps
    updates nothing)."""
    from ajax import SAC

    tiny = ("8", "relu")
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=tiny,
        critic_architecture=tiny,
        expert_policy=_half_torque,
        expert_buffer_n_steps=0,
        extensions=extensions,
    )
    return agent.train(seed=0, n_timesteps=0)[0].actor_state.params["params"]


def _close(a, b):
    pairs = zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    return all(np.allclose(x, y, rtol=1e-5, atol=1e-7) for x, y in pairs)


def test_cloning_pretrain_resets_the_heads_it_is_told_to():
    """``reset_log_std`` puts the cloned actor's log-std head back at its
    initial values; ``reset_actor_head`` re-initialises the mean head too
    (orthogonal(0.01): a unit column scaled to 0.01, zero bias), the trunk
    staying cloned; without either, both heads keep their cloned values."""
    clone = functools.partial(
        CloningPretrain, _half_torque, n_steps=16, epochs=2, batch_size=8
    )
    initial, cloned = _sac_actor(), _sac_actor(clone())
    log_std = _sac_actor(clone(reset_log_std=True))
    head = _sac_actor(clone(reset_actor_head=True))
    assert not _close(cloned["mean"], initial["mean"])
    assert not _close(cloned["log_std"], initial["log_std"])
    assert _close(log_std["log_std"], initial["log_std"])
    assert _close(log_std["mean"], cloned["mean"])
    assert _close(head["log_std"], initial["log_std"])
    kernel = head["mean"]["kernel"]
    np.testing.assert_allclose(np.linalg.norm(kernel), 0.01, rtol=1e-5)
    np.testing.assert_array_equal(head["mean"]["bias"], 0.0)
    assert not _close(kernel, cloned["mean"]["kernel"])
    trunk = [k for k in cloned if k not in ("mean", "log_std")]
    assert trunk and all(_close(head[k], cloned[k]) for k in trunk)


@pytest.mark.parametrize(
    "actor",
    [
        Actor(("8", "relu"), action_dim=2, continuous=True, log_std_init=0.5),
        # APO's and PPO's log-std.
        Actor(
            ("8", "relu"),
            action_dim=2,
            continuous=True,
            log_std_state_independent=True,
            log_std_init=0.0,
        ),
        PIDActor(PIDPolicy(), action_dim=2, log_std_init=0.0),
    ],
    ids=["head", "parameter", "pid"],
)
def test_pre_train_reset_log_std_restores_any_log_std(actor):
    """``reset_log_std`` puts the actor's log-std back at its values before
    cloning, whatever its form: a head over the features at any
    ``log_std_init``, a state-independent parameter or a PID actor's."""
    params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))
    state = train_state.TrainState.create(
        apply_fn=actor.apply, params=params, tx=optax.adam(1e-3)
    )
    clone = functools.partial(
        pre_train,
        jax.random.PRNGKey(1),
        state,
        create_dummy_dataset(),
        actor_epochs=5,
        actor_batch_size=1,
    )
    cloned, reset = clone()[0].params, clone(reset_log_std=True)[0].params
    (name,) = [k for k in params["params"] if k.startswith("log_std")]
    assert not _close(cloned["params"][name], params["params"][name])
    assert _close(reset["params"][name], params["params"][name])
    others = [k for k in params["params"] if k != name]
    assert all(_close(reset["params"][k], cloned["params"][k]) for k in others)
