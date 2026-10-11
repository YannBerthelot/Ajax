import brax
import distrax
import gymnax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from ajax.agents.cloning import (
    collect_experience_from_expert_policy,
    pre_train,
)
from ajax.state import EnvironmentConfig, Transition


@pytest.fixture(scope="session")
def gymnax_env():
    """Provide a simple Gymnax environment and default params."""
    env, env_params = gymnax.make("Pendulum-v1")
    return env, env_params


@pytest.fixture(scope="session")
def brax_env():
    """Provide a simple Brax environment (stateless)."""
    env = brax.envs.create("fast", batch_size=1)
    return env


@pytest.fixture
def gymnax_env_args(gymnax_env):
    """Wrap Gymnax env into EnvironmentConfig."""
    env, env_params = gymnax_env
    n_envs = 2
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=n_envs,
        continuous=False,
    )


@pytest.fixture
def brax_env_args(brax_env):
    """Wrap Brax env into EnvironmentConfig."""
    n_envs = 1
    return EnvironmentConfig(
        env=brax_env,
        env_params=None,
        n_envs=n_envs,
        continuous=True,
    )


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
    obs = jnp.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0]],
        ]
    )  # shape (2, 1, 4)

    actions = jnp.array(
        [
            [[0.5, 0.1]],
            [[0.2, 0.3]],
        ]
    )  # shape (2, 1, 2)

    rewards = jnp.array(
        [
            [[1.0]],
            [[0.5]],
        ]
    )  # shape (2, 1, 1)

    terminated = jnp.array(
        [
            [[0.0]],
            [[1.0]],
        ]
    )  # shape (2, 1, 1)

    truncated = jnp.array(
        [
            [[0.0]],
            [[0.0]],
        ]
    )  # shape (2, 1, 1)

    next_obs = jnp.array(
        [
            [[1.1, 2.1, 3.1, 4.1]],
            [[2.1, 3.1, 4.1, 5.1]],
        ]
    )  # shape (2, 1, 4)

    return Transition(
        obs=obs,
        action=actions,
        reward=rewards,
        terminated=terminated,
        truncated=truncated,
        next_obs=next_obs,
        log_prob=None,
    )


def expand_dataset(dataset, repeat=20):
    return Transition(
        obs=jnp.tile(dataset.obs, (repeat, 1, 1)),
        action=jnp.tile(dataset.action, (repeat, 1, 1)),
        reward=jnp.tile(dataset.reward, (repeat, 1, 1)),
        terminated=jnp.tile(dataset.terminated, (repeat, 1, 1)),
        truncated=jnp.tile(dataset.truncated, (repeat, 1, 1)),
        next_obs=jnp.tile(dataset.next_obs, (repeat, 1, 1)),
        log_prob=(
            None
            if dataset.log_prob is None
            else jnp.tile(dataset.log_prob, (repeat, 1, 1))
        ),
    )


# -----------------------------
# Tests for pre_train
# -----------------------------
def test_pre_train_actor_converges():
    actor_state = create_dummy_actor_state()
    dataset = expand_dataset(create_dummy_dataset(), repeat=1)
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
    dataset = expand_dataset(create_dummy_dataset(), repeat=1)
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
    ):
        assert not jnp.allclose(a_old, a_new)


# -----------------------------
# Tests
# -----------------------------
@pytest.mark.parametrize(
    "mode,env_args_fixture",
    [
        ("gymnax", "gymnax_env_args"),
        ("brax", "brax_env_args"),
    ],
)
def test_collect_experience_shapes_and_types(request, mode, env_args_fixture):
    """Check Transition shapes and dtypes for Gymnax and Brax."""
    env_args = request.getfixturevalue(env_args_fixture)
    rng = jax.random.PRNGKey(0)
    n_steps = 5

    # Expert policy depends on env type
    if env_args.continuous:
        action_dim = (
            env_args.env.action_size if hasattr(env_args.env, "action_size") else 1
        )

        def expert_policy(obs):
            return jnp.zeros((action_dim), dtype=jnp.float32)

    else:

        def expert_policy(obs):
            return jnp.zeros((1), dtype=jnp.int32)

    transitions = collect_experience_from_expert_policy(
        expert_policy, rng, mode, env_args, n_timesteps=n_steps
    )

    assert isinstance(transitions, Transition)
    assert transitions.obs.shape[0] == n_steps
    assert transitions.action.shape[0] == n_steps
    assert transitions.reward.shape[0] == n_steps
    assert transitions.terminated.shape[0] == n_steps
    assert transitions.truncated.shape[0] == n_steps


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
