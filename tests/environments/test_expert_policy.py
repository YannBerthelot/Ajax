import brax
import gymnax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn
from flax import struct
from flax.training import train_state

from ajax.agents.PPO.train_PPO_pre_train import (
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
    env = brax.envs.create("fast", batch_size=1)  # fast = minimal environment in brax
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
        return x


class DummyCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


@struct.dataclass
class FakeTransition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_obs: jax.Array


# -----------------------------
# Fixtures
# -----------------------------
def create_dummy_actor_state(obs_dim=4, action_dim=2):
    rng = jax.random.PRNGKey(0)
    model = DummyPolicy(action_dim=action_dim)
    params = model.init(rng, jnp.ones((1, obs_dim)))
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def create_dummy_critic_state(obs_dim=4):
    rng = jax.random.PRNGKey(0)
    model = DummyCritic()
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
    critic_state = create_dummy_critic_state()
    dataset = expand_dataset(create_dummy_dataset(), repeat=1)
    key = jax.random.PRNGKey(42)
    trained_actor, trained_critic, metrics = pre_train(
        key,
        actor_state,
        critic_state,
        dataset,
        actor_lr=5e-2,
        actor_epochs=100,
        actor_batch_size=2,
        critic_lr=5e-2,
        critic_epochs=50,
        critic_batch_size=4,
    )

    actor_losses = jnp.array(metrics["actor_loss"])
    assert actor_losses[-1] <= actor_losses[0]
    for i in range(2):
        pred_action = trained_actor.apply_fn(trained_actor.params, dataset.obs[i])
        assert jnp.allclose(pred_action, dataset.action[i], atol=0.3)


def test_pre_train_critic_converges():
    actor_state = create_dummy_actor_state()
    critic_state = create_dummy_critic_state()
    dataset = expand_dataset(create_dummy_dataset(), repeat=20)
    key = jax.random.PRNGKey(42)
    trained_actor, trained_critic, metrics = pre_train(
        key,
        actor_state,
        critic_state,
        dataset,
        actor_lr=1e-3,
        actor_epochs=10,
        actor_batch_size=2,
        critic_lr=5e-2,
        critic_epochs=50,
        critic_batch_size=4,
    )

    critic_losses = jnp.array(metrics["critic_loss"])
    assert critic_losses[-1] <= critic_losses[0]

    gamma = 0.99
    for i in range(2):
        v_pred = trained_critic.apply_fn(trained_critic.params, dataset.obs[i])
        v_pred = jnp.squeeze(v_pred)

        v_next = trained_critic.apply_fn(trained_critic.params, dataset.next_obs[i])
        v_next = jnp.squeeze(v_next)

        td_target = jnp.squeeze(dataset.reward[i]) + gamma * v_next * (
            1.0 - jnp.squeeze(dataset.terminated[i])
        )

        # Use allclose or norm-based check
        assert jnp.allclose(
            v_pred, td_target, atol=0.5
        ), f"Critic prediction {v_pred} too far from TD target {td_target}"


def test_pre_train_vmap_compatible():
    n = 3
    rngs = jax.random.split(jax.random.PRNGKey(0), n)
    actor_states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * n), create_dummy_actor_state()
    )
    critic_states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * n), create_dummy_critic_state()
    )

    dataset = expand_dataset(create_dummy_dataset(), repeat=10)

    batched_pre_train = jax.vmap(
        lambda rng, actor, critic: pre_train(
            rng,
            actor,
            critic,
            dataset,
            actor_lr=1e-3,
            actor_epochs=2,
            actor_batch_size=2,
            critic_lr=1e-3,
            critic_epochs=2,
            critic_batch_size=2,
        ),
        in_axes=(0, 0, 0),
    )

    trained_actors, trained_critics, metrics = batched_pre_train(
        rngs, actor_states, critic_states
    )

    assert "actor_loss" in metrics and "critic_loss" in metrics
    assert metrics["actor_loss"].shape[0] == n
    assert metrics["critic_loss"].shape[0] == n

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

    # def expert_policy(obs):
    #     return env_args.env.action_space(env_args.env_params).sample(rng)

    transitions = collect_experience_from_expert_policy(
        expert_policy, rng, mode, env_args, n_timesteps=n_steps
    )

    assert isinstance(transitions, Transition)
    assert transitions.obs.shape[0] == n_steps
    assert transitions.action.shape[0] == n_steps
    assert transitions.reward.shape[0] == n_steps
    assert transitions.terminated.shape[0] == n_steps
    assert transitions.truncated.shape[0] == n_steps
