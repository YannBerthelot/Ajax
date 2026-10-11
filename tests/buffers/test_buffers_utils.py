from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.buffers.utils import get_batch_from_buffer, get_buffer, init_buffer
from ajax.environments.create import build_env_from_id
from ajax.environments.utils import get_state_action_shapes
from ajax.state import EnvironmentConfig


@pytest.fixture(scope="module")
def setup():
    """A CartPole replay of 100 rows over 2 envs, batches of 10."""
    buffer_size, batch_size, n_envs = 100, 10, 2
    buffer = get_buffer(buffer_size, batch_size, n_envs)
    env, env_params = build_env_from_id("CartPole-v1")
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, continuous=False, n_envs=n_envs
    )
    obs_shape, action_shape = get_state_action_shapes(env)
    return SimpleNamespace(
        buffer=buffer,
        state=init_buffer(buffer, env_args),
        buffer_size=buffer_size,
        batch_size=batch_size,
        n_envs=n_envs,
        obs_shape=obs_shape,
        action_shape=action_shape,
    )


def test_get_buffer(setup):
    buffer, n_envs = setup.buffer, setup.n_envs
    assert buffer.init.keywords["add_batch_size"] == n_envs
    assert buffer.init.keywords["max_length_time_axis"] == setup.buffer_size // n_envs
    min_length = buffer.can_sample.keywords["min_length_time_axis"]
    assert min_length == setup.batch_size // n_envs + 1


def test_init_buffer(setup):
    rows = (setup.n_envs, setup.buffer_size // setup.n_envs)
    experience = setup.state.experience
    expected = {
        "obs": setup.obs_shape,
        "next_obs": setup.obs_shape,
        "action": setup.action_shape,
        "reward": (1,),
        "terminated": (1,),
        "truncated": (1,),
    }
    assert set(expected) <= set(experience)
    for key, trailing in expected.items():
        assert experience[key].shape == (*rows, *trailing), key


def test_get_batch_from_buffer(setup):
    obs, terminated, truncated, next_obs, reward, action, raw_obs, is_expert = (
        get_batch_from_buffer(setup.buffer, setup.state, jax.random.PRNGKey(0))
    )
    batch = setup.batch_size
    for sample, trailing in (
        (obs, setup.obs_shape),
        (next_obs, setup.obs_shape),
        (raw_obs, setup.obs_shape),
        (action, setup.action_shape),
        (terminated, (1,)),
        (truncated, (1,)),
        (reward, (1,)),
        (is_expert, (1,)),
    ):
        assert sample.shape == (batch, *trailing)


@pytest.mark.parametrize(
    "env_id, action_dtype", [("Pendulum-v1", jnp.float32), ("CartPole-v1", jnp.int32)]
)
def test_init_buffer_schema_follows_its_options(env_id, action_dtype):
    """A row holds the transition and its raw observation; each option adds
    its fields or widens the stored observations (never the raw ones)."""
    env, params = make_gymnax_env(env_id)
    env_args = EnvironmentConfig(
        env=env, env_params=params, n_envs=1, continuous=env_id == "Pendulum-v1"
    )
    obs_dim = env.observation_space(params).shape[0]
    buffer = get_buffer(buffer_size=8, batch_size=2)

    def row(**options):
        rows = init_buffer(buffer, env_args, **options).experience
        return {k: (v.shape[2:], v.dtype) for k, v in rows.items()}

    plain = row()
    assert set(plain) == {
        "obs",
        "action",
        "reward",
        "terminated",
        "truncated",
        "next_obs",
        "raw_obs",
        "is_expert",
    }
    assert plain["obs"] == plain["next_obs"] == plain["raw_obs"]
    assert plain["obs"] == ((obs_dim,), jnp.float32)
    assert plain["action"] == ((1,), action_dtype)
    full = row(
        add_train_frac=True,
        expert_state_aug_dim=2,
        include_expert_fields=True,
        actor_carry_dim=5,
        include_raw_next_obs=True,
    )
    added = {"a_expert", "next_a_expert", "raw_next_obs", "actor_carry"}
    assert set(full) == set(plain) | added
    assert full["obs"] == full["next_obs"] == ((obs_dim + 1 + 2,), jnp.float32)
    assert full["raw_obs"] == full["raw_next_obs"] == plain["raw_obs"]
    assert full["a_expert"] == full["next_a_expert"] == plain["action"]
    assert full["actor_carry"] == ((5,), jnp.float32)
