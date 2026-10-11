from types import SimpleNamespace

import jax
import pytest

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
