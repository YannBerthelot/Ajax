from typing import Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.environments.utils import get_state_action_shapes
from ajax.state import EnvironmentConfig
from ajax.types import BufferType


def get_buffer(
    buffer_size: int,
    batch_size: int,
    n_envs: int = 1,
):
    return fbx.make_flat_buffer(
        max_length=buffer_size,
        sample_batch_size=batch_size,
        min_length=batch_size,
        add_batch_size=n_envs,
    )


def get_prioritized_buffer(
    buffer_size: int,
    batch_size: int,
    n_envs: int = 1,
    priority_exponent: float = 0.6,
):
    """Prioritized replay buffer (PER). Samples proportionally to priority^exponent.

    Returned buffer has the same add/init interface as the flat buffer, plus
    set_priorities(state, indices, priorities) and sample() that additionally
    returns .indices and .probabilities alongside .experience.
    """
    return fbx.make_prioritised_flat_buffer(
        max_length=buffer_size,
        sample_batch_size=batch_size,
        min_length=batch_size,
        add_batch_size=n_envs,
        priority_exponent=priority_exponent,
    )


def init_buffer(
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    env_args: EnvironmentConfig,
    max_timesteps: Optional[int] = None,
    add_train_frac: Optional[bool] = None,
) -> fbx.flat_buffer.TrajectoryBufferState:
    """
    Initialize the flashbax buffer state with correctly shaped dummy transitions.

    add_train_frac: whether to append the train_time_fraction scalar as the
      last observation dimension. If None (default), inferred from max_timesteps
      for backward compatibility. Prefer passing explicitly.
    """
    observation_shape, action_shape = get_state_action_shapes(env_args.env)
    raw_observation_shape = observation_shape

    # Determine whether to add the train_frac dimension
    if add_train_frac is None:
        add_train_frac = max_timesteps is not None

    if add_train_frac:
        observation_shape = (*observation_shape[:-1], observation_shape[-1] + 1)

    action = jnp.zeros(
        (action_shape[0],),
        dtype=jnp.float32 if env_args.continuous else jnp.int32,
    )
    obsv = jnp.zeros((observation_shape[0],))
    raw_obsv = jnp.zeros((raw_observation_shape[0],))
    reward = jnp.zeros((1,), dtype=jnp.float32)
    done = jnp.zeros((1,), dtype=jnp.float32)

    return buffer.init(
        {
            "obs": obsv,
            "action": action,
            "reward": reward,
            "terminated": done,
            "truncated": done,
            "raw_obs": raw_obsv,
            "is_expert": jnp.zeros((1,), dtype=jnp.float32),
        }
    )


def assert_shape(x, expected_shape, name="tensor"):
    assert (
        x.shape == expected_shape
    ), f"{name} has shape {x.shape}, expected {expected_shape}"


@partial(jax.jit, static_argnames=["buffer"])
def get_batch_from_buffer(
    buffer: BufferType,
    buffer_state,
    key,
):
    batch = buffer.sample(buffer_state, key).experience
    obs = batch.first["obs"]
    act = batch.first["action"]
    rew = batch.first["reward"]
    next_obs = batch.second["obs"]
    terminated = batch.first["terminated"]
    truncated = batch.first["truncated"]
    raw_observations = batch.first["raw_obs"]
    is_expert = batch.first["is_expert"]
    return obs, terminated, truncated, next_obs, rew, act, raw_observations, is_expert


@partial(jax.jit, static_argnames=["buffer"])
def get_batch_from_prioritized_buffer(
    buffer: BufferType,
    buffer_state,
    key,
):
    """Like get_batch_from_buffer but also returns (indices, probabilities) for PER.

    indices:       (batch,)  — buffer positions of sampled transitions
    probabilities: (batch,)  — actual sampling probabilities p(i)^alpha / sum
    """
    sample = buffer.sample(buffer_state, key)
    batch = sample.experience
    obs = batch.first["obs"]
    act = batch.first["action"]
    rew = batch.first["reward"]
    next_obs = batch.second["obs"]
    terminated = batch.first["terminated"]
    truncated = batch.first["truncated"]
    raw_observations = batch.first["raw_obs"]
    is_expert = batch.first["is_expert"]
    return (
        obs, terminated, truncated, next_obs, rew, act, raw_observations, is_expert,
        sample.indices, sample.probabilities,
    )
