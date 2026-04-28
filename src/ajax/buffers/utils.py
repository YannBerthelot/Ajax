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


def init_buffer(
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    env_args: EnvironmentConfig,
    max_timesteps: Optional[int] = None,
    add_train_frac: Optional[bool] = None,
    action_dim_override: Optional[int] = None,
    expert_state_aug_dim: int = 0,
) -> fbx.flat_buffer.TrajectoryBufferState:
    """
    Initialize the flashbax buffer state with correctly shaped dummy transitions.

    add_train_frac: whether to append the train_time_fraction scalar as the
      last observation dimension. If None (default), inferred from max_timesteps
      for backward compatibility. Prefer passing explicitly.
    expert_state_aug_dim: extra trailing obs dims reserved for the
      flattened expert internal state when running with
      `augment_obs_with_expert_state=True`. 0 disables.
    """
    observation_shape, action_shape = get_state_action_shapes(env_args.env)
    if action_dim_override is not None:
        action_shape = (action_dim_override,)
    raw_observation_shape = observation_shape

    # Determine whether to add the train_frac dimension
    if add_train_frac is None:
        add_train_frac = max_timesteps is not None

    if add_train_frac:
        observation_shape = (*observation_shape[:-1], observation_shape[-1] + 1)

    if expert_state_aug_dim > 0:
        observation_shape = (
            *observation_shape[:-1],
            observation_shape[-1] + expert_state_aug_dim,
        )

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
