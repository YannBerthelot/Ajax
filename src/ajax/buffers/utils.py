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
) -> fbx.flat_buffer.TrajectoryBufferState:
    # Get the state and action shapes for the environment
    observation_shape, action_shape = get_state_action_shapes(
        env_args.env,
    )

    # Initialize the action as a single action for a single timestep (not batched)
    action = jnp.zeros(
        (action_shape[0],),  # Shape for a single action (e.g., [action_size])
        dtype=jnp.float32 if env_args.continuous else jnp.int32,
    )

    # Initialize the observation for a single timestep (shape: [observation_size])
    obsv = jnp.zeros((observation_shape[0],))

    # Initialize the reward and done flag for a single timestep
    reward = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single reward
    done = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single done flag

    # Initialize the buffer state with a single transition
    buffer_state = buffer.init(
        {
            "obs": obsv,  # Single observation (shape: [observation_size])
            "action": action,  # Single action (shape: [action_size])
            "reward": reward,  # Single reward (shape: [1])
            "terminated": done,  # Single done flag (shape: [1])
            "truncated": done,  # Single done flag (shape: [1])
        },
    )

    return buffer_state


def assert_shape(x, expected_shape, name="tensor"):
    assert (
        x.shape == expected_shape
    ), f"{name} has shape {x.shape}, expected {expected_shape}"


@partial(
    jax.jit,
    static_argnames=["buffer"],
)
def get_batch_from_buffer(
    buffer: BufferType,
    buffer_state,
    key,
    alternate_buffer: Optional[BufferType] = None,
    mix_fraction=0.5,
):
    normal_key, alt_key = jax.random.split(key, 2)
    batch = buffer.sample(buffer_state, normal_key).experience

    # keys = jax.random.split(key, n_batches)
    # batches = jax.vmap(buffer.sample, in_axes=(None, 0))(buffer_state, keys).experience
    # batch = jax.tree.map(lambda x: x.reshape((x.shape[0] * x.shape[1], -1)), batches)

    obs = batch.first["obs"]
    act = batch.first["action"]
    rew = batch.first["reward"]
    next_obs = batch.second["obs"]
    terminated = batch.first["terminated"]
    truncated = batch.first["truncated"]

    # if alternate_buffer is not None:
    #     len_normal = len(obs)
    #     alt_batch = alternate_buffer.sample(buffer_state, alt_key).experience
    #     alt_obs = alt_batch.first["obs"]
    #     len_alt = len(alt_obs)
    #     import pdb

    #     alt_act = alt_batch.first["action"]
    #     alt_rew = alt_batch.first["reward"]
    #     alt_next_obs = alt_batch.second["obs"]
    #     alt_terminated = alt_batch.first["terminated"]
    #     alt_truncated = alt_batch.first["truncated"]

    #     obs = jnp.concatenate(
    #         [obs, alt_obs],
    #         axis=0,
    #     )
    #     act = jnp.concatenate(
    #         [act, alt_act],
    #         axis=0,
    #     )
    #     rew = jnp.concatenate(
    #         [rew, alt_rew],
    #         axis=0,
    #     )
    #     next_obs = jnp.concatenate(
    #         [next_obs, alt_next_obs],
    #         axis=0,
    #     )
    #     terminated = jnp.concatenate(
    #         [terminated, alt_terminated],
    #         axis=0,
    #     )
    #     truncated = jnp.concatenate(
    #         [truncated, alt_truncated],
    #         axis=0,
    #     )

    return obs, terminated, truncated, next_obs, rew, act
