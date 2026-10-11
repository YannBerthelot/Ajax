from typing import NamedTuple, Optional

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
    sequence_length: Optional[int] = None,
):
    """Replay buffer factory.

    sequence_length=None (default) returns the usual flat buffer of
    independent transitions. When set (recurrent agents), returns a
    trajectory buffer whose ``sample`` yields contiguous per-env sequences
    of that length, shaped (batch_size, sequence_length, ...). The ``add``
    signature is wrapped to accept the same single-step (n_envs, ...)
    transitions the collector already writes.
    """
    if sequence_length is None:
        return fbx.make_flat_buffer(
            max_length=buffer_size,
            sample_batch_size=batch_size,
            min_length=batch_size,
            add_batch_size=n_envs,
        )

    from flashbax.utils import add_dim_to_args

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=n_envs,
        sample_batch_size=batch_size,
        sample_sequence_length=sequence_length,
        period=1,
        min_length_time_axis=sequence_length + 1,
        max_length_time_axis=buffer_size // n_envs,
    )
    # The collector adds one transition of shape (n_envs, ...) per call;
    # the trajectory buffer expects (n_envs, time, ...): insert time axis.
    add_fn = add_dim_to_args(
        buffer.add, axis=1, starting_arg_index=1, ending_arg_index=2
    )
    return buffer.replace(add=add_fn)  # type: ignore[attr-defined]


def init_buffer(
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    env_args: EnvironmentConfig,
    add_train_frac: bool = False,
    action_dim_override: Optional[int] = None,
    expert_state_aug_dim: int = 0,
    include_expert_fields: bool = False,
    actor_carry_dim: int = 0,
    include_raw_next_obs: bool = False,
    include_raw_obs: bool = False,
    include_is_expert: bool = False,
) -> fbx.flat_buffer.TrajectoryBufferState:
    """
    Initialize the flashbax buffer state with correctly shaped dummy transitions.

    Observations are flat (``build_env_from_id`` flattens gymnax ones).

    add_train_frac: whether to append the train_time_fraction scalar as the
      last observation dimension.
    expert_state_aug_dim: extra trailing obs dims reserved for the
      flattened expert internal state when running with
      `augment_obs_with_expert_state=True`. 0 disables.
    include_raw_obs, include_raw_next_obs: also keep the environment's own
      observation, ``raw_obs``, and next observation (the final one where
      an episode ended), ``raw_next_obs``, for an expert to read.
    include_is_expert: also keep whether an expert acted, ``is_expert``.

    A field is stored only for a reader: a row is about 40% raw
    observation, which a run without an expert never reads.
    """
    observation_shape, action_shape = get_state_action_shapes(env_args.env)
    if action_dim_override is not None:
        action_shape = (action_dim_override,)
    raw_observation_shape = observation_shape
    extra = add_train_frac + expert_state_aug_dim
    if extra:
        observation_shape = (*observation_shape[:-1], observation_shape[-1] + extra)

    action = jnp.zeros(
        (action_shape[0],),
        dtype=jnp.float32 if env_args.continuous else jnp.int32,
    )
    obsv = jnp.zeros((observation_shape[0],))
    raw_obsv = jnp.zeros((raw_observation_shape[0],))
    reward = jnp.zeros((1,), dtype=jnp.float32)
    done = jnp.zeros((1,), dtype=jnp.float32)

    schema = {
        "obs": obsv,
        "action": action,
        "reward": reward,
        "terminated": done,
        "truncated": done,
        "next_obs": obsv,
    }
    if include_raw_obs:
        schema["raw_obs"] = raw_obsv
    if include_is_expert:
        schema["is_expert"] = jnp.zeros((1,), dtype=jnp.float32)
    if include_expert_fields:
        schema["a_expert"] = schema["next_a_expert"] = action
    if include_raw_next_obs:
        schema["raw_next_obs"] = raw_obsv
    if actor_carry_dim > 0:
        # R2D2-style stored-state replay: the actor's flattened carry at
        # each step (see ajax.networks.memory.flatten_carry).
        schema["actor_carry"] = jnp.zeros((actor_carry_dim,), dtype=jnp.float32)
    return buffer.init(schema)


class ReplayBatch(NamedTuple):
    """One replay sample: its rows' fields, the optional ones None where
    the buffer does not store them (:func:`init_buffer`)."""

    obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_obs: jax.Array
    reward: jax.Array
    action: jax.Array
    raw_obs: Optional[jax.Array] = None
    is_expert: Optional[jax.Array] = None
    # The expert's action at s and s' (include_expert_fields).
    a_expert: Optional[jax.Array] = None
    next_a_expert: Optional[jax.Array] = None
    # The env's own next observation (include_raw_next_obs).
    raw_next_obs: Optional[jax.Array] = None


@partial(jax.jit, static_argnames=["buffer"])
def get_batch_from_buffer(buffer: BufferType, buffer_state, key) -> ReplayBatch:
    batch = buffer.sample(buffer_state, key).experience
    first = batch.first
    return ReplayBatch(
        obs=first["obs"],
        # Not the next row's obs: after a time limit that is the reset one,
        # and the target bootstraps on the final one (interaction.bootstrap_obs).
        next_obs=first["next_obs"],
        terminated=first["terminated"],
        truncated=first["truncated"],
        reward=first["reward"],
        action=first["action"],
        raw_obs=first.get("raw_obs"),
        is_expert=first.get("is_expert"),
        a_expert=first.get("a_expert"),
        next_a_expert=first.get("next_a_expert"),
        raw_next_obs=first.get("raw_next_obs"),
    )


@partial(jax.jit, static_argnames=["buffer"])
def get_sequence_batch_from_buffer(
    buffer: BufferType,
    buffer_state,
    key,
):
    """Sample contiguous sequences from a trajectory buffer (see
    ``get_buffer(sequence_length=...)``) and return them time-major.

    Returns a dict of arrays shaped (sequence_length, batch_size, ...),
    ready for the recurrent (scan-based) network path.
    """
    batch = buffer.sample(buffer_state, key).experience  # (B, L, ...)
    return jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
