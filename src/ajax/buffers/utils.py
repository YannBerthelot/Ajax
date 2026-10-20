from typing import Any, NamedTuple, Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from jax.tree_util import Partial as partial

from ajax.environments.utils import get_state_action_shapes
from ajax.state import EnvironmentConfig
from ajax.types import BufferType, NormalizationInfo
from ajax.utils import online_normalize


def get_buffer(
    buffer_size: int,
    batch_size: int,
    n_envs: int = 1,
    sequence_length: Optional[int] = None,
):
    """Replay buffer factory: a trajectory buffer of ``buffer_size // n_envs``
    rows per env, whose ``add`` takes the single-step (n_envs, ...)
    transitions the collector writes.

    sequence_length=None (default) samples rows alone, uniformly over every
    stored one: a row holds its next observation, so it needs no successor
    (flashbax's flat buffer samples a row with its successor, which leaves
    each env's newest row out). When set (recurrent agents), ``sample``
    yields contiguous per-env sequences of that length, shaped
    (batch_size, sequence_length, ...).
    """
    from flashbax.utils import add_dim_to_args

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=n_envs,
        sample_batch_size=batch_size,
        sample_sequence_length=sequence_length or 1,
        period=1,
        # Rows per env before can_sample: a sample's own, plus one.
        min_length_time_axis=(sequence_length or batch_size // n_envs) + 1,
        max_length_time_axis=buffer_size // n_envs,
    )
    # The collector adds one transition of shape (n_envs, ...) per call;
    # the trajectory buffer expects (n_envs, time, ...): insert time axis.
    add_fn = add_dim_to_args(
        buffer.add, axis=1, starting_arg_index=1, ending_arg_index=2
    )
    if sequence_length is not None:
        return buffer.replace(add=add_fn)  # type: ignore[attr-defined]

    def sample_rows(state: Any, key: jax.Array) -> TrajectoryBufferSample:
        experience = buffer.sample(state, key).experience  # (B, 1, ...)
        rows = jax.tree.map(lambda x: x[:, 0], experience)
        return TrajectoryBufferSample(experience=rows)

    return buffer.replace(add=add_fn, sample=sample_rows)  # type: ignore[attr-defined]


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


def normalized_rewards(reward: jax.Array, stats: NormalizationInfo) -> jax.Array:
    """The env's own ``reward`` divided by a reward normaliser's statistics,
    which it leaves as they are; while they are empty (no step yet), the
    reward as it is."""
    normed = online_normalize(
        reward, stats.count, stats.mean, stats.mean_2, train=False, shift=False
    )[0]
    return jnp.where(jnp.sum(stats.count) > 0, normed, reward)


@partial(jax.jit, static_argnames=["buffer"])
def get_batch_from_buffer(
    buffer: BufferType,
    buffer_state,
    key,
    reward_stats: Optional[NormalizationInfo] = None,
) -> ReplayBatch:
    """A replay batch. The replay keeps the env's own rewards; with
    ``reward_stats``, a reward normaliser's current statistics
    (:func:`ajax.environments.interaction.reward_statistics`), they come
    divided by them, as SB3's replay buffer does under VecNormalize."""
    first = buffer.sample(buffer_state, key).experience
    reward = first["reward"]
    if reward_stats is not None:
        reward = normalized_rewards(reward, reward_stats)
    return ReplayBatch(
        obs=first["obs"],
        # Not the next row's obs: after a time limit that is the reset one,
        # and the target bootstraps on the final one (interaction.bootstrap_obs).
        next_obs=first["next_obs"],
        terminated=first["terminated"],
        truncated=first["truncated"],
        reward=reward,
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
