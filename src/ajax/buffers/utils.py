from typing import Any, NamedTuple, Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from jax.tree_util import Partial as partial

from ajax.controller import feature_width
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
    include_expert_fields: bool = False,
    actor_carry_dim: int = 0,
    include_raw_next_obs: bool = False,
    include_raw_obs: bool = False,
    include_is_expert: bool = False,
    include_return_to_go: bool = False,
    include_next_action: bool = False,
    include_policy_action: bool = False,
) -> fbx.flat_buffer.TrajectoryBufferState:
    """
    Initialize the flashbax buffer state with correctly shaped dummy transitions.

    Observations are flat (``build_env_from_id`` flattens gymnax ones).

    The observations are as wide as the agent's, their features
    (``env_args.observers``) included.
    include_raw_obs, include_raw_next_obs: also keep the environment's own
      observation, ``raw_obs``, and next observation (the final one where
      an episode ended), ``raw_next_obs``, for an expert to read.
    include_is_expert: also keep whether an expert acted, ``is_expert``.
    include_return_to_go: also keep a dataset row's return to its episode's
      end, ``return_to_go`` (a collected row's is -inf: unknown).
    include_next_action: also keep a dataset row's next action, its
      episode's next row's, ``next_action`` (NaN where unknown).
    include_policy_action: also keep the action the policy chose,
      ``policy_action``, where it differs from the executed one (a residual
      policy's, for an actor fitting its own actions: IQL's).

    A field is stored only for a reader: a row is about 40% raw
    observation, which a run without an expert never reads.
    """
    observation_shape, action_shape = get_state_action_shapes(env_args.env)
    if env_args.policy_action_dim is not None:  # gain mode's gains
        action_shape = (env_args.policy_action_dim,)
    raw_observation_shape = observation_shape
    # The observation features (env_args.observers) are stored with it.
    extra = feature_width(env_args)
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
    if include_return_to_go:
        schema["return_to_go"] = jnp.zeros((1,), dtype=jnp.float32)
    if include_next_action:
        schema["next_action"] = jnp.zeros(action.shape, dtype=jnp.float32)
    if include_policy_action:
        schema["policy_action"] = action
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
    # A dataset row's return to its episode's end (include_return_to_go).
    return_to_go: Optional[jax.Array] = None
    # A dataset row's next action (include_next_action).
    next_action: Optional[jax.Array] = None
    # The action the policy chose (include_policy_action).
    policy_action: Optional[jax.Array] = None


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
        return_to_go=first.get("return_to_go"),
        next_action=first.get("next_action"),
        policy_action=first.get("policy_action"),
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
