from typing import Any, Callable, NamedTuple, Optional

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBufferSample,
    TrajectoryBufferState,
)
from flashbax.buffers.trajectory_buffer import init as init_ring
from jax.tree_util import Partial as partial

from ajax.controller import feature_width
from ajax.environments.utils import check_env_is_gymnax, get_state_action_shapes
from ajax.random_keys import derive_key
from ajax.state import EnvironmentConfig
from ajax.types import BufferType, NormalizationInfo
from ajax.utils import REWARD_CLIP, online_normalize


@chex.dataclass(frozen=True)
class TwoBufferState(TrajectoryBufferState):
    """A replay's ring, which collection writes and evicts from, beside a ring
    of rows that stay for the whole run (RLPD's offline buffer, DDPGfD's
    demonstrations). The readers of a replay see the first ring; sampling
    reads both (:func:`sample_two_buffers`).

    Attributes:
        retained: The ring of kept rows, with the first ring's fields.
        share: The share of each batch drawn from it, exact in every batch;
            None: each row of a batch is drawn over the rows of both rings
            alike, so the share falls as the first ring fills.
    """

    retained: Any = None
    share: Optional[float] = None


def written_rows(buffer_state: Any) -> jax.Array:
    """The rows per env a ring holds: those written, up to its length."""
    held = jax.tree.leaves(buffer_state.experience)[0].shape[1]
    return jnp.where(buffer_state.is_full, held, buffer_state.current_index)


def empty_ring_like(buffer_state: Any, rows_per_env: int) -> Any:
    """An empty ring of rows_per_env rows, with buffer_state's fields."""

    def empty(values: jax.Array) -> jax.Array:
        shape = (values.shape[0], rows_per_env, *values.shape[2:])
        return jnp.zeros(shape, values.dtype)

    return buffer_state.replace(
        experience=jax.tree.map(empty, buffer_state.experience),
        current_index=jnp.zeros_like(buffer_state.current_index),
        is_full=jnp.zeros_like(buffer_state.is_full),
    )


def with_retained_rows(
    buffer_state: Any, retained: Any, share: Optional[float]
) -> TwoBufferState:
    """buffer_state, its next batches also drawing from the ring retained."""
    if isinstance(buffer_state, TwoBufferState):
        raise ValueError("A replay keeps one ring of retained rows.")
    return TwoBufferState(
        experience=buffer_state.experience,
        current_index=buffer_state.current_index,
        is_full=buffer_state.is_full,
        retained=retained,
        share=share,
    )


def sample_two_buffers(
    sample: Callable, state: Any, key: jax.Array
) -> TrajectoryBufferSample:
    """sample(state, key) on a plain ring. On a TwoBufferState, one draw from
    each ring, kept row by row: the first rows of the batch from the retained
    ring where a share is fixed (RLPD's N/2 from each), else each row with the
    retained ring's share of the rows held."""
    online = sample(state, key).experience
    if not isinstance(state, TwoBufferState):
        return TrajectoryBufferSample(experience=online)
    retained_key = derive_key(key, "retained_rows")
    retained = sample(state.retained, retained_key).experience
    size = jax.tree.leaves(online)[0].shape[0]
    if state.share is None:
        kept, written = written_rows(state.retained), written_rows(state)
        share_key = derive_key(key, "retained_share")
        from_retained = jax.random.uniform(share_key, (size,)) < kept / (kept + written)
    else:
        from_retained = jnp.arange(size) < jnp.round(state.share * size)

    def choose(new: jax.Array, old: jax.Array) -> jax.Array:
        mask = from_retained.reshape((size,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, old, new)

    return TrajectoryBufferSample(experience=jax.tree.map(choose, online, retained))


#: A row's fields about the state its target looks ahead to, which an n-step
#: row takes from the step that state follows.
_FIELDS_AHEAD = (
    "terminated",
    "truncated",
    "next_obs",
    "next_a_expert",
    "raw_next_obs",
    "next_action",
)


def n_step_rows(window: Any, gamma: float) -> dict:
    """One row per window of consecutive steps (batch, n, ...): the first step's
    state and action, the discounted sum of the rewards from there, and the
    state n steps ahead or at the episode's end, whichever comes first, as the
    row's next state and flags. A seam ends a window as an episode's end does.
    ``bootstrap_discount`` is gamma to the power of the steps summed, which the
    target's bootstrap takes in place of gamma."""
    ended = (window["terminated"] > 0) | (window["truncated"] > 0)
    if "seam" in window:
        ended = ended | (window["seam"] > 0)
    # The rows the sum reaches: up to the first end, included.
    before_end = jnp.logical_not(ended).astype(jnp.float32)
    reached = jnp.cumprod(
        jnp.concatenate([jnp.ones_like(before_end[:, :1]), before_end[:, :-1]], axis=1),
        axis=1,
    )
    powers = gamma ** jnp.arange(reached.shape[1], dtype=jnp.float32)
    steps = jnp.sum(reached, axis=1)
    last = steps[:, 0].astype(jnp.int32) - 1
    rows = jax.tree.map(lambda field: field[:, 0], window)
    for name in _FIELDS_AHEAD:
        if name in window:
            rows[name] = window[name][jnp.arange(last.shape[0]), last]
    rows["reward"] = jnp.sum(reached * powers[None, :, None] * window["reward"], axis=1)
    rows["bootstrap_discount"] = gamma**steps
    return rows


def get_buffer(
    buffer_size: int,
    batch_size: int,
    n_envs: int = 1,
    sequence_length: Optional[int] = None,
    n_step: int = 1,
    gamma: float = 1.0,
):
    """Replay buffer factory: a trajectory buffer of ``buffer_size // n_envs``
    rows per env, whose ``add`` takes the single-step (n_envs, ...)
    transitions the collector writes. A run's state may hold fewer, the rows
    it writes (init_buffer's rows_per_env): add and sample read the ring's
    length from the state.

    sequence_length=None (default) samples rows alone, uniformly over every
    stored one: a row holds its next observation, so it needs no successor
    (flashbax's flat buffer samples a row with its successor, which leaves
    each env's newest row out). When set (recurrent agents), ``sample``
    yields contiguous per-env sequences of that length, shaped
    (batch_size, sequence_length, ...). Rows alone, a TwoBufferState also
    draws from its retained ring (:func:`sample_two_buffers`); no window
    reads one, as a run with memory refuses the extensions that write it.

    n_step above 1 samples rows as well, each one n_step steps long
    (:func:`n_step_rows`, at gamma), both rings' alike: a window never crosses
    a ring's write position, so a ring needs n_step rows per env first.
    """
    from flashbax.utils import add_dim_to_args

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=n_envs,
        sample_batch_size=batch_size,
        sample_sequence_length=sequence_length or n_step,
        period=1,
        min_length_time_axis=max(
            min_replay_rows(batch_size, n_envs, sequence_length), n_step
        ),
        max_length_time_axis=buffer_size // n_envs,
    )
    # The collector adds one transition of shape (n_envs, ...) per call;
    # the trajectory buffer expects (n_envs, time, ...): insert time axis.
    add_with_time_axis = add_dim_to_args(
        buffer.add, axis=1, starting_arg_index=1, ending_arg_index=2
    )

    def add_fn(state: Any, row: dict) -> Any:
        # The fields the replay keeps as bool (stored_observation_dtype) take
        # the row's 0/1 values as bool. Only those: any other dtype mismatch
        # still fails flashbax's check.
        def kept(value: jax.Array, stored: jax.Array) -> jax.Array:
            if stored.dtype == jnp.bool_:
                return value.astype(jnp.bool_)
            return value

        return add_with_time_axis(state, jax.tree.map(kept, row, state.experience))

    if sequence_length is not None:

        def sample_windows(state: Any, key: jax.Array) -> TrajectoryBufferSample:
            experience = buffer.sample(state, key).experience
            return TrajectoryBufferSample(experience=as_read(experience))

        return buffer.replace(add=add_fn, sample=sample_windows)  # type: ignore[attr-defined]

    def sample_rows(state: Any, key: jax.Array) -> TrajectoryBufferSample:
        # (B, n_step, ...)
        experience = sample_two_buffers(buffer.sample, state, key).experience
        if n_step > 1:
            rows = n_step_rows(experience, gamma)
        else:
            rows = jax.tree.map(lambda x: x[:, 0], experience)
        return TrajectoryBufferSample(experience=as_read(rows))

    return buffer.replace(add=add_fn, sample=sample_rows)  # type: ignore[attr-defined]


def stored_observation_dtype(env_args: EnvironmentConfig) -> Any:
    """The dtype a replay keeps its observations in: bool where the env's
    observation space is boolean (MinAtar's images) and the replay keeps the
    env's own observations, a quarter of float32's memory for the same
    values, 0.0 and 1.0 (as_read reads them back); float32 otherwise, for
    observations normalised or extended with feature columns and every other
    space."""
    if env_args.store_normalized_obs or feature_width(env_args):
        return jnp.float32
    env = env_args.env
    if not check_env_is_gymnax(env):
        return jnp.float32
    # Read at trace time: a space's bounds may be arrays (Pendulum's), whose
    # operations would otherwise join the training program, unused. Its
    # dtype does not depend on the params: the env's defaults, as
    # get_state_action_shapes reads its shape (env_args may hold none).
    with jax.ensure_compile_time_eval():
        space = env.observation_space(env.default_params)
    if jnp.dtype(space.dtype) == jnp.bool_:
        return jnp.bool_
    return jnp.float32


def as_read(experience: Any) -> Any:
    """Replayed fields as the agent reads them: observations kept as bool
    (stored_observation_dtype) in float32, 0.0 and 1.0."""

    def read(values: jax.Array) -> jax.Array:
        if values.dtype == jnp.bool_:
            return values.astype(jnp.float32)
        return values

    return jax.tree.map(read, experience)


def min_replay_rows(
    batch_size: int, n_envs: int, sequence_length: Optional[int] = None
) -> int:
    """The rows per env a replay (get_buffer) holds before it can sample: a
    sample's own, plus one."""
    return (sequence_length or batch_size // n_envs) + 1


def count_replay_rows(buffer_state: Any) -> tuple[int, int]:
    """The rows per env a replay's ring holds, and those written to it (the
    most over the seeds): the rows before its write position, plus the
    ring's once it has filled. The ring axis follows the env axis, after
    the seed axes of current_index."""
    index = jnp.asarray(buffer_state.current_index)
    values = jax.tree.leaves(buffer_state.experience)[0]
    held = int(jnp.shape(values)[jnp.ndim(index) + 1])
    written = index + jnp.where(buffer_state.is_full, held, 0)
    return held, int(jnp.max(written))


def grow_replay(buffer_state: Any, rows: int) -> Any:
    """A replay that has overwritten none of its rows, its ring lengthened to
    rows per env: row a stays at slot a, so it is the ring a replay of rows
    per env holds after the same rows; a ring filled exactly writes its next
    row past its old end. Host-side, before a resumed run."""
    index = jnp.asarray(buffer_state.current_index)
    held, _ = count_replay_rows(buffer_state)
    axis = jnp.ndim(index) + 1

    def pad(values: jax.Array) -> jax.Array:
        widths = [(0, 0)] * jnp.ndim(values)
        widths[axis] = (0, rows - held)
        return jnp.pad(values, widths)

    return buffer_state.replace(
        experience=jax.tree.map(pad, buffer_state.experience),
        current_index=jnp.where(buffer_state.is_full, held, index).astype(index.dtype),
        is_full=jnp.zeros_like(buffer_state.is_full),
    )


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
    include_seam: bool = False,
    rows_per_env: Optional[int] = None,
) -> fbx.flat_buffer.TrajectoryBufferState:
    """
    Initialize the flashbax buffer state with correctly shaped dummy transitions.

    Observations are flat (``build_env_from_id`` flattens gymnax ones).

    The observations are as wide as the agent's, their features
    (``env_args.observers``) included, in stored_observation_dtype: bool for
    a boolean space, which sampling reads back as float32.
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
    include_seam: also keep ``seam``, set on a row the next one written does
      not follow in time: the last row a prefill or a dataset writes, which
      the first step collected comes after. For a replay reading a row's
      successors (:func:`n_step_rows`).
    rows_per_env: the ring's length, the rows a run writes to it
      (ajax.buffers.ring); None: the buffer's capacity.

    A field is stored only for a reader: a row is about 40% raw
    observation, which a run without an expert never reads.
    """
    observation_shape, action_shape = get_state_action_shapes(env_args.env)
    raw_observation_shape = observation_shape
    # The observation features (env_args.observers) are stored with it.
    extra = feature_width(env_args)
    if extra:
        observation_shape = (*observation_shape[:-1], observation_shape[-1] + extra)

    action_dtype = jnp.float32 if env_args.continuous else jnp.int32
    # The expert acts in the env's actions (a_expert); so does the policy,
    # except under gain mode, where it chooses the expert's gains.
    env_action = jnp.zeros((action_shape[0],), dtype=action_dtype)
    action = env_action
    if env_args.policy_action_dim is not None:
        action = jnp.zeros((env_args.policy_action_dim,), dtype=action_dtype)
    obsv = jnp.zeros((observation_shape[0],), stored_observation_dtype(env_args))
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
        schema["a_expert"] = schema["next_a_expert"] = env_action
    if include_raw_next_obs:
        schema["raw_next_obs"] = raw_obsv
    if include_return_to_go:
        schema["return_to_go"] = jnp.zeros((1,), dtype=jnp.float32)
    if include_next_action:
        schema["next_action"] = jnp.zeros(action.shape, dtype=jnp.float32)
    if include_policy_action:
        schema["policy_action"] = action
    if include_seam:
        schema["seam"] = jnp.zeros((1,), dtype=jnp.float32)
    if actor_carry_dim > 0:
        # R2D2-style stored-state replay: the actor's flattened carry at
        # each step (see ajax.networks.memory.flatten_carry).
        schema["actor_carry"] = jnp.zeros((actor_carry_dim,), dtype=jnp.float32)
    if rows_per_env is None:
        return buffer.init(schema)
    return init_ring(schema, env_args.n_envs, rows_per_env)


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
    # An n-step row's gamma to the power of its steps (n_step_rows).
    bootstrap_discount: Optional[jax.Array] = None


def mean_of_written_rows(buffer_state: Any, field: str) -> jax.Array:
    """The mean of a replay field over the rows written so far, every env's
    (all of them once the replay is full): for a 0/1 field (is_expert), the
    share of the rows where it is set. 0 before any row is written."""
    values = buffer_state.experience[field]  # (n_envs, capacity, ...)
    capacity = values.shape[1]
    written = jnp.logical_or(
        jnp.arange(capacity) < buffer_state.current_index, buffer_state.is_full
    )
    written = written.reshape((1, capacity) + (1,) * (values.ndim - 2))
    written = jnp.broadcast_to(written, values.shape)
    total = jnp.sum(jnp.where(written, values, 0.0))
    return total / jnp.maximum(jnp.sum(written), 1)


def normalized_rewards(reward: jax.Array, stats: NormalizationInfo) -> jax.Array:
    """The env's own ``reward`` divided by a reward normaliser's statistics,
    which it leaves as they are, and clipped as collection clips it; while
    they are empty (no step yet), the reward as it is."""
    normed = online_normalize(
        reward, stats.count, stats.mean, stats.mean_2, train=False, shift=False
    )[0]
    normed = jnp.clip(normed, -REWARD_CLIP, REWARD_CLIP)
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
    return as_replay_batch(buffer.sample(buffer_state, key).experience, reward_stats)


def as_replay_batch(
    rows: dict, reward_stats: Optional[NormalizationInfo] = None
) -> ReplayBatch:
    """rows, a replay's fields as an agent reads them (as_read), as a
    ReplayBatch: a sample's, or the rows an agent picks itself (DAgger's
    passes over the whole replay); their rewards as get_batch_from_buffer's."""
    reward = rows["reward"]
    if reward_stats is not None:
        reward = normalized_rewards(reward, reward_stats)
    return ReplayBatch(
        obs=rows["obs"],
        # Not the next row's obs: after a time limit that is the reset one,
        # and the target bootstraps on the final one (interaction.bootstrap_obs).
        next_obs=rows["next_obs"],
        terminated=rows["terminated"],
        truncated=rows["truncated"],
        reward=reward,
        action=rows["action"],
        raw_obs=rows.get("raw_obs"),
        is_expert=rows.get("is_expert"),
        a_expert=rows.get("a_expert"),
        next_a_expert=rows.get("next_a_expert"),
        raw_next_obs=rows.get("raw_next_obs"),
        return_to_go=rows.get("return_to_go"),
        next_action=rows.get("next_action"),
        policy_action=rows.get("policy_action"),
        bootstrap_discount=rows.get("bootstrap_discount"),
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
