"""Shared machinery for recurrent off-policy updates (R2D2-style).

Every off-policy agent with a Q-critic (SAC, ASAC, REDQ, TD3) trains its
memory the same way:

1. Sample contiguous per-env sequences of length
   ``burn_in + sequence_length + 1`` from a trajectory buffer.
2. Warm up ("burn in") the actor / online-critic / target-critic carries
   from zero over the first ``burn_in`` steps with the CURRENT params,
   under ``stop_gradient`` — the carries are constants inside the losses.
3. Train on the next ``sequence_length`` steps with BPTT; the final step
   is not read (each row stores the observation its target bootstraps on,
   ``interaction.bootstrap_obs``).

``sample_and_burnin_sequences`` implements steps 1-3 once for all agents;
the per-agent losses then consume the returned :class:`RecurrentCarries`
through :func:`actor_dist` and :func:`q_values`, which run a network on a
feedforward batch or, given carries, on a replayed sequence.

A critic's memory reads each row's observation and the action before it,
the one taken; the action a loss asks about joins after the memory, at
the critic's head, so a taken or a sampled action is a query on the same
carry and never feeds it (``Critic.query_dim``; the recurrent off-policy
critic of Ni et al. 2022, cited from memory, unverified).
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from ajax.buffers.utils import (
    get_batch_from_buffer,
    get_buffer,
    get_sequence_batch_from_buffer,
)
from ajax.environments.interaction import get_pi, get_pi_sequence
from ajax.networks.memory import (
    MemoryConfig,
    flat_carry_dim,
    unflatten_carry,
    zeros_carry_like,
)
from ajax.networks.networks import (
    action_value_input,
    predict_value,
    predict_value_sequence,
)
from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState, Transition
from ajax.types import BufferType


@partial(struct.dataclass, kw_only=True)
class RecurrentReplayConfig(BaseAgentConfig):
    """An off-policy agent's sequence-replay settings, read only when it has
    memory: replayed windows of ``burn_in + sequence_length + 1`` steps."""

    burn_in: int = 8
    sequence_length: int = 16
    # R2D2 stored-state replay: read actor carries back from the buffer
    # instead of burning them in from zero (Kapturowski et al. 2019).
    stored_state: bool = False


@struct.dataclass
class RecurrentCarries:
    """Per-update recurrent context for replayed sequences (R2D2-style).

    All carries are burned in from zero on the sequence prefix with the
    CURRENT params and treated as constants inside the losses (they are
    stop-gradiented); only the training segment backpropagates through
    time. The carries hold for the training segment's first row; its rows
    (``obs``, the ``actions`` taken, ``resets`` their episode starts) are
    the history a bootstrap runs over (:func:`_bootstrap_stream`).
    """

    resets: jax.Array  # (S, B)
    # Rows whose bootstrap observation starts an episode: the terminated
    # ones (interaction.bootstrap_obs).
    next_resets: jax.Array  # (S, B)
    actor_hidden: Any  # carry valid for obs[0] of the training segment
    critic_hidden: Any  # online-critic carry for (obs, action)[0]
    target_critic_hidden: Any  # target-critic carry for (obs, action)[0]
    obs: jax.Array  # (S, B, O)
    actions: jax.Array  # (S, B, A)
    # What the critics' memory reads with each row's observation: the
    # action taken before it (previous_actions).
    prev_actions: jax.Array  # (S, B, A)
    # Where each row sits in the bootstrap stream (_bootstrap_stream).
    positions: jax.Array  # (S, B)
    # Target-ACTOR carry for obs[0], burned with actor target_params. Only
    # populated for agents whose bootstrap action comes from a target
    # actor (TD3); None otherwise.
    target_actor_hidden: Any = None


def previous_actions(actions: jax.Array, resets: jax.Array) -> jax.Array:
    """Each step's previous action in its episode, ``(T, B, A)``: zero at
    an episode start (``resets``) and at the sequence's first step."""
    previous = jnp.concatenate([jnp.zeros_like(actions[:1]), actions[:-1]])
    return jnp.where(resets[..., None], 0.0, previous)


def stream_positions(cut: jax.Array) -> jax.Array:
    """Each row's position in its window's bootstrap stream, from the rows
    ``cut`` (S, B) where a time limit alone ended the episode: each such
    row's final observation follows it (:func:`_bootstrap_stream`)."""
    return jnp.arange(cut.shape[0])[:, None] + jnp.cumsum(cut, axis=0) - cut


def sample_and_burnin_sequences(
    agent_state: BaseAgentState,
    buffer: BufferType,
    sample_key: jax.Array,
    burn_in: int,
    burn_target_actor: bool = False,
    stored_state: bool = False,
) -> Tuple[Transition, RecurrentCarries]:
    """Sample sequences and burn in all carries; see the module docstring.

    ``stored_state=True`` enables R2D2's stored-state strategy for the
    ACTOR carries: instead of burning them in from zero, the carries the
    actor actually had at collection time are read back from the buffer
    (``actor_carry`` field, written by ``collect_experience`` with
    ``store_hidden=True``). Kapturowski et al. 2019 show stored state
    beats zero-init + burn-in at mitigating recurrent-state staleness.
    The critic carries are still burned in from zero: the critic never
    runs at collection time, so there is nothing stored for it.

    Returns a time-major :class:`Transition` over the training segment
    (leaves shaped ``(sequence_length, batch, ...)``) and the matching
    stop-gradiented :class:`RecurrentCarries`.
    """
    seq = get_sequence_batch_from_buffer(
        buffer,
        agent_state.collector_state.buffer_state,
        sample_key,
    )
    obs_seq = seq["obs"]
    act_seq = seq["action"]
    done_seq = jnp.logical_or(seq["terminated"], seq["truncated"]).squeeze(-1)  # (L, B)
    # Obs-aligned episode starts. The flag for the first step of the
    # sequence is unknown (the buffer slice starts mid-stream); zero
    # carries make step 0 a fresh start regardless.
    resets_seq = jnp.concatenate(
        [jnp.zeros_like(done_seq[:1]), done_seq[:-1]], axis=0
    ).astype(bool)
    batch_size = obs_seq.shape[1]
    prev_act_seq = previous_actions(act_seq, resets_seq)
    # The burn-ins only need the carries: the queried actions are unread.
    xs_seq = action_value_input(obs_seq, jnp.zeros_like(act_seq), prev_act_seq)

    actor_template = zeros_carry_like(
        agent_state.actor_state.hidden_state, batch_size, batch_axis=0
    )
    critic_zero = zeros_carry_like(
        agent_state.critic_state.hidden_state, batch_size, batch_axis=1
    )

    def burn_actor(params: Any) -> Any:
        if burn_in == 0:
            return actor_template
        return get_pi_sequence(
            agent_state.actor_state,
            params,
            obs_seq[:burn_in],
            resets_seq[:burn_in],
            actor_template,
        )[1]

    def burn_critic(params: Any) -> Any:
        if burn_in == 0:
            return critic_zero
        return predict_value_sequence(
            agent_state.critic_state,
            params,
            xs_seq[:burn_in],
            resets_seq[:burn_in],
            critic_zero,
        )[1]

    if stored_state:
        # Actor carries come straight from collection time: exact for the
        # policy that generated the data, stale only w.r.t. subsequent
        # param updates (the trade R2D2 shows is worth making). The reset
        # flags still zero these in-cell at episode starts.
        actor_carry = unflatten_carry(seq["actor_carry"][burn_in], actor_template)
    else:
        actor_carry = burn_actor(agent_state.actor_state.params)
    target_actor_carry = None
    if burn_target_actor:
        # TD3's bootstrap action comes from the TARGET actor. Stored
        # carries were produced by the ONLINE actor; reusing them for the
        # target actor is the standard stored-state staleness trade (R2D2
        # stores one state per step, period).
        target_actor_carry = (
            actor_carry
            if stored_state
            else burn_actor(agent_state.actor_state.target_params)
        )
    rows = slice(burn_in, -1)
    terminated = seq["terminated"][rows].squeeze(-1) > 0
    cut = (seq["truncated"][rows].squeeze(-1) > 0) & ~terminated
    carries = jax.lax.stop_gradient(
        RecurrentCarries(
            resets=resets_seq[rows],
            next_resets=terminated,
            actor_hidden=actor_carry,
            critic_hidden=burn_critic(agent_state.critic_state.params),
            target_critic_hidden=burn_critic(agent_state.critic_state.target_params),
            obs=obs_seq[rows],
            actions=act_seq[rows],
            prev_actions=prev_act_seq[rows],
            positions=stream_positions(cut),
            target_actor_hidden=target_actor_carry,
        )
    )
    # A row's next observation is the one its target bootstraps on: the
    # final one where only a time limit ended the episode.
    transition = Transition(
        obs=obs_seq[rows],
        action=act_seq[rows],
        reward=seq["reward"][rows],
        terminated=seq["terminated"][rows],
        truncated=seq["truncated"][rows],
        next_obs=seq["next_obs"][rows],
    )
    return transition, carries


def sample_replay(
    agent_state: BaseAgentState,
    buffer: BufferType,
    key: jax.Array,
    recurrent: bool = False,
    burn_in: int = 0,
    stored_state: bool = False,
    burn_target_actor: bool = False,
) -> Tuple[Transition, Optional[RecurrentCarries]]:
    """One replay batch: transitions, or burned-in sequences when recurrent.

    Returns ``(batch, carries)``. A feedforward batch keeps its
    pre-normalisation observations in ``batch.raw_obs`` and has no carries
    (``None``); sequences are time-major, with the carries of
    :func:`sample_and_burnin_sequences` and no raw observations.
    """
    if recurrent:
        return sample_and_burnin_sequences(
            agent_state, buffer, key, burn_in, burn_target_actor, stored_state
        )
    obs, terminated, truncated, next_obs, rewards, actions, raw_obs, _ = (
        get_batch_from_buffer(buffer, agent_state.collector_state.buffer_state, key)
    )
    batch = Transition(obs, actions, rewards, terminated, truncated, next_obs, raw_obs)
    return batch, None


def _bootstrap_stream(
    carries: RecurrentCarries, rows: jax.Array, boots: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """The sequence a bootstrap runs over and its reset flags: the rows,
    each followed by the step its target bootstraps on, ``boots``. That is
    the next row's own step, except where only a time limit ended the
    episode: there its final observation is a step of its own, which the
    next row's reset then forgets. Twice the rows long, the tail fresh.
    """
    at, boot = _onehot(carries.positions), _onehot(carries.positions + 1)
    # A bootstrap step and the next row's own step are one (its query wins).
    row = at * (1 - boot.sum(axis=1, keepdims=True))
    xs = _select(row, rows) + _select(boot, boots)
    empty = 1 - row.sum(axis=1) - boot.sum(axis=1)
    flags = _select(row, carries.resets) + _select(boot, carries.next_resets)
    return xs, flags + empty > 0


def _onehot(positions: jax.Array) -> jax.Array:
    """(2S, S, B): stream position p holds row t's entry."""
    stream = jnp.arange(2 * positions.shape[0])[:, None, None]
    return (stream == positions[None]).astype(jnp.float32)


def _select(onehot: jax.Array, values: jax.Array) -> jax.Array:
    """Sum of ``values`` (S, B, ...) over the rows ``onehot`` (P, S, B)
    picks: a select, not a scatter or gather, which XLA runs one index at
    a time on CPU."""
    picks = onehot.reshape(onehot.shape + (1,) * (values.ndim - 2))
    return jnp.sum(picks * values[None], axis=1)


def _bootstrap_steps(carries: RecurrentCarries, out: jax.Array) -> jax.Array:
    """The bootstrap steps of a stream's time-major ``out`` (2S, B, ...):
    row t's, after it (:func:`_bootstrap_stream`)."""
    boot = _onehot(carries.positions + 1)
    picks = boot.reshape(boot.shape + (1,) * (out.ndim - 2))
    return jnp.sum(picks * out[:, None], axis=0)


def actor_dist(
    actor_state: LoadedTrainState,
    params: Any,
    obs: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    *,
    bootstrap: bool = False,
    target_actor: bool = False,
) -> Any:
    """The actor's distribution over ``obs``.

    Without ``carries``, one feedforward batch. With them, a replayed
    sequence started from its burned-in carry (the target actor's when
    ``target_actor``, TD3's bootstrap); when ``bootstrap``, ``obs`` are the
    rows' next observations, each read after its row
    (:func:`_bootstrap_stream`).
    """
    if carries is None:
        return get_pi(actor_state, params, obs)[0]
    hidden = carries.target_actor_hidden if target_actor else carries.actor_hidden
    if not bootstrap:
        return get_pi_sequence(actor_state, params, obs, carries.resets, hidden)[0]
    xs, resets = _bootstrap_stream(carries, carries.obs, obs)
    pi = get_pi_sequence(actor_state, params, xs, resets, hidden)[0]
    # A distribution is a pytree of its parameters, time-major.
    return jax.tree.map(lambda leaf: _bootstrap_steps(carries, leaf), pi)


def q_values(
    critic_state: LoadedTrainState,
    params: Any,
    obs: jax.Array,
    actions: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    *,
    bootstrap: bool = False,
) -> jax.Array:
    """Every critic's ``Q(obs, actions)``, the ensemble axis first.

    With ``carries``, on a replayed sequence from the online critic's
    burned-in carry, or the target critic's when ``bootstrap`` (``obs``
    then being the rows' next observations, each read after its row,
    :func:`_bootstrap_stream`): the memory reads the actions taken before
    the rows, ``actions`` are queried at the head.
    """
    if carries is None:
        return predict_value(critic_state, params, action_value_input(obs, actions))
    if not bootstrap:
        x = action_value_input(obs, actions, carries.prev_actions)
        return predict_value_sequence(
            critic_state, params, x, carries.resets, carries.critic_hidden
        )[0]
    # The rows' queries are unread; after a termination nothing came before.
    rows = action_value_input(carries.obs, actions, carries.prev_actions)
    taken = jnp.where(carries.next_resets[..., None], 0.0, carries.actions)
    xs, resets = _bootstrap_stream(
        carries, rows, action_value_input(obs, actions, taken)
    )
    values = predict_value_sequence(
        critic_state, params, xs, resets, carries.target_critic_hidden
    )[0]
    # Values are (ensemble, T, B, 1): the ensemble axis goes last meanwhile.
    steps = _bootstrap_steps(carries, jnp.moveaxis(values, 0, -1))
    return jnp.moveaxis(steps, -1, 0)


def stored_actor_carry_dim(memory: Optional[MemoryConfig], stored_state: bool) -> int:
    """Width of the actor carry a stored-state replay buffer keeps per step
    (R2D2 stored state), 0 when nothing is stored."""
    return flat_carry_dim(memory) if stored_state and memory is not None else 0


def make_replay_buffer(
    agent_config: Any,
    n_envs: int,
    memory: Optional[MemoryConfig],
    buffer_size: int,
    batch_size: int,
) -> BufferType:
    """The replay buffer of an off-policy agent configured by ``agent_config``
    (a :class:`RecurrentReplayConfig` with ``learning_starts``): transitions,
    or per-env trajectories sampled as windows when ``memory`` is set."""
    burn_in, sequence_length = agent_config.burn_in, agent_config.sequence_length
    if agent_config.stored_state and memory is None:
        raise ValueError(
            "stored_state=True requires a memory config (recurrent networks)."
        )
    learning_starts = agent_config.learning_starts
    if (
        memory is not None
        and learning_starts // n_envs <= burn_in + sequence_length + 1
    ):
        # The trajectory buffer must hold one full window per env by then.
        raise ValueError(
            "learning_starts must exceed n_envs * (burn_in +"
            " sequence_length + 1) so the trajectory buffer holds at"
            " least one full sequence per env before the first update"
            f" (got learning_starts={learning_starts}, n_envs={n_envs},"
            f" burn_in={burn_in}, sequence_length={sequence_length})."
        )
    return get_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        n_envs=n_envs,
        # burn-in prefix + trained segment + bootstrap step
        sequence_length=(burn_in + sequence_length + 1 if memory is not None else None),
    )


def unsupported_recurrent_options(agent_name: str, **options: Optional[Any]) -> None:
    """Raise if any expert-guidance style option is combined with memory."""
    offending = [name for name, value in options.items() if value is not None]
    if offending:
        raise NotImplementedError(
            f"Recurrent {agent_name} does not support these options yet:"
            f" {', '.join(offending)}."
        )
