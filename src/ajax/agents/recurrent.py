"""Shared machinery for recurrent off-policy updates (R2D2-style).

Every off-policy agent with a Q-critic (SAC, ASAC, REDQ, TD3) trains its
memory the same way:

1. Sample contiguous per-env sequences of length
   ``burn_in + sequence_length + 1`` from a trajectory buffer.
2. Warm up ("burn in") the actor / online-critic / target-critic carries
   from zero over the first ``burn_in`` steps with the CURRENT params,
   under ``stop_gradient`` — the carries are constants inside the losses.
3. Train on the next ``sequence_length`` steps with BPTT; the final step
   only provides the bootstrap next-observations.

``sample_and_burnin_sequences`` implements steps 1-3 once for all agents;
the per-agent losses then consume the returned :class:`RecurrentCarries`
through :func:`actor_dist` and :func:`q_values`, which run a network on a
feedforward batch or, given carries, on a replayed sequence.
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
from ajax.networks.networks import predict_value, predict_value_sequence
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
    time. ``resets``/``next_resets`` are obs-aligned episode-start flags
    for the training segment and its one-step-shifted successor.
    """

    resets: jax.Array  # (S, B)
    next_resets: jax.Array  # (S, B)
    actor_hidden: Any  # carry valid for obs[0] of the training segment
    actor_next_hidden: Any  # carry valid for next_obs[0]
    critic_hidden: Any  # online-critic carry for (obs, action)[0]
    target_critic_hidden: Any  # target-critic carry for next inputs
    # Target-ACTOR carry for next_obs[0], burned with actor target_params.
    # Only populated for agents whose bootstrap action comes from a target
    # actor (TD3); None otherwise.
    target_actor_next_hidden: Any = None


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
    xs_seq = jnp.concatenate([obs_seq, act_seq], axis=-1)

    actor_template = zeros_carry_like(
        agent_state.actor_state.hidden_state, batch_size, batch_axis=0
    )
    critic_zero = zeros_carry_like(
        agent_state.critic_state.hidden_state, batch_size, batch_axis=1
    )

    if stored_state:
        # Actor carries come straight from collection time: exact for the
        # policy that generated the data, stale only w.r.t. subsequent
        # param updates (the trade R2D2 shows is worth making). The reset
        # flags still zero these in-cell at episode starts.
        actor_carry = unflatten_carry(seq["actor_carry"][burn_in], actor_template)
        actor_next_carry = unflatten_carry(
            seq["actor_carry"][burn_in + 1], actor_template
        )
    else:
        actor_carry = actor_template
        if burn_in > 0:
            _, actor_carry = get_pi_sequence(
                agent_state.actor_state,
                agent_state.actor_state.params,
                obs_seq[:burn_in],
                resets_seq[:burn_in],
                actor_carry,
            )
        # One extra step for the carry aligned with next_obs[0] = obs[burn_in+1]
        _, actor_next_carry = get_pi_sequence(
            agent_state.actor_state,
            agent_state.actor_state.params,
            obs_seq[burn_in : burn_in + 1],
            resets_seq[burn_in : burn_in + 1],
            actor_carry,
        )

    critic_carry = critic_zero
    if burn_in > 0:
        _, critic_carry = predict_value_sequence(
            agent_state.critic_state,
            agent_state.critic_state.params,
            xs_seq[:burn_in],
            resets_seq[:burn_in],
            critic_zero,
        )
    _, target_critic_carry = predict_value_sequence(
        agent_state.critic_state,
        agent_state.critic_state.target_params,
        xs_seq[: burn_in + 1],
        resets_seq[: burn_in + 1],
        critic_zero,
    )
    target_actor_next_carry = None
    if burn_target_actor:
        if stored_state:
            # Stored carries were produced by the ONLINE actor; reusing
            # them for the target actor is the standard stored-state
            # staleness trade (R2D2 stores one state per step, period).
            target_actor_next_carry = actor_next_carry
        else:
            # TD3's bootstrap action comes from the TARGET actor; burn its
            # carry with the target params over the same prefix.
            actor_zero = zeros_carry_like(
                agent_state.actor_state.hidden_state, batch_size, batch_axis=0
            )
            _, target_actor_next_carry = get_pi_sequence(
                agent_state.actor_state,
                agent_state.actor_state.target_params,
                obs_seq[: burn_in + 1],
                resets_seq[: burn_in + 1],
                actor_zero,
            )
    carries = jax.lax.stop_gradient(
        RecurrentCarries(
            resets=resets_seq[burn_in:-1],
            next_resets=resets_seq[burn_in + 1 :],
            actor_hidden=actor_carry,
            actor_next_hidden=actor_next_carry,
            critic_hidden=critic_carry,
            target_critic_hidden=target_critic_carry,
            target_actor_next_hidden=target_actor_next_carry,
        )
    )
    transition = Transition(
        obs=obs_seq[burn_in:-1],
        action=act_seq[burn_in:-1],
        reward=seq["reward"][burn_in:-1],
        terminated=seq["terminated"][burn_in:-1],
        truncated=seq["truncated"][burn_in:-1],
        next_obs=obs_seq[burn_in + 1 :],
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


def bootstrap_cuts(batch: Transition, carries: Optional[RecurrentCarries]) -> jax.Array:
    """Where a replayed target stops bootstrapping: on termination only, as
    the references do; a truncated row bootstraps on the final observation
    it stores (``interaction.bootstrap_obs``).

    Replayed sequences (``carries``) still cut at time limits too, a
    deliberate workaround: their next observations are the next rows, the
    reset one after an end, and bootstrapping on the final one needs the
    target carries from before the reset, which the burn-in does not keep.
    """
    if carries is None:
        return batch.terminated
    return jnp.logical_or(batch.terminated, batch.truncated)


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
    sequence started from its burned-in carry: the online actor's for the
    observations, or for the next observations when ``bootstrap`` (the
    target actor's as well when ``target_actor``, TD3's bootstrap).
    """
    if carries is None:
        return get_pi(actor_state, params, obs)[0]
    if not bootstrap:
        resets, hidden = carries.resets, carries.actor_hidden
    elif target_actor:
        resets, hidden = carries.next_resets, carries.target_actor_next_hidden
    else:
        resets, hidden = carries.next_resets, carries.actor_next_hidden
    return get_pi_sequence(actor_state, params, obs, resets, hidden)[0]


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
    then being next observations).
    """
    x = jnp.concatenate((obs, actions), axis=-1)
    if carries is None:
        return predict_value(critic_state, params, x)
    if bootstrap:
        resets, hidden = carries.next_resets, carries.target_critic_hidden
    else:
        resets, hidden = carries.resets, carries.critic_hidden
    return predict_value_sequence(critic_state, params, x, resets, hidden)[0]


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
