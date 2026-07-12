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
via their sequence-mode branches.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.buffers.utils import get_sequence_batch_from_buffer
from ajax.environments.interaction import get_pi_sequence
from ajax.networks.memory import zeros_carry_like
from ajax.networks.networks import predict_value_sequence
from ajax.state import BaseAgentState, Transition
from ajax.types import BufferType


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
) -> Tuple[Transition, RecurrentCarries]:
    """Sample sequences and burn in all carries; see the module docstring.

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

    actor_carry = zeros_carry_like(
        agent_state.actor_state.hidden_state, batch_size, batch_axis=0
    )
    critic_zero = zeros_carry_like(
        agent_state.critic_state.hidden_state, batch_size, batch_axis=1
    )
    critic_carry = critic_zero
    if burn_in > 0:
        _, actor_carry = get_pi_sequence(
            agent_state.actor_state,
            agent_state.actor_state.params,
            obs_seq[:burn_in],
            resets_seq[:burn_in],
            actor_carry,
        )
        _, critic_carry = predict_value_sequence(
            agent_state.critic_state,
            agent_state.critic_state.params,
            xs_seq[:burn_in],
            resets_seq[:burn_in],
            critic_zero,
        )
    # One extra step for the carries aligned with next_obs[0] = obs[burn_in+1]
    _, actor_next_carry = get_pi_sequence(
        agent_state.actor_state,
        agent_state.actor_state.params,
        obs_seq[burn_in : burn_in + 1],
        resets_seq[burn_in : burn_in + 1],
        actor_carry,
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


def check_recurrent_learning_starts(
    learning_starts: int, n_envs: int, burn_in: int, sequence_length: int
) -> None:
    """Fail fast if the trajectory buffer cannot hold one full sequence per
    env before the first update."""
    if learning_starts // n_envs <= burn_in + sequence_length + 1:
        raise ValueError(
            "learning_starts must exceed n_envs * (burn_in +"
            " sequence_length + 1) so the trajectory buffer holds at"
            " least one full sequence per env before the first update"
            f" (got learning_starts={learning_starts}, n_envs={n_envs},"
            f" burn_in={burn_in}, sequence_length={sequence_length})."
        )


def unsupported_recurrent_options(agent_name: str, **options: Optional[Any]) -> None:
    """Raise if any expert-guidance style option is combined with memory."""
    offending = [name for name, value in options.items() if value is not None]
    if offending:
        raise NotImplementedError(
            f"Recurrent {agent_name} does not support these options yet:"
            f" {', '.join(offending)}."
        )
