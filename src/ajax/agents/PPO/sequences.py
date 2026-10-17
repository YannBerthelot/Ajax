"""A recurrent on-policy rollout's sequences: its values, reset flags and
start carries (one critic pass), and its minibatches of whole sequences.
PPO's, which APO reuses."""

from typing import Any

import jax
import jax.numpy as jnp

from ajax.agents.PPO.utils import split_fragments
from ajax.environments.interaction import get_pi_sequence
from ajax.networks.networks import predict_value_sequence
from ajax.state import Transition


def recurrent_values(
    agent_state: Any, rollout: Transition, start: Any, length: int
) -> tuple[Any, jax.Array, jax.Array, jax.Array, tuple]:
    """A recurrent rollout's values, next values and reset flags, and the
    actor's and critic's carries at the start of each sequence of
    ``length`` steps.

    One critic pass gives the values and advances the critic carry, which
    collection never steps: it ends as the next rollout's start carry. Run
    as a scan over the sequences, it also yields their start carries. The
    actor's are recomputed sequence by sequence with the current params (one
    more pass, skipped for whole-rollout sequences, whose start carry is
    exact): a sequence never starts from a zero carry mid-episode, the
    variant that makes truncated BPTT worse than none. The pass runs step by
    step (the memories' acting path, equal to the sequence one), and after
    each step reads ``V(s_{t+1})`` from the carry that consumed ``obs[t]``,
    on ``next_obs[t]``: the next observation, or the episode's final one
    where it ended. A time limit thus bootstraps on its own episode, as
    sb3-contrib's RecurrentPPO does (its pre-reset LSTM state on the
    terminal observation), not on the reset observation with a fresh
    carry; GAE masks terminal steps.

    Returns the state (critic carry advanced), the values, the next values,
    the reset flags (``resets[t]``: ``obs[t]`` starts an episode) and the
    ``(actor, critic)`` start carries.
    """
    n_steps, n_envs = rollout.obs.shape[:2]
    n_sequences = n_steps // length
    initial_done = jnp.logical_or(
        start.collector_state.last_terminated, start.collector_state.last_truncated
    ).astype(bool)
    dones_seq = jnp.logical_or(rollout.terminated, rollout.truncated).squeeze(-1)
    resets = jnp.concatenate(
        [initial_done[None], dones_seq[:-1].astype(bool)], axis=0
    )  # (T, B)
    obs_seqs = rollout.obs.reshape(n_sequences, length, *rollout.obs.shape[1:])
    next_obs_seqs = rollout.next_obs.reshape(obs_seqs.shape)
    resets_seqs = resets.reshape(n_sequences, length, n_envs)
    critic, actor = agent_state.critic_state, agent_state.actor_state
    going_on = jnp.zeros((1, n_envs), bool)

    def critic_step(carry: Any, step: tuple) -> tuple[Any, tuple]:
        obs, next_obs, step_resets = step
        value, carry = predict_value_sequence(
            critic, critic.params, obs[None], step_resets[None], carry
        )
        next_value, _ = predict_value_sequence(
            critic, critic.params, next_obs[None], going_on, carry
        )
        # (num = 1, T = 1, B, 1) -> (B, 1)
        return carry, (value.squeeze((0, 1)), next_value.squeeze((0, 1)))

    def critic_chunk(carry: Any, seq: tuple) -> tuple[Any, tuple]:
        carry_next, values = jax.lax.scan(critic_step, carry, seq)
        return carry_next, (values, carry)

    assert start.critic_state.hidden_state is not None
    critic_end, ((seq_values, seq_next_values), critic_starts) = jax.lax.scan(
        critic_chunk,
        jax.lax.stop_gradient(start.critic_state.hidden_state),
        (obs_seqs, next_obs_seqs, resets_seqs),
    )
    # (S, L, B, 1) -> (T, B, 1)
    values = seq_values.reshape(n_steps, *seq_values.shape[2:])
    next_values = seq_next_values.reshape(values.shape)

    initial_actor_hidden = jax.lax.stop_gradient(start.actor_state.hidden_state)
    if n_sequences > 1:

        def actor_chunk(carry: Any, seq: tuple) -> tuple[Any, Any]:
            obs, seq_resets = seq
            _, carry_next = get_pi_sequence(actor, actor.params, obs, seq_resets, carry)
            return carry_next, carry

        _, actor_starts = jax.lax.scan(
            actor_chunk, initial_actor_hidden, (obs_seqs, resets_seqs)
        )
    else:
        actor_starts = jax.tree.map(lambda x: x[None], initial_actor_hidden)

    agent_state = agent_state.replace(
        critic_state=critic.replace(hidden_state=jax.lax.stop_gradient(critic_end))
    )
    carries = jax.lax.stop_gradient((actor_starts, critic_starts))
    return agent_state, values, next_values, resets, carries


def split_carries(x: jax.Array, perm: jax.Array, num_minibatches: int) -> jax.Array:
    """``(S, B, ...)`` start carries to ``(k, S * B // k, ...)``, sequence
    ``s * B + b`` in the order of ``perm`` (as :func:`split_fragments`)."""
    x = x.reshape(-1, *x.shape[2:])
    x = jnp.take(x, perm, axis=0)
    return x.reshape(num_minibatches, -1, *x.shape[1:])


def sequence_minibatches(
    batch: dict, carries: tuple, rng: jax.Array, num_minibatches: int, length: int
) -> dict:
    """A recurrent rollout's minibatches of whole sequences, each with the
    actor's and the critic's carry at its start (the critic's ensemble axis
    second)."""
    n_steps, n_envs = batch["obs"].shape[:2]
    perm = jax.random.permutation(rng, n_steps // length * n_envs)

    def split(x: jax.Array) -> jax.Array:
        return split_carries(x, perm, num_minibatches)

    actor_starts, critic_starts = carries
    return split_fragments(batch, perm, num_minibatches, length) | {
        "actor_hidden": jax.tree.map(split, actor_starts),
        "critic_hidden": jax.tree.map(
            jax.vmap(split, in_axes=1, out_axes=1), critic_starts
        ),
    }
