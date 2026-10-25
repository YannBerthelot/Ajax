"""Shared machinery for agents whose networks have memory (a GRU, LSTM,
transformer or Mamba; a network's memory is also called its carry).

Two kinds of agent use it:
- The off-policy agents with a Q-critic (SAC, ASAC, REDQ, TQC, TD3) train
  on windows of consecutive steps from one environment, as R2D2 does
  (Kapturowski et al. 2019): sample_and_burnin_sequences draws the windows,
  and actor_dist and q_values run the networks over them.
- PPO computes its values on a rollout with compute_rollout_values, and its
  losses run its networks with get_policy_on_minibatch and
  predict_values_on_minibatch, each for a network with or without memory
  (the on-policy section).

How an off-policy agent trains its memory:
1. Sample, for each environment, a window of burn_in + sequence_length + 1
   consecutive steps from the replay.
2. Warm up ("burn in") each network's memory: start it from zero and run
   the network over the first burn_in steps with the current parameters.
   Training does not change these warmed-up memories (no gradient flows
   back into them).
3. Train on the next sequence_length steps, backpropagating through time.
   The window's last step is never trained on: each row already stores the
   next observation its target looks ahead to (interaction.bootstrap_obs).

A TD target adds to the reward the estimated value of the next state: the
bootstrap. With memory, that value must come from the memory that has read
the rows up to the current one. Where a time limit cut an episode, that
needs the time-limit section below.

A critic's memory reads each row's observation together with the action
taken before it. The action a loss asks about (the taken one, or one the
policy samples) enters after the memory, at the critic's output layer, so
asking about another action never changes the memory (Critic.query_dim;
the recurrent off-policy critic of Ni et al. 2022, cited from memory,
unverified).
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.buffers.utils import (
    get_batch_from_buffer,
    get_buffer,
    get_sequence_batch_from_buffer,
    normalized_rewards,
)
from ajax.environments.interaction import get_pi, get_pi_sequence, reward_statistics
from ajax.networks.memory import (
    MemoryConfig,
    flat_carry_dim,
    unflatten_carry,
    zeros_carry_like,
)
from ajax.networks.networks import (
    action_value_input,
    in_critic_units,
    predict_value,
    predict_value_sequence,
)
from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState, Transition
from ajax.types import BufferType

# ----------------------------------------------------------------------------
# Replayed windows for the off-policy agents: sampling them and warming
#
# up each network's memory (steps 1 and 2 of the module docstring).
# ----------------------------------------------------------------------------


@struct.dataclass(kw_only=True)
class RecurrentReplayConfig(BaseAgentConfig):
    """An off-policy agent's sequence-replay settings, read only when it has
    memory: replayed windows of burn_in + sequence_length + 1 steps."""

    burn_in: int = 8
    sequence_length: int = 16
    # R2D2 stored-state replay: read actor carries back from the buffer
    # instead of burning them in from zero (Kapturowski et al. 2019).
    stored_state: bool = False


@struct.dataclass
class RecurrentCarries:
    """What one update needs to replay a window of S trained rows in B
    environments (R2D2-style): each network's warmed-up memory at the first
    trained row, and the rows themselves.

    The memories were warmed up with the current parameters, and no
    gradient flows back into them: only the trained rows backpropagate
    through time. The rows' observations, actions and episode starts are
    kept because the targets replay the networks over them (the
    time-limit section).
    """

    # (S, B): True where the row's observation starts an episode.
    resets: jax.Array
    # (S, B): True where the row's episode really ended (terminated), so the
    # observation after it starts a new episode
    # (interaction.bootstrap_obs).
    next_resets: jax.Array
    actor_hidden: Any  # the actor's memory before the first trained row
    critic_hidden: Any  # the online critic's memory before the first trained row
    target_critic_hidden: Any  # the target critic's memory, likewise
    obs: jax.Array  # (S, B, observation size)
    actions: jax.Array  # (S, B, action size): the actions taken
    # What the critics' memory reads with each row's observation: the
    # action taken before it (previous_actions).
    prev_actions: jax.Array  # (S, B, action size)
    # (S, B): each row's position once the final observations of the
    # episodes a time limit cut are inserted (the time-limit section).
    row_positions_with_final_observations: jax.Array
    # The target actor's memory before the first trained row, warmed up
    # with its target parameters: only for an agent whose target asks a
    # target actor for the next action (TD3); None otherwise.
    target_actor_hidden: Any = None


def previous_actions(actions: jax.Array, resets: jax.Array) -> jax.Array:
    """Each step's previous action in its episode, (T, B, A): zero at
    an episode start (resets) and at the sequence's first step."""
    previous = jnp.concatenate([jnp.zeros_like(actions[:1]), actions[:-1]])
    return jnp.where(resets[..., None], 0.0, previous)


def sample_and_burnin_sequences(
    agent_state: BaseAgentState,
    buffer: BufferType,
    sample_key: jax.Array,
    burn_in: int,
    burn_target_actor: bool = False,
    stored_state: bool = False,
) -> Tuple[Transition, RecurrentCarries]:
    """Sample windows and warm up every network's memory (steps 1 and 2 of
    the module docstring).

    With stored_state (R2D2's stored state), the actor's memory is not
    warmed up from zero: the memory the actor had when it collected the
    data is read back from the replay (the actor_carry field, written by
    collect_experience with store_hidden). Kapturowski et al. 2019 show it
    goes stale less than a zero start with a warm-up. The critics are
    still warmed up from zero: they never run during collection, so nothing
    is stored for them.

    Returns the trained rows as a Transition whose arrays have time as
    their first axis, (sequence_length, batch, ...), and the matching
    RecurrentCarries, which no gradient flows back into.
    """
    window = get_sequence_batch_from_buffer(
        buffer,
        agent_state.collector_state.buffer_state,
        sample_key,
    )
    # The replay keeps the env's own rewards (get_batch_from_buffer).
    reward_stats = reward_statistics(agent_state.collector_state.env_state)
    if reward_stats is not None:
        window = {
            **window,
            "reward": normalized_rewards(window["reward"], reward_stats),
        }
    window_obs = window["obs"]
    window_actions = window["action"]
    window_dones = jnp.logical_or(window["terminated"], window["truncated"]).squeeze(
        -1
    )  # (L, B)
    # Episode starts, aligned with the observations. The window's first
    # step may sit mid-episode, so its flag is unknown; the warm-up starts
    # every memory from zero there anyway.
    window_resets = jnp.concatenate(
        [jnp.zeros_like(window_dones[:1]), window_dones[:-1]], axis=0
    ).astype(bool)
    batch_size = window_obs.shape[1]
    window_previous_actions = previous_actions(window_actions, window_resets)
    # The warm-ups only need the memories: the asked-about actions are never
    # read, so zeros stand in for them.
    window_critic_inputs = action_value_input(
        window_obs, jnp.zeros_like(window_actions), window_previous_actions
    )

    actor_template = zeros_carry_like(
        agent_state.actor_state.hidden_state, batch_size, batch_axis=0
    )
    critic_zero_carry = zeros_carry_like(
        agent_state.critic_state.hidden_state, batch_size, batch_axis=1
    )

    def burn_actor(params: Any) -> Any:
        if burn_in == 0:
            return actor_template
        return get_pi_sequence(
            agent_state.actor_state,
            params,
            window_obs[:burn_in],
            window_resets[:burn_in],
            actor_template,
        )[1]

    def burn_critic(params: Any) -> Any:
        if burn_in == 0:
            return critic_zero_carry
        return predict_value_sequence(
            agent_state.critic_state,
            params,
            window_critic_inputs[:burn_in],
            window_resets[:burn_in],
            critic_zero_carry,
        )[1]

    if stored_state:
        # The actor's memories as they were during collection: exact for
        # the policy that collected the data, stale only by the updates
        # since (the trade R2D2 shows is worth making). The reset flags
        # still zero them at episode starts.
        actor_carry = unflatten_carry(window["actor_carry"][burn_in], actor_template)
    else:
        actor_carry = burn_actor(agent_state.actor_state.params)
    target_actor_carry = None
    if burn_target_actor:
        # TD3's target asks its target actor for the next action, so the
        # target actor needs a warmed-up memory of its own. With stored
        # memories, the online actor's are reused for it, though the online
        # actor produced them: the usual staleness of stored memories (R2D2
        # stores one memory per step).
        target_actor_carry = (
            actor_carry
            if stored_state
            else burn_actor(agent_state.actor_state.target_params)
        )
    rows = slice(burn_in, -1)
    terminated = window["terminated"][rows].squeeze(-1) > 0
    cut_by_time_limit = (window["truncated"][rows].squeeze(-1) > 0) & ~terminated
    carries = jax.lax.stop_gradient(
        RecurrentCarries(
            resets=window_resets[rows],
            next_resets=terminated,
            actor_hidden=actor_carry,
            critic_hidden=burn_critic(agent_state.critic_state.params),
            target_critic_hidden=burn_critic(agent_state.critic_state.target_params),
            obs=window_obs[rows],
            actions=window_actions[rows],
            prev_actions=window_previous_actions[rows],
            row_positions_with_final_observations=row_positions_with_final_observations(
                cut_by_time_limit
            ),
            target_actor_hidden=target_actor_carry,
        )
    )
    # A row's next observation is the one its target looks ahead to: the
    # episode's final observation where a time limit ended the episode.
    transition = Transition(
        obs=window_obs[rows],
        action=window_actions[rows],
        reward=window["reward"][rows],
        terminated=window["terminated"][rows],
        truncated=window["truncated"][rows],
        next_obs=window["next_obs"][rows],
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

    Returns (batch, carries). A feedforward batch keeps its
    pre-normalisation observations in batch.raw_obs and has no carries
    (None); sequences are time-major, with the carries of
    sample_and_burnin_sequences and no raw observations.
    """
    if recurrent:
        return sample_and_burnin_sequences(
            agent_state, buffer, key, burn_in, burn_target_actor, stored_state
        )
    collector = agent_state.collector_state
    sampled = get_batch_from_buffer(
        buffer, collector.buffer_state, key, reward_statistics(collector.env_state)
    )
    batch = Transition(
        sampled.obs,
        sampled.action,
        sampled.reward,
        sampled.terminated,
        sampled.truncated,
        sampled.next_obs,
        sampled.raw_obs,
        a_expert=sampled.a_expert,
        next_a_expert=sampled.next_a_expert,
        raw_next_obs=sampled.raw_next_obs,
        return_to_go=sampled.return_to_go,
        next_action=sampled.next_action,
        is_expert=sampled.is_expert,
        policy_action=sampled.policy_action,
    )
    return batch, None


def stored_actor_carry_dim(memory: Optional[MemoryConfig], stored_state: bool) -> int:
    """Width of the actor carry a stored-state replay buffer keeps per step
    (R2D2 stored state), 0 when nothing is stored."""
    return flat_carry_dim(memory) if stored_state and memory is not None else 0


def make_replay_buffer(
    agent: Any,
    n_envs: int,
    memory: Optional[MemoryConfig],
    buffer_size: int,
    batch_size: int,
) -> BufferType:
    """The replay buffer of an off-policy agent: transitions, or per-env
    trajectories sampled as windows when memory is set. agent holds
    burn_in, sequence_length, stored_state and
    learning_starts: a declared agent (its fields), or an older agent's
    RecurrentReplayConfig."""
    burn_in, sequence_length = agent.burn_in, agent.sequence_length
    if agent.stored_state and memory is None:
        raise ValueError(
            "stored_state=True requires a memory config (recurrent networks)."
        )
    learning_starts = agent.learning_starts
    if (
        memory is not None
        and learning_starts // n_envs <= burn_in + sequence_length + 1
    ):
        raise ValueError(
            "learning_starts // n_envs (each env's steps before the first"
            " update) must exceed burn_in + sequence_length + 1, so the"
            " trajectory buffer holds a full sequence per env"
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


# ----------------------------------------------------------------------------
# Networks with memory, run over a replayed window: the actor's
#
# distribution and the critics' Q-values.
# ----------------------------------------------------------------------------


def actor_dist(
    actor_state: LoadedTrainState,
    params: Any,
    obs: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    *,
    bootstrap: bool = False,
    target_actor: bool = False,
) -> Any:
    """The actor's action distribution at obs.

    Without carries (an actor without memory), one batch of observations.
    With carries (an actor with memory), over a replayed window, starting
    from the warmed-up memory:
    - bootstrap False: at the rows themselves (obs are their observations),
      for the actor's own loss;
    - bootstrap True: at each row's next observation (obs are those next
      observations), for the critic's target (see the time-limit
      section).
    target_actor starts from the target actor's memory instead of the online
    actor's: TD3's target asks its target actor for the next action.
    """
    if carries is None:
        return get_pi(actor_state, params, obs)[0]
    hidden = carries.target_actor_hidden if target_actor else carries.actor_hidden
    if not bootstrap:
        return get_pi_sequence(actor_state, params, obs, carries.resets, hidden)[0]
    sequence_observations, resets = _insert_final_observations(
        carries, carries.obs, obs
    )
    policy = get_pi_sequence(
        actor_state, params, sequence_observations, resets, hidden
    )[0]
    # A distribution is a tree of its parameters (a Gaussian's mean and
    # scale), each with time as its first axis: read each one's outputs
    # at the rows' next observations.
    return jax.tree.map(
        lambda leaf: _read_outputs_at_next_observations(carries, leaf), policy
    )


def q_values(
    critic_state: LoadedTrainState,
    params: Any,
    obs: jax.Array,
    actions: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    *,
    bootstrap: bool = False,
    quantiles: bool = False,
) -> jax.Array:
    """Every critic's Q(obs, actions), with the critics of the ensemble on
    the first axis (quantiles: a distributional critic's quantiles, on the
    last axis).

    With carries (critics with memory), over a replayed window:
    - bootstrap False: the online critic, from its warmed-up memory, at the
      rows (obs are their observations, actions the ones the loss asks
      about);
    - bootstrap True: the target critic, from its warmed-up memory, at each
      row's next observation (obs are those next observations, actions the
      next actions the target asks about), see the time-limit section.
    The memory reads each observation with the action taken before it; the
    asked-about action enters at the output layer. Every action reaches the
    critic in its own units (in_critic_units; predict_value converts a
    critic without memory's).
    """
    if carries is None:
        critic_input = action_value_input(obs, actions)
        return predict_value(critic_state, params, critic_input, quantiles=quantiles)
    actions = in_critic_units(critic_state, actions)
    previous_actions_in_critic_units = in_critic_units(
        critic_state, carries.prev_actions
    )
    if not bootstrap:
        critic_input = action_value_input(
            obs, actions, previous_actions_in_critic_units
        )
        return predict_value_sequence(
            critic_state,
            params,
            critic_input,
            carries.resets,
            carries.critic_hidden,
            quantiles=quantiles,
        )[0]
    # At the rows only the memory matters, so their asked-about actions are
    # never read. At a row's next step the memory reads the action taken at
    # the row, except after a real termination: the next step then starts a
    # new episode, with no action before it.
    rows = action_value_input(carries.obs, actions, previous_actions_in_critic_units)
    action_taken_before = jnp.where(
        carries.next_resets[..., None],
        0.0,
        in_critic_units(critic_state, carries.actions),
    )
    sequence_inputs, resets = _insert_final_observations(
        carries, rows, action_value_input(obs, actions, action_taken_before)
    )
    values = predict_value_sequence(
        critic_state,
        params,
        sequence_inputs,
        resets,
        carries.target_critic_hidden,
        quantiles=quantiles,
    )[0]
    # values are (critics, time, B, 1 or quantiles): move the critics' axis
    # last while reading the outputs at the next observations along time,
    # then back.
    next_q_values = _read_outputs_at_next_observations(
        carries, jnp.moveaxis(values, 0, -1)
    )
    return jnp.moveaxis(next_q_values, -1, 0)


# ----------------------------------------------------------------------------
# Time limits: the rows with each cut episode's final observation inserted.
#
# This section serves one case only, not memory itself. A TD target for
# row t needs the network's output at row t's next observation, from the
# memory that has read the rows up to t. Usually that next observation is
# row t + 1, one place later. But where a time limit (not a real
# termination) ended the episode at row t, the environment has already
# reset: row t + 1 starts a new episode, and the old episode's final
# observation is stored only as row t's next observation. So the network
# runs over the rows with that final observation inserted after row t;
# row t + 1 starts with a reset, so the inserted step never reaches the
# new episode's memory. One pass then gives every row's output at its
# next observation, one place after the row.
# ----------------------------------------------------------------------------


def row_positions_with_final_observations(cut_by_time_limit: jax.Array) -> jax.Array:
    """Each row's position once a final observation is inserted after
    every row a time limit cut.

    cut_by_time_limit (S, B) is True where a time limit, not a real
    termination, ended the episode at that row. Each insertion moves every
    later row one position further, so a row's position is its own index
    plus the number of final observations inserted before it."""
    rows = jnp.arange(cut_by_time_limit.shape[0])[:, None]
    # The extra steps up to and including the row's own, minus its own.
    return rows + jnp.cumsum(cut_by_time_limit, axis=0) - cut_by_time_limit


# The former name, released in 0.4.0; 0.5.0 removes it.
stream_positions = row_positions_with_final_observations


def _insert_final_observations(
    carries: RecurrentCarries, rows: jax.Array, next_step_inputs: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """The network's inputs at the rows, with the final observation of each
    episode a time limit cut inserted after its last row, and the matching
    reset flags.

    rows (S, B, ...) are the network's inputs at the rows, next_step_inputs
    (S, B, ...) its inputs at each row's next observation. There is room for
    one insertion per row, so the result is 2 * S long; its unused positions
    at the end are marked as resets.
    """
    row_positions = _make_row_to_position_table(
        carries.row_positions_with_final_observations
    )
    next_step_positions = _make_row_to_position_table(
        carries.row_positions_with_final_observations + 1
    )
    # Where row t's next step is row t + 1 itself, that position takes the
    # next-step input: the same observation, with the action the target asks
    # about (the rows' own asked-about actions are never read).
    row_positions = row_positions * (1 - next_step_positions.sum(axis=1, keepdims=True))
    sequence_inputs = _move_rows_to_positions(
        row_positions, rows
    ) + _move_rows_to_positions(next_step_positions, next_step_inputs)
    unused_positions = 1 - row_positions.sum(axis=1) - next_step_positions.sum(axis=1)
    reset_flags = _move_rows_to_positions(
        row_positions, carries.resets
    ) + _move_rows_to_positions(next_step_positions, carries.next_resets)
    return sequence_inputs, reset_flags + unused_positions > 0


def _make_row_to_position_table(row_positions: jax.Array) -> jax.Array:
    """A 0/1 table (2 * S, S, B): table[p, r, b] is 1 where row r of
    environment b sits at position p, from each row's position (S, B)."""
    positions = jnp.arange(2 * row_positions.shape[0])[:, None, None]
    return (positions == row_positions[None]).astype(jnp.float32)


def _move_rows_to_positions(
    row_to_position: jax.Array, row_values: jax.Array
) -> jax.Array:
    """row_values (S, B, ...) moved to the positions the table
    row_to_position (2 * S, S, B) gives them. Written as a multiply-and-sum
    because XLA on CPU runs a scatter or a gather one index at a time."""
    row_to_position = row_to_position.reshape(
        row_to_position.shape + (1,) * (row_values.ndim - 2)
    )
    return jnp.sum(row_to_position * row_values[None], axis=1)


def _read_outputs_at_next_observations(
    carries: RecurrentCarries, sequence_outputs: jax.Array
) -> jax.Array:
    """Every row's output at its next observation, from a network's outputs
    over the rows with the final observations inserted (2 * S, B, ...): the
    output one position after the row."""
    next_step_positions = _make_row_to_position_table(
        carries.row_positions_with_final_observations + 1
    )
    row_to_position = next_step_positions.reshape(
        next_step_positions.shape + (1,) * (sequence_outputs.ndim - 2)
    )
    return jnp.sum(row_to_position * sequence_outputs[:, None], axis=0)


# ----------------------------------------------------------------------------
# On-policy agents (PPO): the critic's values on a rollout, and each network
# over a minibatch.
#
# Each call here serves a network with or without memory, so the agent's own
# file never asks which.
# ----------------------------------------------------------------------------


def compute_rollout_values(
    agent_state: Any,
    rollout: Transition,
    agent_state_at_rollout_start: Any,
    sequence_length: int,
) -> tuple[Any, jax.Array, jax.Array, Optional[dict]]:
    """Before an on-policy update: the critic's values V(s_t) on the rollout
    and its next values V(s_{t+1}), which GAE reads.

    A critic without memory values each observation on its own. A critic
    with memory reads the rollout in order
    (compute_values_from_rollout_sequence), which also moves its memory to
    the rollout's end and gives what the minibatches replay both networks
    from: the episode starts (resets, one per step) and each network's
    memory at the start of every sequence of sequence_length steps
    (actor_hidden and critic_hidden).

    Returns the agent state, the values, the next values and those memories
    (None without memory).
    """
    critic = agent_state.critic_state
    if critic.hidden_state is None:
        values = predict_value(critic, critic.params, rollout.obs).squeeze(0)
        next_values = predict_value(critic, critic.params, rollout.next_obs).squeeze(0)
        return agent_state, values, next_values, None
    agent_state, values, next_values, resets, (actor_starts, critic_starts) = (
        compute_values_from_rollout_sequence(
            agent_state, rollout, agent_state_at_rollout_start, sequence_length
        )
    )
    memories = {
        "resets": resets,
        "actor_hidden": actor_starts,
        "critic_hidden": critic_starts,
    }
    return agent_state, values, next_values, memories


def compute_values_from_rollout_sequence(
    agent_state: Any,
    rollout: Transition,
    agent_state_at_rollout_start: Any,
    length: int,
) -> tuple[Any, jax.Array, jax.Array, jax.Array, tuple]:
    """Before a recurrent PPO update: the critic's values on the rollout,
    its next values, the episode-start flags, and each network's memory at
    the start of each training sequence of length steps.

    While collecting, only the actor runs, so its memory is up to date and
    the critic's is not. This pass runs the critic over the rollout one step
    at a time, from its memory at the rollout's start, which gives:
    - the value V(s_t) at every step;
    - the next value V(s_{t+1}): after reading step t, the critic is asked
      about step t's next observation, which is the episode's final
      observation where the episode ended. So a step cut by a time limit is
      valued with its own episode's memory, as sb3-contrib's RecurrentPPO
      does, not with a fresh memory on the next episode's first observation
      (GAE ignores the next value where the episode really terminated);
    - the critic's memory at the start of every training sequence, which the
      loss replays the critic from;
    - the critic's memory at the rollout's end, kept as the next rollout's
      starting memory.
    It runs step by step, as acting does, because each next value needs the
    memory right after its own step.

    The actor's memory at the start of each training sequence is recomputed
    with the current parameters by one more pass (skipped when one sequence
    is the whole rollout, whose start memory is the stored one exactly). So
    no sequence starts from a zero memory in the middle of an episode, which
    makes truncated backpropagation through time worse than none.

    Returns the agent state with the critic's memory moved to the rollout's
    end, the values, the next values, the episode-start flags (resets[t]:
    obs[t] starts an episode) and the (actor, critic) start memories.
    """
    n_steps, n_envs = rollout.obs.shape[:2]
    n_sequences = n_steps // length
    initial_done = jnp.logical_or(
        agent_state_at_rollout_start.collector_state.last_terminated,
        agent_state_at_rollout_start.collector_state.last_truncated,
    ).astype(bool)
    dones = jnp.logical_or(rollout.terminated, rollout.truncated).squeeze(-1)
    resets = jnp.concatenate(
        [initial_done[None], dones[:-1].astype(bool)], axis=0
    )  # (T, B)
    obs_sequences = rollout.obs.reshape(n_sequences, length, *rollout.obs.shape[1:])
    next_obs_sequences = rollout.next_obs.reshape(obs_sequences.shape)
    reset_sequences = resets.reshape(n_sequences, length, n_envs)
    critic, actor = agent_state.critic_state, agent_state.actor_state
    no_resets = jnp.zeros((1, n_envs), bool)

    def step_critic(carry: Any, step: tuple) -> tuple[Any, tuple]:
        obs, next_obs, step_resets = step
        value, carry = predict_value_sequence(
            critic, critic.params, obs[None], step_resets[None], carry
        )
        next_value, _ = predict_value_sequence(
            critic, critic.params, next_obs[None], no_resets, carry
        )
        # (num = 1, T = 1, B, 1) -> (B, 1)
        return carry, (value.squeeze((0, 1)), next_value.squeeze((0, 1)))

    def run_critic_over_sequence(carry: Any, sequence: tuple) -> tuple[Any, tuple]:
        next_carry, values = jax.lax.scan(step_critic, carry, sequence)
        return next_carry, (values, carry)

    assert agent_state_at_rollout_start.critic_state.hidden_state is not None
    critic_end, ((sequence_values, sequence_next_values), critic_starts) = jax.lax.scan(
        run_critic_over_sequence,
        jax.lax.stop_gradient(agent_state_at_rollout_start.critic_state.hidden_state),
        (obs_sequences, next_obs_sequences, reset_sequences),
    )
    # (S, L, B, 1) -> (T, B, 1)
    values = sequence_values.reshape(n_steps, *sequence_values.shape[2:])
    next_values = sequence_next_values.reshape(values.shape)

    initial_actor_hidden = jax.lax.stop_gradient(
        agent_state_at_rollout_start.actor_state.hidden_state
    )
    if n_sequences > 1:

        def run_actor_over_sequence(carry: Any, sequence: tuple) -> tuple[Any, Any]:
            obs, sequence_resets = sequence
            _, next_carry = get_pi_sequence(
                actor, actor.params, obs, sequence_resets, carry
            )
            return next_carry, carry

        _, actor_starts = jax.lax.scan(
            run_actor_over_sequence,
            initial_actor_hidden,
            (obs_sequences, reset_sequences),
        )
    else:
        actor_starts = jax.tree.map(lambda leaf: leaf[None], initial_actor_hidden)

    agent_state = agent_state.replace(
        critic_state=critic.replace(hidden_state=jax.lax.stop_gradient(critic_end))
    )
    carries = jax.lax.stop_gradient((actor_starts, critic_starts))
    return agent_state, values, next_values, resets, carries


def get_policy_on_minibatch(
    actor_state: LoadedTrainState, actor_params: Any, minibatch: dict
) -> Any:
    """The actor's action distribution at the minibatch's observations (obs),
    for an on-policy loss.

    An actor without memory reads each observation on its own. An actor with
    memory replays each of the minibatch's sequences from its memory at the
    sequence's start (actor_hidden, from compute_rollout_values: the live
    memory on actor_state has moved past this rollout), resets flagging the
    observations that start an episode."""
    if actor_state.hidden_state is None:
        return get_pi(actor_state, actor_params, minibatch["obs"])[0]
    return get_pi_sequence(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=minibatch["obs"],
        resets=minibatch["resets"],
        initial_hidden=minibatch["actor_hidden"],
    )[0]


def predict_values_on_minibatch(
    critic_state: LoadedTrainState, critic_params: Any, minibatch: dict
) -> jax.Array:
    """The critic's value predictions at the minibatch's observations (obs),
    for an on-policy loss: one critic, so without the ensemble's axis.

    A critic with memory replays the minibatch's sequences from critic_hidden,
    as get_policy_on_minibatch replays the actor's."""
    if critic_state.hidden_state is None:
        value_preds = predict_value(critic_state, critic_params, minibatch["obs"])
    else:
        value_preds, _ = predict_value_sequence(
            critic_state,
            critic_params,
            minibatch["obs"],
            minibatch["resets"],
            minibatch["critic_hidden"],
        )
    return value_preds.squeeze(0)


# ----------------------------------------------------------------------------
# Refusals.
#
# ----------------------------------------------------------------------------


def unsupported_recurrent_options(agent_name: str, **options: Optional[Any]) -> None:
    """Raise if any expert-guidance style option is combined with memory."""
    offending = [name for name, value in options.items() if value is not None]
    if offending:
        raise NotImplementedError(
            f"Recurrent {agent_name} does not support these options yet:"
            f" {', '.join(offending)}."
        )
