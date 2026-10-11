"""A replayed window of the recurrent off-policy agents (recurrent.py): a
critic's memory reads each row's observation and the action taken before
it; the action a loss asks about is a query at the critic's head, never
fed to its carry; a row's target bootstraps on the observation it stores,
read after the row, through a time limit."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    previous_actions,
    q_values,
    sample_and_burnin_sequences,
    stream_positions,
)
from ajax.buffers.utils import get_buffer
from ajax.environments.interaction import get_pi_sequence
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig, zeros_carry_like
from ajax.networks.networks import (
    Actor,
    MultiCritic,
    action_value_input,
    init_network_state,
    predict_value_sequence,
)

S, B, OBS, ACT = 4, 3, 2, 1
_KEYS = jax.random.split(jax.random.PRNGKey(0), 5)
OBS_SEQ, AFTER = jax.random.normal(_KEYS[0], (2, S, B, OBS))
TAKEN = jax.random.uniform(_KEYS[2], (S, B, ACT), minval=-1.0, maxval=1.0)
QUERIES = jax.random.uniform(_KEYS[3], (S, B, ACT), minval=-1.0, maxval=1.0)
# Env 0's time limit ends an episode at row 1, env 1's termination one at
# row 2, env 2's time limits episodes at rows 0 and 2; the next rows start
# episodes. A row's next observation is the next row's, except a time
# limit's final one (interaction.bootstrap_obs).
CUT = jnp.zeros((S, B), bool).at[1, 0].set(True).at[0, 2].set(True).at[2, 2].set(True)
NEXT_OBS = jnp.where(CUT[..., None], AFTER, jnp.concatenate([OBS_SEQ[1:], AFTER[-1:]]))
TERMINATED = jnp.zeros((S, B), bool).at[2, 1].set(True)
RESETS = jnp.concatenate([jnp.zeros((1, B), bool), CUT[:-1] | TERMINATED[:-1]])


def _memory(kind: str) -> MemoryConfig:
    return MemoryConfig(kind, hidden_size=8, window=4)


def _critic(kind: str) -> Any:
    critic = MultiCritic(
        input_architecture=("16", "relu"), num=2, memory=_memory(kind), query_dim=ACT
    )
    x = jnp.zeros((1, OBS + 2 * ACT))
    return init_network_state(x, critic, _KEYS[4], optax.sgd(0.1), memory=_memory(kind))


def _actor(kind: str) -> Any:
    actor = Actor(
        input_architecture=("16", "relu"),
        action_dim=ACT,
        continuous=True,
        squash=True,
        memory=_memory(kind),
    )
    x = jnp.zeros((1, OBS))
    return init_network_state(
        x, actor, _KEYS[4], optax.sgd(0.1), n_envs=B, memory=_memory(kind)
    )


def _carries(critic: Any, actor: Any = None, taken: jax.Array = TAKEN) -> Any:
    """A window whose rows took ``taken``, started from fresh carries."""
    hidden = zeros_carry_like(critic.hidden_state, B, batch_axis=1)
    actor_hidden = None if actor is None else zeros_carry_like(actor.hidden_state, B)
    positions = stream_positions(CUT)
    return RecurrentCarries(
        resets=RESETS,
        next_resets=TERMINATED,
        actor_hidden=actor_hidden,
        critic_hidden=hidden,
        target_critic_hidden=hidden,
        obs=OBS_SEQ,
        actions=taken,
        prev_actions=previous_actions(taken, RESETS),
        positions=positions,
    )


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_a_queried_action_moves_its_own_rows_value_only(
    kind: str, bootstrap: bool
) -> None:
    state = _critic(kind)
    carries, obs = _carries(state), NEXT_OBS if bootstrap else OBS_SEQ

    def q(queries: jax.Array) -> jax.Array:
        return q_values(state, state.params, obs, queries, carries, bootstrap=bootstrap)

    values, moved = q(QUERIES), q(QUERIES.at[1].add(0.5))
    rows = np.arange(S) != 1
    np.testing.assert_array_equal(values[:, rows], moved[:, rows])
    assert not np.allclose(values[:, 1], moved[:, 1])


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_the_memory_reads_the_action_taken_before_each_row(kind: str) -> None:
    """Row 1's taken action reaches the later rows of its episode (envs 1
    and 2), not row 1 itself nor env 0's next episode (row 2 on)."""
    state = _critic(kind)
    taken = q_values(state, state.params, OBS_SEQ, QUERIES, _carries(state))
    changed = _carries(state, taken=TAKEN.at[1].add(0.5))
    moved = q_values(state, state.params, OBS_SEQ, QUERIES, changed)
    np.testing.assert_array_equal(taken[:, :2], moved[:, :2])
    np.testing.assert_array_equal(taken[:, 2:, 0], moved[:, 2:, 0])
    assert not np.allclose(taken[:, 2:, 1:], moved[:, 2:, 1:])


def _after(rows: jax.Array, t: int, boot: jax.Array) -> tuple:
    """Rows 0..t, then row t's bootstrap step: the reference history."""
    xs = jnp.concatenate([rows[: t + 1], boot[None]])
    return xs, jnp.concatenate([RESETS[: t + 1], TERMINATED[t][None]])


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_a_bootstrap_reads_each_rows_next_observation_after_the_row(
    kind: str,
) -> None:
    """The target critic and the actor answer row t's next observation from
    the history up to row t, through env 0's time limit and env 2's two;
    after env 1's termination at row 2, from a fresh start."""
    critic, actor = _critic(kind), _actor(kind)
    carries = _carries(critic, actor)
    values = q_values(
        critic, critic.target_params, NEXT_OBS, QUERIES, carries, bootstrap=True
    )
    pi = actor_dist(actor, actor.params, NEXT_OBS, carries, bootstrap=True)
    rows = action_value_input(OBS_SEQ, QUERIES, carries.prev_actions)
    taken = jnp.where(TERMINATED[..., None], 0.0, TAKEN)
    for t in range(S):
        boot = action_value_input(NEXT_OBS[t], QUERIES[t], taken[t])
        xs, resets = _after(rows, t, boot)
        want = predict_value_sequence(
            critic, critic.target_params, xs, resets, carries.target_critic_hidden
        )[0][:, -1]
        np.testing.assert_allclose(values[:, t], want, atol=1e-5)
        obs, resets = _after(OBS_SEQ, t, NEXT_OBS[t])
        mean = get_pi_sequence(actor, actor.params, obs, resets, carries.actor_hidden)
        np.testing.assert_allclose(pi.mean()[t], mean[0].mean()[-1], atol=1e-5)


def test_a_replayed_window_bootstraps_a_time_limit_on_its_final_observation() -> None:
    """Rows store the observation their target bootstraps on; the window
    reads it, flags a bootstrap reset after terminations only, and places
    each time limit's final observation after its row."""
    burn_in, length = 1, 3
    buffer = get_buffer(64, 16, sequence_length=burn_in + length + 1)
    # Each end's row: (terminated, truncated).
    ends = {3: (0.0, 1.0), 7: (1.0, 0.0), 11: (1.0, 1.0)}

    def row(t: int) -> dict:
        terminated, truncated = ends.get(t, (0.0, 0.0))
        return {
            "obs": jnp.full((1, OBS), float(t)),
            "action": jnp.full((1, ACT), 0.1 * t),
            "reward": jnp.zeros((1, 1)),
            "terminated": jnp.full((1, 1), terminated),
            "truncated": jnp.full((1, 1), truncated),
            # The final observation where the time limit alone fired.
            "next_obs": jnp.full((1, OBS), 100.0 + t if t == 3 else t + 1.0),
        }

    buffer_state = buffer.init(jax.tree.map(lambda x: x[0], row(0)))
    for t in range(16):
        buffer_state = buffer.add(buffer_state, row(t))
    agent_state: Any = SimpleNamespace(
        collector_state=SimpleNamespace(buffer_state=buffer_state),
        actor_state=_actor("gru"),
        critic_state=_critic("gru"),
    )
    batch, carries = sample_and_burnin_sequences(
        agent_state, buffer, jax.random.PRNGKey(1), burn_in
    )
    t = batch.obs[..., 0]
    np.testing.assert_array_equal(batch.next_obs[..., 0], jnp.where(t == 3, 103, t + 1))
    np.testing.assert_array_equal(carries.next_resets, jnp.isin(t, jnp.array([7, 11])))
    np.testing.assert_array_equal(carries.resets, jnp.isin(t, jnp.array([4, 8, 12])))
    after_limit = jnp.cumsum(t == 3, 0) - (t == 3)
    np.testing.assert_array_equal(
        carries.positions, jnp.arange(length)[:, None] + after_limit
    )
