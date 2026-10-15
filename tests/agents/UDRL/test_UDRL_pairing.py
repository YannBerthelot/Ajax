"""UDRL keeps each env's column its own, on two envs of the identity task
(tests/environments/identity_env.py: the observation [t, c] says which step
of which episode a row is; an episode ends at t = 3 or 5 by its own draw,
so the columns' ends fall apart), through 12 segments into a FIFO of 8
slots, the commands pinned (``command_target_tau`` 0, the exploration std
0) so every episode starts at (R0, H0):

(i) within an episode, a column's next stored observation continues its own
episode, [t + 1, c];
(ii) every stored command follows its own column's recurrence: d_r' = d_r -
r and d_h' = max(d_h - 1, 1) within an episode, (R0, H0) at each start;
(iii) every command the sampler hands the actor is its row's hindsight in
its own column: d_h steps of the row's episode, from the row on, and d_r
the rewards they paid; the action is the row's.
"""

import functools
from typing import Any

import jax
import numpy as np

from ajax.agents.UDRL.buffer import episode_stream, sample_training_batch
from ajax.agents.UDRL.UDRL import UDRL
from tests.environments.identity_env import IdentityEnv

R0, H0, SEG, CAPACITY, N_ENVS = 2.5, 8.0, 4, 8, 2


@functools.cache
def _buffer() -> Any:
    env = IdentityEnv()
    agent = UDRL(
        env_id=env,
        env_params=env.default_params,
        n_envs=N_ENVS,
        n_steps=SEG,
        buffer_capacity=CAPACITY,
        batch_size=8,
        n_updates_per_iter=1,
        actor_architecture=("16", "relu"),
        command_target_tau=0.0,
        command_return_init=R0,
        command_horizon_init=H0,
    )
    state, _ = agent.train(seed=0, n_timesteps=12 * SEG * N_ENVS)
    return jax.tree.map(lambda x: x[0], state.buffer)


def _stream(buffer: Any) -> tuple[np.ndarray, ...]:
    """The FIFO's slots oldest first, as one stream per column: the stored
    observation (S, n_envs, 4), action, reward and done flag."""
    assert int(buffer.fill_count) == CAPACITY  # it wrapped: 12 segments in 8
    order = (int(buffer.write_idx) + np.arange(CAPACITY)) % CAPACITY
    leaves = (buffer.obs, buffer.actions, buffer.rewards, buffer.dones)
    obs, action, reward, done = (
        np.asarray(v)[order].reshape(CAPACITY * SEG, N_ENVS, -1) for v in leaves
    )
    return obs, action, reward[..., 0], done[..., 0]


def test_each_column_continues_its_own_episodes_and_commands() -> None:
    obs, _, reward, done = _stream(_buffer())
    ended = done[:-1] > 0
    assert (ended[:, 0] != ended[:, 1]).any(), "the columns' ends coincide"
    assert ended.sum(0).min() >= 3, ended.sum(0)
    # (i)
    nxt, now = obs[1:, :, :2], obs[:-1, :, :2]
    np.testing.assert_array_equal(nxt[~ended], now[~ended] + [1.0, 0.0])
    np.testing.assert_array_equal(nxt[ended][:, 0], 0.0)
    # (ii)
    d_r, d_h = obs[..., 2], obs[..., 3]
    want_r = np.where(ended, R0, d_r[:-1] - reward[:-1])
    want_h = np.where(ended, H0, np.maximum(d_h[:-1] - 1.0, 1.0))
    np.testing.assert_allclose(d_r[1:], want_r, atol=1e-5)
    np.testing.assert_array_equal(d_h[1:], want_h)


def _indexed(obs: np.ndarray) -> list:
    """((stream row, column), [t, c]) of every stored row; [t, c] is unique."""
    pairs = [
        ((k, b), tuple(obs[k, b, :2])) for k in range(len(obs)) for b in range(N_ENVS)
    ]
    assert len({p for _, p in pairs}) == len(pairs), "two rows share [t, c]"
    return pairs


def test_each_sampled_command_is_its_rows_hindsight_in_its_own_column() -> None:
    buffer = _buffer()
    obs, action, reward, done = _stream(buffer)
    batch_obs, batch_action = sample_training_batch(
        jax.random.PRNGKey(1), buffer, episode_stream(buffer), 512
    )
    rows = {(float(t), float(c)): (k, b) for (k, b), (t, c) in _indexed(obs)}
    columns = set()
    for (t, c, d_r, d_h), a in zip(
        np.asarray(batch_obs), np.asarray(batch_action), strict=True
    ):
        k, b = rows[(float(t), float(c))]
        n = int(d_h)
        assert n == d_h >= 1 and k + n <= len(done), (k, d_h)
        assert not done[k : k + n - 1, b].any(), "the command spans an episode end"
        np.testing.assert_allclose(d_r, reward[k : k + n, b].sum(), atol=1e-4)
        np.testing.assert_array_equal(a, action[k, b])
        columns.add(b)
    assert columns == {0, 1}
