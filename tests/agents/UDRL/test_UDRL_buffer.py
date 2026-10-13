"""UDRL's segment buffer read as one stream per env: episodes span slots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.UDRL.buffer import (
    add_segment,
    episode_stream,
    init_buffer,
    sample_training_batch,
    topk_command_stats,
)

T = 4  # steps per segment


def _buffer(
    capacity: int, segments: int, dones_at: tuple[int, ...], reward_is_step=False
):
    """``segments`` segments of one env; step ``g`` of the run has observation
    ``g`` (then the command), action ``g`` and reward 1 (``g`` with
    ``reward_is_step``), and ends an episode where ``g`` is in ``dones_at``."""
    buffer = init_buffer(capacity, T, n_envs=1, obs_dim=3, action_dim=1)
    for first in range(0, segments * T, T):
        g = jnp.arange(first, first + T, dtype=jnp.float32)[:, None, None]
        obs = jnp.concatenate([g, jnp.zeros((T, 1, 2))], axis=-1)
        dones = jnp.isin(g, jnp.asarray(dones_at)).astype(jnp.float32)
        reward = g if reward_is_step else jnp.ones_like(g)
        buffer = add_segment(buffer, obs, g, reward, dones)
    return buffer


def test_topk_counts_an_episode_spanning_segments_whole():
    """Three segments of 4 steps, one done at step 9: the one whole episode
    returns 10 in 10 steps (not the last segment's fragment of 2)."""
    mean_r, std_r, mean_h = topk_command_stats(episode_stream(_buffer(4, 3, (9,))), k=4)
    np.testing.assert_allclose((mean_r, std_r, mean_h), (10.0, 0.0, 10.0))


def test_topk_skips_the_fragment_whose_start_was_evicted():
    """Five segments into three slots (steps 8 .. 19 held), dones at 9 and
    15: the episode ending at 9 lost its start; the one of steps 10 .. 15
    is the only whole one."""
    stream = episode_stream(_buffer(3, 5, (9, 15)))
    np.testing.assert_array_equal(stream.order, [2, 0, 1])
    mean_r, _, mean_h = topk_command_stats(stream, k=4)
    np.testing.assert_allclose((mean_r, mean_h), (6.0, 6.0))


@pytest.mark.parametrize(
    "capacity, segments, dones_at, held, episodes",
    [
        # one episode over three segments, then a running one
        (4, 3, (9,), range(12), ((0, 9), (10, 11))),
        # five segments into three slots: steps 8 .. 19, oldest first
        (3, 5, (9, 15), range(8, 20), ((8, 9), (10, 15), (16, 19))),
    ],
    ids=["spanning", "wrapped"],
)
def test_the_sampler_draws_intervals_inside_episodes_across_segments(
    capacity, segments, dones_at, held, episodes
):
    """Every held step starts intervals, uniformly; an interval ends at
    most on its episode's last step, wherever the segments split it, and
    every interval of every episode is drawn; ``dr`` sums its rewards
    (reward = step), ``dh`` counts its steps, and they replace the
    command of the observation at its start."""
    buffer = _buffer(capacity, segments, dones_at, reward_is_step=True)
    stream = episode_stream(buffer)
    obs, action = jax.jit(sample_training_batch, static_argnums=3)(
        jax.random.PRNGKey(0), buffer, stream, 6000
    )
    t1, dr, dh = np.asarray(obs).T  # the command is (dr, dh)
    dh = dh.astype(int)
    t2 = t1 + dh  # exclusive
    np.testing.assert_array_equal(np.asarray(action[:, 0]), t1)
    np.testing.assert_allclose(dr, (t1 + t2 - 1) * dh / 2)  # sum of t1 .. t2 - 1
    counts = np.bincount(t1.astype(int) - held[0], minlength=len(held))
    assert len(counts) == len(held) and counts.min() > 6000 / len(held) * 0.8
    drawn = set(zip(t1.astype(int), t2.astype(int), strict=True))
    allowed = {
        (a, b)
        for first, last in episodes
        for a in range(first, last + 1)
        for b in range(a + 1, last + 2)
    }
    assert drawn == allowed


def _serial_episode_stream(buffer):
    """``episode_stream`` as one serial scan, before its associative scans:
    the oracle they must match up to the rounding of the returns' sums."""
    B, T, n_envs = buffer.rewards.shape[:3]
    order = (buffer.write_idx - buffer.fill_count + jnp.arange(B)) % B
    filled = (jnp.arange(B) < buffer.fill_count).astype(jnp.float32)
    rewards = buffer.rewards[order].reshape(B * T, n_envs, 1)
    dones = buffer.dones[order] * filled[:, None, None, None]
    dones = dones.reshape(B * T, n_envs, 1)

    def body(carry, x):
        running_r, running_h, whole = carry
        r, d = x
        running_r, running_h = running_r + r, running_h + 1.0
        out, keep = (running_r, running_h, d * whole), 1.0 - d
        return (running_r * keep, running_h * keep, jnp.maximum(whole, d)), out

    zeros = jnp.zeros_like(rewards[0])
    init = (zeros, zeros, buffer.episode_start[order[0]])
    _, outs = jax.lax.scan(body, init, (rewards, dones))
    steps, length = jnp.arange(B * T)[:, None], buffer.fill_count * T
    last = jnp.where(dones[..., 0] > 0, steps, length - 1)
    return (*(o[..., 0] for o in outs), jax.lax.cummin(last, axis=0, reverse=True))


@pytest.mark.parametrize("segments", [5, 11], ids=["filling", "wrapped"])
def test_the_stream_matches_the_serial_scan(segments):
    """Pendulum-like rewards in [-16, 0], 3 envs, dones at rate 1/20 except
    in env 0 (one episode over the whole stream): the returns are the
    serial scan's within 8 float32 eps, the rest is equal."""
    B, T, n_envs = 8, 16, 3
    buffer = init_buffer(B, T, n_envs, obs_dim=3, action_dim=1)
    for key in jax.random.split(jax.random.PRNGKey(0), segments):
        k_r, k_d = jax.random.split(key)
        rewards = jax.random.uniform(k_r, (T, n_envs, 1), minval=-16.0, maxval=0.0)
        dones = (jax.random.uniform(k_d, (T, n_envs, 1)) < 0.05).at[:, 0].set(False)
        obs = jnp.zeros((T, n_envs, 3))
        buffer = add_segment(buffer, obs, obs[..., :1], rewards, dones * 1.0)
    stream = jax.jit(episode_stream)(buffer)
    returns, *exact = jax.jit(_serial_episode_stream)(buffer)
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(stream.returns, returns, rtol=8 * eps)
    for new, old in zip(
        (stream.horizons, stream.ends_whole, stream.last), exact, strict=True
    ):
        np.testing.assert_array_equal(new, old)
