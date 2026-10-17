"""UDRL's segment buffer read as one stream per env: episodes span slots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.UDRL.buffer import (
    add_segment,
    episode_stream,
    init_buffer,
    sample_raw_observations,
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


def test_the_raw_observations_are_drawn_at_the_batch_rows():
    """A buffer keeping the env's own observations (here minus the stored
    ones, over two envs and five segments into three slots) returns, given
    the batch's key, those of the rows the training batch reads."""
    buffer = init_buffer(3, T, n_envs=2, obs_dim=3, action_dim=1, keep_raw_obs=True)
    for first in range(0, 5 * T, T):
        g = jnp.arange(first, first + T, dtype=jnp.float32)[:, None, None]
        g = g + jnp.asarray([0.0, 0.5])[None, :, None]  # (T, 2 envs, 1)
        obs = jnp.concatenate([g, jnp.zeros((T, 2, 2))], axis=-1)
        buffer = add_segment(buffer, obs, g, jnp.ones_like(g), 0 * g, raw_obs=-g)
    stream, key = episode_stream(buffer), jax.random.PRNGKey(0)
    obs, _ = sample_training_batch(key, buffer, stream, 64)
    raw = sample_raw_observations(key, buffer, stream, 64)
    np.testing.assert_array_equal(raw, -obs[:, :1])
    with pytest.raises(ValueError, match="keep_raw_obs"):
        add_segment(buffer, obs[:T, None], obs[:T, None, :1], 0 * g, 0 * g)
    with pytest.raises(ValueError, match="keep_raw_obs"):
        sample_raw_observations(key, _buffer(3, 1, ()), stream, 4)
