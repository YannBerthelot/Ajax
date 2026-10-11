"""UDRL's segment buffer read as one stream per env: episodes span slots."""

import jax.numpy as jnp
import numpy as np

from ajax.agents.UDRL.buffer import (
    add_segment,
    episode_stream,
    init_buffer,
    topk_command_stats,
)

T = 4  # steps per segment


def _buffer(capacity: int, segments: int, dones_at: tuple[int, ...]):
    """``segments`` segments of one env; step ``g`` of the run has observation
    ``g`` (then the command), action ``g`` and reward 1, and ends an episode
    where ``g`` is in ``dones_at``."""
    buffer = init_buffer(capacity, T, n_envs=1, obs_dim=3, action_dim=1)
    for first in range(0, segments * T, T):
        g = jnp.arange(first, first + T, dtype=jnp.float32)[:, None, None]
        obs = jnp.concatenate([g, jnp.zeros((T, 1, 2))], axis=-1)
        dones = jnp.isin(g, jnp.asarray(dones_at)).astype(jnp.float32)
        buffer = add_segment(buffer, obs, g, jnp.ones_like(g), dones)
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
