"""The house training return: each env's rolling window of episode returns
(init_rolling_mean, update_episodic_return), for two envs."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.environments.interaction import init_rolling_mean, update_episodic_return


def test_init_rolling_mean():
    state = init_rolling_mean(
        window_size=5,
        cumulative_reward=jnp.zeros((3, 1)),
        last_return=jnp.zeros((3, 1)),
    )
    assert state.buffer.shape == (5, 3, 1)
    for x in (state.buffer, state.index, state.count, state.sum):
        assert jnp.all(x == 0)


def _episodes(*returns):
    """Steps ending one episode per env with each of ``returns``."""
    return [(r, [True, True]) for r in returns]


# window, (reward, done) steps -> buffer, index, count, sum, mean over envs.
CASES = {
    "one episode": (
        5,
        [([1.0, 2.0], [False, False]), ([9.0, 18.0], [True, True])],
        [[10, 20], [0, 0], [0, 0], [0, 0], [0, 0]],
        [1, 1],
        [1, 1],
        [10, 20],
        15.0,
    ),
    "partial window": (
        5,
        _episodes([10.0, 20.0], [30.0, 40.0]),
        [[10, 20], [30, 40], [0, 0], [0, 0], [0, 0]],
        [2, 2],
        [2, 2],
        [40, 60],
        25.0,
    ),
    "wrap-around": (
        3,
        _episodes([10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]),
        [[70, 80], [30, 40], [50, 60]],
        [1, 1],
        [3, 3],
        [150, 180],
        55.0,
    ),
    # An env whose episode runs on keeps its window as it is.
    "envs ending apart": (
        3,
        [([10.0, 20.0], [True, False]), ([3.0, 5.0], [False, True])],
        [[10, 25], [0, 0], [0, 0]],
        [1, 1],
        [1, 1],
        [10, 25],
        17.5,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_update_episodic_return(name):
    window, steps, buffer, index, count, total, mean = CASES[name]
    state = init_rolling_mean(
        window_size=window,
        cumulative_reward=jnp.zeros((2, 1)),
        last_return=jnp.nan * jnp.zeros((2, 1)),
    )
    for reward, done in steps:
        state, got = update_episodic_return(state, jnp.array(reward), jnp.array(done))
    np.testing.assert_array_equal(state.buffer[..., 0], buffer)
    np.testing.assert_array_equal(state.index[:, 0], index)
    np.testing.assert_array_equal(state.count[:, 0], count)
    np.testing.assert_array_equal(state.sum[:, 0], total)
    assert float(got) == mean


def test_an_env_without_a_finished_episode_has_no_mean_yet():
    state = init_rolling_mean(
        window_size=3,
        cumulative_reward=jnp.zeros((2, 1)),
        last_return=jnp.nan * jnp.zeros((2, 1)),
    )
    state, mean = update_episodic_return(
        state, jnp.array([10.0, 20.0]), jnp.array([True, False])
    )
    assert jnp.isnan(mean)
    np.testing.assert_array_equal(state.cumulative_reward[:, 0], [0.0, 20.0])
