"""Unit tests for UDRL helpers: returns-to-go + horizons and command updates,
on hand-computed values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.UDRL.utils import (
    compute_returns_to_go_horizons,
    update_command,
)

COLUMN = [[1.0], [2.0], [3.0]]  # (T=3, n_envs=1)


@pytest.mark.parametrize(
    "rewards, dones, gamma, rtg, horizon",
    [
        # no episode end: RTG accumulates to the end, the horizon counts down
        (COLUMN, [[0.0]] * 3, 1.0, [[6.0], [5.0], [3.0]], [[3.0], [2.0], [1.0]]),
        # an end at t=1: t=0 covers [r0, r1] only; r2 belongs to the next episode
        (
            COLUMN,
            [[0.0], [1.0], [0.0]],
            1.0,
            [[3.0], [2.0], [3.0]],
            [[2.0], [1.0], [1.0]],
        ),
        # gamma < 1 discounts within a segment, the horizon still counts steps
        ([[1.0]] * 3, [[0.0]] * 3, 0.5, [[1.75], [1.5], [1.0]], [[3.0], [2.0], [1.0]]),
        # (T, n_envs=2, 1): env 0 never ends, env 1 ends at t=0 and restarts
        (
            [[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]],
            [[[0.0], [1.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
            1.0,
            [[[6.0], [10.0]], [[5.0], [50.0]], [[3.0], [30.0]]],
            [[[3.0], [1.0]], [[2.0], [2.0]], [[1.0], [1.0]]],
        ),
    ],
    ids=["no-done", "done", "discount", "multi-env"],
)
def test_returns_to_go_and_horizons(rewards, dones, gamma, rtg, horizon):
    got_rtg, got_horizon = compute_returns_to_go_horizons(
        jnp.array(rewards), jnp.array(dones), gamma=gamma
    )
    np.testing.assert_allclose(got_rtg, rtg)
    np.testing.assert_allclose(got_horizon, horizon)
    assert got_rtg.shape == got_horizon.shape == np.shape(rewards)


@pytest.mark.parametrize(
    "d_r, d_h, reward, done, new_d_r, new_d_h",
    [
        # within an episode: d_r loses the reward, d_h one step
        ([10.0, 5.0], [20.0, 10.0], [1.0, 2.0], [0.0, 0.0], [9.0, 3.0], [19.0, 9.0]),
        # an episode end resets the command to its initial values
        ([10.0, 5.0], [20.0, 10.0], [1.0, 2.0], [1.0, 0.0], [100.0, 3.0], [50.0, 9.0]),
        # d_h floors at 1 past the expected horizon (no 0 or negative command)
        ([1.0], [1.0], [0.0], [0.0], [1.0], [1.0]),
    ],
    ids=["decay", "reset", "horizon-floor"],
)
def test_update_command(d_r, d_h, reward, done, new_d_r, new_d_h):
    """Eager and jitted alike: update_command has no Python branching."""
    jitted = jax.jit(update_command, static_argnames=("return_init", "horizon_init"))
    args = [jnp.array(x) for x in (d_r, d_h, reward, done)]
    for fn in (update_command, jitted):
        got_d_r, got_d_h = fn(*args, return_init=100.0, horizon_init=50.0)
        assert got_d_r.shape == got_d_h.shape == (len(d_r),)
        np.testing.assert_allclose(got_d_r, new_d_r)
        np.testing.assert_allclose(got_d_h, new_d_h)
