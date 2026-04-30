"""Unit tests for UDRL helpers: returns-to-go + horizons and command updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.UDRL.utils import (
    compute_returns_to_go_horizons,
    update_command,
)


def test_compute_returns_to_go_horizons_no_done():
    """No episode terminations: RTG accumulates to end of segment, horizon counts down."""
    rewards = jnp.array([[1.0], [2.0], [3.0]])  # (T=3, n_envs=1)
    dones = jnp.zeros_like(rewards)
    rtg, horizon = compute_returns_to_go_horizons(rewards, dones, gamma=1.0)
    np.testing.assert_allclose(np.array(rtg).flatten(), [6.0, 5.0, 3.0])
    np.testing.assert_allclose(np.array(horizon).flatten(), [3.0, 2.0, 1.0])


def test_compute_returns_to_go_horizons_with_done():
    """Termination at t=1: RTG and horizon at t=0 cover only [r0, r1]; r2 belongs to next ep."""
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    dones = jnp.array([[0.0], [1.0], [0.0]])
    rtg, horizon = compute_returns_to_go_horizons(rewards, dones, gamma=1.0)
    np.testing.assert_allclose(np.array(rtg).flatten(), [3.0, 2.0, 3.0])
    np.testing.assert_allclose(np.array(horizon).flatten(), [2.0, 1.0, 1.0])


def test_compute_returns_to_go_horizons_discount():
    """Gamma < 1.0 discounts future rewards correctly within a segment."""
    rewards = jnp.array([[1.0], [1.0], [1.0]])
    dones = jnp.zeros_like(rewards)
    rtg, _ = compute_returns_to_go_horizons(rewards, dones, gamma=0.5)
    expected = [1.0 + 0.5 + 0.25, 1.0 + 0.5, 1.0]
    np.testing.assert_allclose(np.array(rtg).flatten(), expected)


def test_compute_returns_to_go_horizons_multi_env():
    """Two parallel segments with different termination structures."""
    rewards = jnp.array([[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]])  # (T, n_envs, 1)
    dones = jnp.array([[[0.0], [1.0]], [[0.0], [0.0]], [[0.0], [0.0]]])
    rtg, horizon = compute_returns_to_go_horizons(rewards, dones, gamma=1.0)
    # env 0: no done, full sum
    np.testing.assert_allclose(np.array(rtg[:, 0, 0]), [6.0, 5.0, 3.0])
    np.testing.assert_allclose(np.array(horizon[:, 0, 0]), [3.0, 2.0, 1.0])
    # env 1: done at t=0 -> only first reward; subsequent rtg restart
    np.testing.assert_allclose(np.array(rtg[:, 1, 0]), [10.0, 50.0, 30.0])
    np.testing.assert_allclose(np.array(horizon[:, 1, 0]), [1.0, 2.0, 1.0])


def test_update_command_decay():
    """Within an episode: d_r decreases by reward, d_h decreases by 1."""
    prev_d_r = jnp.array([10.0, 5.0])
    prev_d_h = jnp.array([20.0, 10.0])
    reward = jnp.array([1.0, 2.0])
    done = jnp.array([0.0, 0.0])
    new_d_r, new_d_h = update_command(
        prev_d_r, prev_d_h, reward, done,
        return_init=100.0, horizon_init=50.0,
    )
    np.testing.assert_allclose(np.array(new_d_r), [9.0, 3.0])
    np.testing.assert_allclose(np.array(new_d_h), [19.0, 9.0])


def test_update_command_reset_on_done():
    """On done, command resets to init values regardless of previous state."""
    prev_d_r = jnp.array([10.0, 5.0])
    prev_d_h = jnp.array([20.0, 10.0])
    reward = jnp.array([1.0, 2.0])
    done = jnp.array([1.0, 0.0])
    new_d_r, new_d_h = update_command(
        prev_d_r, prev_d_h, reward, done,
        return_init=100.0, horizon_init=50.0,
    )
    np.testing.assert_allclose(np.array(new_d_r), [100.0, 3.0])
    np.testing.assert_allclose(np.array(new_d_h), [50.0, 9.0])


def test_update_command_horizon_floor():
    """d_h is floored at 1.0 even past expected horizon, to avoid 0/negative conditioning."""
    prev_d_r = jnp.array([1.0])
    prev_d_h = jnp.array([1.0])
    reward = jnp.array([0.0])
    done = jnp.array([0.0])
    _, new_d_h = update_command(
        prev_d_r, prev_d_h, reward, done,
        return_init=100.0, horizon_init=50.0,
    )
    np.testing.assert_allclose(np.array(new_d_h), [1.0])


def test_update_command_jax_pure():
    """update_command must jit cleanly (no Python branching)."""
    f = jax.jit(update_command, static_argnames=("return_init", "horizon_init"))
    prev_d_r = jnp.array([10.0])
    prev_d_h = jnp.array([20.0])
    reward = jnp.array([1.0])
    done = jnp.array([0.0])
    new_d_r, new_d_h = f(prev_d_r, prev_d_h, reward, done,
                         return_init=100.0, horizon_init=50.0)
    assert new_d_r.shape == (1,)
    assert new_d_h.shape == (1,)
