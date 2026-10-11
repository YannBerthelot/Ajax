"""ASAC's episode-termination penalty on hand-computed values: the running
penalty moves by ``tau`` towards ``p_0`` times the mean reward of the steps
that did not terminate, and a terminated step's reward pays it."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.ASAC.utils import (
    compute_episode_termination_penalty,
    get_episode_termination_penalized_rewards,
)

R = [1.0, 2.0, 3.0]
NONE = [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "rewards, terminated, p_0, tau, expected",
    [
        (R, NONE, 0.1, 0.5, 0.5 * 0.5 + 0.5 * 0.1 * 2.0),
        (R, NONE, 0.1, 0.0, 0.5),  # tau = 0 keeps the penalty
        (R, NONE, 0.2, 1.0, 0.2 * 2.0),  # tau = 1 jumps to p_0 * mean
        # a terminated step's reward leaves the mean (still over 3 steps)
        (R, [1.0, 0.0, 1.0], 0.1, 0.5, 0.5 * 0.5 + 0.5 * 0.1 * 2.0 / 3),
        # a batch of minibatches: the mean over all of them, a scalar
        ([R, [2.0, 4.0, 6.0]], [NONE, NONE], 0.1, 0.3, 0.5 * 0.7 + 0.3 * 0.1 * 3.0),
    ],
    ids=["basic", "tau-zero", "tau-one", "terminated", "batched"],
)
def test_the_penalty_moves_towards_p0_times_the_mean_reward(
    rewards, terminated, p_0, tau, expected
):
    penalty = compute_episode_termination_penalty(
        jnp.array(0.5), jnp.array(rewards), jnp.array(terminated), p_0, tau
    )
    assert penalty.shape == ()
    np.testing.assert_allclose(penalty, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "penalty, rewards, terminated, expected",
    [
        (0.5, R, NONE, R),
        (0.5, R, [1.0, 0.0, 1.0], [0.5, 2.0, 2.5]),
        (
            0.2,
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.8, 2.0], [3.0, 3.8]],
        ),
    ],
    ids=["no-termination", "termination", "2d"],
)
def test_a_terminated_step_pays_the_penalty(penalty, rewards, terminated, expected):
    rewards = jnp.array(rewards)
    out = get_episode_termination_penalized_rewards(
        jnp.array(penalty), rewards, jnp.array(terminated)
    )
    assert out.shape == rewards.shape and out.dtype == rewards.dtype
    np.testing.assert_allclose(out, expected, rtol=1e-6)
