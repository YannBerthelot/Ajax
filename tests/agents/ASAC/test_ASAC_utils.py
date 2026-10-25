"""ASAC's episode-termination penalty on hand-computed values: the running
penalty moves by ``tau`` towards ``p_0`` times the largest reward
``gather(rewards, 0, terminated)`` reads (row 0's for a step that did not
terminate, row 1's for one that did, as the authors' code), and a
terminated step's reward pays it."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.ASAC.ASAC import (
    compute_episode_termination_penalty,
    get_episode_termination_penalized_rewards,
)

R = [1.0, 2.0, 3.0]
NONE = [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "rewards, terminated, p_0, tau, expected",
    [
        (R, NONE, 0.1, 0.5, 0.5 * 0.5 + 0.5 * 0.1 * 1.0),  # row 0's reward
        (R, NONE, 0.1, 0.0, 0.5),  # tau = 0 keeps the penalty
        (R, NONE, 0.2, 1.0, 0.2 * 1.0),  # tau = 1 jumps to p_0 * r[0]
        # a terminated row reads row 1's reward: max(r[1], r[0], r[1])
        (R, [1.0, 0.0, 1.0], 0.1, 0.5, 0.5 * 0.5 + 0.5 * 0.1 * 2.0),
        # not the largest non-terminal reward (5), nor their mean (4)
        ([3.0, -1.0, 5.0], [0.0, 1.0, 0.0], 1.0, 1.0, 3.0),
        ([3.0, -1.0, 5.0], [1.0, 1.0, 1.0], 1.0, 1.0, -1.0),  # all terminate
        # a batch of minibatches: rows of the flattened batch, a scalar
        ([[2.0, 4.0], [6.0, 8.0]], [[0.0, 1.0], [0.0, 0.0]], 0.1, 0.3, 0.35 + 0.12),
    ],
    ids=["basic", "tau-zero", "tau-one", "terminated", "rows", "all-end", "batched"],
)
def test_the_penalty_moves_towards_p0_times_the_rows_the_original_reads(
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
