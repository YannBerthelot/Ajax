"""AVG's TD-error scale is the authors' ``TDErrorScaler``
(gauthamvasan/avg, ``incremental_rl/td_error_scaler.py``): ``sigma =
sqrt(max(var(r) + E[G^2] var(gamma), 1e-4))``, or 1 while that variance is
at most 0.01 and no return has been seen. Checked on hand-computed values,
one row per env, a NaN return marking an env whose episode did not end."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.AVG.state import RunningMoments
from ajax.agents.AVG.utils import compute_td_error_scaling

NAN = float("nan")


def _moments(value, count=0.0, mean=0.0, mean_2=0.0) -> RunningMoments:
    """A statistic's latest ``value`` (a row per env) and its running count,
    mean and sum of squared deviations, shaped as AVG's state holds them."""
    return RunningMoments(
        value=jnp.array(value),
        count=jnp.full((1,), count),
        mean=jnp.full((1, 1), mean),
        mean_2=jnp.full((1, 1), mean_2),
    )


@pytest.mark.parametrize(
    "reward, gamma, G_return, scale, G_count",
    [
        # No return yet and a variance under 0.01: unscaled.
        ([[1.0], [1.05]], [[0.99], [0.99]], [[NAN], [NAN]], 1.0, 0),
        # No return yet, but var(r) = 4 > 0.01: scaled by 2, as the authors'
        # code does on a task without terminations.
        ([[0.0], [4.0]], [[0.99], [0.99]], [[NAN], [NAN]], 2.0, 0),
        # One of two envs ended with G = 3: one return, E[G^2] = 9, and the
        # discount's variance (0.99 / 2)^2 enters.
        ([[1.0], [1.0]], [[0.0], [0.99]], [[3.0], [NAN]], 3 * 0.99 / 2, 1),
        # A return seen and no variance: the 1e-4 floor, sigma = 0.01.
        ([[1.0]], [[0.0]], [[2.0]], 0.01, 1),
    ],
    ids=["unscaled", "no-return-scaled", "one-return-of-two-envs", "floor"],
)
def test_the_scale_is_the_authors_td_error_scaler(
    reward, gamma, G_return, scale, G_count
):
    scaling, _, _, G = compute_td_error_scaling(
        _moments(reward), _moments(gamma), _moments(G_return)
    )
    np.testing.assert_allclose(scaling, [[scale]], rtol=1e-5)
    np.testing.assert_array_equal(G.count, [G_count])


def test_the_statistics_fold_each_value_in_once():
    """Chan et al.'s parallel update: (0, 2) then 4 give mean 2 and a sum of
    squared deviations 8; a NaN return leaves the return's statistics as
    they were."""
    reward = _moments([[4.0]], count=2.0, mean=1.0, mean_2=2.0)
    G = _moments([[NAN]], count=2.0, mean=5.0, mean_2=3.0)
    _, reward, gamma, G_after = compute_td_error_scaling(reward, _moments([[0.5]]), G)
    np.testing.assert_allclose(reward.count, [3.0])
    np.testing.assert_allclose(reward.mean, [[2.0]])
    np.testing.assert_allclose(reward.mean_2, [[8.0]])
    np.testing.assert_allclose(gamma.count, [1.0])
    np.testing.assert_allclose(gamma.mean, [[0.5]])
    for before, after in zip(
        (G.count, G.mean, G.mean_2),
        (G_after.count, G_after.mean, G_after.mean_2),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
