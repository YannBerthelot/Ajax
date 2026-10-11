"""AVG's TD-error scaling folds the reward, the discount and the squared
return into running means (one row per env), on hand-computed values."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.AVG.utils import (
    NormalizationInfo,
    compute_td_error_scaling,
)


def _info(value, count=0, mean=0.0) -> NormalizationInfo:
    """Running statistics of ``value`` (a column per env) after ``count``
    values of mean ``mean``."""
    value = jnp.array(value)
    return NormalizationInfo(
        value=value,
        count=jnp.full(value.shape, count),
        mean=jnp.full(value.shape, mean),
        mean_2=jnp.zeros_like(value),
    )


@pytest.mark.parametrize(
    "reward, gamma, G_return, count, means, unit_scaling",
    [
        # the first values: their own means, and a scale of 1 until a second
        # return arrives (the return's is squared)
        (_info([[1.0]]), _info([[0.99]]), _info([[10.0]]), 1, (1.0, 0.99, 100.0), True),
        (
            _info([[2.0]], 1, 1.0),
            _info([[0.98]], 1, 0.99),
            _info([[20.0]], 1, 10.0),
            2,
            (1.5, 0.985, (10.0 + 20.0**2) / 2),
            None,
        ),
        # a batch of two values a statistic: both fold in, the scale is live
        (
            _info([[1.0], [2.0]]),
            _info([[0.99], [0.95]]),
            _info([[10.0], [20.0]]),
            2,
            (1.5, 0.97, (10.0**2 + 20.0**2) / 2),
            False,
        ),
        (
            _info([[2.0], [4.0]], 1, 1.0),
            _info([[0.98], [0.96]], 1, 0.99),
            _info([[20.0], [30.0]], 1, 10.0),
            3,
            (7 / 3, (0.99 + 0.98 + 0.96) / 3, (10 + 20**2 + 30**2) / 3),
            None,
        ),
    ],
    ids=["initial", "update", "initial-batched", "update-batched"],
)
def test_compute_td_error_scaling(reward, gamma, G_return, count, means, unit_scaling):
    scaling, *updated = compute_td_error_scaling(reward, gamma, G_return)
    for info, mean in zip(updated, means, strict=True):
        assert jnp.all(info.count == count)
        np.testing.assert_allclose(
            info.mean, jnp.full(info.value.shape, mean), rtol=1e-6
        )
    if unit_scaling is not None:
        assert bool(jnp.allclose(scaling, 1.0)) == unit_scaling
