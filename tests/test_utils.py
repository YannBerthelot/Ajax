import jax.numpy as jnp
from jax import random

from ajax.utils import online_normalize


def test_online_normalize_chains_welford_updates():
    """Three single-sample batches, each update's statistics fed to the next:
    ``count``, ``mean``, ``M2``, the variance ``M2 / count`` and the output
    ``(x - mean) / std`` after each."""
    count, mean, mean_2 = 0, jnp.zeros((1, 3)), jnp.zeros((1, 3))
    expected = (
        # Single-sample first batch: Welford's M2 is correctly 0 (no
        # within-batch variance to accumulate). The previous implementation
        # patched M2 to 1 via ``replace_zeros_with_ones``, which silently
        # corrupted running variance for any feature with a zero-variance
        # batch. Removed to match brax acme.running_statistics; std is now
        # clipped to [1e-6, 1e6] (max(sqrt(0 + eps), 1e-6) = 1e-4 here), so
        # the normalised output (0 / std = 0) stays finite without
        # falsifying the running stat.
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0, 0.0),
        ([4.0, 5.0, 6.0], [2.5, 3.5, 4.5], 4.5, 2.25),
        ([7.0, 8.0, 9.0], [4.0, 5.0, 6.0], 18.0, 6.0),
    )
    for step, (x, new_mean, new_mean_2, variance) in enumerate(expected, 1):
        x = jnp.array([x])
        x_norm, count, mean, mean_2, sigma = online_normalize(x, count, mean, mean_2)
        assert count == step
        assert jnp.allclose(mean, jnp.array([new_mean]))
        assert jnp.allclose(mean_2, new_mean_2)
        assert jnp.allclose(sigma, variance)
        std = jnp.sqrt(variance) if variance else 1.0
        assert jnp.allclose(x_norm, (x - mean) / std)


def test_online_normalize_keeps_raw_columns():
    """The leading raw columns come out unchanged, their statistics pinned
    at mean 0, variance 1; the others normalise as without them."""
    x = random.normal(random.PRNGKey(0), (8, 4)) * 5.0 + 3.0
    zeros = jnp.zeros((1, 4))
    plain = online_normalize(x, 0, zeros, zeros)
    kept = online_normalize(x, 0, zeros, zeros, raw_columns=2)
    assert jnp.array_equal(kept[0][:, :2], x[:, :2])
    assert jnp.array_equal(kept[0][:, 2:], plain[0][:, 2:])
    assert jnp.all(kept[2][:, :2] == 0.0) and jnp.all(kept[4][:, :2] == 1.0)
    frozen = online_normalize(x, *kept[1:4], train=False, raw_columns=2)
    assert jnp.array_equal(frozen[0][:, :2], x[:, :2])
