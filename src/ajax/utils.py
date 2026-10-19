from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(
    jax.jit,
    static_argnames=["train", "eps", "shift", "nan_safe", "raw_columns"],
)
def online_normalize(
    x: jax.Array,
    count: jax.Array,
    mean: jax.Array,
    mean_2: jax.Array,
    eps: float = 1e-8,
    train: bool = True,
    shift: bool = True,
    returns: Optional[jax.Array] = None,
    nan_safe: bool = True,
    raw_columns: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Welford-style running mean / var update.

    ``nan_safe`` (static): when True (default), the means skip NaN
    entries via ``jnp.nanmean``; the count still counts every row, so NaN
    is no way to leave a row out (AVG's TD-error scale counts its own,
    :func:`ajax.agents.AVG.utils.compute_td_error_scaling`). Set to False
    at call sites that pass guaranteed-clean data (e.g. the agent-side obs
    normalizer) to drop the per-reduction mask cost.

    ``raw_columns`` (static): the leading feature columns an env declares
    raw (:func:`ajax.environments.utils.raw_obs_columns`) come out
    unchanged, and their statistics are pinned at mean 0, variance 1 so
    every other reader of them (``apply_obs_norm``) leaves them raw too.
    """
    input_x = x
    _mean = jnp.nanmean if nan_safe else jnp.mean

    if train:
        x = x if returns is None else returns
        x = x.reshape(1, -1) if len(x.shape) < 2 else x

        batch_size = x.shape[0]
        batch_mean = _mean(x, axis=0, keepdims=True)
        batch_mean_2 = _mean((x - batch_mean) ** 2, axis=0, keepdims=True)

        total_count = count + batch_size

        delta = batch_mean - mean
        mean = mean + delta * batch_size / total_count
        mean_2 = (
            mean_2
            + batch_mean_2 * batch_size
            + (delta**2) * count * batch_size / total_count
        )
        count = total_count

    if raw_columns:
        raw = jnp.arange(jnp.shape(mean_2)[-1]) < raw_columns
        mean = jnp.where(raw, 0.0, mean)
        mean_2 = jnp.where(raw, count, mean_2)
    variance = mean_2 / count
    std = jnp.sqrt(variance + eps)
    # Match brax acme.running_statistics: clip std to [1e-6, 1e6] so
    # zero-variance features at init don't blow up the normalized obs.
    std = jnp.clip(std, 1e-6, 1e6)
    x_norm = (input_x - _mean(mean, axis=0) * shift) / _mean(std, axis=0)

    x_norm = x_norm.reshape(input_x.shape)
    if raw_columns:
        # Set, not computed: a compiler free to fuse the division by the
        # averaged std may otherwise move them by an ulp.
        x_norm = x_norm.at[..., :raw_columns].set(input_x[..., :raw_columns])
    assert x_norm.shape == input_x.shape, (
        f"x_norm shape {x_norm.shape} does not match input_x shape {input_x.shape}"
    )

    return (
        x_norm,
        count,
        mean,
        mean_2,
        variance,
    )


def online_denormalize(
    x_norm: jax.Array, count: Any, mean: Any, mean_2: Any, eps: float = 1e-8
) -> jax.Array:
    """``online_normalize``'s input from its output and the statistics it
    returned (rows along the second-to-last axis, as it averages them).
    Raw columns, pinned at mean 0 and variance 1, return unchanged to
    within ``eps``."""
    std = jnp.clip(jnp.sqrt(mean_2 / count + eps), 1e-6, 1e6)
    return x_norm * jnp.mean(std, axis=-2, keepdims=True) + jnp.mean(
        mean, axis=-2, keepdims=True
    )
