import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(
    jax.jit,
    static_argnames=[
        "train",
        "eps",
    ],
)
def online_normalize(
    x: jnp.array,
    count: int,
    mean: float,
    mean_2: float,
    eps: float = 1e-8,
    train: bool = True,
) -> tuple[jnp.array, int, float, float, float]:
    if train:
        batch_size = x.shape[0]
        batch_mean = jnp.nanmean(
            x, axis=0
        )  # (D,) nanmean to handle possible means in G_return, as expected for AVG
        batch_mean_2 = jnp.nanmean((x - batch_mean) ** 2, axis=0)  # (D,)

        total_count = count + batch_size
        delta = batch_mean - mean
        mean = mean + delta * batch_size / total_count
        mean_2 = (
            mean_2
            + batch_mean_2 * batch_size
            + (delta**2) * count * batch_size / total_count
        )
        count = total_count

    variance = mean_2 / count
    std = jnp.sqrt(variance + eps)
    x_norm = (x - mean) / std
    return x_norm, count, mean, mean_2, variance
