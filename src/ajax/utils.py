from dataclasses import fields
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.tree_util import Partial as partial


@partial(
    jax.jit,
    static_argnames=["train", "eps", "shift", "nan_safe"],
)
def online_normalize(
    x: jnp.array,
    count: int,
    mean: float,
    mean_2: float,
    eps: float = 1e-8,
    train: bool = True,
    shift: bool = True,
    returns: Optional[jax.Array] = None,
    nan_safe: bool = True,
) -> tuple[jnp.array, int, float, float, float]:
    """Welford-style running mean / var update.

    ``nan_safe`` (static): when True (default), reductions skip NaN
    entries via ``jnp.nanmean``. This is load-bearing for AVG, which
    feeds NaN-sentinel values for non-terminal transitions in
    ``G_return`` ([agents/AVG/utils.py:41-55](src/ajax/agents/AVG/utils.py#L41-L55)).
    Set to False at call sites that pass guaranteed-clean data (e.g. the
    agent-side obs normalizer) to drop the per-reduction mask cost.
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

    variance = mean_2 / count
    std = jnp.sqrt(variance + eps)
    # Match brax acme.running_statistics: clip std to [1e-6, 1e6] so
    # zero-variance features at init don't blow up the normalized obs.
    std = jnp.clip(std, 1e-6, 1e6)
    x_norm = (input_x - _mean(mean, axis=0) * shift) / _mean(std, axis=0)

    x_norm = x_norm.reshape(input_x.shape)  # Ensure output shape matches input shape
    assert (
        x_norm.shape == input_x.shape
    ), f"x_norm shape {x_norm.shape} does not match input_x shape {input_x.shape}"

    return (
        x_norm,
        count,
        mean,
        mean_2,
        variance,
    )


def fill_with_nan(dataclass):
    """
    Recursively fills all fields of a dataclass with jnp.nan.
    """
    nan = jnp.ones(1) * jnp.nan
    dict = {}
    for field in fields(dataclass):
        sub_dataclass = field.type
        if hasattr(
            sub_dataclass, "__dataclass_fields__"
        ):  # Check if the field is another dataclass
            dict[field.name] = fill_with_nan(sub_dataclass)
        else:
            dict[field.name] = nan
    return dataclass(**dict)


def compare_frozen_dicts(dict1: FrozenDict, dict2: FrozenDict) -> bool:
    """
    Compares two FrozenDicts to check if they are equal.

    Args:
        dict1 (FrozenDict): The first FrozenDict.
        dict2 (FrozenDict): The second FrozenDict.

    Returns:
        bool: True if the FrozenDicts are equal, False otherwise.
    """
    for key in dict1.keys():
        if key not in dict2:
            return False
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, FrozenDict) and isinstance(value2, FrozenDict):
            if not compare_frozen_dicts(value1, value2):
                return False
        elif not jnp.allclose(value1, value2):
            return False
    return True


def get_one(_: Any) -> float:
    return jnp.ones(1)
