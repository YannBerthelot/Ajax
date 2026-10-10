import jax
import jax.numpy as jnp

from ajax.agents.AVG.state import NormalizationInfo
from ajax.utils import online_normalize


def _normalize_and_update(
    info: NormalizationInfo, square_value: bool
) -> tuple[NormalizationInfo, jnp.array]:
    """Fold ``info.value`` (squared when ``square_value``) into its running
    statistics; returns them and the running variance."""
    value = jnp.square(info.value) if square_value else info.value
    _, count, mean, mean_2, var = online_normalize(
        value, info.count, info.mean, info.mean_2
    )
    return info.replace(count=count, mean=mean, mean_2=mean_2), var


def compute_td_error_scaling(
    reward: NormalizationInfo,
    gamma: NormalizationInfo,
    G_return: NormalizationInfo,
) -> tuple[jnp.array, NormalizationInfo, NormalizationInfo, NormalizationInfo]:
    """AVG's TD-error scale ``sqrt(var(r) + E[G^2] var(gamma))`` (1 until two
    returns were seen), with the reward, discount and squared-return
    statistics updated; ``G_return.value`` is NaN unless an episode just
    ended, and then leaves its statistics as they were."""
    reward, variance_reward = _normalize_and_update(reward, square_value=False)
    gamma, variance_gamma = _normalize_and_update(gamma, square_value=False)
    updated_G_return, _ = _normalize_and_update(G_return, square_value=True)
    no_return = jnp.all(jnp.isnan(G_return.value))
    G_return = jax.tree.map(
        lambda old, new: jnp.where(no_return, old, new), G_return, updated_G_return
    )
    scaling = jnp.sqrt(variance_reward + G_return.mean * variance_gamma)
    td_error_scaling = jnp.where(G_return.count > 1, scaling, jnp.ones_like(scaling))
    return td_error_scaling, reward, gamma, G_return
