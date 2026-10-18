import jax
import jax.numpy as jnp

from ajax.agents.AVG.state import RunningMoments


def _fold(moments: RunningMoments, x: jax.Array) -> RunningMoments:
    """Fold the entries of ``x`` (one per env) into ``moments``' running
    count, mean and sum of squared deviations (Chan et al.'s parallel
    update). A NaN entry is absent: it neither moves the mean nor counts."""
    valid = ~jnp.isnan(x)
    k = jnp.sum(valid)
    x = jnp.where(valid, x, 0.0)
    batch_mean = jnp.sum(x) / jnp.maximum(k, 1)
    batch_m2 = jnp.sum(jnp.where(valid, jnp.square(x - batch_mean), 0.0))
    total = moments.count + k
    share = k / jnp.maximum(total, 1)
    delta = batch_mean - moments.mean
    return moments.replace(
        count=total,
        mean=moments.mean + delta * share,
        mean_2=moments.mean_2 + batch_m2 + jnp.square(delta) * moments.count * share,
    )


def _variance(moments: RunningMoments) -> jax.Array:
    """The population variance, 0 before any value."""
    return moments.mean_2 / jnp.maximum(moments.count, 1)


def compute_td_error_scaling(
    reward: RunningMoments,
    gamma: RunningMoments,
    G_return: RunningMoments,
) -> tuple[jax.Array, RunningMoments, RunningMoments, RunningMoments]:
    """AVG's TD-error scale, the authors' ``TDErrorScaler``
    (gauthamvasan/avg, ``incremental_rl/td_error_scaler.py``, after Schaul
    et al. 2021): ``sqrt(max(var(r) + E[G^2] var(gamma), 1e-4))``, or 1
    while that variance is at most 0.01 and no return has been seen. The
    reward, discount and squared-return statistics are updated first.
    ``G_return.value`` is NaN unless an episode just ended: only the
    returns of the episodes that ended count, however many envs run."""
    reward = _fold(reward, reward.value)
    gamma = _fold(gamma, gamma.value)
    G_return = _fold(G_return, jnp.square(G_return.value))
    variance = jnp.maximum(_variance(reward) + G_return.mean * _variance(gamma), 1e-4)
    unscaled = (variance <= 0.01) & (G_return.count == 0)
    scaling = jnp.where(unscaled, jnp.ones_like(variance), jnp.sqrt(variance))
    return scaling, reward, gamma, G_return
