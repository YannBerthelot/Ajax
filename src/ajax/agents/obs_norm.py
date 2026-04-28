"""Agent-side running observation normalization.

Lives inside ``CollectorState.obs_norm_info`` so stats are part of
``agent_state`` and thread through naturally. Normalization is applied
at every actor/critic ``apply_fn`` boundary; stats are updated only at
collection time (online interaction and BC dataset prep). The replay
buffer stores RAW augmented observations so changing stats don't poison
old samples; we re-normalize at sample time.

Why agent-side and not env-wrapper-side: when ``augment_obs_with_expert_state``
is on, the ``expert_state`` (PID integrators etc.) is appended AFTER the
env step, in the collector. The env wrapper can't see those dims, so
its running stats don't cover them. Agent-side normalization wraps the
full augmented vector.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ajax.utils import online_normalize
from ajax.wrappers import NormalizationInfo, init_norm_info


def init_agent_obs_norm(n_envs: int, obs_dim: int) -> NormalizationInfo:
    """Initialise running stats for an obs vector of size ``obs_dim`` over
    ``n_envs`` parallel envs. Mirrors ``init_norm_info`` from the env
    wrapper, sized to the FULL augmented obs (env + expert_state)."""
    return init_norm_info(batch_size=n_envs, obs_shape=(obs_dim,))


def update_obs_norm(
    obs: jnp.ndarray, info: Optional[NormalizationInfo]
) -> Tuple[jnp.ndarray, Optional[NormalizationInfo]]:
    """Update running stats with ``obs`` and return (normalized_obs, new_info).

    ``obs`` shape: ``(n_envs, obs_dim)``. When ``info is None`` the call is
    a no-op (returns input unchanged) so call sites can stay unconditional.
    """
    if info is None:
        return obs, None
    new_obs, count, mean, mean_2, var = online_normalize(
        obs, info.count, info.mean, info.mean_2, train=True
    )
    new_info = NormalizationInfo(
        count=count, mean=mean, mean_2=mean_2, var=var,
        returns=info.returns,
    )
    return new_obs, new_info


def apply_obs_norm(
    obs: jnp.ndarray, info: Optional[NormalizationInfo]
) -> jnp.ndarray:
    """Normalize ``obs`` using existing stats (no update). Used at every
    actor/critic ``apply_fn`` site that consumes obs sampled from the
    buffer or carried by the eval scan, so stats updates stay localised
    to the collection sites. Falls back to raw obs while the running
    stats are still empty (count == 0), avoiding a divide-by-≈0 at
    step 1 before any update."""
    if info is None or info.var is None:
        return obs
    count_total = jnp.sum(info.count)
    mean = jnp.nanmean(info.mean, axis=0)
    std = jnp.sqrt(jnp.nanmean(info.var, axis=0) + 1e-8)
    normalized = (obs - mean) / std
    return jnp.where(count_total > 0, normalized, obs)


def seed_obs_norm_from_dataset(
    dataset_obs: jnp.ndarray, n_envs: int
) -> NormalizationInfo:
    """Compute one-shot stats from a fixed BC dataset and emit a
    NormalizationInfo seeded with those values, so the agent's
    online running stats start at the BC dataset's distribution.
    Subsequent online updates evolve the stats from there.

    ``dataset_obs`` shape: ``(T, n_envs, obs_dim)`` or ``(N, obs_dim)``.
    """
    flat = dataset_obs.reshape(-1, dataset_obs.shape[-1])
    mean = flat.mean(axis=0)
    var = flat.var(axis=0) + 1e-6
    n = flat.shape[0]
    # Broadcast to (n_envs, obs_dim) — online_normalize expects this layout.
    mean_b = jnp.broadcast_to(mean, (n_envs, mean.shape[0]))
    var_b = jnp.broadcast_to(var, (n_envs, var.shape[0]))
    # mean_2 = sum of squared deviations = var * count.
    mean_2_b = var_b * n
    count_b = jnp.full((n_envs, 1), float(n))
    return NormalizationInfo(
        count=count_b, mean=mean_b, mean_2=mean_2_b, var=var_b, returns=None,
    )
