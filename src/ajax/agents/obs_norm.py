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

import jax.numpy as jnp

from ajax.utils import online_normalize
from ajax.wrappers import NormalizationInfo, init_norm_info


def init_agent_obs_norm(n_envs: int, obs_dim: int) -> NormalizationInfo:
    """Initialise running stats for an obs vector of size ``obs_dim``.

    Stats are stored with leading shape 1 (not ``n_envs``): online_normalize
    reduces the per-env axis on every update so all envs would share the
    same scalar stats anyway, and ``apply_obs_norm`` only needs the
    per-feature mean/var. The size-1 leading axis preserves broadcasting
    against batched obs ``(*, obs_dim)`` while saving an n_envs× memory
    factor on every checkpoint and vmap broadcast. ``n_envs`` is kept in
    the signature for backwards compatibility but ignored.
    """
    del n_envs
    return init_norm_info(batch_size=1, obs_shape=(obs_dim,))


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
        obs, info.count, info.mean, info.mean_2, train=True, nan_safe=False
    )
    new_info = NormalizationInfo(
        count=count,
        mean=mean,
        mean_2=mean_2,
        var=var,
        returns=info.returns,
    )
    return new_obs, new_info


def apply_obs_norm(obs: jnp.ndarray, info: Optional[NormalizationInfo]) -> jnp.ndarray:
    """Normalize ``obs`` using existing stats (no update). Used at every
    actor/critic ``apply_fn`` site that consumes obs sampled from the
    buffer or carried by the eval scan, so stats updates stay localised
    to the collection sites. Falls back to raw obs while the running
    stats are still empty (count == 0), avoiding a divide-by-≈0 at
    step 1 before any update."""
    if info is None or info.var is None:
        return obs
    count_total = jnp.sum(info.count)
    std = jnp.sqrt(info.var + 1e-8)
    normalized = (obs - info.mean) / std
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
    del n_envs  # kept for signature compatibility; stats are not per-env
    flat = dataset_obs.reshape(-1, dataset_obs.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    var = flat.var(axis=0, keepdims=True) + 1e-6
    n = flat.shape[0]
    # Stats live with leading axis size 1 (see init_agent_obs_norm).
    return NormalizationInfo(
        count=jnp.full((1, 1), float(n)),
        mean=mean,
        mean_2=var * n,
        var=var,
        returns=None,
    )
