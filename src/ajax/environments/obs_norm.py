"""Agent-side running observation normalization.

Lives inside ``CollectorState.obs_norm_info`` so stats are part of
``agent_state`` and thread through naturally. Normalization is applied
at every actor/critic ``apply_fn`` boundary; stats are updated only at
collection time (online interaction and BC dataset prep). The replay
buffer stores raw observations (their features included) so changing stats do not poison
old samples; we re-normalize at sample time.

Agent-side so that the statistics are the agent's, shipped with its policy,
and cover its observation features (``ajax.controller``), which the env never
sees: the env's columns and the normalised features; the raw features (an
expert's action) come last and stay raw.
"""

from typing import Optional, Tuple

import jax.numpy as jnp

from ajax.utils import online_normalize
from ajax.wrappers import NormalizationInfo, init_norm_info


def init_agent_obs_norm(obs_dim: int) -> NormalizationInfo:
    """Initialise running stats for an obs vector of size ``obs_dim``.

    Stats are stored with leading shape 1 (not ``n_envs``): online_normalize
    reduces the per-env axis on every update so all envs would share the
    same scalar stats anyway, and ``apply_obs_norm`` only needs the
    per-feature mean/var. The size-1 leading axis preserves broadcasting
    against batched obs ``(*, obs_dim)`` while saving an n_envs× memory
    factor on every checkpoint and vmap broadcast.
    """
    return init_norm_info(batch_size=1, obs_shape=(obs_dim,))


def _covered(obs: jnp.ndarray, info: NormalizationInfo) -> tuple:
    """``obs``'s leading columns, which ``info``'s statistics cover, and the
    raw features after them (None when there are none)."""
    span = info.mean.shape[-1]
    if obs.shape[-1] == span:
        return obs, None
    return obs[..., :span], obs[..., span:]


def _joined(normalised: jnp.ndarray, raw: Optional[jnp.ndarray]) -> jnp.ndarray:
    return normalised if raw is None else jnp.concatenate([normalised, raw], -1)


def update_obs_norm(
    obs: jnp.ndarray, info: Optional[NormalizationInfo], raw_columns: int = 0
) -> Tuple[jnp.ndarray, Optional[NormalizationInfo]]:
    """Update running stats with ``obs`` and return (normalized_obs, new_info).

    ``obs`` shape: ``(n_envs, obs_dim)``. When ``info is None`` the call is
    a no-op (returns input unchanged) so call sites can stay unconditional.
    The leading ``raw_columns`` stay raw (see ``online_normalize``), as do
    the raw features after the columns the statistics cover.
    """
    if info is None:
        return obs, None
    own, raw = _covered(obs, info)
    new_obs, count, mean, mean_2, var = online_normalize(
        own,
        info.count,
        info.mean,
        info.mean_2,
        train=True,
        nan_safe=False,
        raw_columns=raw_columns,
    )
    new_info = NormalizationInfo(
        count=count,
        mean=mean,
        mean_2=mean_2,
        var=var,
        returns=info.returns,
    )
    return _joined(new_obs, raw), new_info


def normalized_obs(
    obs: jnp.ndarray, info: Optional[NormalizationInfo], raw_columns: int = 0
) -> jnp.ndarray:
    """``obs`` normalised with the statistics ``info``, not updated, as
    :func:`update_obs_norm` normalises (``online_normalize``): the frozen
    statistics of an on-policy agent's evaluation and of its final and
    reset observations. ``info`` None: ``obs`` as it is."""
    if info is None:
        return obs
    own, raw = _covered(obs, info)
    normalised = online_normalize(
        own,
        info.count,
        info.mean,
        info.mean_2,
        train=False,
        nan_safe=False,
        raw_columns=raw_columns,
    )[0]
    return _joined(normalised, raw)


def apply_obs_norm(obs: jnp.ndarray, info: Optional[NormalizationInfo]) -> jnp.ndarray:
    """Normalize ``obs`` using existing stats (no update). Used at every
    actor/critic ``apply_fn`` site that consumes obs sampled from the
    buffer or carried by the eval scan, so stats updates stay localised
    to the collection sites. Falls back to raw obs while the running
    stats are still empty (count == 0), avoiding a divide-by-≈0 at
    step 1 before any update.

    The statistics cover the leading columns: the env's and the normalised
    features'; what follows stays raw: a raw feature (an expert's action)
    and a critic's action, already in action
    units, as the env's ``raw_columns`` keep a PID's terms raw."""
    if info is None or info.var is None:
        return obs
    count_total = jnp.sum(info.count)
    std = jnp.sqrt(info.var + 1e-8)
    own, appended = obs[..., : info.mean.shape[-1]], obs[..., info.mean.shape[-1] :]
    normalized = jnp.where(count_total > 0, (own - info.mean) / std, own)
    return jnp.concatenate([normalized, appended], axis=-1)
