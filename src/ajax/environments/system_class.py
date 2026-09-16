"""System classes: distributions over environment parameters.

A *system class* is a distribution ``p(S)`` over the dynamics of an
environment, represented as a distribution over its ``EnvParams``. It
is the object that turns one environment into a family of environments
for domain randomisation, meta-learning and in-context control (Busetto
et al. 2024, "One controller to rule them all", arXiv:2411.06482).

Sampling returns *batched* params: every leaf carries a leading axis of
size ``n``, one entry per parallel environment. :func:`ajax.environments
.interaction.reset` / :func:`~ajax.environments.interaction.step` detect
that leading axis and vmap over it, so a batched ``EnvParams`` can be
used anywhere an unbatched one is accepted (gymnax only: brax envs do not
take per-step params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

EnvParamsT = Any


def env_params_is_batched(params: EnvParamsT) -> bool:
    """True when ``params`` carries one system per env along a leading axis.

    A batched params (from :meth:`SystemClass.sample` or
    :func:`broadcast_env_params`) has *every* leaf with the same leading
    axis. An unbatched one always has at least one rank-0 leaf (gymnax's
    ``max_steps_in_episode`` at minimum), even when other fields are
    vectors or matrices (e.g. a plant's state-space model), so "any leaf has
    rank >= 1" would misclassify those.
    """
    if params is None:
        return False
    leaves = jax.tree.leaves(params)
    if not leaves or any(jnp.ndim(leaf) == 0 for leaf in leaves):
        return False
    first = jnp.shape(leaves[0])[0]
    return all(jnp.shape(leaf)[0] == first for leaf in leaves)


def broadcast_env_params(params: EnvParamsT, n: int) -> EnvParamsT:
    """Replicate an unbatched ``params`` ``n`` times along a new leading axis.

    Used to run a single nominal system through the batched code path
    (e.g. a fixed system during the first curriculum stage). Every leaf
    (scalar, vector or matrix) gets the new leading axis; an already
    batched params is returned unchanged.
    """
    if env_params_is_batched(params):
        return params

    def _tile(leaf):
        arr = jnp.asarray(leaf)
        return jnp.broadcast_to(arr, (n, *arr.shape))

    return jax.tree.map(_tile, params)


def select_env_params(params: EnvParamsT, index: int | jax.Array) -> EnvParamsT:
    """Pick one system out of a batched ``params`` (inverse of broadcasting)."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf)[index], params)


class SystemClass:
    """Base class: a distribution over ``EnvParams``.

    Subclasses implement :meth:`sample`. The nominal params are kept so
    callers can always fall back to the reference system (evaluation on
    the nominal plant, curriculum stage 1, ...).
    """

    nominal: EnvParamsT

    def sample(self, rng: jax.Array, n: int) -> EnvParamsT:
        """Draw ``n`` systems; returns batched params (leading axis ``n``)."""
        raise NotImplementedError


@dataclass(frozen=True)
class FixedSystem(SystemClass):
    """Degenerate class containing a single system (the nominal one)."""

    nominal: EnvParamsT

    def sample(self, rng: jax.Array, n: int) -> EnvParamsT:
        del rng
        return broadcast_env_params(self.nominal, n)


@dataclass(frozen=True)
class UniformPerturbation(SystemClass):
    """Independent uniform relative perturbation of selected parameter fields.

    Each listed field ``p`` is drawn as ``p * (1 + U(-scale, scale))``,
    independently per field and per sampled system; every other field
    stays at its nominal value. This is the meta-dataset construction of
    Busetto et al. 2024 (there: ``scale=0.05`` on the 19 evaporator
    constants).

    ``fields`` must name attributes of the nominal ``EnvParams``; a typo
    raises at construction rather than silently sampling nothing.
    """

    nominal: EnvParamsT
    fields: Sequence[str]
    scale: float = 0.05

    def __post_init__(self):
        if self.scale < 0:
            raise ValueError(f"scale must be non-negative, got {self.scale}")
        available = set(_field_names(self.nominal))
        missing = [f for f in self.fields if f not in available]
        if missing:
            raise ValueError(
                f"Unknown EnvParams fields {missing}; available: {sorted(available)}"
            )
        # Freeze the field order so sampling is deterministic under a key.
        object.__setattr__(self, "fields", tuple(self.fields))

    def sample(self, rng: jax.Array, n: int) -> EnvParamsT:
        batched = broadcast_env_params(self.nominal, n)
        if not self.fields:
            return batched
        keys = jax.random.split(rng, len(self.fields))
        updates = {}
        for key, name in zip(keys, self.fields):
            nominal_value = jnp.asarray(getattr(self.nominal, name), dtype=jnp.float32)
            factor = 1.0 + jax.random.uniform(
                key, (n,), minval=-self.scale, maxval=self.scale
            )
            updates[name] = nominal_value * factor
        return batched.replace(**updates)


def _field_names(params: EnvParamsT) -> list[str]:
    """Attribute names of a (flax/dataclass) ``EnvParams`` pytree."""
    fields = getattr(params, "__dataclass_fields__", None)
    if fields is not None:
        return list(fields)
    return list(vars(params))


__all__ = [
    "FixedSystem",
    "SystemClass",
    "UniformPerturbation",
    "broadcast_env_params",
    "env_params_is_batched",
    "select_env_params",
]
