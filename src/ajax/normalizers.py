"""Percentile-range scale normalisers (DreamerV3 retnorm, TD-MPC2 RunningScale).

Both world-model agents divide a quantity by a running estimate of the
spread between its 5th and 95th percentiles, floored at 1, so that the
actor's learning signal is invariant to the reward scale without
amplifying small returns. The papers describe the same idea ("an EMA of
the 5-95 percentile range, floored at 1"), but the two implementations
differ, and each normaliser below reproduces its own reference:

* :class:`ReturnNormalizer`, DreamerV3's return normaliser (arXiv:2301.04104v2
  p.6 and Eq. 7, Table 4; ``Moments(impl='perc', rate=0.01, limit=1.0)`` at
  ``danijar/dreamerv3@2411f7d:dreamerv3/jaxutils.py:301-395``,
  ``configs.yaml:141``): separate EMAs of the 5th and 95th percentiles,
  both starting at 0, no bias correction; the floor is applied when the
  scale is read, ``max(1, hi - lo)``. Bit-identical to the reference run
  op by op.
* :class:`RunningScale`, TD-MPC2's Q normaliser (arXiv:2310.16828v2 Sec. 3.1,
  Table 8; ``nicklashansen/tdmpc2@5f6fade:tdmpc2/common/scale.py:4-45``):
  one EMA starting at 1 of the range floored *before* averaging,
  ``S <- lerp(S, max(1, p95 - p5), tau)``. Equal to the reference up to
  float32 rounding of the EMA (deviation T25 in
  ``docs/world_models/deviations.md``).

Clamping before or after the EMA, and starting at 0 or 1, give different
scales on the same data (ranges alternating 0 and 3: 1.5075 vs 2.0050).
Both EMAs are ``optax.incremental_update``.

In both references the statistics are updated *before* the scale is read
in the same training step, so the current batch already counts:
DreamerV3 ``agent.py:329`` (``retnorm(ret, update)``), TD-MPC2
``tdmpc2.py:188-189`` (``update`` then divide). Percentiles are
``jnp.percentile`` with linear interpolation over all elements of the
stop-gradiented float32 input; TD-MPC2's sort-and-interpolate
``_percentile`` (``scale.py:23-35``) is the same formula, and it always
receives a ``(B, 1)`` batch, so "all elements" is "over the batch".
Neither normaliser passes gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import struct

# Both papers use the 5th and 95th percentiles (DreamerV3 Table 4,
# TD-MPC2 Table 8).
_PERCENTILES = (5.0, 95.0)


def _percentile_range(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The 5th and 95th percentiles of all elements of ``sg(f32(x))``.

    The statistics both normalisers average: ``jnp.percentile`` with
    linear interpolation of the stop-gradiented float32 input (DreamerV3
    ``jaxutils.py:343-344, :357``; TD-MPC2 ``scale.py:23-38``, which
    receives float32 already).
    """
    x = jax.lax.stop_gradient(jnp.asarray(x, jnp.float32))
    p5, p95 = jnp.percentile(x, jnp.asarray(_PERCENTILES), method="linear")
    return p5, p95


@struct.dataclass
class ReturnNormalizer:
    """DreamerV3 return normaliser state; see the module docstring.

    ``lo`` and ``hi`` are the EMAs of the 5th and 95th percentiles of the
    returns. Use as ``norm = norm.update(returns)`` followed by
    ``norm.scale()`` in the same step (update, then read). DreamerV3 divides
    advantages by the scale and does not subtract the offset ``lo``.

    Attributes:
        lo, hi: float32 scalars, both 0 at :meth:`create`.
        rate: EMA rate, 0.01 (decay 0.99).
        limit: floor of the scale, 1.0.
    """

    lo: jax.Array
    hi: jax.Array
    rate: float = struct.field(pytree_node=False, default=0.01)
    limit: float = struct.field(pytree_node=False, default=1.0)

    @classmethod
    def create(cls, rate: float = 0.01, limit: float = 1.0) -> ReturnNormalizer:
        """Initial state: ``lo = hi = 0`` (``jaxutils.py:319-321``).

        ``lo`` and ``hi`` are two arrays, not one aliased twice, so that a
        state holding the normaliser can be donated to a jitted function.
        """
        return cls(
            lo=jnp.zeros((), jnp.float32),
            hi=jnp.zeros((), jnp.float32),
            rate=rate,
            limit=limit,
        )

    def update(self, x: jax.Array) -> ReturnNormalizer:
        """Fold the percentiles of all elements of ``x`` into the EMAs.

        ``lo <- (1 - rate) lo + rate P5(x)``, ``hi`` likewise with P95
        (``jaxutils.py:356-359``). No bias correction.
        ``optax.incremental_update`` adds the same two products in the
        other order, which is exact: floating-point addition commutes.
        """
        lo, hi = optax.incremental_update(
            _percentile_range(x), (self.lo, self.hi), self.rate
        )
        return self.replace(lo=lo, hi=hi)

    def scale(self) -> jax.Array:
        """``max(limit, hi - lo)``, stop-gradiented (``jaxutils.py:382-386``)."""
        return jax.lax.stop_gradient(jnp.maximum(self.limit, self.hi - self.lo))


@struct.dataclass
class RunningScale:
    """TD-MPC2 running Q scale state; see the module docstring.

    Use as ``scale = scale.update(q)`` then ``q / scale.scale()`` in the
    same step (update before divide), with ``q`` the ``(B, 1)`` batch of
    t = 0 policy-loss Q values (``tdmpc2.py:188-189``).

    Attributes:
        value: the float32 scale ``S``, 1 at :meth:`create`; always >= 1.
        rate: EMA rate ``tau``, 0.01 (TD-MPC2 reuses the target-network
            ``tau``).
    """

    value: jax.Array
    rate: float = struct.field(pytree_node=False, default=0.01)

    @classmethod
    def create(cls, rate: float = 0.01) -> RunningScale:
        """Initial state: ``S = 1`` (``scale.py:9``)."""
        return cls(value=jnp.ones((), jnp.float32), rate=rate)

    def update(self, x: jax.Array) -> RunningScale:
        """``S <- (1 - rate) S + rate max(1, P95(x) - P5(x))`` (``scale.py:37-40``).

        The range is floored at 1 *before* it enters the average, so ``S``
        never drops below 1. torch's ``lerp_`` computes ``S + rate (v - S)``,
        the same average up to float32 rounding (deviation T25).
        """
        p5, p95 = _percentile_range(x)
        value = optax.incremental_update(
            jnp.maximum(1.0, p95 - p5), self.value, self.rate
        )
        return self.replace(value=value)

    def scale(self) -> jax.Array:
        """The current scale ``S``, stop-gradiented."""
        return jax.lax.stop_gradient(self.value)
