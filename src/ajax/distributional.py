"""Symlog transforms and two-hot discrete regression (DreamerV3, TD-MPC2).

Both world-model agents predict rewards and values as a categorical
distribution over fixed bins and train it with a cross-entropy against a
*two-hot* target: the scalar target is spread over its two neighbouring
bins by linear interpolation. The papers share the idea but not the
details, and the details matter for fidelity:

* **DreamerV3** (Hafner et al., arXiv:2301.04104v2, Eqs. 9-12 and p.18;
  paper-era code ``danijar/dreamerv3@2411f7d``) places 255 bins at
  ``symexp(linspace(-20, 20, 255))`` (``dreamerv3/nets.py:462-466``),
  interpolates the **raw** target between them and predicts the
  expectation of the raw bins (``dreamerv3/jaxutils.py:210-274``).
* **TD-MPC2** (Hansen et al., arXiv:2310.16828v2, Sec. 3.1 and Table 8;
  paper-era code ``nicklashansen/tdmpc2@5f6fade``) places 101 bins at
  ``linspace(-10, 10, 101)`` in **symlog** space, interpolates
  ``clip(symlog(y), -10, 10)`` between them and predicts
  ``symexp(E_p[bins])`` (``tdmpc2/common/math.py:5-9, 66-95``).

In raw space both bin sets are ``symexp(linspace(-limit, limit, n))``;
they differ in the bin count, the limit and the space in which the
target is interpolated and the expectation taken. :class:`TwoHot`
captures exactly these three parameters. For the same target the two
interpolation spaces give different weights even on identical bins
(y = 1 on DreamerV3's bins: upper weight 0.3827 in raw space, 0.4015 in
symlog space), and ``symexp(E[b]) != E[symexp(b)]``, so neither can be
simplified into the other.

Both bin sets are built by mirroring their negative half, so the middle
bin is exactly 0 and the bins are exactly antisymmetric. The expectation
is DreamerV3's symmetric sum over mirror bins, written so that uniform
probabilities give exactly 0 in eager and in jitted code alike: a
zero-initialised head predicts 0 (:meth:`TwoHot.decode`).

Fidelity, tested against literal ports of both references
(``tests/world_models/reference_impls.py``), with the float32-level
departures registered as deviations D22 (DreamerV3) and T24 (TD-MPC2) in
``docs/world_models/deviations.md``:

* The bins are the float32 roundings of their float64 values, computed
  once with numpy, so they are the same constants in eager and jitted
  code and on every backend. The references evaluate the same expressions
  in float32 on the device: DreamerV3's ``jnp`` bins differ from these by
  up to 1.3e-6 relative (mostly float32 ``linspace`` rounding near 0),
  TD-MPC2's ``torch.linspace`` bins by up to 4.8e-7 (its middle bin is
  -1.5e-7, not 0).
* Given the same bins, :meth:`TwoHot.encode` and :meth:`TwoHot.loss` are
  bit-identical to DreamerV3's code run op by op, and
  :meth:`TwoHot.decode` is within ``3 eps E_p|b|`` of it (``eps`` the
  float32 machine epsilon), eagerly and under ``jax.jit``.
* TD-MPC2 encodes with a floor formula on its uniform grid. The generic
  interpolation used here is equal in exact arithmetic, and in float32
  the more accurate of the two (within 2e-6 of the exact weights, against
  up to 8.6e-6 for the floor formula, whose positions near 100 have an
  ulp of 7.6e-6); they agree to 1.1e-5. TD-MPC2's naive expectation is
  matched within ``eps (1 + 6 E_p|b|)`` in symlog space (4.8e-6 at most;
  the ``eps`` term is its ``exp(|x|) - 1`` rounding near 0), and its
  ``F.log_softmax`` (max subtracted first) within
  ``3 eps (max|logits| + loss)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


def symlog(x: jax.Array) -> jax.Array:
    """``sign(x) * log(1 + |x|)``, elementwise and dtype-preserving.

    DreamerV3 Eq. 9 (``2411f7d:dreamerv3/jaxutils.py:76-77``); TD-MPC2
    ``5f6fade:tdmpc2/common/math.py:48-54`` writes ``log(1 + |x|)``, equal
    to rounding. As in both references the derivative at exactly 0 is 0
    (``sign'(0) = sign(0) = 0``); the limit from either side is 1.
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """``sign(x) * (exp(|x|) - 1)``, the inverse of :func:`symlog`.

    DreamerV3 ``2411f7d:dreamerv3/jaxutils.py:80-81``; TD-MPC2
    ``5f6fade:tdmpc2/common/math.py:57-63`` (``exp(|x|) - 1``).
    """
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


@dataclass(frozen=True)
class TwoHot:
    """Two-hot discrete regression over symlog-spaced bins.

    Frozen and hashable, so it can be a ``jax.jit`` static argument or a
    field of a frozen module config. Use the paper constructors
    :meth:`dreamerv3` and :meth:`tdmpc2` rather than the raw fields.

    Args:
        num_bins: number of bins ``n``; odd and at least 3, so that the
            middle bin is 0 and the symmetric summation pairs every bin.
        limit: the bins span ``[-limit, limit]`` in symlog space, i.e.
            ``[-symexp(limit), symexp(limit)]`` in raw space. Only
            symmetric ranges are representable.
        transform: the space in which targets are interpolated and the
            expectation is taken. ``"identity"`` (DreamerV3): raw space,
            the bins are ``symexp(linspace(-limit, limit, n))``.
            ``"symlog"`` (TD-MPC2): symlog space, the bins are
            ``linspace(-limit, limit, n)``, the target is ``symlog(y)``
            and the prediction is ``symexp`` of the expectation.

    All methods compute in float32 and return float32.
    """

    num_bins: int
    limit: float
    transform: Literal["identity", "symlog"]

    def __post_init__(self) -> None:
        if self.num_bins < 3 or self.num_bins % 2 == 0:
            raise ValueError(
                f"TwoHot needs an odd num_bins >= 3 (got {self.num_bins}): the "
                "middle bin must be 0 for the symmetric summation."
            )
        if not self.limit > 0:
            raise ValueError(f"TwoHot limit must be positive, got {self.limit}")
        if self.transform not in ("identity", "symlog"):
            raise ValueError(
                f"TwoHot transform must be 'identity' or 'symlog', got "
                f"{self.transform!r}"
            )

    @classmethod
    def dreamerv3(cls, num_bins: int = 255) -> TwoHot:
        """DreamerV3's ``symexp_twohot``: raw-space bins and interpolation.

        Bins ``symexp(linspace(-20, 20, num_bins))`` (range about
        ``±4.85e8``), raw targets, raw-space expectation
        (``2411f7d:dreamerv3/nets.py:462-466``, ``jaxutils.py:210-274``;
        ``bins: 255`` in ``configs.yaml``).
        """
        return cls(num_bins=num_bins, limit=20.0, transform="identity")

    @classmethod
    def tdmpc2(cls, num_bins: int = 101, limit: float = 10.0) -> TwoHot:
        """TD-MPC2's ``two_hot``: symlog-space bins and interpolation.

        Bins ``linspace(-limit, limit, num_bins)`` in symlog space (step
        0.2 by default, raw range about ``±22025``), target
        ``clip(symlog(y), -limit, limit)``, prediction
        ``symexp(E_p[bins])`` (``5f6fade:tdmpc2/common/math.py:66-95``).
        TD-MPC2's config has separate ``vmin: -10`` and ``vmax: +10``
        (``config.yaml:48-50``); the range must be symmetric here, which
        every TD-MPC2 config is: ``limit = vmax = -vmin`` (deviation T24).
        """
        return cls(num_bins=num_bins, limit=limit, transform="symlog")

    def bins(self) -> jax.Array:
        """The ``[num_bins]`` bin locations in the interpolation space.

        Built as DreamerV3 builds its bins (``nets.py:464-466``): the
        negative half ``linspace(-limit, 0, (n + 1) // 2)`` (mapped by
        ``symexp`` for the identity transform) concatenated with its
        negated mirror, so ``bins[n // 2] == 0`` and ``bins == -bins[::-1]``
        exactly. The half is computed in float64 with numpy and rounded to
        float32 once, so the bins are trace-time constants, identical in
        eager and jitted code (deviations D22, T24; module docstring).
        """
        half = np.linspace(-self.limit, 0.0, (self.num_bins + 1) // 2)
        if self.transform == "identity":
            half = np.sign(half) * np.expm1(np.abs(half))
        bins = np.concatenate([half, -half[:-1][::-1]]).astype(np.float32)
        return jnp.asarray(bins)

    def _forward(self, y: jax.Array) -> jax.Array:
        return symlog(y) if self.transform == "symlog" else y

    def _inverse(self, x: jax.Array) -> jax.Array:
        return symexp(x) if self.transform == "symlog" else x

    def encode(self, y: jax.Array) -> jax.Array:
        """Two-hot weights ``[..., num_bins]`` for scalar targets ``y [...]``.

        DreamerV3's algorithm (``2411f7d:dreamerv3/jaxutils.py:257-271``;
        dreamerv3_spec Algorithm D) applied to the transformed target:
        linear interpolation between the neighbouring bins
        ``b_below <= y < b_above``; a target outside the bin range puts all
        its mass on the edge bin; a target equal to a bin puts all its mass
        on that bin. On TD-MPC2's uniform symlog grid this equals, in exact
        arithmetic, its floor formula ``k = floor((y - vmin) / step)``,
        ``t[k] = 1 - o``, ``t[k + 1] = o`` after ``clip(y, vmin, vmax)``
        (``5f6fade:tdmpc2/common/math.py:72-77``): a clipped target lands
        exactly on the edge bin. See the module docstring for the float32
        agreement (deviation T24).
        """
        x = self._forward(jnp.asarray(y, jnp.float32))
        bins = self.bins()
        n = self.num_bins
        below = jnp.sum(bins <= x[..., None], -1, dtype=jnp.int32) - 1
        above = n - jnp.sum(bins > x[..., None], -1, dtype=jnp.int32)
        below = jnp.clip(below, 0, n - 1)
        above = jnp.clip(above, 0, n - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1.0, jnp.abs(bins[below] - x))
        dist_to_above = jnp.where(equal, 1.0, jnp.abs(bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return (
            jax.nn.one_hot(below, n, dtype=jnp.float32) * weight_below[..., None]
            + jax.nn.one_hot(above, n, dtype=jnp.float32) * weight_above[..., None]
        )

    def decode(self, logits: jax.Array) -> jax.Array:
        """Predicted scalars ``[...]`` from logits ``[..., num_bins]``.

        ``inverse_transform(E_p[bins])`` with ``p = softmax(logits)`` and
        the expectation summed symmetrically over mirror bins, as DreamerV3
        does (``2411f7d:dreamerv3/jaxutils.py:226-243``; paper p.18) so
        that a zero-initialised head predicts exactly 0. DreamerV3 sums
        ``p_m b_m + sum_i (p_i b_i + p_j b_j)`` over the mirror pairs
        ``(i, j)``. Under ``jax.jit`` XLA contracts each pair into a fused
        multiply-add, which skips the rounding of one of the two products,
        so the pairs no longer cancel: for uniform logits on DreamerV3's
        bins the reference code then gives 0.07 to 0.16 on CPU, depending
        on the batch shape. With
        ``b_m = 0`` and ``b_i = -b_j`` exactly, the same sum is
        ``sum_j (p_j - p_i) b_j``; the difference is exactly 0 for uniform
        ``p`` before any multiplication, so the result is exactly 0 however
        the operations are fused (deviation D22). A naive ``jnp.sum`` of
        ``p * bins`` gives 2.0 on DreamerV3's bins (CPU).
        """
        probs = jax.nn.softmax(jnp.asarray(logits, jnp.float32), -1)
        m = self.num_bins // 2
        upper_bins = self.bins()[m + 1 :]
        mirror_differences = probs[..., m + 1 :] - probs[..., :m][..., ::-1]
        return self._inverse(jnp.sum(mirror_differences * upper_bins, -1))

    def loss(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """Soft cross-entropy ``-sum(w * log_softmax(logits), -1)``, shape ``[...]``.

        The log-probabilities are ``logits - logsumexp(logits)`` as in
        DreamerV3 (``2411f7d:dreamerv3/jaxutils.py:272-274``); TD-MPC2's
        ``F.log_softmax`` (``5f6fade:tdmpc2/common/math.py:5-9``) subtracts
        the max first, the same up to float32 rounding (deviation T24).
        The weights ``w = encode(y)`` are stop-gradiented. In both
        references the callers pass targets that carry no gradient
        (DreamerV3 ``agent.py:257`` replay data, ``:349-350`` and
        ``:371-373`` ``log_prob(sg(...))``; TD-MPC2 ``tdmpc2.py:201-202``
        and ``:231-233``, TD targets under ``torch.no_grad``, and replay
        rewards at ``:257``); Ajax makes that explicit. There is no
        reduction over the batch; callers apply each paper's own weighting
        and averaging.
        """
        weights = jax.lax.stop_gradient(self.encode(y))
        logits = jnp.asarray(logits, jnp.float32)
        log_probs = logits - jax.scipy.special.logsumexp(logits, -1, keepdims=True)
        return -jnp.sum(weights * log_probs, -1)
