"""Compare two training steps leaf by leaf, on the change each step makes, within a tolerance set by
the rounding each side shows on its own.

A leaf's difference is ``max |delta_a - delta_b| / max |delta_b|``, ``delta`` the change the step made
(new minus old). Comparing changes rather than values keeps a leaf near zero (a bias early in
training) from inflating the error, and keeps a large leaf from hiding a wrong step. A leaf one side
changes and the other does not differs infinitely. The null of a leaf is the same measure between
two runs of one side that compute the same maths in a different order (jitted against eager); its
tolerance is ``max(factor * null, floor)``. Discrete leaves (actions, done flags, counters) must be
equal. Equal NaNs are equal: bookkeeping leaves hold NaN until an episode ends.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import numpy as np

FACTOR, FLOOR = 10.0, 1e-6


def _array(x: Any) -> np.ndarray:
    return np.asarray(x, np.float64)


def is_discrete(x: Any) -> bool:
    return not np.issubdtype(np.asarray(x).dtype, np.floating)


def change_diff(a: Any, b: Any, old: Optional[Any] = None) -> float:
    """The relative difference of the changes from ``old`` to ``a`` and to ``b`` (of the values
    when ``old`` is None); for discrete leaves, the share of elements that differ."""
    if is_discrete(b):
        return float(np.mean(np.asarray(a) != np.asarray(b))) if np.size(b) else 0.0
    base = 0.0 if old is None else _array(old)
    da, db = _array(a) - base, _array(b) - base
    both_nan = np.isnan(da) & np.isnan(db)
    da, db = np.where(both_nan, 0.0, da), np.where(both_nan, 0.0, db)
    gap = float(np.max(np.abs(da - db), initial=0.0))
    scale = float(np.max(np.abs(db), initial=0.0))
    if scale == 0.0:
        return 0.0 if gap == 0.0 else float("inf")
    return gap / scale


@dataclasses.dataclass(frozen=True)
class Check:
    leaf: str
    diff: float
    null: float
    tau: float

    @property
    def ok(self) -> bool:
        return self.diff <= self.tau


@dataclasses.dataclass(frozen=True)
class Report:
    """The checks of one comparison, and what it compared, for the failure message."""

    checks: tuple[Check, ...]
    context: str = ""

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def __str__(self) -> str:
        worst = sorted(
            self.checks, key=lambda c: c.diff / max(c.tau, 1e-300), reverse=True
        )
        rows = [f"{'leaf':60s} {'diff':>9s} {'tau':>9s} {'null':>9s}"]
        rows += [
            f"{c.leaf[-60:]:60s} {c.diff:9.2e} {c.tau:9.2e} {c.null:9.2e}{'' if c.ok else '  <-'}"
            for c in worst[:12]
        ]
        head = f"{len(self.failures)} of {len(self.checks)} leaves beyond tolerance"
        return "\n".join([head + (f" ({self.context})" if self.context else ""), *rows])


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    nulls: Mapping[str, float],
    old: Optional[Mapping[str, Any]] = None,
    factor: float = FACTOR,
    floor: float = FLOOR,
    context: str = "",
) -> Report:
    """Compare ``a`` with ``b`` on every leaf of ``b``; ``nulls`` maps each float leaf to its null."""
    checks = []
    for leaf, vb in b.items():
        diff = change_diff(a[leaf], vb, None if old is None else old[leaf])
        if is_discrete(vb):
            checks.append(Check(leaf, diff, 0.0, 0.0))
        else:
            null = nulls[leaf]
            checks.append(Check(leaf, diff, null, max(factor * null, floor)))
    return Report(tuple(checks), context)


def null_of(
    *pairs: tuple[Mapping[str, Any], Mapping[str, Any]],
    old: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Per leaf, the largest :func:`change_diff` within each pair (one side jitted, the same side
    eager): the rounding the comparison must tolerate."""
    leaves = pairs[0][1].keys()
    return {
        leaf: max(
            change_diff(x[leaf], y[leaf], None if old is None else old[leaf])
            for x, y in pairs
        )
        for leaf in leaves
        if not is_discrete(pairs[0][1][leaf])
    }
