"""Warnings for public keywords on their way out (CLAUDE.md rule 10, expand
then contract): a keyword that no longer has an effect stays accepted, with
a FutureWarning, until the downstream projects stop using it."""

import warnings
from typing import Any


def warn_inert(owner: str, reason: str, stacklevel: int = 3, **values: Any) -> None:
    """A FutureWarning for each of ``values`` set (not None): a keyword of
    ``owner`` with no effect (``reason``), kept until callers stop passing
    it."""
    for name, value in values.items():
        if value is not None:
            warnings.warn(
                f"{owner}'s {name} has no effect and will be removed: {reason}",
                FutureWarning,
                stacklevel=stacklevel,
            )
