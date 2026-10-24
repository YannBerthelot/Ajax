"""Warnings for public keywords and names on their way out (CLAUDE.md rule
10, expand then contract): a keyword that no longer has an effect stays
accepted, and a name moved to another module still imports from its old
one, each with a FutureWarning, until the downstream projects stop using
it."""

import importlib
import warnings
from collections.abc import Callable
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


def moved_names(old: str, homes: dict[str, str]) -> Callable[[str], Any]:
    """A module ``__getattr__`` (PEP 562) for the names moved out of module
    ``old``: ``homes`` maps each to its new module (``"module:new_name"``
    when renamed, ``"module:Class.method"`` for a function that became a
    method), from which it resolves, with a FutureWarning, on every access.
    The old module must not import them itself, or the import binds them and
    the warning never fires."""

    def __getattr__(name: str) -> Any:
        if name not in homes:
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        module, _, new_name = homes[name].partition(":")
        new_name = new_name or name
        warnings.warn(
            f"{old}.{name} moved to {module}.{new_name}; the old path will be"
            " removed in 0.5.0",
            FutureWarning,
            stacklevel=2,
        )
        found = importlib.import_module(module)
        for part in new_name.split("."):
            found = getattr(found, part)
        return found

    return __getattr__
