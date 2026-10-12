"""Old import paths kept while downstream projects migrate (CLAUDE.md rule
10, expand then contract): a moved or renamed public name stays reachable
where it was, with a FutureWarning naming its new home."""

import warnings
from typing import Any, Callable


def moved_names(module: str, **new: Any) -> Callable[[str], Any]:
    """A module ``__getattr__`` serving each old name of ``new`` (old name ->
    the object now defined elsewhere) with a FutureWarning."""

    def __getattr__(name: str) -> Any:
        if name not in new:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        target = new[name]
        warnings.warn(
            f"{module}.{name} is now {target.__module__}.{target.__qualname__};"
            " the old path will be removed.",
            FutureWarning,
            stacklevel=2,
        )
        return target

    return __getattr__


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
