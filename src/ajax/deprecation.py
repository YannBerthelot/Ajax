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
