"""Warnings for public keywords and names on their way out (CLAUDE.md rule
10, expand then contract): a keyword that no longer has an effect stays
accepted, and a name moved to another module still imports from its old
one, each with a FutureWarning, until the downstream projects stop using
it."""

import importlib
import inspect
import sys
import types
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
    ``old``, which calls it from its own body: ``homes`` maps each to its
    new module (``"module:new_name"`` when renamed), from which it resolves,
    with a FutureWarning, on every access. A function that became a method
    (``"module:Class.method"``) takes other arguments, so it resolves to a
    function raising TypeError with the method's call form. Writing a moved
    name on the old module raises AttributeError: nothing reads it there,
    so a patch (monkeypatch.setattr) would change nothing. The old module
    must not import the names itself, or the import binds them and the
    warning never fires."""

    def home(name: str) -> tuple[str, str]:
        module, _, new_name = homes[name].partition(":")
        return module, new_name or name

    def __getattr__(name: str) -> Any:
        if name not in homes:
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        module, new_name = home(name)
        warnings.warn(
            f"{old}.{name} moved to {module}.{new_name}; the old path will be"
            " removed in 0.5.0",
            FutureWarning,
            stacklevel=2,
        )
        found: Any = importlib.import_module(module)
        for part in new_name.split("."):
            found = getattr(found, part)
        if "." in new_name:
            return became_a_method(f"{old}.{name}", f"{module}.{new_name}", found)
        return found

    class MovedNames(types.ModuleType):
        def __setattr__(self, name: str, value: Any) -> None:
            if name in homes:
                module, new_name = home(name)
                raise AttributeError(
                    f"{old}.{name} moved to {module}.{new_name}: set it there,"
                    " where it is read (nothing reads the old path)."
                )
            super().__setattr__(name, value)

    sys.modules[old].__class__ = MovedNames
    return __getattr__


def became_a_method(old: str, new: str, method: Callable) -> Callable:
    """A stand-in for the function ``old``, now the method ``new``
    (``module.Class.method``) with other parameters: calling it raises
    TypeError with the method's call form, instead of binding the old
    arguments to the method's and failing far from the call."""
    owner, _, name = new.rpartition(".")
    parameters = list(inspect.signature(method).parameters.values())
    call = [p.name for p in parameters[1:] if p.default is p.empty]
    if len(call) < len(parameters) - 1:
        call.append("...")
    message = (
        f"{old} became {new}({', '.join(map(_annotated, parameters))}), with"
        f" other parameters: call it on an instance of {owner.rpartition('.')[2]},"
        f" instance.{name}({', '.join(call)})."
    )

    def moved(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError(message)

    moved.__name__ = moved.__qualname__ = old.rpartition(".")[2]
    moved.__doc__ = message
    return moved


def _annotated(parameter: inspect.Parameter) -> str:
    """``name: annotation`` (a string under ``from __future__ import
    annotations``), or the name alone."""
    if parameter.annotation is parameter.empty:
        return parameter.name
    annotation = parameter.annotation
    return f"{parameter.name}: {getattr(annotation, '__name__', annotation)}"
