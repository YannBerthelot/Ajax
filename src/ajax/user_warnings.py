"""Warnings about how Ajax is called, shown at the caller's own line.

A warning names the line it is attributed to. With a fixed stacklevel that
line is an Ajax one whenever the code that noticed the problem sits a frame
deeper than usual, or runs while JAX traces a function. These warnings name
the first line outside Ajax and JAX: the user's call."""

from __future__ import annotations

import os
import warnings

import jax


def _folder(module_file: str) -> str:
    # With the separator, a sibling folder whose name starts the same
    # ("ajax_tools") is not taken for the package.
    return os.path.dirname(os.path.abspath(module_file)) + os.sep


#: The frames a warning skips: Ajax's own, and JAX's while it traces.
SKIPPED_FOLDERS = (_folder(__file__), _folder(jax.__file__))


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """``warnings.warn``, attributed to the first frame outside Ajax and JAX."""
    warnings.warn(message, category, skip_file_prefixes=SKIPPED_FOLDERS)
