"""Composable research-feature extensions for Ajax agents.

See :mod:`ajax.extensions.base` for the framework. Concrete extensions are
imported from their own modules (``expert``, ``exploration``,
``target_mods``, ``pretrain``, ``instrumentation``, ``ensemble``).
"""

from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionContext,
    ExtensionStack,
)

__all__ = ["PHASES", "Extension", "ExtensionContext", "ExtensionStack"]
