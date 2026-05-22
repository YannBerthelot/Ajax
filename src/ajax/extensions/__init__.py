"""Composable research-feature extensions for Ajax agents.

See :mod:`ajax.extensions.base` for the framework. Concrete extensions
(expert guidance, observation augmentation, online BC, target modifiers,
exploration schemes, instrumentation) are added in their own modules and
re-exported here as they are migrated.
"""

from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionContext,
    ExtensionStack,
)

__all__ = ["Extension", "ExtensionContext", "ExtensionStack", "PHASES"]
