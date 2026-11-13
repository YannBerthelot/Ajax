"""Per-task presets: the settings an agent's reference code tunes for one
task, kept apart from the agent's defaults, which are its reference's general
config. An agent's module keeps them in a ``PRESETS`` table, task name to
:class:`Preset`, and ``Agent.preset(name, env_id, **overrides)``
(:meth:`ajax.agents.families.DeclaredAgent.preset`) builds the agent from
one.

A task goes by the name its reference gives it ("Hopper", "AntMaze",
"Atari"), not by an env id: one task has ids in gymnasium, brax and
playground.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Preset:
    """A task's settings in an agent's reference.

    Args:
        overrides: The agent's fields its reference sets for the task. A
            table gives each of its presets the same fields, the ones the
            reference varies across its tasks, even where a value equals the
            default: a default that changes later leaves the preset as its
            reference.
        citation: Where the values come from (repo@commit, file:line).
    """

    overrides: Mapping[str, Any]
    citation: str


def presets_of(agent_class: type) -> Mapping[str, Preset]:
    """The PRESETS table of the module that defines agent_class, empty when it
    has none: a subclass defined elsewhere tunes other things."""
    return getattr(sys.modules[agent_class.__module__], "PRESETS", {})
