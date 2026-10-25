"""The agents' surface, pinned while their files are restructured: every
public name an agent module defines still imports from its path, every
agent built on its defaults logs the run config and holds the agent_config
it did, and every agent class keeps the contract the extensions and the
loop read off it (a value a family base supplies is the one the agent
declared).

After a deliberate change (a default changed in its own pull request, a new
name), rewrite the records::

    uv run python -m tests.agents.test_agent_surface

The rewrite keeps every recorded name: a name removed on purpose is struck
by hand.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import ajax
from tests.agents.test_agent_contract import CASES
from tests.agents.test_agent_pins import _multitask

HERE = Path(__file__).parent
AGENTS = Path(ajax.__file__).parent / "agents"
SURFACE = HERE / "agent_surface.json"
CONFIGS = HERE / "agent_configs.json"
CONTRACTS = HERE / "agent_contracts.json"
#: The exported agents test_agent_contract has no row for: the env each is
#: built on (TDMPC2MultiTask on the pins' two toy tasks).
OTHERS = {"APG": "Pendulum-v1", "TDMPC2": "Pendulum-v1", "DreamerV3": "CartPole-v1"}
#: ActorCritic's class attributes an agent declares for the extensions and
#: the loop (with its make_train, :func:`contract`).
CONTRACT = (
    "name capabilities supported_extension_phases phase_batch_keys"
    " extension_context pretrain_context supported_buffer_fields"
    " replays_transitions stores_expert_actions supports_memory trains_batch_norm"
    " supports_batched_env_params per_seed_update_counts"
).split()


def defined_names(path: Path) -> set[str]:
    """The public names a module's top level defines (not those it imports)."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update({t.id for t in node.targets if isinstance(t, ast.Name)})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not n.startswith("_")}


def surface() -> list[str]:
    lines = []
    for path in sorted(AGENTS.rglob("*.py")):
        parts = path.relative_to(AGENTS.parent.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        lines += [f"{module}:{name}" for name in sorted(defined_names(path))]
    return lines


def canonical(value: Any) -> Any:
    """A JSON form of a config value: the value itself, a dataclass's fields
    under its class name, else the type's name."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return {
            type(value).__name__: {
                f.name: canonical(getattr(value, f.name)) for f in fields
            }
        }
    if getattr(value, "shape", None) == ():
        return value.item()
    return f"<{type(value).__name__}>"


def configs(name: str) -> dict[str, Any]:
    """The run config the loggers record and the agent's ``agent_config``,
    the agent built on its defaults."""
    cls = getattr(ajax, name)
    if name == "TDMPC2MultiTask":
        agent = _multitask()
    elif name in CASES:
        agent = cls(env_id=CASES[name].envs[0], **CASES[name].required)
    else:
        agent = cls(env_id=OTHERS[name])
    return {
        "config": canonical(agent.config),
        "agent_config": canonical(agent.agent_config),
    }


def contract(name: str) -> dict[str, Any]:
    """An agent class's :data:`CONTRACT` (sets as sorted lists) and the
    ``module:qualified name`` of its make_train."""

    def plain(value: Any) -> Any:
        if isinstance(value, frozenset | set):
            return sorted(value)
        if isinstance(value, Mapping):
            return {k: plain(v) for k, v in sorted(value.items())}
        return value

    cls = getattr(ajax, name)  # TDMPC2MultiTask is no ActorCritic: what it has
    make_train = getattr(cls, "make_train", None)
    home = make_train and f"{make_train.__module__}:{make_train.__qualname__}"
    found = {n: plain(getattr(cls, n)) for n in CONTRACT if hasattr(cls, n)}
    return {**found, "make_train": home}


def test_every_recorded_name_still_imports() -> None:
    missing = []
    for line in json.loads(SURFACE.read_text()):
        module, name = line.split(":")
        if not hasattr(importlib.import_module(module), name):
            missing.append(line)
    assert not missing, missing


@pytest.mark.parametrize("name", ajax.__all__)
def test_an_agent_keeps_its_recorded_configs(name: str) -> None:
    assert configs(name) == json.loads(CONFIGS.read_text())[name]


@pytest.mark.parametrize("name", ajax.__all__)
def test_an_agent_keeps_its_recorded_contract(name: str) -> None:
    assert contract(name) == json.loads(CONTRACTS.read_text())[name]


if __name__ == "__main__":
    names = sorted(set(json.loads(SURFACE.read_text())).union(surface()))
    SURFACE.write_text(json.dumps(names, indent=0) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = {name: configs(name) for name in ajax.__all__}
    CONFIGS.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")
    contracts = {name: contract(name) for name in ajax.__all__}
    CONTRACTS.write_text(json.dumps(contracts, indent=1, sort_keys=True) + "\n")
