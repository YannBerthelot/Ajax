"""Every module on every agent.

Each agent is built with each extension of :mod:`ajax.extensions` (found by
import, so a new one is in the matrix from its first commit) and with each
network option (``memory``, ``cnn``). Every cell must be one of:

* supported: the agent builds;
* refused for capabilities its algorithm (or a module form, ``FORM_GAPS``)
  lacks: :class:`~ajax.extensions.capabilities.MissingCapability`;
* a gap the agent's code has not wired yet, pinned in ``GAPS`` with the
  work that closes it.

Anything else fails: another exception, a gap not pinned, a pinned gap that
now works. The matrix is written to ``docs/guide/compatibility.md``, which
must match (``AJAX_WRITE_COMPATIBILITY=1`` rewrites it).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import pkgutil
from functools import cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import ajax
import ajax.extensions as extensions_package
from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.base import Extension
from ajax.extensions.capabilities import MissingCapability, NotWired
from ajax.extensions.pretrain import MCPretrain
from ajax.networks.memory import MemoryConfig
from tests.agents.TDMPC2.test_TDMPC2MultiTask import make_dataset

#: (agent, environment): each agent on each action kind it takes.
ROWS = [
    *((name, "Pendulum-v1") for name in "SAC TD3 REDQ ASAC AVG APG TDMPC2".split()),
    *(
        (name, env)
        for name in "PPO APO UDRL DreamerV3".split()
        for env in ("Pendulum-v1", "CartPole-v1")
    ),
    ("DQN", "CartPole-v1"),
    ("PQN", "CartPole-v1"),
    ("TDMPC2MultiTask", "dataset"),
]
NETWORK_OPTIONS = ("memory", "cnn")

# Why each gap is open, and the work that closes it.
ACTION = "only SAC folds action and transform_action (batch module-reach)"
MC = "MCPretrain binds only on SAC (batch module-reach)"
CONTEXT = "the agent binds no context for it yet"
EXPERT = "the agent takes no expert_policy yet (expert reach)"
FORM = (
    "no form for this agent yet (cross-entropy for discrete actors; a policy's encoder)"
)
WORLD = "world models fold no loss phase for extensions and build their own encoders"
MEMORY = "memory is not wired on this agent yet (batch network-reach)"
DEFERRED = "memory on this agent has no published form (deferred)"
PRIOR = "the agent's own replay is not handed to its pretraining yet"

#: Module forms not written yet: a refusal for these capabilities alone is
#: a gap, not inherent.
FORM_GAPS = {
    ("ImitationLoss", "continuous_actions"),
    ("CloningPretrain", "continuous_actions"),
    ("ConditioningMetrics", "critic"),  # it could read a policy's encoder
}

#: The pinned gaps: reason -> module -> the rows (agents, and "/discrete"
#: for an agent taking both action kinds) where it is not wired yet.
GAPS: dict[str, dict[str, str]] = {
    ACTION: {
        "EDGEExploration": "TD3 REDQ ASAC AVG UDRL UDRL/discrete DQN PQN",
        "JSRLCurriculum": "TD3 REDQ ASAC AVG UDRL UDRL/discrete DQN PQN",
        "ResidualPolicy": "TD3 REDQ ASAC AVG APG PPO APO UDRL",
        "ValueBox": "TD3 REDQ AVG DQN PQN",
    },
    MC: {
        "CriticBlend": "TD3 REDQ AVG DQN PQN",
        "MCPretrain": "TD3 REDQ AVG PPO PPO/discrete DQN PQN",
        "MCVarianceCorrection": "TD3 REDQ",
        "OnlineBC": "TD3 REDQ AVG",
        "PhiRefresh": "TD3 REDQ DQN",
    },
    CONTEXT: {
        "BellmanPretrain": "TD3 REDQ ASAC",
        "CloningPretrain": "APG PPO UDRL",
    },
    EXPERT: {
        "BellmanPretrain": "DQN",
        "ExpertValueMetrics": "DQN",
    },
    FORM: {
        "CloningPretrain": "PPO/discrete APO/discrete UDRL/discrete DreamerV3/discrete",
        "ImitationLoss": "PPO/discrete APO/discrete UDRL/discrete DreamerV3/discrete",
        "ConditioningMetrics": "APG UDRL UDRL/discrete",
    },
    WORLD: {
        **dict.fromkeys(
            "CriticBlend IBRL KernelRepulsion LCBGatedBootstrap MCVarianceCorrection"
            " OnlineBC PhiRefresh ValueBox ExpertValueMetrics".split(),
            "TDMPC2 TDMPC2MultiTask",
        ),
        **dict.fromkeys(
            "EDGEExploration JSRLCurriculum MCPretrain BellmanPretrain"
            " BiasVorePenalty cnn".split(),
            "TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask",
        ),
        **dict.fromkeys(
            "CloningPretrain ImitationLoss ResidualPolicy".split(),
            "TDMPC2 DreamerV3 TDMPC2MultiTask",
        ),
    },
    MEMORY: {"memory": "DQN PQN"},
    DEFERRED: {"memory": "AVG UDRL UDRL/discrete"},
    PRIOR: {
        "PriorData": "UDRL UDRL/discrete TDMPC2 DreamerV3 DreamerV3/discrete"
        " TDMPC2MultiTask"
    },
}


def _expert(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(obs.shape[:-1] + (1,))


def _discrete_expert(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(obs.shape[:-1], jnp.int32)


@cache
def extension_classes() -> dict[str, type[Extension]]:
    found = {}
    for info in pkgutil.iter_modules(extensions_package.__path__):
        module = importlib.import_module(f"ajax.extensions.{info.name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Extension) and cls.__module__ == module.__name__:
                found[name] = cls
    found.pop("Extension", None)
    return found


def agent_class(name: str) -> Any:
    return getattr(ajax, name)


#: Values for an extension's fields without a default.
def field_values(env: str) -> dict[str, Any]:
    expert = _discrete_expert if env == "CartPole-v1" else _expert
    one = np.zeros((1, 1))
    episode = TransitionDataset(one, one, one[:, 0], one[:, 0] + 1, one[:, 0], one)
    return {"expert_policy": expert, "threshold": 1.0, "dataset": episode}


def build_extension(cls: type[Extension], env: str) -> Extension:
    values = field_values(env)
    required = [
        f.name
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.init
    ]
    unknown = [f for f in required if f not in values]
    assert not unknown, f"give the guard a value for {cls.__name__}'s {unknown}"
    return cls(**{f: values[f] for f in required})


def build(agent: str, env: str, **kwargs: Any) -> Any:
    cls = agent_class(agent)
    if agent == "TDMPC2MultiTask":
        return cls(make_dataset(), **kwargs)
    if "expert_policy" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("expert_policy", field_values(env)["expert_policy"])
    return cls(env, **kwargs)


def verdict(agent: str, env: str, column: str) -> tuple[str, str]:
    """``(kind, detail)`` of one cell: kind ``supported``, ``inherent`` or
    ``gap``; raises on any other outcome."""
    if column == "memory":
        if "obs_networks" not in agent_class(agent).capabilities:
            return "inherent", "obs_networks"
        kwargs: dict[str, Any] = {"memory": MemoryConfig()}
    elif column == "cnn":
        kwargs = {"cnn_image_shape": (1, 1, 1)}
    else:
        ext = build_extension(extension_classes()[column], env)
        expert = field_values(env)["expert_policy"]
        companions = (
            (MCPretrain(expert_policy=expert),) if "phi_star" in ext.needs() else ()
        )
        kwargs = {"extensions": (*companions, ext)}
    try:
        build(agent, env, **kwargs)
    except MissingCapability as error:
        lacks = frozenset().union(*error.missing.values())
        if all((column, cap) in FORM_GAPS for cap in lacks):
            return "gap", ", ".join(sorted(lacks))
        return "inherent", ", ".join(sorted(lacks))
    except NotWired as error:
        return "gap", "; ".join(f"{k}: {v}" for k, v in error.unwired.items())
    except (NotImplementedError, TypeError) as error:
        option = column in NETWORK_OPTIONS and (
            isinstance(error, NotImplementedError) or column in str(error)
        )
        if not option:
            raise
        return "gap", f"{column} option"
    return "supported", ""


def columns() -> list[str]:
    return [*sorted(extension_classes()), *NETWORK_OPTIONS]


def row_label(agent: str, env: str) -> str:
    discrete = env == "CartPole-v1" and agent in ("PPO", "APO", "UDRL", "DreamerV3")
    return f"{agent}/discrete" if discrete else agent


@cache
def matrix() -> dict[tuple[str, str], tuple[str, str]]:
    return {
        (row_label(agent, env), column): verdict(agent, env, column)
        for agent, env in ROWS
        for column in columns()
    }


def pinned() -> dict[tuple[str, str], str]:
    """The cells of ``GAPS``, each with its (first) reason."""
    cells: dict[tuple[str, str], str] = {}
    for reason, modules in GAPS.items():
        for column, rows in modules.items():
            for row in rows.split():
                cells.setdefault((row, column), reason)
    return cells


def test_every_cell_is_supported_inherent_or_a_pinned_gap() -> None:
    gaps = {cell for cell, (kind, _) in matrix().items() if kind == "gap"}
    unpinned = sorted(gaps - set(pinned()))
    closed = sorted(set(pinned()) - gaps)
    assert not unpinned, f"wire these, or pin them in GAPS with their work: {unpinned}"
    assert not closed, f"these gaps are closed: remove them from GAPS: {closed}"


def render() -> str:
    rows = [row_label(agent, env) for agent, env in ROWS]
    symbol = {"supported": "yes", "inherent": "no", "gap": "not yet"}
    head = "| Module | " + " | ".join(rows) + " |\n"
    rule = "| --- |" + " --- |" * len(rows) + "\n"
    body = "".join(
        f"| `{column}` | "
        + " | ".join(symbol[matrix()[(row, column)][0]] for row in rows)
        + " |\n"
        for column in columns()
    )
    lacking: dict[tuple[str, str], list[str]] = {}
    for (row, column), (kind, detail) in matrix().items():
        if kind == "inherent":
            lacking.setdefault((column, detail), []).append(row)
    inherent = [
        f"- `{column}` needs {detail}: not on {', '.join(rows)}"
        for (column, detail), rows in sorted(lacking.items())
    ]
    gaps = [
        f"- {reason}: "
        + "; ".join(f"`{column}` on {rows}" for column, rows in modules.items())
        for reason, modules in GAPS.items()
    ]
    return (
        "# Compatibility\n\n"
        "Which module runs on which agent, as the agent is built. Generated by"
        " `tests/agents/test_compatibility.py`; do not edit by hand.\n\n"
        "**yes**: supported. **no**: the algorithm lacks what the module needs"
        " (listed below). **not yet**: the agent has it, and its code does not"
        " pass it on yet.\n\n"
        + head
        + rule
        + body
        + "\n## Refused: what the algorithm lacks\n\n"
        + "\n".join(inherent)
        + "\n\n## Not yet wired\n\n"
        + "\n".join(gaps)
        + "\n"
    )


DOCS = Path(__file__).parents[2] / "docs" / "guide" / "compatibility.md"


def test_the_docs_table_matches_the_matrix() -> None:
    text = render()
    if os.environ.get("AJAX_WRITE_COMPATIBILITY"):
        DOCS.write_text(text)
    assert DOCS.exists() and DOCS.read_text() == text, (
        "docs/guide/compatibility.md is stale: rerun this test with"
        " AJAX_WRITE_COMPATIBILITY=1"
    )


if __name__ == "__main__":  # print the gap cells, to pin them
    for (row, column), (kind, detail) in sorted(matrix().items()):
        if kind == "gap":
            print(f"{row}:{column}  ({detail})")
