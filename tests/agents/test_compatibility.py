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
must match (``AJAX_WRITE_COMPATIBILITY=1`` rewrites it), and the catalogue
of ``docs/guide/extensions.md``, written by hand, must name for each module
the agents the matrix runs it on.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import pkgutil
import re
from collections.abc import Mapping
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
from ajax.extensions.pretrain import ExpertPrefill, MCPretrain
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import CNNSpec
from tests.agents.TDMPC2.test_TDMPC2MultiTask import make_dataset
from tests.test_docs_agent_sets import PROSE_NAMES, SETS, agents_where

#: (agent, environment): each agent on each action kind it takes.
ROWS = [
    *(
        (name, "Pendulum-v1")
        for name in "SAC TD3 REDQ ASAC RVISAC AVG CrossQ TQC IQL APG TDMPC2".split()
    ),
    *(
        (name, env)
        for name in "PPO APO REINFORCE UDRL DAgger DreamerV3".split()
        for env in ("Pendulum-v1", "CartPole-v1")
    ),
    ("DQN", "CartPole-v1"),
    ("PQN", "CartPole-v1"),
    ("TDMPC2MultiTask", "dataset"),
]
NETWORK_OPTIONS = ("memory", "cnn")
#: Module options needing more than the module, each a column of its own.
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "ImitationLoss(q_filter=True)": ("ImitationLoss", {"q_filter": True})
}
VARIANTS["ImitationLoss(target='replay', alpha=2.5)"] = (
    "ImitationLoss",
    {"expert_policy": None, "target": "replay", "alpha": 2.5},
)

# Why each gap is open, and the work that closes it.
MC = "MCPretrain binds only on SAC"
CONTEXT = "the agent binds no context for it yet"
FORM = (
    "no form for this agent yet (cross-entropy for discrete actors; a policy's encoder)"
)
OWN_COLLECTOR = "its collector steps the env itself, with no track_memory phase yet"
WORLD = "world models fold no loss phase for extensions and build their own encoders"
CNN = "the cnn options are not on this agent yet"
MEMORY = "memory is not wired on this agent yet"
DEFERRED = "memory on this agent has no published form to follow yet"
PRIOR = "the agent's own replay is not handed to its pretraining yet"
REWARD = "only PPO folds the reward phase (GAIL's AdversarialReward)"
FIRST = "REINFORCE takes no expert and folds no loss phase for extensions yet"
WARMUP = (
    "TD-MPC2's own collector does not run the choose_warmup_action phase in its"
    " seed phase yet, and it takes no expert_policy"
)
OWN_STORE = "the agent's own collector does not run choose_replay_action yet"
ROLLOUT_REWARD = "APG's closed-loop rollout passes the features no reward yet"
FEATURES = (
    "ExpertActionInput ExpertStateInput TrainingProgressInput PIDTermsInput"
    " PreviousAction PreviousReward".split()
)
# The expert features cannot reach a world model's imagined steps; the
# training progress and PID terms join the observations it collects, and
# imagination reads latents alone.
IMAGINED = (
    "imagination has no observation for the expert to read: inherent, refused"
    " as unwired"
)
LOGGED = (
    "it trains offline on logged observations, which hold no feature columns:"
    " inherent, refused as unwired"
)
WORLD_ROWS = "TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask"

#: Module forms not written yet: a refusal for these capabilities alone is
#: a gap, not inherent.
FORM_GAPS = {
    ("ImitationLoss", "continuous_actions"),
    ("CloningPretrain", "continuous_actions"),
    ("CriticDiagnostics", "critic"),  # it could read a policy's encoder
}

#: The pinned gaps: reason -> module -> the rows (agents, and "/discrete"
#: for an agent taking both action kinds) where it is not wired yet.
GAPS: dict[str, dict[str, str]] = {
    MC: {
        "CriticBlend": "TD3 REDQ CrossQ AVG DQN PQN IQL",
        "MCPretrain": "TD3 REDQ TQC CrossQ AVG PPO PPO/discrete DQN PQN IQL"
        " REINFORCE REINFORCE/discrete",
        "MCVarianceCorrection": "TD3 REDQ CrossQ IQL",
        "OnlineBC": "TD3 REDQ TQC CrossQ AVG IQL",
        "PhiRefresh": "TD3 REDQ TQC CrossQ DQN IQL",
        "ValueBox": "TD3 REDQ TQC CrossQ AVG DQN PQN IQL",
    },
    CONTEXT: {
        "BellmanPretrain": "ASAC RVISAC DQN IQL",
        "CloningPretrain": "APG PPO UDRL",
        # UDRL's networks read a command the env's rollouts cannot give, and
        # APG's controller may keep a state between steps.
        "RepresentationProbe": "APG UDRL UDRL/discrete",
    },
    OWN_COLLECTOR: {"BumplessTracking": "APG UDRL UDRL/discrete"},
    FORM: {
        "CloningPretrain": "PPO/discrete APO/discrete UDRL/discrete DAgger/discrete"
        " DreamerV3/discrete",
        "ImitationLoss": "PPO/discrete APO/discrete UDRL/discrete DAgger/discrete"
        " DreamerV3/discrete",
        "CriticDiagnostics": "UDRL UDRL/discrete DAgger DAgger/discrete",
    },
    WORLD: {
        **dict.fromkeys(
            "CriticBlend IBRL KernelRepulsion LCBGatedBootstrap MCVarianceCorrection"
            " OnlineBC ValueBox ExpertValueMetrics"
            " ImitationLoss(q_filter=True)".split(),
            "TDMPC2 TDMPC2MultiTask",
        ),
        **dict.fromkeys(
            [
                "ImitationLoss(target='replay', alpha=2.5)",
                *("ConservativeQ", "CalQL", "TargetBC"),
            ],
            "TDMPC2 TDMPC2MultiTask",
        ),
        **dict.fromkeys(
            ("JSRLCurriculum", "MCPretrain", "BiasVorePenalty", "cnn"),
            "TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask",
        ),
        # Neither has a discrete form: DreamerV3's discrete row refuses them.
        **dict.fromkeys(
            ("EDGEExploration", "EpsilonGreedyPresample"),
            "TDMPC2 DreamerV3 TDMPC2MultiTask",
        ),
        "BellmanPretrain": "TDMPC2 TDMPC2MultiTask",
        **dict.fromkeys(
            "CloningPretrain ImitationLoss".split(),
            "TDMPC2 DreamerV3 TDMPC2MultiTask",
        ),
    },
    IMAGINED: dict.fromkeys(FEATURES[:2], WORLD_ROWS),
    LOGGED: dict.fromkeys(FEATURES[2:], "TDMPC2MultiTask"),
    CNN: {
        "cnn": "SAC TD3 REDQ TQC CrossQ ASAC RVISAC AVG APG APO APO/discrete DAgger"
        " DAgger/discrete IQL REINFORCE REINFORCE/discrete"
    },
    MEMORY: {
        "memory": "APO APO/discrete DQN PQN DAgger DAgger/discrete IQL REINFORCE"
        " REINFORCE/discrete"
    },
    DEFERRED: {"memory": "AVG CrossQ UDRL UDRL/discrete"},
    WARMUP: {"ExpertWarmup": "TDMPC2"},
    FIRST: dict.fromkeys(
        "BiasVorePenalty BumplessTracking CloningPretrain ExpertStateInput"
        " ImitationLoss".split(),
        "REINFORCE REINFORCE/discrete",
    ),
    ROLLOUT_REWARD: {"PreviousReward": "APG"},
    REWARD: {
        "AdversarialReward": "SAC TD3 REDQ TQC CrossQ ASAC RVISAC AVG APG APO UDRL DAgger"
        " IQL DreamerV3 REINFORCE"
        " TDMPC2 TDMPC2MultiTask"
    },
    OWN_STORE: {
        "StorePolicyAction": "DreamerV3 DreamerV3/discrete TDMPC2 TDMPC2MultiTask"
    },
    PRIOR: dict.fromkeys(
        ("PriorData", "ExpertPrefill", "RLPD", "RetainedDemonstrations"),
        "UDRL UDRL/discrete DAgger DAgger/discrete TDMPC2 DreamerV3"
        " DreamerV3/discrete TDMPC2MultiTask",
    ),
}


def _expert(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(obs.shape[:-1] + (1,))


def _discrete_expert(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(obs.shape[:-1] + (1,), jnp.int32)


class _GainExpert:
    """A stateful expert with one learnable gain, which gain mode
    (ExpertGains) needs."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs: int) -> jnp.ndarray:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jnp.ndarray, obs: jnp.ndarray) -> tuple:
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state: Any, obs: jnp.ndarray, gains: Any) -> tuple:
        return -gains[0] * obs[..., :1], state


#: The agent's expert a module needs beyond the stateless one.
MODULE_EXPERTS = {"ExpertGains": _GainExpert(), "BumplessTracking": _GainExpert()}


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
    return {
        "expert_policy": expert,
        "target_channels": (0,),
        "reading_channels": (0,),
        "max_change": (0.1,),
        "threshold": 1.0,
        "n_steps": 4,
        "fraction": 0.5,
        "run_directory": "unused",
        "probe_interval": 1,
        "dataset": episode,
        "demonstrations": episode,
        # A PID's error x[1] - x[0], on every row's observation.
        "current_idx": (0,),
        "target_idx": (1,),
    }


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


def verdict(
    agent: str, env: str, column: str, agent_options: Mapping[str, Any] | None = None
) -> tuple[str, str]:
    """``(kind, detail)`` of one cell, the agent built with its defaults or
    with ``agent_options``: kind ``supported``, ``inherent`` or ``gap``;
    raises on any other outcome."""
    if column == "memory":
        if "obs_networks" not in agent_class(agent).capabilities:
            return "inherent", "obs_networks"
        kwargs: dict[str, Any] = {"memory": MemoryConfig()}
    elif column == "cnn":
        kwargs = {"cnn_image_shape": (1, 1, 1), "cnn_extra_obs_dim": 0}
        # DQN's and PQN's reference networks need 3 pixels a side.
        if "cnn_spec" in inspect.signature(agent_class(agent)).parameters:
            kwargs["cnn_spec"] = CNNSpec((2,), (1,), (1,), 8)
    else:
        name, options = VARIANTS.get(column, (column, {}))
        ext = build_extension(extension_classes()[name], env)
        ext = dataclasses.replace(ext, **options)  # type: ignore[type-var]
        expert = field_values(env)["expert_policy"]
        companions = (
            (MCPretrain(expert_policy=expert),) if "phi_star" in ext.needs() else ()
        )
        # A module reading a seeded replay (BellmanPretrain) gets the expert's
        # prefill, which fills it.
        if "seeded_replay" in ext.needs():
            companions = (*companions, ExpertPrefill(n_steps=4))
        kwargs = {"extensions": (*companions, ext)}
        takes = inspect.signature(agent_class(agent).__init__).parameters
        if name in MODULE_EXPERTS and env != "CartPole-v1" and "expert_policy" in takes:
            kwargs["expert_policy"] = MODULE_EXPERTS[name]
    kwargs.update(agent_options or {})
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
    return [*sorted(extension_classes()), *VARIANTS, *NETWORK_OPTIONS]


def row_label(agent: str, env: str) -> str:
    discrete = env == "CartPole-v1" and agent in (
        *("PPO", "APO", "REINFORCE", "UDRL", "DAgger", "DreamerV3"),
    )
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
    unpinned = sorted(gaps.difference(pinned()))
    closed = sorted(set(pinned()).difference(gaps))
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
        "Which module runs on which agent: every [extension](extensions.md)"
        " and network option (`memory`, `cnn`), as the agent is built. Check"
        " here before combining a module with an agent. The page is generated"
        " from the code by `tests/agents/test_compatibility.py`.\n\n"
        "**yes**: supported. **no**: the algorithm lacks a"
        " [capability](extensions.md#what-an-agent-has-and-what-an-extension-needs)"
        " the module needs (listed below). **not yet**: the agent has what the"
        " module needs, and its code does not pass it on yet (the reasons are"
        " below). An agent taking both kinds of action has a second column,"
        " `/discrete`, for discrete actions.\n\n"
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


EXTENSIONS_GUIDE = DOCS.with_name("extensions.md")
#: The capabilities an agent's option gives where the default build lacks
#: them: the option, a value giving it, and the words a catalogue cell
#: states that condition in.
OPTIONS_GIVING = {
    "critic_ensemble": ("num_critics", 2, "two critics or more"),
    "warmup_steps": ("learning_starts", 4, "`learning_starts` above 0"),
}


def catalogue() -> dict[str, str]:
    """Each module of the guide's catalogue with its "Works with" cell, the
    last of its row."""
    text = EXTENSIONS_GUIDE.read_text().split("\n## Catalogue\n", 1)[1]
    cells = {}
    for line in text.split("\n## ", 1)[0].splitlines():
        if line.startswith("| `"):
            row = line.strip().strip("|").split("|")
            cells[row[0].strip(" `")] = row[-1].strip()
    return cells


def catalogue_phrases() -> dict[str, set[str]]:
    """The phrases a cell names a set of agents by, each with its agents:
    the longer of two phrases starting alike comes first."""
    agents = {agent for agent, _ in ROWS}
    takes_expert = {
        agent
        for agent in agents
        if "expert_policy" in inspect.signature(agent_class(agent).__init__).parameters
    }
    critic = {"q_critic", "v_critic"}
    return {
        "every agent with a stateful expert": takes_expert,
        "every agent with an expert": takes_expert,
        "every agent with a critic": {
            a for a in agents if critic.intersection(agent_class(a).capabilities)
        },
        "every agent whose networks read the observation": {
            a for a in agents if "obs_networks" in agent_class(a).capabilities
        },
        "every agent": agents,
        "the off-policy actor-critics": agents_where(SETS["off-policy-actor-critics"]),
        "the agents with a warm-up": agents_where(SETS["warmup-action"]),
    }


def stated_agents(module: str, cells: Mapping[str, str]) -> set[str]:
    """The agents ``module``'s cell names: agents, catalogue_phrases and
    "as `Other`" (that module's), less those after "but". The set comes
    first; its conditions (", with ...", "; ...", an aside in parentheses)
    and a heading ending in a colon name no agent."""
    phrases = catalogue_phrases()
    text = re.sub(r"<!--.*?-->", "", cells[module])
    for index, phrase in enumerate(phrases):
        text = text.replace(phrase, f"#{index}")
    text = re.sub(r"\([^)]*\)", "", text).split(";")[0].split(" with ")[0]
    named, _, excluded = text.split(":")[-1].partition(" but ")

    def agents_in(part: str) -> set[str]:
        found: set[str] = set()
        for item in re.split(r",|\band\b", part.replace("`", "")):
            item = item.strip()
            if item.startswith("as "):
                found.update(stated_agents(item.removeprefix("as "), cells))
            elif item.startswith("#"):
                found.update(list(phrases.values())[int(item[1:])])
            elif item:
                found.add(PROSE_NAMES.get(item, item))
        return found

    return agents_in(named).difference(agents_in(excluded))


def accepting_agents(module: str, cell: str) -> set[str]:
    """The agents the matrix runs ``module`` on, with either action kind;
    an agent refused only for capabilities an option of its own gives
    (OPTIONS_GIVING) is built again with it, where the cell states it."""
    accepting = set()
    for agent, env in ROWS:
        kind, detail = matrix()[(row_label(agent, env), module)]
        if kind == "inherent":
            fields = inspect.signature(agent_class(agent)).parameters
            lacks = detail.split(", ")
            given = [OPTIONS_GIVING.get(capability) for capability in lacks]
            if all(g is not None and g[0] in fields and g[2] in cell for g in given):
                options = {g[0]: g[1] for g in given if g is not None}
                kind = verdict(agent, env, module, options)[0]
        if kind == "supported":
            accepting.add(agent)
    return accepting


def test_the_catalogue_names_the_agents_each_module_runs_on() -> None:
    """Each "Works with" cell of the guide's catalogue names exactly the
    agents the matrix runs its module on, and every module has a row.
    Catches a cell left behind when an agent gains or loses a module, which
    the 0.6.0 review found in several rows."""
    cells = catalogue()
    assert set(cells) == set(extension_classes())
    wrong = {}
    for module, cell in sorted(cells.items()):
        stated, accepting = stated_agents(module, cells), accepting_agents(module, cell)
        if stated != accepting:
            wrong[module] = {
                "named, not run": sorted(stated.difference(accepting)),
                "run, not named": sorted(accepting.difference(stated)),
            }
    assert not wrong, f"docs/guide/extensions.md's catalogue: {wrong}"


if __name__ == "__main__":  # print the gap cells, to pin them
    for (row, column), (kind, detail) in sorted(matrix().items()):
        if kind == "gap":
            print(f"{row}:{column}  ({detail})")
