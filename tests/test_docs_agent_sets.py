"""The lists of agents the documentation states are the code's.

The contract (docs/algorithms/index.md, "Agent families"): a family is the
class of ``ajax.agents.families`` an agent is built on, and a property a page
attributes to a list of agents holds for exactly those agents. A page marks
such a list ``<!-- agents: KEY -->SAC, TD3 and DQN<!-- /agents -->``; each
KEY below says how the code answers. ``train_offline``'s refusal names the
agents that accept a dataset, by the same rule. The presets table of
docs/guide/training.md names, per agent, the presets of its ``PRESETS`` table
and the fields they set.

Each test catches a list left behind when an agent joins or leaves a set, as
the review of 2026-11-14 found on three pages.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from typing import Callable

import pytest

import ajax
from ajax.agents import families
from ajax.agents.base import ActorCritic
from ajax.agents.presets import presets_of
from tests.doc_examples import ROOT
from tests.test_exports import library_agents

MARKED = re.compile(r"<!-- agents: ([a-z-]+) -->(.*?)<!-- /agents -->", re.DOTALL)
#: The names the prose gives two agents, beside their class names.
PROSE_NAMES = {"RVI-SAC": "RVISAC", "TD-MPC2": "TDMPC2"}


def built_on(family: type) -> Callable[[type[ActorCritic]], bool]:
    return lambda agent: issubclass(agent, family)


def folds(phase: str) -> Callable[[type[ActorCritic]], bool]:
    return lambda agent: phase in agent.supported_extension_phases


def accepts_a_dataset(agent: type[ActorCritic]) -> bool:
    return "offline_data" in inspect.signature(agent.make_train).parameters


SETS: dict[str, Callable[[type[ActorCritic]], bool]] = {
    "replay": built_on(families.ReplayAgent),
    "sequence-replay": built_on(families.RecurrentReplayAgent),
    "on-policy": built_on(families.OnPolicyAgent),
    "incremental": built_on(families.IncrementalAgent),
    "own-iteration": built_on(families.OwnIterationAgent),
    "own-replay": built_on(families.OwnReplayAgent),
    "offline": accepts_a_dataset,
    "raw-replay": lambda agent: agent.replays_transitions,
    "warmup-action": folds("choose_warmup_action"),
    "replay-action": folds("choose_replay_action"),
    # The agents a stateful expert reaches extensions on, through the actions
    # they store from its state.
    "expert-actions": lambda agent: agent.stores_expert_actions,
}


def agents_where(holds: Callable[[type[ActorCritic]], bool]) -> set[str]:
    classes = [getattr(ajax, name) for name in ajax.__all__]
    return {
        agent.__name__
        for agent in classes
        if inspect.isclass(agent) and issubclass(agent, ActorCritic) and holds(agent)
    }


def names_in(text: str) -> set[str]:
    """The agents of a list in prose: links, backticks and "and" removed."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text).replace("`", "")
    names = re.split(r",|\band\b", text)
    return {PROSE_NAMES.get(name.strip(), name.strip()) for name in names} - {""}


def marked_lists() -> list:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    return [
        pytest.param(key, listed, id=f"{page.relative_to(ROOT)}:{key}:{place}")
        for page in pages
        for place, (key, listed) in enumerate(MARKED.findall(page.read_text()))
    ]


@pytest.mark.parametrize("key, listed", marked_lists())
def test_a_marked_list_names_exactly_the_agents_of_its_set(key: str, listed: str):
    assert names_in(listed) == agents_where(SETS[key])


def test_every_set_is_stated_somewhere():
    """Catches a marker renamed on a page, which would leave its set
    unchecked without failing anything."""
    assert {case.values[0] for case in marked_lists()} == set(SETS)


def test_the_offline_refusal_names_the_agents_that_accept_a_dataset():
    """Catches the message sending a reader to an agent that refuses too, or
    leaving out one that would have served."""
    with pytest.raises(ValueError, match="cannot train offline") as refusal:
        ajax.PPO("CartPole-v1").train_offline(None, n_updates=1)
    named = re.search(r"\(([^)]*)\)", str(refusal.value))
    assert named is not None
    assert names_in(named.group(1)) == agents_where(accepts_a_dataset)


PRESET_ROW = re.compile(r"^\| \[(\w+)\]\([^)]*#presets\) \| (.*?) \| (.*?) \|", re.M)
NAME = re.compile(r"`(\w+)`")


def test_the_presets_table_names_each_agents_presets_and_their_fields():
    """Catches a preset added, renamed or given another field without the
    guide's table following."""
    guide = (ROOT / "docs" / "guide" / "training.md").read_text()
    stated = {
        agent: (set(NAME.findall(presets)), set(NAME.findall(fields)))
        for agent, presets, fields in PRESET_ROW.findall(guide)
    }
    shipped = {
        agent.__name__: (
            set(presets_of(agent)),
            {
                name
                for preset in presets_of(agent).values()
                for name in preset.overrides
            },
        )
        for agent in library_agents()
        if presets_of(agent)
    }
    assert stated == shipped


# Two tables of docs/guide state defaults per agent. The contract is the
# agents' own fields: a table row names a value and the agents whose default
# it is, and every agent that has the field is in one row.
def _default_rows(page: str, header: str) -> list:
    """(value cell, agents) of the table of ``page`` whose header starts with
    ``header``."""
    table = (ROOT / "docs" / "guide" / page).read_text().split(header, 1)[1]
    rows = []
    for line in table.split("\n\n", 1)[0].splitlines()[2:]:
        value, agents = (cell.strip() for cell in line.strip("|").split("|"))
        rows.append((value, names_in(re.sub(r"\([^)]*\)", "", agents))))
    return rows


def _defaults(field: str) -> dict:
    found = {}
    for agent in library_agents():
        if dataclasses.is_dataclass(agent):
            for declared in dataclasses.fields(agent):
                if declared.name == field:
                    found[agent.__name__] = declared.default
    return found


def test_the_n_envs_table_states_each_agents_default():
    stated = {
        agent: int(value)
        for value, agents in _default_rows("training.md", "| `n_envs` default |")
        for agent in agents
    }
    actual = _defaults("n_envs")
    assert stated == {agent: actual[agent] for agent in stated}
    assert set(stated) == {a for a, v in actual.items() if isinstance(v, int)}


def test_the_architecture_table_states_each_agents_default():
    stated = {
        agent: ast.literal_eval(value.strip("`"))
        for value, agents in _default_rows("networks.md", "| Default | Agents |")
        for agent in agents
    }
    actual = _defaults("actor_architecture")
    actual.update(_defaults("architecture"))
    actual = {a: v for a, v in actual.items() if isinstance(v, tuple)}
    assert stated == actual
