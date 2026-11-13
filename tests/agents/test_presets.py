"""Per-task presets (ajax.agents.presets, DeclaredAgent.preset).

The contract, from the API's documentation:

- ``Agent.preset(name, env_id, **overrides)`` builds the agent with the
  values of the task ``name`` in the PRESETS table of the agent's module;
- the overrides given win over the preset's values; the agent's other
  fields keep their defaults;
- an unknown name raises ValueError naming the presets that exist;
- the run config records the preset's name under "preset";
- every entry of every table the library ships sets the agent's own fields,
  and builds; the agent's algorithm page names it under "## Presets".
"""

import dataclasses
import re
import sys
import types

import pytest

from ajax import SAC
from ajax.agents.families import DeclaredAgent
from ajax.agents.presets import Preset, presets_of
from tests.doc_examples import ROOT
from tests.test_exports import library_agents

#: The table of TunedSAC, the module's agent: a task setting two fields.
PRESETS = {"Swing": Preset({"tau": 0.01, "gamma": 0.9}, "this test")}


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class TunedSAC(SAC):
    """SAC defined in this module, so that this module's PRESETS is its table."""


def test_a_preset_sets_its_values_and_leaves_the_rest_at_the_defaults():
    """Catches a preset whose values do not reach the agent, or one that
    resets fields it does not name."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1")

    assert (agent.tau, agent.gamma) == (0.01, 0.9)
    assert agent.batch_size == TunedSAC("Pendulum-v1").batch_size


def test_the_overrides_given_win_over_the_preset():
    """Catches a preset applied over the caller's own arguments."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1", tau=0.5, batch_size=32)

    assert (agent.tau, agent.gamma, agent.batch_size) == (0.5, 0.9, 32)


def test_the_run_config_records_the_preset_name():
    """Catches a logged run that does not say which preset built it."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1")

    assert agent.config["preset"] == "Swing"
    assert agent.config["tau"] == 0.01


def test_an_unknown_name_raises_and_names_the_presets():
    """Catches a misspelt task silently building the defaults."""
    with pytest.raises(ValueError, match="'Swnig'.*Swing"):
        TunedSAC.preset("Swnig", "Pendulum-v1")


def test_a_subclass_defined_elsewhere_does_not_take_its_parents_table(
    monkeypatch,
):
    """Catches a subclass in another module (a variant of the algorithm)
    built with values tuned for its parent."""
    module = types.ModuleType("a_module_without_presets")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Elsewhere(TunedSAC):
        __module__ = module.__name__

    with pytest.raises(ValueError, match="its presets: none"):
        Elsewhere.preset("Swing", "Pendulum-v1")


SHIPPED = sorted(
    (
        (cls, name)
        for cls in library_agents()
        if issubclass(cls, DeclaredAgent)
        for name in presets_of(cls)
    ),
    key=lambda case: (case[0].__name__, case[1]),
)
SHIPPED_IDS = [f"{cls.__name__}-{name}" for cls, name in SHIPPED]


def test_the_library_ships_presets():
    """Catches the tables dropping out of their modules (a renamed PRESETS)."""
    assert {cls.__name__ for cls, _ in SHIPPED} >= {"ASAC", "PPO", "PQN", "TQC"}


@pytest.mark.parametrize(("cls", "name"), SHIPPED, ids=SHIPPED_IDS)
def test_every_shipped_preset_sets_the_agents_fields_and_builds(cls, name):
    """Catches an entry naming a field the agent does not have (a typo, a
    renamed field) or a value the agent refuses. The preset builds on a
    stand-in env the agent acts in, a field's check needing no simulator of
    the task: brax's fast, which also takes an action repeat, unless the
    agent only takes discrete actions."""
    preset = presets_of(cls)[name]
    init_fields = {field.name for field in dataclasses.fields(cls) if field.init}
    assert set(preset.overrides) <= init_fields
    assert preset.citation

    env_id = "CartPole-v1" if cls.action_space == "discrete" else "fast"
    agent = cls.preset(name, env_id)

    for field, value in preset.overrides.items():
        assert getattr(agent, field) == value, field


@pytest.mark.parametrize(("cls", "name"), SHIPPED, ids=SHIPPED_IDS)
def test_every_shipped_preset_is_named_on_its_algorithm_page(cls, name):
    """Catches a preset added or renamed without its page saying so."""
    page = ROOT / "docs" / "algorithms" / f"{cls.__name__.lower()}.md"
    section = page.read_text().split("\n## Presets\n", 1)[1].split("\n## ", 1)[0]

    assert re.search(f'[`"]{name}[`"]', section)
