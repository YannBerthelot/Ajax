"""``from ajax import <agent>`` works for every agent Ajax ships."""

import importlib
import inspect
from pathlib import Path

import ajax
from ajax.agents.base import ActorCritic


def test_every_agent_class_is_exported():
    # CONTRIBUTING's agent anatomy: an agent folder holds its public class in
    # the module of the same name (SAC/SAC.py, TDMPC2/TDMPC2MultiTask.py).
    agents = {}
    for path in sorted((Path(ajax.__file__).parent / "agents").glob("*/[!_]*.py")):
        module = importlib.import_module(f"ajax.agents.{path.parent.name}.{path.stem}")
        cls = getattr(module, path.stem, None)
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            agents[cls.__name__] = cls
    # An ActorCritic agent named otherwise would escape the scan above.
    library = {
        c for c in ActorCritic.__subclasses__() if c.__module__.startswith("ajax.")
    }
    assert library <= set(agents.values())

    assert sorted(set(agents) - set(ajax.__all__)) == []
    assert all(getattr(ajax, name) is cls for name, cls in agents.items())
