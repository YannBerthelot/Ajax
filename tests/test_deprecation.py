"""A name moved out of a module still imports from it, with a FutureWarning
naming its new home (ajax.deprecation.moved_names); one that became a method
raises with the method's call form, and none can be written there."""

import importlib
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from ajax.deprecation import moved_names

#: The public names every agent module defines or serves (test_agent_surface).
SURFACE = Path(__file__).parent / "agents" / "agent_surface.json"


@pytest.fixture
def old_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("old_home")
    monkeypatch.setitem(sys.modules, "old_home", module)  # as while it runs
    module.__getattr__ = moved_names(  # type: ignore[method-assign]
        "old_home",
        {"warn_inert": "ajax.deprecation", "gone": "ajax.deprecation:moved_names"}
        | {"train_agent": "ajax.agents.base:ActorCritic.train"},
    )
    return module


def test_a_moved_name_resolves_from_its_new_home_with_a_warning(
    old_module: types.ModuleType,
) -> None:
    from ajax import deprecation

    with pytest.warns(FutureWarning, match="old_home.warn_inert moved to ajax"):
        assert old_module.warn_inert is deprecation.warn_inert
    with pytest.warns(FutureWarning, match="deprecation.moved_names; the old path"):
        from old_home import gone  # type: ignore[import-not-found]
    assert gone is moved_names


def test_a_function_that_became_a_method_raises_with_its_call_form(
    old_module: types.ModuleType,
) -> None:
    """Its old arguments cannot bind to the method's: a call says how to
    call the method instead of failing far inside it."""
    with pytest.warns(FutureWarning, match="moved to ajax.agents.base.ActorCritic"):
        train_agent = old_module.train_agent
    call_form = r"old_home.train_agent became ajax.agents.base.ActorCritic.train\(self"
    with pytest.raises(TypeError, match=call_form + r".*instance of ActorCritic"):
        train_agent("Pendulum-v1", 0)


def test_writing_a_moved_name_on_its_old_module_raises(
    old_module: types.ModuleType,
) -> None:
    """Nothing reads a moved name on its old module, so a write there (as
    monkeypatch.setattr does) would change nothing."""
    with pytest.raises(AttributeError, match="old_home.warn_inert moved to ajax"):
        old_module.warn_inert = print  # type: ignore[attr-defined]
    old_module.other = 1  # type: ignore[attr-defined]  # never moved: writable
    assert old_module.other == 1  # type: ignore[attr-defined]


def test_a_name_that_never_lived_there_is_still_an_attribute_error(
    old_module: types.ModuleType,
) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'other'"):
        old_module.other  # noqa: B018


#: The old functions of IQL's, CrossQ's and TD3's train modules that became
#: methods (their update, init and make_train).
BECAME_METHODS = {
    "init_IQL",
    "init_CrossQ",
    "init_TD3",
    "update_agent",
    "update_critics",
    "update_actor_and_temperature",
    "update_value_functions",
    "update_policy",
    "make_train",
}


@pytest.mark.parametrize("agent", ["IQL", "CrossQ", "TD3", "PQN"])
def test_the_declared_agents_old_names_resolve_or_give_the_new_call(
    agent: str,
) -> None:
    """Every name the surface record holds for the old modules of a declared
    agent resolves to its home in the agent's file, or, become a method,
    raises TypeError naming the method; none can be written there."""
    olds = tuple(
        f"ajax.agents.{agent}.{m}" for m in (f"train_{agent}", "state", "utils")
    )
    recorded = json.loads(SURFACE.read_text())
    lines = [line.split(":") for line in recorded if line.startswith(olds)]
    assert lines
    for path, name in lines:
        module = importlib.import_module(path)
        with pytest.warns(FutureWarning, match=f"{path}.{name} moved to"):
            found = getattr(module, name)
        if name in BECAME_METHODS:
            with pytest.raises(TypeError, match=f"{path}.{name} became .*self"):
                found()
        else:
            home = inspect.getmodule(found)
            assert home is not None
            assert home.__name__ == f"ajax.agents.{agent}.{agent}"
        with pytest.raises(AttributeError, match=f"{path}.{name} moved to"):
            setattr(module, name, found)


def test_ppos_old_names_resolve_or_give_the_new_call() -> None:
    """PPO's old modules serve every name the surface record holds for them
    (and _compute_gae, which downstream reads) from PPO.py or a shared
    module; its init, update and make_train became methods. core keeps one
    legacy definition, resolve_clip_coef."""
    olds = tuple(
        f"ajax.agents.PPO.{m}" for m in ("train_PPO", "state", "core", "utils")
    )
    recorded = json.loads(SURFACE.read_text())
    lines = [line.split(":") for line in recorded if line.startswith(olds)]
    lines += [["ajax.agents.PPO.utils", "_compute_gae"]]
    assert len(lines) > 20
    for path, name in lines:
        module = importlib.import_module(path)
        if name == "resolve_clip_coef":
            assert inspect.getmodule(module.resolve_clip_coef) is module
            continue
        with pytest.warns(FutureWarning, match=f"{path}.{name} moved to"):
            found = getattr(module, name)
        if name in {"init_PPO", "update_agent", "make_train"}:
            with pytest.raises(TypeError, match=f"{path}.{name} became .*self"):
                found()
        with pytest.raises(AttributeError, match=f"{path}.{name} moved to"):
            setattr(module, name, found)
