"""A name moved out of a module still imports from it, with a FutureWarning
naming its new home (ajax.deprecation.moved_names)."""

import sys
import types

import pytest

from ajax.deprecation import moved_names


@pytest.fixture
def old_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("old_home")
    module.__getattr__ = moved_names(  # type: ignore[method-assign]
        "old_home",
        {"warn_inert": "ajax.deprecation", "gone": "ajax.deprecation:moved_names"}
        | {"train_agent": "ajax.agents.base:ActorCritic.train"},
    )
    monkeypatch.setitem(sys.modules, "old_home", module)
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
    from ajax.agents.base import ActorCritic

    with pytest.warns(FutureWarning, match="moved to ajax.agents.base.ActorCritic"):
        assert old_module.train_agent is ActorCritic.train  # a function, a method


def test_a_name_that_never_lived_there_is_still_an_attribute_error(
    old_module: types.ModuleType,
) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'other'"):
        old_module.other  # noqa: B018
