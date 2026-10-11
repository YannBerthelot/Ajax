"""A moved public name stays importable from its old path, with a
FutureWarning, until the downstream projects migrate (CLAUDE.md rule 10)."""

import importlib

import pytest

from ajax.networks.distributions import SquashedNormal

MOVED = {
    ("ajax.agents.SAC.utils", "SquashedNormal"): SquashedNormal,
}


@pytest.mark.parametrize("old", sorted(MOVED), ids="{0[0]}.{0[1]}".format)
def test_an_old_path_serves_the_name_with_a_warning(old: tuple[str, str]) -> None:
    module = importlib.import_module(old[0])
    with pytest.warns(FutureWarning, match=MOVED[old].__qualname__):
        assert getattr(module, old[1]) is MOVED[old]
    with pytest.raises(AttributeError, match="no_such_name"):
        module.no_such_name  # noqa: B018
