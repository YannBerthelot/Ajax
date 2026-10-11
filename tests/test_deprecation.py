"""A moved public name stays importable from its old path, with a
FutureWarning, until the downstream projects migrate (CLAUDE.md rule 10)."""

import importlib

import pytest

from ajax.environments import obs_norm
from ajax.networks.distributions import SquashedNormal

MOVED = {
    ("ajax.agents.SAC.utils", "SquashedNormal"): SquashedNormal,
    **{
        ("ajax.agents.obs_norm", name): getattr(obs_norm, name)
        for name in ("apply_obs_norm", "init_agent_obs_norm", "update_obs_norm")
    },
}


@pytest.mark.parametrize("old", sorted(MOVED), ids="{0[0]}.{0[1]}".format)
def test_an_old_path_serves_the_name_with_a_warning(old: tuple[str, str]) -> None:
    module = importlib.import_module(old[0])
    with pytest.warns(FutureWarning, match=MOVED[old].__qualname__):
        assert getattr(module, old[1]) is MOVED[old]
    with pytest.raises(AttributeError, match="no_such_name"):
        module.no_such_name  # noqa: B018
