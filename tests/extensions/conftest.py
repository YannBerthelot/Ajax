"""A stand-in critic the extension maths tests share."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def critic_stub():
    """``critic_stub(q)``: a critic state whose Q at input ``x`` is ``q(x)``,
    an ensemble of one. The maths under test only call
    ``critic_state.apply_fn(params, x)``."""

    def make(q):
        return SimpleNamespace(
            apply_fn=lambda params, x: q(x)[None],
            params={"online": 1.0},
            target_params={"target": 1.0},
        )

    return make
