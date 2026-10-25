"""The entropy temperature every soft actor-critic shares
(ajax.agents.temperature): alpha read from its learned log and one step
down the agent's temperature loss."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ajax.agents.SAC.SAC import compute_temperature_loss
from ajax.agents.temperature import (
    create_alpha_train_state,
    get_alpha,
    update_temperature,
)


def _state(alpha_init: float = 0.5) -> SimpleNamespace:
    alpha = create_alpha_train_state(learning_rate=0.1, alpha_init=alpha_init)

    def replace(**changes):
        return SimpleNamespace(**{"alpha": alpha, "replace": replace, **changes})

    return replace()


def test_alpha_is_the_exponential_of_the_learned_log():
    np.testing.assert_allclose(get_alpha(_state(0.25)), 0.25, rtol=1e-6)


def test_the_step_goes_down_the_given_loss_on_the_summed_log_probs():
    """log_probs per action dimension are summed before the loss reads them;
    an entropy above the target lowers alpha under SAC's loss and the step's
    logs carry no gradient."""
    log_probs = jnp.full((4, 2), -1.0)  # entropy 2 per row, target -2
    seen = []

    def loss(params, summed, target):
        seen.append(summed.shape)
        return compute_temperature_loss(params, summed, target)

    stepped, aux = update_temperature(_state(), log_probs, jnp.asarray(-2.0), loss)
    assert seen and seen[0] == (4,)
    assert float(get_alpha(stepped)) < 0.5
    np.testing.assert_allclose(aux.alpha, 0.5, rtol=1e-6)
