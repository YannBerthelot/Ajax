"""The entropy temperature every soft actor-critic shares
(ajax.agents.temperature): alpha read from its learned log, one step down
the agent's temperature loss, and the 0.4.0 form of the step SAC's core
keeps until 0.5.0."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.SAC import core
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
        return core.temperature_loss_fn(params, summed, target)

    stepped, aux = update_temperature(_state(), log_probs, jnp.asarray(-2.0), loss)
    assert seen and seen[0] == (4,)
    assert float(get_alpha(stepped)) < 0.5
    np.testing.assert_allclose(aux.alpha, 0.5, rtol=1e-6)


def test_sacs_core_keeps_the_040_three_argument_step_on_sacs_loss():
    """core.update_temperature(agent_state, log_probs, target_entropy), as
    0.4.0 called it, is the shared step on SAC's temperature loss."""
    log_probs, target = jnp.full((3, 1), -0.3), jnp.asarray(-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a function kept, not a moved name
        legacy, legacy_aux = core.update_temperature(_state(), log_probs, target)
    shared, shared_aux = update_temperature(
        _state(), log_probs, target, core.temperature_loss_fn
    )
    leaves = jax.tree.leaves((legacy.alpha.params, legacy_aux))
    expected = jax.tree.leaves((shared.alpha.params, shared_aux))
    for got, want in zip(leaves, expected, strict=True):
        np.testing.assert_array_equal(got, want)
