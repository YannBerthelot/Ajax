"""The automatic reset cost's steps (ajax.agents.reset_cost), against Hisaki & Ono (ICML 2024).

Contract (their Section 3.3, Eqs. 22-24, and Table 1):
- It starts with the rate xi_reset and the cost c at 0, the reset critic's target equal to it.
- A terminated row's reward is charged c, any other row nothing.
- The reset critic regresses on 1(terminated) - xi_reset + Q_reset'(s', a') by its mean squared error;
  xi_reset then moves to (1 - kappa) xi_reset + kappa gain mean Q_reset'(s', a').
- c steps by Adam down -c (xi_reset - eps): up while the rate is above eps, down while below, never
  below 0.
- The target follows the reset critic by Polyak averaging at tau.
- A discrete action reaches the reset critic one-hot.
Expected values are worked by hand below; zero weights make the reset critic output 0 everywhere.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.reset_cost import (
    ResetCostSettings,
    create_reset_cost_state,
    reset_charge,
    reset_critic_input,
    step_reset_cost,
    step_reset_critic,
    update_reset_target,
)


def fresh(width: int = 3):
    return create_reset_cost_state(jax.random.PRNGKey(0), width, ("8", "relu"), 3e-4)


def with_cost(state, cost: float):
    params = state.cost.params.copy({"cost": jnp.asarray(cost)})
    return state.replace(cost=state.cost.replace(params=params))


def silent(state):
    """The reset critic and its target with zero weights: Q_reset = 0 everywhere."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, state.critic.params)
    return state.replace(critic=state.critic.replace(params=zeros), target_params=zeros)


def test_it_starts_at_zero_with_its_target_a_copy_and_the_authors_settings():
    state = fresh()
    assert float(state.rate) == 0.0 and float(state.cost.params["cost"]) == 0.0
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, state.critic.params, state.target_params
    )
    settings = (
        ResetCostSettings()
    )  # Table 1: eps 1e-3, kappa 5e-3, a 64-64 reset critic, lr 3e-4
    assert (settings.target_rate, settings.rate_tau, settings.learning_rate) == (
        1e-3,
        5e-3,
        3e-4,
    )
    assert settings.architecture == ("64", "relu", "64", "relu")


def test_a_terminated_row_is_charged_the_cost_and_no_other():
    charge = reset_charge(with_cost(fresh(), 0.5), jnp.array([[1.0], [0.0]]))
    np.testing.assert_array_equal(charge, [[0.5], [0.0]])


def test_the_reset_critic_regresses_on_the_termination_indicator_minus_the_rate():
    # Q = Q' = 0, rate 0.2: targets 1 - 0.2 = 0.8 on terminated rows, -0.2 elsewhere;
    # loss (0.64 + 0.04 + 0.04 + 0.64) / 4 = 0.34; rate (1 - 0.5) 0.2 + 0.5 * mean Q' = 0.1.
    state = silent(fresh()).replace(rate=jnp.asarray(0.2))
    inputs = jnp.zeros((4, 3))
    terminated = jnp.array([[1.0], [0.0], [0.0], [1.0]])
    stepped, metrics = step_reset_critic(state, inputs, inputs, terminated, 0.5)
    np.testing.assert_allclose(metrics["reset_critic_loss"], 0.34, rtol=1e-6)
    np.testing.assert_allclose(stepped.rate, 0.1, rtol=1e-6)


@pytest.mark.parametrize("rate, rises", [(0.01, True), (1e-4, False)])
def test_the_cost_rises_above_the_target_rate_and_falls_below_it(rate, rises):
    # Adam's first step moves c by about the learning rate against the gradient -(rate - eps).
    state = with_cost(fresh(), 0.1).replace(rate=jnp.asarray(rate))
    cost = float(step_reset_cost(state, 1e-3).cost.params["cost"])
    np.testing.assert_allclose(abs(cost - 0.1), 3e-4, rtol=1e-3)
    assert (cost > 0.1) == rises


def test_the_cost_never_goes_below_zero():
    state = fresh().replace(
        rate=jnp.asarray(0.0)
    )  # below eps: the step pushes c down from 0
    for _ in range(3):
        state = step_reset_cost(state, 1e-3)
        assert float(state.cost.params["cost"]) == 0.0


def test_the_target_follows_the_reset_critic_at_tau():
    state = silent(fresh())
    ones = jax.tree_util.tree_map(jnp.ones_like, state.critic.params)
    state = state.replace(critic=state.critic.replace(params=ones))
    target = update_reset_target(state, 0.25).target_params
    jax.tree_util.tree_map(lambda t: np.testing.assert_allclose(t, 0.25), target)


def test_a_discrete_action_reaches_the_reset_critic_one_hot():
    observations = jnp.array([[0.5], [-1.0]])
    np.testing.assert_array_equal(
        reset_critic_input(observations, jnp.array([[2], [0]]), 3),
        [[0.5, 0.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 0.0]],
    )
