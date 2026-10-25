"""SAC-specific training tests.

Tests functions and mechanics unique to SAC: temperature (alpha) loss and
update, and soft target-network updates. Shared behaviors (loss shapes,
agent-state updates, training loop, make_train) are covered by the probing
suite (``tests/probing``) and the tiny runs of
``tests/agents/test_agent_contract.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState

from ajax.agents.critics import compute_q_regression_loss
from ajax.agents.recurrent import actor_dist
from ajax.agents.SAC.SAC import (
    SAC,
    compute_alpha_temperature_loss,
    compute_soft_bellman_target,
    compute_soft_policy_loss,
    compute_soft_policy_objective,
    compute_temperature_loss,
)
from ajax.agents.temperature import create_alpha_train_state, update_temperature
from ajax.agents.update_steps import update_target_networks


def _allclose(a, b) -> bool:
    return jax.tree.all(jax.tree.map(jnp.allclose, a, b))


@pytest.fixture
def SAC_state(env_config):
    """A fresh SAC state on the env's config (SAC.init alone)."""
    agent = SAC(
        env_config.env,
        env_params=env_config.env_params,
        actor_architecture=("64", "relu", "64"),
        critic_architecture=("64", "relu", "64"),
        buffer_size=1000,
        batch_size=32,
    )
    _, context = agent.make_context(1)
    return agent.init(jax.random.PRNGKey(0), context)


def test_create_alpha_train_state():
    learning_rate = 3e-4
    alpha_init = 1.0

    train_state = create_alpha_train_state(
        learning_rate=learning_rate, alpha_init=alpha_init
    )

    assert isinstance(train_state, TrainState)
    assert jnp.isclose(train_state.params["log_alpha"], jnp.log(alpha_init))
    assert train_state.tx is not None


@pytest.mark.parametrize(
    "log_alpha, target_entropy, log_probs, grad",
    [
        # grad = mean(-log_probs) - target_entropy: positive (alpha shrinks)
        # while the entropy is above the target, negative below it
        (0.0, -1.0, [-0.5, -1.5, -1.0], 1.0 + 1.0),
        (-1.0, -2.0, [-1.0, -2.0, -1.5], 1.5 + 2.0),
        (0.5, 1.0, [-0.1, -0.2], 0.15 - 1.0),
    ],
    ids=["above-target", "above-target-2", "below-target"],
)
def test_temperature_loss_function(log_alpha, target_entropy, log_probs, grad):
    """``log_alpha * mean(-log_probs - target_entropy)``, the bracket held
    constant: its gradient in ``log_alpha`` is the bracket."""
    (loss, aux), grads = jax.value_and_grad(compute_temperature_loss, has_aux=True)(
        FrozenDict({"log_alpha": jnp.array(log_alpha)}),
        log_probs=jnp.array(log_probs),
        target_entropy=target_entropy,
    )
    aux = to_state_dict(aux)
    assert isinstance(grads, FrozenDict)
    np.testing.assert_allclose(grads["log_alpha"], grad, rtol=1e-6)
    np.testing.assert_allclose(loss, log_alpha * grad, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(aux["alpha"], np.exp(log_alpha), rtol=1e-6)
    np.testing.assert_allclose(aux["log_alpha"], log_alpha)


def test_update_temperature(env_config, SAC_state):
    log_probs = jnp.ones((env_config.n_envs, 1)) * -0.5
    target_entropy = jnp.array(-1.0)

    original_alpha_params = SAC_state.alpha.params

    updated_state, aux = update_temperature(
        agent_state=SAC_state,
        log_probs=log_probs,
        target_entropy=target_entropy,
        loss=compute_temperature_loss,
    )
    aux = to_state_dict(aux)
    assert not _allclose(updated_state.alpha.params, original_alpha_params)
    assert "alpha" in aux
    assert aux["alpha"] > 0


def test_update_target_networks(env_config, SAC_state):
    tau = 0.05

    critic_state = SAC_state.critic_state
    shifted_params = FrozenDict(
        jax.tree_util.tree_map(lambda x: x + 1, critic_state.params)
    )
    critic_state = critic_state.replace(params=shifted_params)
    SAC_state = SAC_state.replace(critic_state=critic_state)

    original_params = SAC_state.critic_state.params
    original_target_params = SAC_state.critic_state.target_params

    updated_state = update_target_networks(SAC_state, tau=tau)

    new_target = updated_state.critic_state.target_params
    expected_target = jax.tree.map(
        lambda old, cur: tau * cur + (1 - tau) * old,
        original_target_params,
        original_params,
    )
    assert _allclose(updated_state.critic_state.params, original_params)
    assert not _allclose(new_target, original_target_params)
    assert _allclose(new_target, expected_target)


def test_the_soft_bellman_target_from_a_bootstrap_value():
    """r scaled by 2, then 0.9 (1 - d)(Q' - 0.5 log pi): a terminal row
    keeps its reward alone."""
    target = compute_soft_bellman_target(
        next_q=jnp.array([[2.0], [2.0]]),
        next_log_probs=jnp.array([[-1.0], [-1.0]]),
        rewards=jnp.array([[1.0], [1.0]]),
        dones=jnp.array([[0.0], [1.0]]),
        gamma=0.9,
        alpha=jnp.asarray(0.5),
        reward_scale=2.0,
    )
    np.testing.assert_allclose(target, [[4.25], [2.0]], rtol=1e-6)


def test_the_q_regression_loss_from_predictions():
    q_preds = jnp.array([[[1.0], [3.0]], [[2.0], [5.0]]])  # 2 critics, 2 rows
    loss, aux = compute_q_regression_loss(q_preds, jnp.array([[2.0], [4.0]]))
    assert float(loss) == pytest.approx((1 + 1 + 0 + 1) / 4)
    assert float(aux.q_gap[0]) == pytest.approx((1 + 2) / 2)


def test_the_soft_policy_objective_is_the_loss_on_a_given_policy(SAC_state):
    actor, critic = SAC_state.actor_state, SAC_state.critic_state
    obs = SAC_state.collector_state.last_obs
    key, alpha = jax.random.PRNGKey(1), jnp.asarray(0.2)
    pi = actor_dist(actor, actor.params, obs)
    loss, (log_probs, q, _) = compute_soft_policy_loss(
        actor.params, actor, critic, obs, alpha, key
    )
    same, (same_log_probs, same_q, _) = compute_soft_policy_objective(
        pi, critic, obs, alpha, key
    )
    assert float(loss) == float(same)
    assert _allclose((log_probs, q), (same_log_probs, same_q))


def test_the_alpha_temperature_loss_scales_sacs_gradient_by_alpha():
    """CrossQ's authors' loss, alpha (H - H_target): its gradient on log alpha
    is alpha times SAC's, (H - H_target)."""
    params = FrozenDict({"log_alpha": jnp.log(0.5)})
    log_probs, target = jnp.array([-1.0, -3.0]), jnp.asarray(-1.0)
    grad = jax.grad(lambda p: compute_alpha_temperature_loss(p, log_probs, target)[0])
    sac_grad = jax.grad(lambda p: compute_temperature_loss(p, log_probs, target)[0])
    entropy_gap = 2.0 - (-1.0)
    assert float(grad(params)["log_alpha"]) == pytest.approx(0.5 * entropy_gap)
    assert float(sac_grad(params)["log_alpha"]) == pytest.approx(entropy_gap)
