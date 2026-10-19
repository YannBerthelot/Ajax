"""DQN's and PQN's diagnostics: the TD error's size, the gradient norm and
the exploration rate equal their definitions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import predict_q
from ajax.agents.DQN.train_DQN import (
    init_DQN,
    linear_epsilon,
    mse_td_loss,
    q_gradient_step,
    q_loss_fn,
)
from ajax.agents.PQN.PQN import PQN
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig

NET = ("16", "relu")


def test_epsilon_decays_linearly_then_stays() -> None:
    def eps(t: float) -> float:
        return float(linear_epsilon(jnp.asarray(t), 100, 1.0, 0.1, 0.5))

    assert eps(0) == 1.0
    np.testing.assert_allclose(eps(25), 0.55, rtol=1e-6)
    assert eps(50) == pytest.approx(0.1) and eps(90) == pytest.approx(0.1)


def test_a_q_step_reports_the_td_error_size_and_gradient_norm() -> None:
    agent = DQN("CartPole-v1", n_envs=1, architecture=NET)
    state = init_DQN(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.critic_optimizer_args,
        agent.network_args,
        n_actions=2,
    )
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    obs = jax.random.normal(keys[0], (8, *obs_shape))
    actions = jax.random.randint(keys[1], (8, 1), 0, 2)
    targets = jax.random.normal(keys[2], (8, 1))

    _, aux, _ = q_gradient_step(
        state, obs, actions, targets, mse_td_loss, ExtensionStack(()), None, 1
    )
    q = state.actor_state
    q_taken = jnp.take_along_axis(predict_q(q, q.params, obs), actions, axis=-1)
    np.testing.assert_allclose(
        aux.abs_td_error, np.mean(np.abs(q_taken - targets)), rtol=1e-6
    )
    grads = jax.grad(lambda p: q_loss_fn(p, q, obs, actions, targets, mse_td_loss)[0])(
        q.params
    )
    np.testing.assert_allclose(aux.grad_norm, optax.global_norm(grads), rtol=1e-6)


@pytest.mark.parametrize(
    "agent",
    [
        lambda: DQN(
            "CartPole-v1",
            n_envs=2,
            architecture=NET,
            learning_starts=16,
            batch_size=8,
            buffer_size=256,
            epsilon_decay_frac=0.5,
        ),
        lambda: PQN(
            "CartPole-v1",
            n_envs=2,
            architecture=NET,
            n_steps=8,
            n_epochs=2,
            num_minibatches=2,
            epsilon_decay_frac=0.5,
        ),
    ],
    ids=["DQN", "PQN"],
)
def test_the_logged_epsilon_is_the_schedules_at_each_evaluation(agent) -> None:
    """Epsilon falls from 1 to 0.05 over the first half of 256 steps."""
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent().train(
        seed=0, n_timesteps=256, num_episode_test=1, logging_config=logging
    )
    timestep = np.asarray(out["timestep"]).reshape(-1)
    evaluated = timestep >= 0
    epsilon = np.asarray(out["policy/epsilon"]).reshape(-1)[evaluated]
    expected = linear_epsilon(jnp.asarray(timestep[evaluated]), 256, 1.0, 0.05, 0.5)
    assert evaluated.sum() >= 4
    np.testing.assert_allclose(epsilon, expected, rtol=1e-6)
