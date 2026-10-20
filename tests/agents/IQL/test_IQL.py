"""IQL's contracts: its expectile loss, its reference defaults and its
refusals. Whether it learns is the probes' (tests/probing/test_offline)."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.agents.IQL.train_IQL import expectile_loss


def test_the_expectile_loss_weighs_each_side_of_zero() -> None:
    """``tau`` above the value, ``1 - tau`` below (critic.py ``loss``)."""
    diff = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    want = np.where(np.asarray(diff) > 0, 0.9, 0.1) * np.asarray(diff) ** 2
    np.testing.assert_allclose(expectile_loss(diff, 0.9), want, rtol=1e-6)


def test_the_defaults_are_the_reference_codes() -> None:
    """ikostrikov/implicit_q_learning@09d7002: a Gaussian on a tanh mean
    with one log-std per dimension in [-5, 2], orthogonal(sqrt 2) layers,
    optax's Adam eps, the locomotion expectile and temperature."""
    agent = IQL("Pendulum-v1")
    net = agent.network_args
    assert net.tanh_mean and net.log_std_state_independent and not net.squash
    assert (net.log_std_init, net.log_std_bounds) == (0.0, (-5.0, 2.0))
    assert net.encoder_kernel_init == net.mean_kernel_init == "orthogonal(np.sqrt(2))"
    assert agent.actor_optimizer_args.eps == agent.critic_optimizer_args.eps == 1e-8
    value = agent.train_kwargs["value_optimizer_args"]
    assert (value.eps, value.learning_rate) == (1e-8, 3e-4)
    config = agent.agent_config
    assert (config.expectile, config.temperature, config.tau) == (0.7, 3.0, 0.005)


def test_it_refuses_discrete_actions_and_an_expectile_outside_the_open_unit() -> None:
    with pytest.raises(ValueError, match="continuous"):
        IQL("CartPole-v1")
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="expectile"):
            IQL("Pendulum-v1", expectile=bad)
