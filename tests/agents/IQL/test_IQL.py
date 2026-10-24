"""IQL's contracts: its expectile loss, the actor's advantage weights, its
reference defaults and its refusals. Whether it learns is the probes'
(tests/probing/test_offline)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.agents.families import TrainContext
from ajax.agents.IQL.IQL import expectile_loss


def test_the_expectile_loss_weighs_each_side_of_zero() -> None:
    """``tau`` above the value, ``1 - tau`` below (critic.py ``loss``)."""
    diff = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    want = np.where(np.asarray(diff) > 0, 0.9, 0.1) * np.asarray(diff) ** 2
    np.testing.assert_allclose(expectile_loss(diff, 0.9), want, rtol=1e-6)


def _constant(params, value: float):
    """Every weight zero and the output layer's bias ``value``: the network
    gives ``value`` everywhere."""

    def leaf(path, x):
        names = [getattr(k, "key", None) for k in path]
        return jnp.full_like(x, value if names[-2:] == ["model", "bias"] else 0.0)

    return jax.tree_util.tree_map_with_path(leaf, params)


def test_the_actor_weighs_each_row_by_its_clipped_advantage() -> None:
    """The actor's loss is ``-mean(w log pi(a | s))`` with ``w = min(exp(
    temperature (q - V(s))), 100)`` (actor.py). With the target critics at q
    and V at 0 everywhere, q = 10 is past the clip (exp(30)) and q = 0 weighs
    each row about 1 (V's one step, on Adam's momentum alone, barely moves
    it): on the same batch and actor, the first loss is 100 times the second."""
    agent = IQL("Pendulum-v1", batch_size=8, actor_architecture=("8", "relu"))
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=16)[0])
    value = state.value_state.replace(params=_constant(state.value_state.params, 0.0))
    ctx = TrainContext(  # what the update reads: the replay, the extensions
        env_args=agent.env_args,
        network_args=agent.network_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        extensions=agent.extension_stack,
        total_timesteps=16,
        collector=agent.collector,
        buffer=agent.buffer,
    )

    def actor_loss(q: float) -> float:
        critic = state.critic_state
        critic = critic.replace(target_params=_constant(critic.target_params, q))
        _, aux = agent.update(
            state.replace(critic_state=critic, value_state=value), ctx
        )
        return float(aux.value.actor_loss)

    np.testing.assert_allclose(actor_loss(10.0) / actor_loss(0.0), 100.0, rtol=1e-3)


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
    value = agent.value_optimizer_args
    assert (value.eps, value.learning_rate) == (1e-8, 3e-4)
    assert (agent.expectile, agent.temperature, agent.tau) == (0.7, 3.0, 0.005)


def test_it_refuses_discrete_actions_and_an_expectile_outside_the_open_unit() -> None:
    with pytest.raises(ValueError, match="continuous"):
        IQL("CartPole-v1")
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="expectile"):
            IQL("Pendulum-v1", expectile=bad)
