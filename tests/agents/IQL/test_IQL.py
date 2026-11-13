"""IQL's contracts: its expectile loss, the actor's advantage weights, its
reference defaults and its refusals. Whether it learns is the probes'
(tests/probing/test_offline)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.agents.families import TrainContext
from ajax.agents.IQL.IQL import compute_expectile_loss
from ajax.extensions.pretrain import ExpertPrefill


def test_the_expectile_loss_weighs_each_side_of_zero() -> None:
    """``tau`` above the value, ``1 - tau`` below (critic.py ``loss``)."""
    diff = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    want = np.where(np.asarray(diff) > 0, 0.9, 0.1) * np.asarray(diff) ** 2
    np.testing.assert_allclose(compute_expectile_loss(diff, 0.9), want, rtol=1e-6)


def _constant(params, value: float):
    """Every weight zero and the output layer's bias ``value``: the network
    gives ``value`` everywhere."""

    def leaf(path, x):
        names = [getattr(k, "key", None) for k in path]
        return jnp.full_like(x, value if names[-2:] == ["model", "bias"] else 0.0)

    return jax.tree_util.tree_map_with_path(leaf, params)


@pytest.fixture(scope="module")
def update_with_values():
    """One update of a tiny trained IQL with its target critics at ``q`` and
    its V at ``v`` everywhere, on the same batch and actor: its logs."""
    agent = IQL("Pendulum-v1", batch_size=8, actor_architecture=("8", "relu"))
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=16)[0])
    context = TrainContext(  # what the update reads: the replay, the extensions
        env_args=agent.env_args,
        network_args=agent.network_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        extensions=agent.extension_stack,
        total_timesteps=16,
        make_collector=agent.make_collector,
        buffer=agent.buffer,
    )

    def update(q: float, v: float):
        critic = state.critic_state
        critic = critic.replace(target_params=_constant(critic.target_params, q))
        value = state.value_state
        value = value.replace(params=_constant(value.params, v))
        return agent.update(
            state.replace(critic_state=critic, value_state=value), context
        )[1].value

    return update


def test_the_actor_weighs_each_row_by_its_clipped_advantage(update_with_values) -> None:
    """The actor's loss is ``-mean(w log pi(a | s))`` with ``w = min(exp(
    temperature (q - V(s))), 100)`` (actor.py). With the target critics at q
    and V at 0 everywhere, q = 10 is past the clip (exp(30)) and q = 0 weighs
    each row about 1 (V's one step, on Adam's momentum alone, barely moves
    it): on the same batch and actor, the first loss is 100 times the second."""
    clipped = update_with_values(10.0, 0.0).actor_loss
    unit = update_with_values(0.0, 0.0).actor_loss
    np.testing.assert_allclose(float(clipped) / float(unit), 100.0, rtol=1e-3)


def test_the_logged_advantage_is_the_critics_value_less_v(update_with_values) -> None:
    """Catches a logged advantage that adds V: q = 10 less V = 4 is 6. V's
    one step moves it by about a learning rate, 3e-4, well within the
    tolerance."""
    assert float(update_with_values(10.0, 4.0).advantage) == pytest.approx(
        6.0, abs=0.01
    )


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
    assert agent.actor_lr_decay == "cosine"


@pytest.mark.parametrize("offline, updates", [(False, 50), (True, 100)])
def test_the_actor_rate_falls_to_zero_by_a_cosine_over_the_runs_updates(
    offline: bool, updates: int
) -> None:
    """learner.py:85-86 steps the actor with optax.cosine_decay_schedule
    over the run's max_steps: 3e-4 (1 + cos(pi k / N)) / 2 at update k of N,
    so 3e-4, then 1.5e-4 halfway, then 0. A run of 100 timesteps on 2 envs
    makes N = 50 updates online (one per tick), N = 100 offline (one per
    timestep). With actor_lr_decay=None the rate stays the float."""
    agent = IQL("Pendulum-v1", n_envs=2)
    context = dataclasses.replace(agent.make_context(100)[1], offline=offline)
    rate = agent.with_decayed_actor_rate(context).actor_optimizer_args.learning_rate
    assert callable(rate)
    got = [float(rate(k)) for k in (0, updates // 2, updates)]
    np.testing.assert_allclose(got, [3e-4, 1.5e-4, 0.0], atol=1e-12)
    constant = IQL("Pendulum-v1", n_envs=2, actor_lr_decay=None)
    kept = constant.with_decayed_actor_rate(constant.make_context(100)[1])
    assert kept.actor_optimizer_args.learning_rate == 3e-4


def test_the_cosine_also_spans_the_updates_an_extension_runs_before_env_steps() -> None:
    """ExpertPrefill's offline_updates step the actor before any env step, so
    they count among the run's updates: 120 of them and no env step give
    N = 120, so 1.5e-4 at the 60th and 0 at the 120th, not 0 after the first."""
    prefill = ExpertPrefill(n_steps=8, offline_updates=120)
    expert = lambda obs: jnp.full(obs.shape[:-1] + (1,), 0.5)  # noqa: E731
    agent = IQL("Pendulum-v1", expert_policy=expert, extensions=(prefill,))
    context = agent.make_context(0, extensions=(prefill,))[1]
    rate = agent.with_decayed_actor_rate(context).actor_optimizer_args.learning_rate
    assert callable(rate)
    np.testing.assert_allclose([float(rate(60)), float(rate(120))], [1.5e-4, 0.0])


def test_it_refuses_discrete_actions_and_an_expectile_outside_the_open_unit() -> None:
    with pytest.raises(ValueError, match="continuous"):
        IQL("CartPole-v1")
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="expectile"):
            IQL("Pendulum-v1", expectile=bad)
    with pytest.raises(ValueError, match="actor_lr_decay"):
        IQL("Pendulum-v1", actor_learning_rate=lambda step: 3e-4)
