"""DAgger's contracts: its beta schedule, the mixing of the expert's and the
actor's actions at collection, the expert's labels on every stored row, the
labels as what the actor fits, its fits (when they run, over which rows, on
which loss) and its refusals. Whether it learns is the probes'
(tests/probing/test_imitation).

The fits' contract is the reference code's, HumanCompatibleAI/imitation
@e5ef188 (SimpleDAggerTrainer.train, algorithms/dagger.py:615-694;
DAggerTrainer.extend_and_update, :455-497; BC.train and its loss,
algorithms/bc.py:93-156, :381-510): after each round of collection, behaviour
cloning makes DEFAULT_N_EPOCHS = 4 passes over the whole aggregated dataset,
in shuffled minibatches of batch_size, a short last one dropped, on
``-mean log pi(a_E | s) - ent_weight * mean entropy`` (ent_weight 1e-3). Its
policy, SB3's ActorCriticPolicy, is a Gaussian with one log-std per action
dimension starting at 0 (or a categorical), and acts by sampling."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import DAgger
from ajax.agents.DAgger.DAgger import (
    ExponentialBetaSchedule,
    IterationFit,
    LinearBetaSchedule,
    OnlineFit,
)
from ajax.environments.interaction import get_pi
from ajax.logging.wandb_logging import LoggingConfig

N_ENVS = 2
#: The paper's beta_i = I(i = 1): the expert acts in the first iteration only.
FIRST_ONLY = ExponentialBetaSchedule(0.0)
#: One epoch: the fast tier's cap of 128 gradient updates a test (the
#: default's 4, a fit every tick, would take 480 in the learning test).
FITS = (IterationFit(epochs=1), OnlineFit())
FIT_NAMES = ["IterationFit", "OnlineFit"]
#: A Gaussian's entropy per dimension at std 1: (1 + log(2 pi)) / 2.
UNIT_GAUSSIAN_ENTROPY = 0.5 * (1.0 + math.log(2.0 * math.pi))


def push(obs):
    """0.5 in env units on Pendulum's [-2, 2] torque: 0.25 in agent units."""
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def choose_one(obs):
    return jnp.ones(obs.shape[:-1] + (1,), jnp.int32)


def replay(state, first: int = 0) -> dict:
    """Seed 0's stored rows from each env's ``first`` step on, envs joined
    (the replay is ``[seed, env, step, ...]``)."""
    exp = state.collector_state.buffer_state.experience
    end = int(state.collector_state.buffer_state.current_index[0])
    return {
        k: np.asarray(v[0, :, first:end]).reshape(-1, v.shape[-1])
        for k, v in exp.items()
    }


@pytest.mark.parametrize("decay", [0.0, 0.5, 1.0])
def test_beta_is_the_decay_to_the_iterations_done(decay):
    """beta_i = decay ** (i - 1) over iterations of iteration_steps
    timesteps; 0 ** 0 = 1: the paper's beta_i = I(i = 1) at decay 0."""
    schedule = ExponentialBetaSchedule(decay)
    agent = DAgger(
        "Pendulum-v1", expert_policy=push, beta_schedule=schedule, iteration_steps=10
    )
    got = [float(agent.compute_beta(jnp.asarray(t))) for t in (0, 9, 10, 25)]
    np.testing.assert_allclose(got, [1.0, 1.0, decay, decay**2])


def test_beta_falls_linearly_to_zero_over_fifteen_iterations_by_default():
    """The reference's LinearBetaSchedule(15) (imitation@e5ef188
    dagger.py:56-67, :355): beta = clip((15 - k) / 15, 0, 1) in the
    iteration k counted from 0, so 1, 14/15, 8/15, 1/15, then 0 for good."""
    agent = DAgger("Pendulum-v1", expert_policy=push, iteration_steps=10)
    assert agent.beta_schedule == LinearBetaSchedule(rampdown_rounds=15)
    timesteps = (0, 9, 10, 75, 149, 150, 400)
    got = [float(agent.compute_beta(jnp.asarray(t))) for t in timesteps]
    np.testing.assert_allclose(got, [1.0, 1.0, 14 / 15, 8 / 15, 1 / 15, 0.0, 0.0])


@pytest.mark.parametrize(
    "iteration_steps, expert_acts", [(10_000, True), (N_ENVS, False)]
)
def test_rows_hold_the_experts_label_whoever_acted(iteration_steps, expert_acts):
    """Every row stores the expert's action at its state, in agent units.
    While beta is 1 the expert's action is the executed one; once it is 0
    (after the first tick) the actor's, flagged as not the expert's. A
    batch of 16 keeps the fits, every tick at iteration_steps 2, under the
    fast tier's 128 gradient updates (40)."""
    agent = DAgger(
        "Pendulum-v1",
        expert_policy=push,
        n_envs=N_ENVS,
        iteration_steps=iteration_steps,
        beta_schedule=FIRST_ONLY,
        batch_size=16,
        buffer_size=64,
    )
    state, _ = agent.train(seed=[0], n_timesteps=32)
    np.testing.assert_allclose(replay(state)["a_expert"], 0.25, rtol=1e-6)
    later = replay(state, first=1)  # the first tick has beta 1 either way
    np.testing.assert_array_equal(later["is_expert"], float(expert_acts))
    assert np.isclose(later["action"], 0.25).all() == expert_acts


def test_a_discrete_run_stores_integer_actions_and_labels():
    agent = DAgger(
        "CartPole-v1",
        expert_policy=choose_one,
        n_envs=N_ENVS,
        batch_size=4,
        buffer_size=64,
    )
    state, _ = agent.train(seed=[0], n_timesteps=32)
    rows = replay(state)
    assert rows["action"].dtype == rows["a_expert"].dtype == np.int32
    np.testing.assert_array_equal(rows["a_expert"], 1)
    np.testing.assert_array_equal(rows["action"], 1)  # beta 1 for 500 steps


def _pi(state, obs):
    actor = jax.tree.map(lambda x: x[0], state.actor_state)  # seed 0
    return get_pi(actor, actor.params, jnp.asarray(obs))[0]


@pytest.mark.parametrize("fit", FITS, ids=FIT_NAMES)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_the_actor_learns_the_experts_labels_not_the_actions_taken(env_id, fit):
    """The supervised step fits the actor to the stored ``a_expert``, not to
    the executed action (Ross et al., 2011: the expert labels the states the
    learner visits). From the second tick on (beta 0) the actor acts alone, so
    its executed actions are its own: fitting them would hold it where it
    started, while fitting the labels moves it towards the expert's action.
    The discrete expert picks the action the untrained actor does not: fitting
    the labels takes it to that action (1.0 on seeds 0-3), fitting its own
    samples leaves it near 0.5 (0.40-0.58), hence a 0.25 margin. Either fit:
    an IterationFit fits every tick here (one iteration a tick)."""
    continuous = env_id == "Pendulum-v1"
    kwargs = {"n_envs": N_ENVS, "iteration_steps": N_ENVS, "batch_size": 8}
    kwargs.update({"beta_schedule": FIRST_ONLY, "fit": fit})
    kwargs.update({"buffer_size": 256, "actor_learning_rate": 1e-2})
    # The initial actor, built beside an expert of the env's action kind
    # (collection stores its actions in the replay's layout).
    first_expert = push if continuous else choose_one
    init, _ = DAgger(env_id, expert_policy=first_expert, **kwargs).train(
        seed=[0], n_timesteps=0
    )
    if continuous:
        expert, label = push, 0.25
    else:
        label = 1 - int(_pi(init, jnp.zeros((1, 4))).mode()[0])
        expert = lambda obs: jnp.full(obs.shape[:-1] + (1,), label, jnp.int32)  # noqa: E731
    state, _ = DAgger(env_id, expert_policy=expert, **kwargs).train(
        seed=[0], n_timesteps=64
    )
    obs = replay(state)["obs"]
    if continuous:
        before, after = (abs(_pi(s, obs).mean() - label).mean() for s in (init, state))
        assert after < 0.5 * before, (before, after)
    else:
        before, after = (_pi(s, obs).probs[:, label].mean() for s in (init, state))
        assert after > before + 0.25, (before, after)


def test_the_default_fit_is_the_references_cloning_after_each_round():
    """imitation@e5ef188: rounds of 500 steps (rollout_round_min_timesteps,
    dagger.py:620), each followed by DEFAULT_N_EPOCHS = 4 of behaviour
    cloning (dagger.py:326), whose batch is 32 and entropy weight 1e-3
    (bc.py:282, :286)."""
    agent = DAgger("Pendulum-v1", expert_policy=push)
    assert agent.fit == IterationFit(epochs=4, entropy_weight=1e-3)
    assert (agent.iteration_steps, agent.batch_size) == (500, 32)


def actor_steps(state) -> int:
    """The optimiser steps the actor took, seed 0."""
    return int(np.asarray(state.actor_state.step).reshape(-1)[0])


@pytest.mark.parametrize(
    "n_timesteps, fits, steps",
    # 40 steps over 2 envs end iteration 1: 4 epochs of 40 // 16 = 2
    # minibatches (8 rows dropped); then 80 and 120 rows give 5 and 7 a pass.
    [(38, 0, 0), (40, 1, 4 * 2), (120, 3, 4 * (2 + 5 + 7))],
)
def test_a_fit_ends_each_iteration_with_epochs_over_every_stored_row(
    n_timesteps, fits, steps
):
    """The reference fits once per round, after collecting it (dagger.py:
    659-694): no step before the first iteration ends, one fit at each
    iteration's end, each 4 passes over every row stored by then, all envs'
    (its dataset aggregates every round's), in minibatches of batch_size
    with the short last one dropped (dagger.py:445-451)."""
    agent = DAgger(
        "Pendulum-v1",
        expert_policy=push,
        n_envs=N_ENVS,
        iteration_steps=40,
        batch_size=16,
        buffer_size=1000,
        fit=IterationFit(),
    )
    state, _ = agent.train(seed=[0], n_timesteps=n_timesteps)
    np.testing.assert_array_equal(state.n_updates, fits)
    assert actor_steps(state) == steps


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_the_cloning_loss_is_the_likelihood_minus_the_weighted_entropy(env_id):
    """loss = -mean log pi(a_E | s) - entropy_weight * mean entropy
    (bc.py:130-146), read on an actor that does not move (learning rate 0):
    at its start, a Gaussian of std 1 around a mean near 0 (0.01-gain head)
    or a near-uniform categorical. The expert's label is 0.25 in agent units
    (Pendulum), or action 1 of 2 (CartPole), so the likelihood term is
    log(2 pi) / 2 + 0.25^2 / 2 = 0.950, or log 2 = 0.693, both within 0.02
    (the mean's and logits' small spread); a squared error would log 0.0625.
    Between entropy weights 0 and 1 the loss falls by the entropy itself:
    (1 + log(2 pi)) / 2 for std 1, log 2 for two even actions."""
    continuous = env_id == "Pendulum-v1"
    expert = push if continuous else choose_one
    nll, entropy = (
        (0.5 * math.log(2 * math.pi) + 0.5 * 0.25**2, UNIT_GAUSSIAN_ENTROPY)
        if continuous
        else (math.log(2.0), math.log(2.0))
    )

    def logged_loss(entropy_weight: float) -> float:
        agent = DAgger(
            env_id,
            expert_policy=expert,
            n_envs=N_ENVS,
            iteration_steps=16,
            batch_size=16,
            buffer_size=1000,
            actor_learning_rate=0.0,
            fit=IterationFit(entropy_weight=entropy_weight),
        )
        logging = LoggingConfig(log_frequency=32, backends=())
        _, out = agent.train(seed=[0], n_timesteps=32, logging_config=logging)
        return float(np.asarray(out["policy/imitation_loss"]).reshape(-1)[-1])

    without, with_unit_weight = logged_loss(0.0), logged_loss(1.0)
    np.testing.assert_allclose(without, nll, atol=0.02)
    np.testing.assert_allclose(without - with_unit_weight, entropy, atol=1e-3)


def test_an_iteration_fit_actor_starts_as_the_references_gaussian():
    """SB3's ActorCriticPolicy, the reference's: an unsquashed Gaussian whose
    log-std is one parameter per action dimension starting at 0, so its std
    is 1 at every state."""
    state, _ = DAgger("Pendulum-v1", expert_policy=push, fit=IterationFit()).train(
        seed=[0], n_timesteps=0
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (8, 3)) * 5.0
    np.testing.assert_allclose(_pi(state, obs).stddev(), 1.0)


@pytest.mark.parametrize(
    "fit, spread",
    list(zip(FITS, [(0.7, 1.3), (0.0, 0.0)], strict=True)),
    ids=FIT_NAMES,
)
def test_the_actor_acts_by_sampling_under_an_iteration_fit(fit, spread):
    """The reference's robot acts through policy.predict, whose default
    samples (dagger.py:511): where the actor acted, the stored action is a
    draw around its mean, of std 1 at its start (62 rows: within 0.3). An
    OnlineFit's actor acts with its mean. A batch of 64 holds the fits back
    to the last tick."""
    agent = DAgger(
        "Pendulum-v1",
        expert_policy=push,
        n_envs=N_ENVS,
        iteration_steps=N_ENVS,
        beta_schedule=FIRST_ONLY,
        batch_size=64,
        buffer_size=1000,
        actor_learning_rate=0.0,
        fit=fit,
    )
    state, _ = agent.train(seed=[0], n_timesteps=64)
    rows = replay(state, first=1)  # the actor acts from the second tick
    np.testing.assert_array_equal(rows["is_expert"], 0.0)
    residual = rows["action"] - np.asarray(_pi(state, rows["obs"]).mean())
    low, high = spread
    assert low <= residual.std() <= high, residual.std()


def test_it_refuses_no_expert_and_a_schedule_outside_its_range():
    with pytest.raises(ValueError, match="expert_policy"):
        DAgger("Pendulum-v1", expert_policy=None)
    with pytest.raises(ValueError, match="decay"):
        ExponentialBetaSchedule(1.5)
    with pytest.raises(ValueError, match="rampdown_rounds"):
        LinearBetaSchedule(0)
    with pytest.raises(ValueError, match="epoch"):
        IterationFit(epochs=0)
