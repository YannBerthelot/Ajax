"""DAgger's contracts: its beta schedule, the mixing of the expert's and the
actor's actions at collection, the expert's labels on every stored row, the
labels as what the actor fits, and its refusals. Whether it learns is the
probes' (tests/probing/test_imitation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import DAgger
from ajax.environments.interaction import get_pi

N_ENVS = 2


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
    agent = DAgger(
        "Pendulum-v1", expert_policy=push, beta_decay=decay, iteration_steps=10
    )
    got = [float(agent.compute_beta(jnp.asarray(t))) for t in (0, 9, 10, 25)]
    np.testing.assert_allclose(got, [1.0, 1.0, decay, decay**2])


@pytest.mark.parametrize(
    "iteration_steps, expert_acts", [(10_000, True), (N_ENVS, False)]
)
def test_rows_hold_the_experts_label_whoever_acted(iteration_steps, expert_acts):
    """Every row stores the expert's action at its state, in agent units.
    While beta is 1 the expert's action is the executed one; once it is 0
    (after the first tick) the actor's, flagged as not the expert's."""
    agent = DAgger(
        "Pendulum-v1",
        expert_policy=push,
        n_envs=N_ENVS,
        iteration_steps=iteration_steps,
        batch_size=4,
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


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_the_actor_learns_the_experts_labels_not_the_actions_taken(env_id):
    """The supervised step fits the actor to the stored ``a_expert``, not to
    the executed action (Ross et al., 2011: the expert labels the states the
    learner visits). From the second tick on (beta 0) the actor acts alone, so
    its executed actions are its own: fitting them would hold it where it
    started, while fitting the labels moves it towards the expert's action.
    The discrete expert picks the action the untrained actor does not: fitting
    the labels takes it to that action (1.0 on seeds 0-3), fitting its own
    samples leaves it near 0.5 (0.40-0.58), hence a 0.25 margin."""
    continuous = env_id == "Pendulum-v1"
    kwargs = {"n_envs": N_ENVS, "iteration_steps": N_ENVS, "batch_size": 8}
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


def test_it_refuses_no_expert_and_a_decay_outside_the_unit_interval():
    with pytest.raises(ValueError, match="expert_policy"):
        DAgger("Pendulum-v1", expert_policy=None)
    with pytest.raises(ValueError, match="beta_decay"):
        DAgger("Pendulum-v1", expert_policy=push, beta_decay=1.5)
