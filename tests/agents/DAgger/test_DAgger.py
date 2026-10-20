"""DAgger's contracts: its beta schedule, the mixing of the expert's and the
actor's actions at collection, the expert's labels on every stored row, and
its refusals. Whether it learns is the probes' (tests/probing/test_imitation)."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax import DAgger
from ajax.agents.DAgger.state import DAggerConfig
from ajax.agents.DAgger.train_DAgger import beta

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
    config = DAggerConfig(beta_decay=decay, iteration_steps=10)
    got = [float(beta(jnp.asarray(t), config)) for t in (0, 9, 10, 25)]
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


def test_it_refuses_no_expert_and_a_decay_outside_the_unit_interval():
    with pytest.raises(ValueError, match="expert_policy"):
        DAgger("Pendulum-v1", expert_policy=None)
    with pytest.raises(ValueError, match="beta_decay"):
        DAgger("Pendulum-v1", expert_policy=push, beta_decay=1.5)
