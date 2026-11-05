"""The fixed states activations are read on.

Contract (ajax.representation.probe_set): a seed alone fixes the states; they
come with the actions a random policy took there, in the agent's units, and
the env's own fields at each one; a directory keeps one probe set, reused and
never rebuilt. Pendulum's observation is (cos theta, sin theta, theta dot)
(gymnax.environments.classic_control.pendulum), which gives an oracle
independent of the code: the ground truth must explain the observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ajax import PPO, SAC
from ajax.representation.probe_set import (
    PROBE_SET_FILE,
    ProbeSet,
    make_probe_set,
    probe_set_for,
)

N_STATES = 64


@pytest.fixture(scope="module")
def pendulum():
    return SAC("Pendulum-v1", n_envs=2).env_args


@pytest.fixture(scope="module")
def probe_set(pendulum):
    return make_probe_set(pendulum, N_STATES, seed=3)


def test_a_seed_fixes_the_states(pendulum, probe_set):
    """Catches a generation drawing on any other randomness: two seeds of a
    run, or two checkpoints, would read different states."""
    again = make_probe_set(pendulum, N_STATES, seed=3)
    other = make_probe_set(pendulum, N_STATES, seed=4)

    np.testing.assert_array_equal(again.observations, probe_set.observations)
    assert np.abs(other.observations - probe_set.observations).max() > 0.1


def test_the_ground_truth_describes_the_observation_it_comes_with(probe_set):
    """Catches a row pairing an observation with the state after it: every
    colouring of a manifold by the env's state would be one step off."""
    theta = probe_set.ground_truth["theta"]
    theta_dot = probe_set.ground_truth["theta_dot"]

    expected = np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=-1)
    np.testing.assert_allclose(probe_set.observations, expected, atol=1e-5)


def test_states_are_distinct_and_leave_the_episode_start(probe_set):
    """Catches rollouts reset too often, or one state repeated: a probe set
    of first states would hide every representation of the swing-up."""
    assert len(np.unique(probe_set.observations, axis=0)) == N_STATES
    assert probe_set.ground_truth["time"].max() > 100


def test_continuous_actions_are_in_the_agents_units(probe_set):
    """Catches actions stored in the env's units: Pendulum's torque goes to
    +-2, and a Q-critic reads [-1, 1]."""
    assert probe_set.actions.shape == (N_STATES, 1)
    assert np.abs(probe_set.actions).max() <= 1.0
    assert np.abs(probe_set.actions).max() > 0.5


def test_discrete_actions_are_one_index_column():
    """Catches a one-hot or float action reaching a discrete env's step."""
    env_args = PPO("CartPole-v1", n_envs=2).env_args

    actions = make_probe_set(env_args, N_STATES, seed=0).actions

    assert actions.shape == (N_STATES, 1)
    assert set(np.unique(actions)) == {0.0, 1.0}


def test_a_brax_env_gives_states_too():
    """Catches a generation written for gymnax alone: brax resets on one key
    and carries its observation on the state."""
    env_args = SAC("fast", n_envs=2).env_args

    probe_set = make_probe_set(env_args, N_STATES, seed=0)

    assert probe_set.observations.shape[0] == N_STATES
    assert probe_set.actions.shape == (N_STATES, 1)


def test_a_saved_probe_set_comes_back_whole(tmp_path, probe_set):
    """Catches a field lost on disk, the seed included, which says what the
    run's states were."""
    path = tmp_path / "states.npz"

    probe_set.save(path)
    loaded = ProbeSet.load(path)

    np.testing.assert_array_equal(loaded.observations, probe_set.observations)
    np.testing.assert_array_equal(loaded.actions, probe_set.actions)
    assert loaded.ground_truth.keys() == probe_set.ground_truth.keys()
    assert loaded.seed == 3


def test_a_directory_keeps_one_probe_set(tmp_path, pendulum):
    """Catches a second run rebuilding the states: runs of one directory are
    only comparable on the same ones. The second call has no env to roll out."""
    made = probe_set_for(tmp_path, pendulum, N_STATES, seed=1)

    reused = probe_set_for(tmp_path, None, N_STATES, seed=1)

    assert (tmp_path / PROBE_SET_FILE).exists()
    np.testing.assert_array_equal(reused.observations, made.observations)


@pytest.mark.parametrize("n_states, seed", [(N_STATES, 2), (N_STATES // 2, 1)])
def test_a_directory_refuses_other_settings(tmp_path, pendulum, n_states, seed):
    """Catches settings silently ignored: asking for other states and reading
    the saved ones would mislabel the run."""
    probe_set_for(tmp_path, pendulum, N_STATES, seed=1)

    with pytest.raises(ValueError, match="holds"):
        probe_set_for(tmp_path, pendulum, n_states, seed=seed)
