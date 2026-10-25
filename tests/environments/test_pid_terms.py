"""PIDTermsInput: a PID's terms (e, I, D) after the observation, on a
scripted episode; the agent-side normaliser keeps them raw. (Each episode
restarting them: tests/environments/test_row_collector.py.)"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.controller import ObserveInputs
from ajax.environments.obs_norm import (
    apply_obs_norm,
    init_agent_obs_norm,
    update_obs_norm,
)
from ajax.environments.pid_terms import split_pid_terms
from ajax.extensions.observe import PIDTermsInput


def _terms(positions: list[float], **kwargs) -> np.ndarray:
    """The feature's (e, I, D) along one episode of a plant at positions x,
    its observation [x, 1] with the set-point 1 last."""
    feature = PIDTermsInput((0,), (1,), **kwargs)
    state, rows = feature.init_observer(1), []
    for x in positions:
        obs = jnp.array([[x, 1.0]])
        columns, state = feature.observe(ObserveInputs(obs), state)
        rows.append(np.asarray(columns[0]))
    return np.stack(rows)


def test_terms_follow_a_scripted_trajectory():
    """x: 0, .5, .75, .75 -> e: 1, .5, .25, .25; dt = .5 integrates and
    divides; the first difference is 0."""
    terms = _terms([0.0, 0.5, 0.75, 0.75], dt=0.5)
    np.testing.assert_allclose(
        terms,
        [[1.0, 0.5, 0.0], [0.5, 0.75, -1.0], [0.25, 0.875, -0.5], [0.25, 1.0, 0.0]],
    )
    np.testing.assert_allclose(split_pid_terms(jnp.asarray(terms), 1)[:, :, 0], terms)


def test_integral_limit_clamps_and_unwinds_at_once():
    """e: 1, 1, 1, -1 -> I: .5, .6, .6, .1 under the .6 clamp (1.0 unclamped)."""
    positions = [0.0, 0.0, 0.0, 2.0]
    clamped = _terms(positions, dt=0.5, integral_limit=0.6)
    np.testing.assert_allclose(clamped[:, 1], [0.5, 0.6, 0.6, 0.1], rtol=1e-6)
    np.testing.assert_allclose(_terms(positions, dt=0.5)[:, 1], [0.5, 1.0, 1.5, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_idx": (0,), "target_idx": (1, 0)},
        {"current_idx": (), "target_idx": ()},
        {"current_idx": (0,), "target_idx": (1,), "dt": 0.0},
        {"current_idx": (0,), "target_idx": (1,), "integral_limit": -1.0},
    ],
)
def test_bad_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        PIDTermsInput(**kwargs)


def test_the_agent_side_normaliser_keeps_the_terms_raw():
    obs = jax.random.normal(jax.random.PRNGKey(0), (16, 5)) * 4.0 + 1.0
    _, info = update_obs_norm(obs, init_agent_obs_norm(5), raw_columns=3)
    applied = apply_obs_norm(obs, info)
    np.testing.assert_allclose(applied[:, :3], obs[:, :3], rtol=1e-6)
    assert float(jnp.abs(applied[:, 3:].mean(0)).max()) < 1e-5
