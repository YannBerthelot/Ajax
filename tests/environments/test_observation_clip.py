"""The observation clip, baselines' VecNormalize clipob (common/vec_env/
vec_normalize.py @ea25b9e: clip((obs - mean) / sqrt(var + 1e-8), -10, 10)).

Contract: an on-policy agent storing observations normalised at collection
clips them to +-observation_clip, at collection and when its frozen
statistics normalise an evaluation's observations; the env's leading raw
columns stay as they are; None clips nothing; a replay agent, which
normalises inside its networks, refuses a clip."""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.environments.obs_norm import (
    init_agent_obs_norm,
    normalized_obs,
    update_obs_norm,
)


def _fitted(clip):
    """Statistics of one feature after rows 0 and 2: mean 1, variance 1."""
    info = init_agent_obs_norm(1, clip)
    _, info = update_obs_norm(jnp.array([[0.0], [2.0]]), info)
    return info


def test_a_normalised_observation_is_clipped_at_the_bound() -> None:
    """With mean 1 and variance 1, 51 normalises to (51 - 1) / 1 = 50,
    clipped to 10; -49 to -50, clipped to -10; 3 to 2, inside the bound."""
    info = _fitted(10.0)
    got = normalized_obs(jnp.array([[51.0], [-49.0], [3.0]]), info)
    np.testing.assert_allclose(got[:, 0], [10.0, -10.0, 2.0], rtol=1e-6)
    unclipped = normalized_obs(jnp.array([[51.0]]), _fitted(None))
    np.testing.assert_allclose(unclipped[:, 0], [50.0], rtol=1e-6)


def test_collection_clips_what_it_stores_and_keeps_the_bound() -> None:
    """The update both clips the rows it normalises and hands the bound on."""
    info = init_agent_obs_norm(1, 1.0)
    stored, updated = update_obs_norm(jnp.array([[0.0], [100.0]]), info)
    assert float(jnp.abs(stored).max()) <= 1.0
    assert updated is not None and updated.clip == 1.0


def test_the_envs_raw_columns_are_not_clipped() -> None:
    """A raw leading column (a PID's terms) passes through, beyond the bound."""
    info = init_agent_obs_norm(2, 1.0)
    obs = jnp.array([[50.0, 0.0], [-50.0, 2.0]])
    stored, _ = update_obs_norm(obs, info, raw_columns=1)
    np.testing.assert_allclose(stored[:, 0], [50.0, -50.0])
    assert float(jnp.abs(stored[:, 1]).max()) <= 1.0


def test_a_replay_agent_refuses_a_clip() -> None:
    with pytest.raises(ValueError, match="observation_clip"):
        SAC("Pendulum-v1", normalize_observations=True, observation_clip=10.0)
