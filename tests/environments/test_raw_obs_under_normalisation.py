"""get_raw_obs gives an expert the environment's own observation, also
where an observation normaliser replaced the state's (brax, playground)."""

import jax
import jax.numpy as jnp
import numpy as np

from ajax.environments.create import prepare_env
from ajax.environments.interaction import get_raw_obs, reset, step


def _rollout(normalize_obs: bool) -> tuple[np.ndarray, np.ndarray]:
    """The observation and get_raw_obs after three steps of brax ``fast``
    (no ``_get_obs``, as playground's state.obs path), two envs."""
    env, params, _, _ = prepare_env("fast", n_envs=2, normalize_obs=normalize_obs)
    obs, state = reset(jax.random.PRNGKey(0), env, "brax", params)
    action = jnp.full((2, env.action_size), 0.3)
    for t in range(3):
        obs, state, *_ = step(jax.random.PRNGKey(t), state, action, env, "brax", params)
    return np.asarray(obs), np.asarray(get_raw_obs(state, env, "brax"))


def test_get_raw_obs_reads_through_a_brax_observation_normaliser():
    raw_obs, raw = _rollout(normalize_obs=False)
    normalised_obs, raw_under_normaliser = _rollout(normalize_obs=True)
    assert not np.allclose(normalised_obs, raw_obs)  # the normaliser acted
    np.testing.assert_allclose(raw, raw_obs)
    np.testing.assert_allclose(raw_under_normaliser, raw_obs)
