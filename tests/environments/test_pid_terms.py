"""PIDTermsWrapper: (e, I, D) in front of the observation, on scripted runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import environment, spaces

from ajax.environments.interaction import reset, step
from ajax.environments.pid_terms import PIDTermsWrapper, split_pid_terms
from ajax.evaluate import setup_environment


@struct.dataclass
class _State(environment.EnvState):
    x: jax.Array


@struct.dataclass
class _Params(environment.EnvParams):
    max_steps_in_episode: int = 10


class _Integrator(environment.Environment):
    """``x_{t+1} = x_t + a_t`` from ``x_0 = 0``; observation ``[x, 1]``, the
    set-point 1 last; terminates past ``x = 10``."""

    @property
    def default_params(self) -> _Params:
        return _Params()

    def reset_env(self, key, params):
        state = _State(time=0, x=jnp.float32(0.0))
        return self.get_obs(state), state

    def step_env(self, key, state, action, params):
        state = state.replace(x=state.x + action[0], time=state.time + 1)
        return self.get_obs(state), state, -state.x, state.x > 10.0, {}

    def get_obs(self, state, params=None, key=None):
        return jnp.stack([state.x, jnp.float32(1.0)])

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (2,))

    def action_space(self, params):
        return spaces.Box(-1.0, 1.0, (1,))


def _run(actions, max_steps=10, **kwargs):
    """Observations (reset first) and infos of a scripted run."""
    env = PIDTermsWrapper(_Integrator(), (0,), (1,), **kwargs)
    params = _Params(max_steps_in_episode=max_steps)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    observations, infos = [obs], []
    for a in actions:
        obs, state, _, _, _, info = env.step(key, state, jnp.array([a]), params)
        observations.append(obs)
        infos.append(info)
    return np.asarray(observations), infos, env, state, params


def test_terms_follow_a_scripted_trajectory():
    """x: 0, .5, .75, .75 -> e: 1, .5, .25, .25; dt = .5 integrates and
    divides; the first difference is 0."""
    obs, *_ = _run([0.5, 0.25, 0.0], dt=0.5)
    np.testing.assert_allclose(
        obs,
        [
            [1.0, 0.5, 0.0, 0.0, 1.0],
            [0.5, 0.75, -1.0, 0.5, 1.0],
            [0.25, 0.875, -0.5, 0.75, 1.0],
            [0.25, 1.0, 0.0, 0.75, 1.0],
        ],
    )
    terms = split_pid_terms(jnp.asarray(obs), 1)
    np.testing.assert_allclose(terms[:, :, 0], obs[:, :3])


def test_terms_restart_at_every_episode_and_end_on_the_final_observation():
    """A 2-step time limit: the step that ends the episode returns the next
    episode's fresh terms; ``final_observation`` keeps the ended one's."""
    obs, infos, *_ = _run([0.5, 0.25, 0.5], max_steps=2)
    np.testing.assert_allclose(obs[2], [1.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        infos[1]["final_observation"], [0.25, 1.75, -0.25, 0.75, 1.0]
    )
    np.testing.assert_allclose(obs[3], [0.5, 1.5, -0.5, 0.5, 1.0])
    np.testing.assert_allclose(infos[2]["final_observation"], obs[3])


def test_terms_restart_after_a_termination():
    obs, infos, *_ = _run([1.0] * 11 + [0.0], max_steps=100, dt=0.1)
    assert bool(infos[10]["terminated"]) and not bool(infos[9]["terminated"])
    np.testing.assert_allclose(obs[11], [1.0, 0.1, 0.0, 0.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(infos[10]["final_observation"][2], -10.0, rtol=1e-5)


def test_integral_limit_clamps_and_unwinds_at_once():
    """e: 1, 1, 1, -1 -> I: .5, .6, .6, .1 under the .6 clamp (1.0 unclamped)."""
    clamped, *_ = _run([0.0, 0.0, 2.0], dt=0.5, integral_limit=0.6)
    free, *_ = _run([0.0, 0.0, 2.0], dt=0.5)
    np.testing.assert_allclose(clamped[:, 1], [0.5, 0.6, 0.6, 0.1], rtol=1e-6)
    np.testing.assert_allclose(free[:, 1], [0.5, 1.0, 1.5, 1.0])


def test_get_obs_and_space_match_the_observation():
    obs, _, env, state, params = _run([0.5, 0.25])
    np.testing.assert_allclose(env.get_obs(state, params), obs[-1])
    assert env.observation_space(params).shape == obs[-1].shape == (5,)
    assert env.n_errors == 1


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
        PIDTermsWrapper(_Integrator(), **kwargs)


def test_ajax_steps_the_wrapper_in_training_and_evaluation():
    """Ajax's vmapped reset/step drive it, and the evaluation env keeps it."""
    env = PIDTermsWrapper(_Integrator(), (0,), (1,))
    params = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    for built in (env, setup_environment(env, params, 3, None, 0.99)[0]):
        obs, state = reset(keys, built, "gymnax", params)
        obs, *_ = step(keys, state, jnp.full((3, 1), 0.5), built, "gymnax", params)
        np.testing.assert_allclose(obs[0], [0.5, 1.5, -0.5, 0.5, 1.0])
