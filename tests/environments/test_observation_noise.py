"""Observation noise: ObservationNoiseWrapper (gymnax) and its brax variant."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from gymnax import make as make_gymnax_env

from ajax.environments.create import build_env_from_id
from ajax.environments.interaction import reset, step
from ajax.wrappers import ObservationNoiseWrapper, ObservationNoiseWrapperBrax

SCALE = jnp.array([1.0, 2.0, 0.5])


def _draw(key, shape, sigma=0.1):
    return sigma * SCALE * jax.random.normal(jax.random.fold_in(key, 0), shape)


def test_gymnax_noise_is_drawn_per_call_and_leaves_the_env_unchanged():
    """Reset and step return the env's observation plus sigma * scale times
    a draw off the call's key; the state, the reward and get_obs are the
    noiseless env's, and the final observation gets the step's draw."""
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(max_steps_in_episode=2)
    noisy = ObservationNoiseWrapper(env, sigma=0.1, scale=SCALE)
    key = jax.random.PRNGKey(0)
    obs, state = noisy.reset(key, params)
    clean_obs, clean_state = env.reset(key, params)
    np.testing.assert_allclose(obs, clean_obs + _draw(key, (3,)), rtol=1e-6)
    jax.tree.map(np.testing.assert_array_equal, state, clean_state)
    for i in range(2):  # the second step ends the episode
        k, action = jax.random.fold_in(key, i + 1), jnp.array([0.5])
        out = noisy.step(k, state, action, params)
        ref = env.step(k, state, action, params)
        noise = _draw(k, (3,))
        np.testing.assert_allclose(out[0], ref[0] + noise, rtol=1e-6)
        final = out[5]["final_observation"]
        np.testing.assert_allclose(
            final, ref[5]["final_observation"] + noise, rtol=1e-6
        )
        assert out[2] == ref[2] and out[3] == ref[3] and out[4] == ref[4]
        state = out[1]
        np.testing.assert_array_equal(noisy.get_obs(state, params), env.get_obs(state))
    assert bool(out[4])  # truncated: the final observation is the pre-reset one


def test_gymnax_noise_differs_across_envs_and_steps():
    env, params = make_gymnax_env("Pendulum-v1")
    noisy = ObservationNoiseWrapper(env, sigma=1.0)
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    obs, state = reset(keys, noisy, "gymnax", params)
    clean, _ = reset(keys, env, "gymnax", params)
    noise = np.asarray(obs - clean)
    assert len(np.unique(noise.round(5), axis=0)) == 4
    step_keys = jax.random.split(jax.random.PRNGKey(2), 4)
    obs2, *_ = step(step_keys, state, jnp.zeros((4, 1)), noisy, "gymnax", params)
    assert not np.allclose(obs2, obs)


def test_brax_noise_advances_its_key_and_noises_the_final_observation():
    env, _ = build_env_from_id("fast", n_envs=3, episode_length=4)
    noisy = ObservationNoiseWrapperBrax(env, sigma=0.5)
    key = jax.random.PRNGKey(0)
    state, clean = noisy.reset(key), env.reset(key)
    keys = [state.info[ObservationNoiseWrapperBrax._RNG_KEY]]
    noises = [state.obs - clean.obs]
    for _ in range(4):
        action = jnp.zeros((3, env.action_size))
        state, clean = noisy.step(state, action), env.step(clean, action)
        keys.append(state.info[ObservationNoiseWrapperBrax._RNG_KEY])
        noises.append(state.obs - clean.obs)
        np.testing.assert_allclose(
            state.info["final_obs"] - clean.info["final_obs"], noises[-1], atol=1e-6
        )
        np.testing.assert_array_equal(state.reward, clean.reward)
    assert keys[0].shape == (3, 2)
    assert len({tuple(np.asarray(k[0])) for k in keys}) == len(keys)
    assert all(not np.allclose(a, b) for a, b in itertools.pairwise(noises))


def test_an_agent_trains_on_a_noisy_env():
    from ajax import PPO

    env, params = make_gymnax_env("Pendulum-v1")
    agent = PPO(
        ObservationNoiseWrapper(env, sigma=0.1),
        env_params=params,
        n_envs=2,
        n_steps=8,
        batch_size=8,
        actor_architecture=("8", "tanh"),
        critic_architecture=("8", "tanh"),
    )
    state, _ = agent.train(seed=0, n_timesteps=32)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state.actor_state.params))
