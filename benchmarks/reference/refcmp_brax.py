"""Brax's trainers as references, run on Brax or gymnax environments.

Imports nothing from Ajax, so the references stay independent of the code
they check.
"""

import gymnax
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from refcmp_jax import evaluate_final


class GymnaxAsBrax(Env):
    """A gymnax env as a Brax ``Env`` with transition gradients on, so Brax's
    APG trainer differentiates the same dynamics as Ajax's APG. Actions in
    ``[-1, 1]`` are mapped onto the env's bounds; ``done`` is the gymnax time
    limit (Brax's EpisodeWrapper cuts episodes at the same length)."""

    def __init__(self, env_id: str):
        env, self._params = gymnax.make(env_id)
        self._env = env.with_transition_gradients()
        self._space = env.action_space(self._params)
        self._obs_size = env.observation_space(self._params).shape[0]

    def reset(self, rng):
        obs, state = self._env.reset_env(rng, self._params)
        zero = jnp.zeros(())
        return State(pipeline_state=state, obs=obs, reward=zero, done=zero)

    def step(self, state, action):
        low, high = self._space.low, self._space.high
        u = low + (action + 1.0) / 2.0 * (high - low)
        obs, inner, reward, done, _ = self._env.step_env(
            jax.random.PRNGKey(0), state.pipeline_state, u, self._params
        )
        return state.replace(
            pipeline_state=inner, obs=obs, reward=reward, done=done.astype(jnp.float32)
        )

    @property
    def observation_size(self):
        return self._obs_size

    @property
    def action_size(self):
        return self._space.shape[0]

    @property
    def backend(self):
        return "gymnax"


def run_brax_apg(spec, seeds):
    from brax.training.agents.apg import train as apg

    finals = []
    for seed in seeds:  # Brax's trainer has no seed axis
        make_policy, params, _ = apg.train(
            environment=GymnaxAsBrax(spec.env_id),
            seed=seed,
            num_evals=1,
            **spec.reference_config,
        )
        policy = make_policy(params, deterministic=True)
        act = lambda obs, policy=policy: policy(obs, jax.random.PRNGKey(0))[0]  # noqa: E731
        key = jax.random.PRNGKey(10_000 + seed)
        returns = evaluate_final(act, spec.env_id, key, spec.n_eval_episodes)
        finals.append([float(returns.mean())])
    return finals


def run_brax_ppo(spec, seeds):
    """Brax's PPO trainer on a Brax env, one seed at a time (it has no seed
    axis), with its own deterministic evaluation every ``eval_every`` steps;
    the evaluation of the initial policy is left out, as Ajax makes none."""
    from brax import envs
    from brax.training.agents.ppo import train as ppo

    curves = []
    for seed in seeds:
        returns: list = []
        ppo.train(
            environment=envs.get_environment(spec.env_id),
            num_timesteps=spec.n_timesteps,
            num_evals=spec.n_timesteps // spec.eval_every + 1,
            num_eval_envs=spec.n_eval_episodes,
            deterministic_eval=True,
            seed=seed,
            progress_fn=lambda _, m, r=returns: r.append(m["eval/episode_reward"]),
            **spec.reference_config,
        )
        curves.append([float(r) for r in returns[1:]])
    return curves


RUNNERS = {"APG": run_brax_apg, "PPO": run_brax_ppo}
