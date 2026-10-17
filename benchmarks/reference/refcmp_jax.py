"""The JAX references: purejaxql's PQN and Brax's APG, run on gymnax.

usage: python refcmp_jax.py AGENT FIRST_SEED N_SEEDS >> results/<ref>.jsonl

Runs in Ajax's environment (jax, gymnax, brax) but imports nothing from
Ajax, so the references stay independent of the code they check.
"""

import json
import pathlib
import sys
import time
import types

import gymnax
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from gymnax.wrappers.purerl import LogWrapper
from refcmp_specs import COMPARISONS, N_EVAL_EPISODES

VENDOR = pathlib.Path(__file__).parent / "vendor" / "purejaxql"


def evaluate_final(policy, env_id: str, key: jax.Array, n: int = N_EVAL_EPISODES):
    """Returns ``(n,)`` of one deterministic episode each: ``policy(obs)``
    answers in ``[-1, 1]``, mapped onto the env's action bounds. The shared
    evaluator for agents whose own evaluation differs (APG)."""
    env, params = gymnax.make(env_id)
    space = env.action_space(params)
    obs, state = jax.vmap(env.reset, (0, None))(jax.random.split(key, n), params)

    def step(carry, key):
        obs, state, ret, alive = carry
        action = space.low + (policy(obs) + 1.0) / 2.0 * (space.high - space.low)
        keys = jax.random.split(key, n)
        obs, state, reward, term, trunc, _ = jax.vmap(env.step, (0, 0, 0, None))(
            keys, state, action, params
        )
        ret = ret + reward * alive
        return (obs, state, ret, alive * (1.0 - (term | trunc))), None

    init = (obs, state, jnp.zeros(n), jnp.ones(n))
    keys = jax.random.split(key, params.max_steps_in_episode)
    (_, _, ret, _), _ = jax.lax.scan(step, init, keys)
    return ret


class _FiveValueStep:
    """gymnax >= 1.0 steps return ``terminated, truncated``; purejaxql, written
    for gymnax 0.0.x, expects one ``done``. Merges them (purejaxql treats a
    time limit as a termination) and keeps only LogWrapper's episode stats,
    whose leaves are per env as purejaxql's test metrics need."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, key, state, action, params=None):
        obs, state, reward, term, trunc, info = self._env.step(
            key, state, action, params
        )
        info = {k: v for k, v in info.items() if k.startswith("returned_episode")}
        return obs, state, reward, jnp.logical_or(term, trunc), info


def load_purejaxql():
    """purejaxql's pqn_gymnax module, unmodified. Its CLI imports (hydra,
    omegaconf, wandb, safetensors) serve only its entry point and are stubbed
    when missing; LogWrapper is swapped for the five-value adapter."""
    for name in ("hydra", "omegaconf", "wandb", "safetensors", "safetensors.flax"):
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.main = lambda **_: lambda f: f
            stub.OmegaConf = stub.load_file = stub.save_file = None
            sys.modules[name] = stub
    sys.path.insert(0, str(VENDOR))
    import pqn_gymnax

    pqn_gymnax.LogWrapper = lambda env: _FiveValueStep(LogWrapper(env))
    return pqn_gymnax


def run_purejaxql(spec, seeds):
    config = dict(spec.reference_config, NUM_SEEDS=len(seeds))
    train = jax.jit(jax.vmap(load_purejaxql().make_train(config)))
    metrics = train(jnp.stack([jax.random.PRNGKey(s) for s in seeds]))["metrics"]
    every = spec.eval_every // (config["NUM_ENVS"] * config["NUM_STEPS"])
    tests = metrics["test/returned_episode_returns"][:, every - 1 :: every]
    return tests.tolist()


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
        returns = evaluate_final(act, spec.env_id, jax.random.PRNGKey(10_000 + seed))
        finals.append([float(returns.mean())])
    return finals


RUNNERS = {"purejaxql": run_purejaxql, "brax": run_brax_apg}

if __name__ == "__main__":
    spec = COMPARISONS[sys.argv[1]]
    seeds = list(range(int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3])))
    t0 = time.perf_counter()
    curve = RUNNERS[spec.reference](spec, seeds)
    print(
        json.dumps(
            {
                "lib": spec.reference,
                "agent": spec.agent,
                "arm": "reference",
                "seeds": seeds,
                "eval_every": spec.eval_every,
                "versions": {"jax": jax.__version__},
                "wall_s": round(time.perf_counter() - t0, 1),
                "curve": [[round(float(x), 2) for x in row] for row in curve],
            }
        )
    )
