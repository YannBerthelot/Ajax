"""The Stable-Baselines3 references (PPO, DQN, SAC, TD3).

usage: uv run --project envs/sb3 python refcmp_sb3.py AGENT FIRST_SEED N_SEEDS
           >> results/sb3.jsonl

Imports nothing from Ajax. SB3 trains and evaluates on the gymnax
environment Ajax trains on, through :func:`gymnax_vec_env`. One torch thread
per process; run several processes for several seed blocks.
"""

import json
import sys
import time

from refcmp_specs import COMPARISONS


def sb3_kwargs(spec) -> dict:
    """The SB3 constructor's keywords for ``spec.reference_config``; the
    harness's own keys (n_envs, net_arch, activation, action_noise_std) are
    returned under ``_harness`` for the runner to apply."""
    kwargs = dict(spec.reference_config)
    harness = {
        k: kwargs.pop(k)
        for k in ("n_envs", "net_arch", "activation", "action_noise_std")
        if k in kwargs
    }
    return {**kwargs, "_harness": harness}


def gymnax_vec_env(env_id: str, n_envs: int, seed: int, base=None):
    """``n_envs`` copies of a gymnax environment as an SB3 ``VecEnv`` (or
    ``base``), stepped by one jitted, vmapped call. gymnax restarts an env
    that ends, as SB3's own vectorised envs do; the env's last observation
    goes into ``terminal_observation``, and a time limit sets
    ``TimeLimit.truncated``, from which SB3 bootstraps."""
    import gymnasium
    import gymnax
    import jax
    import numpy as np
    from gymnax.environments import spaces

    if base is None:
        from stable_baselines3.common.vec_env import VecEnv as base

    env, params = gymnax.make(env_id)

    def batched(f):
        """``f`` over the envs, its key split inside one jitted call."""

        def call(key, *args):
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, n_envs)
            return key, jax.vmap(f, (0, *[0] * len(args), None))(keys, *args, params)

        return jax.jit(call)

    reset, step = batched(env.reset), batched(env.step)

    def space(s):
        if isinstance(s, spaces.Discrete):
            return gymnasium.spaces.Discrete(s.n)
        low, high = (
            np.broadcast_to(b, s.shape).astype(np.float32) for b in (s.low, s.high)
        )
        return gymnasium.spaces.Box(low, high, dtype=np.float32)

    class GymnaxVecEnv(base):
        render_mode = None

        def __init__(self):
            obs_space, act_space = (
                env.observation_space(params),
                env.action_space(params),
            )
            super().__init__(n_envs, space(obs_space), space(act_space))
            self.seed(seed)

        def seed(self, seed=None):
            self.key = jax.random.PRNGKey(0 if seed is None else seed)
            return [seed] * n_envs

        def reset(self):
            self.key, (obs, self.state) = reset(self.key)
            return np.array(obs)

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            self.key, (obs, self.state, reward, term, trunc, info) = step(
                self.key, self.state, self.actions
            )
            term, trunc = np.asarray(term), np.asarray(trunc)
            final = np.asarray(info["final_observation"])
            infos = [
                {
                    "terminal_observation": final[i],
                    "TimeLimit.truncated": trunc[i] and not term[i],
                }
                if term[i] or trunc[i]
                else {}
                for i in range(n_envs)
            ]
            return np.array(obs), np.array(reward, np.float32), term | trunc, infos

        def close(self):
            pass

        def get_attr(self, name, *_):
            return [getattr(self, name)] * n_envs

        def set_attr(self, name, value, *_):
            setattr(self, name, value)

        def env_method(self, *_, **__):
            return [None] * n_envs

        def env_is_wrapped(self, *_):
            return [False] * n_envs

    return GymnaxVecEnv()


def run(spec, seeds: list) -> list:
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.noise import NormalActionNoise

    torch.set_num_threads(1)

    class Evaluate(BaseCallback):
        """Evaluates every ``eval_every`` env steps inside one ``learn`` call
        (learning in chunks would restart SB3's schedules at each chunk), at
        the start of a rollout, so after the update of the steps before it,
        as Ajax evaluates; the caller evaluates the end of the run."""

        def __init__(self, env, eval_every):
            super().__init__()
            self.env, self.eval_every, self.curve = env, eval_every, []

        def _on_rollout_start(self) -> None:
            n = self.num_timesteps
            if 0 < n < spec.n_timesteps and n % self.eval_every == 0:
                self.evaluate()

        def _on_step(self) -> bool:
            return True

        def evaluate(self) -> None:
            mean, _ = evaluate_policy(
                self.model, self.env, spec.n_eval_episodes, deterministic=True
            )
            self.curve.append(float(mean))

    curve = []
    for seed in seeds:
        kwargs = sb3_kwargs(spec)
        harness = kwargs.pop("_harness")
        policy_kwargs = {"net_arch": harness["net_arch"]}
        if "activation" in harness:
            policy_kwargs["activation_fn"] = {"tanh": torch.nn.Tanh}[
                harness["activation"]
            ]
        env = gymnax_vec_env(spec.env_id, harness.get("n_envs", 1), seed)
        if "action_noise_std" in harness:
            n = env.action_space.shape[0]
            kwargs["action_noise"] = NormalActionNoise(
                np.zeros(n), harness["action_noise_std"] * np.ones(n)
            )
        model = getattr(sb3, spec.agent)(
            "MlpPolicy",
            env,
            seed=seed,
            device="cpu",
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        callback = Evaluate(
            gymnax_vec_env(spec.env_id, 1, 10_000 + seed), spec.eval_every
        )
        model.learn(spec.n_timesteps, callback=callback)
        callback.evaluate()
        curve.append(callback.curve)
    return curve


if __name__ == "__main__":
    import stable_baselines3
    import torch

    spec = COMPARISONS[sys.argv[1]]
    seeds = list(range(int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3])))
    t0 = time.perf_counter()
    curve = run(spec, seeds)
    print(
        json.dumps(
            {
                "lib": "sb3",
                "agent": spec.agent,
                "arm": "reference",
                "seeds": seeds,
                "eval_every": spec.eval_every,
                "versions": {
                    "sb3": stable_baselines3.__version__,
                    "torch": torch.__version__,
                },
                "wall_s": round(time.perf_counter() - t0, 1),
                "curve": [[round(x, 2) for x in row] for row in curve],
            }
        )
    )
