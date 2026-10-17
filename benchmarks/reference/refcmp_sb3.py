"""The Stable-Baselines3 references (PPO, DQN, SAC, TD3).

usage: <python with stable-baselines3> refcmp_sb3.py AGENT FIRST_SEED N_SEEDS
           >> results/sb3.jsonl

Imports neither Ajax nor JAX: it runs in any environment with SB3. One torch
thread per process; run several processes for several seed blocks.
"""

import json
import sys
import time

from refcmp_specs import COMPARISONS, N_EVAL_EPISODES


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


def run(spec, seeds: list) -> list:
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
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
                self.model, self.env, N_EVAL_EPISODES, deterministic=True
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
        env = make_vec_env(spec.env_id, n_envs=harness.get("n_envs", 1), seed=seed)
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
        # SB3 seeds env i of seed s with s + i, so seeds' envs would share
        # streams; s * n_envs + i keeps them apart (the same for one env).
        env.seed(seed * env.num_envs)
        callback = Evaluate(
            make_vec_env(spec.env_id, n_envs=1, seed=10_000 + seed), spec.eval_every
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
