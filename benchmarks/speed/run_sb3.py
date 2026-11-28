"""Trains one seed of a Stable-Baselines3 or SBX agent, as their docs do:
``Algorithm("MlpPolicy", env).learn(n)`` on Gymnasium environments in a
DummyVecEnv, verbose off, on the GPU when one is visible (envs/*-cuda).
speed_clock times the segment.

usage: python run_sb3.py {SB3,SBX} ALGORITHM TASK SEED END_STEP
"""

import importlib
import os
import sys

import numpy as np
from speed_specs import settings


def keywords(algorithm: str, task: str, env) -> dict:
    """The constructor's keywords for the comparison's settings."""
    s = settings(algorithm, task)
    if algorithm in ("SAC", "TD3"):
        out = {
            "learning_rate": s["learning_rate"],
            "buffer_size": s["buffer_size"],
            "learning_starts": s["learning_starts"],
            "batch_size": s["batch_size"],
            "tau": s["tau"],
            "gamma": s["gamma"],
            "train_freq": 1,
            "gradient_steps": s["updates_per_step"],
            "policy_kwargs": {"net_arch": list(s["hidden"])},
        }
        if algorithm == "TD3":
            from stable_baselines3.common.noise import NormalActionNoise

            n = env.action_space.shape[0]
            out.update(
                {
                    "policy_delay": s["policy_delay"],
                    "target_policy_noise": s["target_noise"],
                    "target_noise_clip": s["target_noise_clip"],
                    # SB3 adds the noise to actions scaled to [-1, 1].
                    "action_noise": NormalActionNoise(
                        np.zeros(n), s["exploration_noise"] * np.ones(n)
                    ),
                }
            )
        return out
    if algorithm == "PPO":
        rollout = s["n_envs"] * s["n_steps"]
        return {
            "n_steps": s["n_steps"],
            "batch_size": rollout // s["n_minibatches"],
            "n_epochs": s["n_epochs"],
            "learning_rate": s["learning_rate"],
            "gamma": s["gamma"],
            "gae_lambda": s["gae_lambda"],
            "clip_range": s["clip_range"],
            "ent_coef": s["ent_coef"],
            "vf_coef": s["vf_coef"],
            "max_grad_norm": s["max_grad_norm"],
            # Both default to tanh; their pi and vf networks are separate.
            "policy_kwargs": {"net_arch": list(s["hidden"])},
        }
    start, end, fraction = s["epsilon"]
    return {
        "learning_rate": s["learning_rate"],
        "buffer_size": s["buffer_size"],
        "learning_starts": s["learning_starts"],
        "batch_size": s["batch_size"],
        "gamma": s["gamma"],
        "train_freq": s["train_frequency"],
        "gradient_steps": 1,
        "target_update_interval": s["target_update"],
        "exploration_initial_eps": start,
        "exploration_final_eps": end,
        "exploration_fraction": fraction,
        "policy_kwargs": {"net_arch": list(s["hidden"])},
    }


def main() -> None:
    library, algorithm, task, seed, end = sys.argv[1:6]
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    module = importlib.import_module(
        {"SB3": "stable_baselines3", "SBX": "sbx"}[library]
    )
    s = settings(algorithm, task)
    env = make_vec_env(f"speed_clock:speed/{task}", n_envs=s["n_envs"], seed=int(seed))
    if s.get("normalize"):
        env = VecNormalize(env, gamma=s["gamma"])
    model = getattr(module, algorithm)(
        "MlpPolicy",
        env,
        seed=int(seed),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        verbose=0,
        **keywords(algorithm, task, env),
    )
    model.learn(int(end))


if __name__ == "__main__":
    main()
