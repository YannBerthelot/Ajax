"""Times an Ajax agent, every seed in one compiled program: ``train(seed=0)``
for one seed, ``train(seed=list(range(10)))`` for ten.

usage: uv run python benchmarks/speed/run_ajax.py ALGORITHM TASK N_SEEDS WARMUP SEGMENT

A first call trains WARMUP steps per seed and compiles. Two calls then resume
from it, one after the other: a short one, about a quarter of SEGMENT, and
one of SEGMENT steps. Each is timed less the seconds JAX spends tracing and
compiling its program (each resumed call compiles a program of its own); the
difference of the two is the stretch's time, so a call's fixed cost (seconds
on HalfCheetah) cancels, as the other libraries' timings leave out their
set-up. No evaluation runs (logging_config=None). Prints one JSON line.
"""

import json
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from agent_bench import compile_spans, covered_seconds
from speed_specs import AJAX_IDS, settings


def layers(hidden: tuple, activation: str) -> tuple:
    return tuple(x for size in hidden for x in (str(size), activation))


def keywords(algorithm: str, task: str) -> dict:
    s = settings(algorithm, task)
    net = layers(s["hidden"], s["activation"])
    if algorithm in ("SAC", "TD3"):
        out = {
            "n_envs": s["n_envs"],
            "buffer_size": s["buffer_size"],
            "batch_size": s["batch_size"],
            "learning_starts": s["learning_starts"],
            "train_frequency": 1,
            "actor_learning_rate": s["learning_rate"],
            "critic_learning_rate": s["learning_rate"],
            "gamma": s["gamma"],
            "tau": s["tau"],
            "actor_architecture": net,
            "critic_architecture": net,
        }
        if algorithm == "SAC":
            out["alpha_learning_rate"] = s["learning_rate"]
        else:
            out.update(
                {
                    "policy_delay": s["policy_delay"],
                    "target_policy_noise": s["target_noise"],
                    "target_noise_clip": s["target_noise_clip"],
                    "exploration_noise": s["exploration_noise"],
                }
            )
        return out
    if algorithm == "PPO":
        return {
            "n_envs": s["n_envs"],
            "n_steps": s["n_steps"],
            "n_epochs": s["n_epochs"],
            "num_minibatches": s["n_minibatches"],
            "actor_learning_rate": s["learning_rate"],
            "critic_learning_rate": s["learning_rate"],
            "gamma": s["gamma"],
            "gae_lambda": s["gae_lambda"],
            "clip_range": s["clip_range"],
            "ent_coef": s["ent_coef"],
            "vf_coef": s["vf_coef"],
            "max_grad_norm": s["max_grad_norm"],
            "actor_architecture": net,
            "critic_architecture": net,
            "normalize_observations": s["normalize"],
            "normalize_rewards": s["normalize"],
        }
    start, end, fraction = s["epsilon"]
    common = {
        "n_envs": s["n_envs"],
        "learning_rate": s["learning_rate"],
        "gamma": s["gamma"],
        "architecture": net,
        "epsilon_start": start,
        "epsilon_end": end,
        "epsilon_decay_frac": fraction,
    }
    if algorithm == "DQN":
        common.update(
            {
                "buffer_size": s["buffer_size"],
                "batch_size": s["batch_size"],
                "learning_starts": s["learning_starts"],
                "train_frequency": s["train_frequency"],
                "n_gradient_steps": 1,
                # Counted in gradient steps: one every train_frequency env steps.
                "target_update_interval": s["target_update"] // s["train_frequency"],
            }
        )
        return common
    common.update(
        {  # PQN, whose layers carry layer norm
            "n_steps": s["n_steps"],
            "n_epochs": s["n_epochs"],
            "num_minibatches": s["n_minibatches"],
            "q_lambda": s["q_lambda"],
            "max_grad_norm": s["max_grad_norm"],
        }
    )
    return common


def main() -> None:
    import ajax

    algorithm, task = sys.argv[1:3]
    n_seeds, warmup, segment = map(int, sys.argv[3:6])
    s = settings(algorithm, task)
    unit = s["n_envs"] * s.get("n_steps", 1)  # a whole rollout per call
    short = unit * max(1, round(segment / unit / 4))
    agent = getattr(ajax, algorithm)(
        AJAX_IDS.get(task, task), **keywords(algorithm, task)
    )
    seed = 0 if n_seeds == 1 else list(range(n_seeds))
    run = {"total_timesteps": warmup + short + segment, "logging_config": None}
    spans = compile_spans()
    state, _ = agent.train(seed=seed, n_timesteps=warmup, **run)
    windows = []
    for steps in (short, segment):
        spans.clear()
        start, cpu = time.time(), time.process_time()
        state, _ = agent.train(seed=seed, n_timesteps=steps, initial_state=state, **run)
        end = time.time()
        windows.append(
            {
                "steps": steps,
                "start": start,
                "end": end,
                "cpu_s": time.process_time() - cpu,
                "compile_s": covered_seconds(spans),
            }
        )
    short_s, long_s = (w["end"] - w["start"] - w["compile_s"] for w in windows)
    record = {
        "steps": n_seeds * (segment - short),
        "seconds": long_s - short_s,
        "windows": windows,
        "device_kind": ajax_device(),
    }
    print(json.dumps(record), flush=True)


def ajax_device() -> str:
    import jax

    return jax.devices()[0].device_kind


if __name__ == "__main__":
    main()
