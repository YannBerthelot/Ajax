"""Times an Ajax agent, every seed in one compiled program: ``train(seed=0)``
for one seed, ``train(seed=list(range(10)))`` for ten.

usage: uv run python benchmarks/speed/run_ajax.py ALGORITHM TASK N_SEEDS WARMUP SEGMENT

The method of benchmarks/suite/results/timing-0.6: a first call trains WARMUP
steps per seed and compiles; a second resumes for SEGMENT steps and is timed,
less the seconds JAX spends tracing and compiling the resumed program. No
evaluation runs (logging_config=None). Prints one JSON line.
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
    agent = getattr(ajax, algorithm)(
        AJAX_IDS.get(task, task), **keywords(algorithm, task)
    )
    seed = 0 if n_seeds == 1 else list(range(n_seeds))
    run = {"total_timesteps": warmup + segment, "logging_config": None}
    spans = compile_spans()
    state, _ = agent.train(seed=seed, n_timesteps=warmup, **run)
    spans.clear()
    start, cpu = time.time(), time.process_time()
    agent.train(seed=seed, n_timesteps=segment, initial_state=state, **run)
    end = time.time()
    compile_s = covered_seconds(spans)
    window = {"start": start, "end": end, "cpu_s": time.process_time() - cpu}
    record = {
        "steps": n_seeds * segment,
        "seconds": end - start - compile_s,
        "resume_compile_s": compile_s,
        "windows": [window],
        "device_kind": ajax_device(),
    }
    print(json.dumps(record), flush=True)


def ajax_device() -> str:
    import jax

    return jax.devices()[0].device_kind


if __name__ == "__main__":
    main()
