import os
import sys
import json
import time
import subprocess
import warnings
from functools import partial

import jax.numpy as jnp
from utils import AGENT_MAP, get_args, get_study, get_log_config_for_sweep
from train import train
from target_gym import Plane

warnings.filterwarnings(
    "ignore",
    message="Explicitly requested dtype float64 requested in asarray is not available.*",
)


def optuna_config_to_actual_config(config: dict) -> dict:
    """Convert Optuna flat config to actual agent architecture config."""
    N_NEURONS = config.pop("n_neurons", 128)
    activation = config.pop("activation", "relu")
    config["actor_architecture"] = (
        f"{N_NEURONS}",
        activation,
        f"{N_NEURONS}",
        activation,
    )

    config["critic_architecture"] = (
        f"{N_NEURONS}",
        activation,
        f"{N_NEURONS}",
        activation,
    )
    return config


def get_best_n_configs(study, n=5):
    """Return best N configurations from Optuna study."""
    reverse = study.direction.name == "MAXIMIZE"
    valid_trials = [t for t in study.trials if t.value is not None]
    best_trials = sorted(valid_trials, key=lambda t: t.value, reverse=reverse)[:n]
    results = []
    for t in best_trials:
        results.append({"config": t.params.copy(), "value": t.value})
    return results


def run_worker(args, idx):
    """Worker process: run a single configuration."""
    with open(f"sweep/{args.agent}/best_n_configs.json", "r") as f:
        best_n_configs = json.load(f)

    entry = best_n_configs[idx]
    config = optuna_config_to_actual_config(entry["config"])

    if args.env_id.lower() == "plane":
        env_id = Plane(integration_method="rk4_1")
    else:
        env_id = args.env_id

    print(
        f"[Worker {idx}] Running config on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    reward = train(
        config=config,
        agent=AGENT_MAP[args.agent],
        env_id=env_id,
        n_seeds=args.n_seeds_long,
        n_timesteps=args.n_timesteps_long,
        num_episode_test=args.n_episode_test,
        env_params=None,
        logging_config=get_log_config_for_sweep(args.n_seeds, args.agent, short=False),
    )

    if not jnp.isfinite(reward):
        reward = float("nan")

    result = {"index": idx, "config": config, "reward": float(reward)}

    os.makedirs(f"sweep/{args.agent}/results", exist_ok=True)
    with open(f"sweep/{args.agent}/results/result_{idx}.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"[Worker {idx}] Finished with reward {reward:.3f}")
    return reward


def main(args):
    # Step 1: Load best N configs and save to JSON
    study = get_study(args)
    best_n = get_best_n_configs(study, n=args.subset_size)
    os.makedirs(f"sweep/{args.agent}", exist_ok=True)
    with open(f"sweep/{args.agent}/best_n_configs.json", "w") as f:
        json.dump(best_n, f, indent=4)
    print(f"Saved best {args.subset_size} configs to JSON.")

    # Step 2: Distributed execution setup (same as original)
    GPU_MEMORY_FRAC = "0.45"
    GPU_IDS = ["0", "1"]
    MAX_JOBS_PER_GPU = int(1 / float(GPU_MEMORY_FRAC))
    TOTAL_RUNS = len(best_n)
    MAX_WORKERS = args.n_workers

    gpu_job_counts = {gpu: 0 for gpu in GPU_IDS}
    active_processes = []
    run_idx = 0

    print("Starting distributed evaluation of best configs!")
    while run_idx < TOTAL_RUNS or active_processes:
        # Clean up finished processes
        alive = []
        for p, gpu_id in active_processes:
            if p.poll() is None:
                alive.append((p, gpu_id))
            else:
                gpu_job_counts[gpu_id] -= 1
        active_processes = alive

        # Launch new jobs if capacity allows
        while run_idx < TOTAL_RUNS and len(active_processes) < MAX_WORKERS:
            available_gpus = [
                (gpu_id, count)
                for gpu_id, count in gpu_job_counts.items()
                if count < MAX_JOBS_PER_GPU
            ]
            if not available_gpus:
                break

            gpu_id = min(available_gpus, key=lambda x: x[1])[0]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            env["TF_CPP_MIN_LOG_LEVEL"] = "3"

            cmd = [sys.executable, __file__, "--worker-trial", str(run_idx)]
            for arg in sys.argv[1:]:
                if arg not in ["--worker-trial"]:
                    cmd.append(arg)

            print(f"[Scheduler] Launching config {run_idx} on GPU {gpu_id}")
            p = subprocess.Popen(cmd, env=env)
            active_processes.append((p, gpu_id))
            gpu_job_counts[gpu_id] += 1
            run_idx += 1
            time.sleep(0.5)

        time.sleep(1)

    print("✅ All configs evaluated.")

    # Step 3: Aggregate results
    results_dir = f"sweep/{args.agent}/results"
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r") as f:
                results.append(json.load(f))

    results = sorted(results, key=lambda x: x["reward"], reverse=True)
    with open(f"sweep/{args.agent}/final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"🏁 Done! Top result: {results[0]}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = get_args()

    if "--worker-trial" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--worker-trial") + 1])
        run_worker(args, idx)
        sys.exit(0)
    else:
        main(args)
