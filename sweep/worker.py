import time
import os
from functools import partial
import multiprocessing
import optuna
import yaml
from train import train
from utils import AGENT_MAP, get_args, get_log_config_for_sweep, get_study
import subprocess
import sys
from target_gym import Plane
import warnings

warnings.filterwarnings(
    "ignore",
    message="Explicitly requested dtype float64 requested in asarray is not available.*",
)


def load_search_space(yaml_path, env_id):
    """Load hyperparameter definitions from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config[env_id]


def suggest_from_yaml(trial: optuna.trial.Trial, search_space):
    """Use Optuna trial to suggest values based on the YAML spec."""
    config = {}

    for name, params in search_space.items():
        ptype = params["type"]

        if ptype == "float":
            low = float(params["low"])
            high = float(params["high"])
            log = params.get("log", False)
            config[name] = trial.suggest_float(name, low, high, log=log)

        elif ptype == "int":
            low = int(params["low"])
            high = int(params["high"])
            log = params.get("log", False)
            config[name] = trial.suggest_int(name, low, high, log=log)

        elif ptype == "categorical":
            choices = params["choices"]
            config[name] = trial.suggest_categorical(name, choices)

        else:
            raise ValueError(f"Unknown parameter type: {ptype}")

    return config


def objective(
    trial,
    agent,
    env_id,
    n_seeds,
    n_timesteps,
    num_episode_test,
    env_params,
    logging_config,
    config_name,
):
    # try:
    config = suggest_from_yaml(
        trial,
        load_search_space(f"sweep/configs/{agent.name}/{config_name}.yaml", env_id),
    )
    if env_id.lower() == "plane":
        env_id = Plane(integration_method="rk4_1")
    else:
        env_id = args.env_id
    reward = train(
        config,
        agent,
        env_id,
        n_seeds,
        n_timesteps,
        num_episode_test,
        env_params,
        logging_config,
    )

    import jax.numpy as jnp

    if not jnp.isfinite(reward):
        raise optuna.TrialPruned()

    return reward


# except Exception as e:
#     print(f"[Trial failed] {e}")
#     raise optuna.TrialPruned()


def run_worker_trial(args, trial_idx):
    """Run a single trial in this process (called with subprocess)."""
    print(
        f"[Worker {trial_idx}] Starting. GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    start_time = time.time()
    study = get_study(args)

    objective_fn = partial(
        objective,
        agent=AGENT_MAP[args.agent],
        env_id=args.env_id,
        n_seeds=args.n_seeds,
        n_timesteps=args.n_timesteps,
        num_episode_test=args.n_episode_test,
        env_params=None,
        logging_config=get_log_config_for_sweep(
            args.n_seeds, agent_name=args.agent, short=True
        ),
        config_name=args.config_name,
    )
    study.optimize(objective_fn, n_trials=1)
    print(f"[Worker {trial_idx}] Finished in {time.time() - start_time:.1f}s.")


def main(args):
    GPU_MEMORY_FRAC = "0.125"
    GPU_IDS = ["1"]
    MAX_JOBS_PER_GPU = int(1 / float(GPU_MEMORY_FRAC))
    TOTAL_TRIALS = args.n_trials
    MAX_WORKERS = args.n_workers

    gpu_job_counts = {gpu: 0 for gpu in GPU_IDS}
    active_processes = []
    trial_idx = 0

    print("Starting trials!")
    while trial_idx < TOTAL_TRIALS or active_processes:
        # Clean up finished processes
        alive = []
        for p, gpu_id in active_processes:
            if p.poll() is None:
                alive.append((p, gpu_id))
            else:
                gpu_job_counts[gpu_id] -= 1
        active_processes = alive

        # Launch new trials if GPUs have capacity and worker limit not reached
        while trial_idx < TOTAL_TRIALS and len(active_processes) < MAX_WORKERS:
            # Find GPU with least number of jobs that has capacity
            available_gpus = [
                (gpu_id, count)
                for gpu_id, count in gpu_job_counts.items()
                if count < MAX_JOBS_PER_GPU
            ]

            if not available_gpus:
                break

            # Sort by job count (ascending) to get GPU with least jobs
            gpu_id = min(available_gpus, key=lambda x: x[1])[0]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            # env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = GPU_MEMORY_FRAC
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            env["TF_CPP_MIN_LOG_LEVEL"] = "3"

            # Build command excluding --worker-trial from original args
            cmd = [sys.executable, __file__, "--worker-trial", str(trial_idx)]

            # Add all original arguments except --worker-trial
            skip_next = False
            for i, arg in enumerate(sys.argv[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--worker-trial":
                    skip_next = True
                    continue
                cmd.append(arg)

            print(
                f"[Scheduler] Launching trial {trial_idx} on GPU {gpu_id} (current jobs: {gpu_job_counts[gpu_id]})"
            )
            p = subprocess.Popen(cmd, env=env)
            active_processes.append((p, gpu_id))
            gpu_job_counts[gpu_id] += 1
            trial_idx += 1
            time.sleep(0.5)

        time.sleep(1)

    print("✅ All trials finished.")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = get_args()

    # Check if this is a worker process
    if "--worker-trial" in sys.argv:
        trial_idx = int(sys.argv[sys.argv.index("--worker-trial") + 1])
        run_worker_trial(args, trial_idx)
        sys.exit(0)  # Exit after completing the worker trial
    else:
        # This is the main scheduler process
        main(args)
