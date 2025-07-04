import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
from time import sleep

from dotenv import load_dotenv
load_dotenv()  # If you still want to load .env variables

# Parse command line args
parser = argparse.ArgumentParser(description="Launcher for DynaSAC sweep")
parser.add_argument("--GPU", type=int, default=0, help="Which GPU device ID to use (default=0)")
parser.add_argument("--n_exps", type=int, default=10, help="Number of parallel experiments")
args = parser.parse_args()

N_MAX_RUNS = args.n_exps
SWEEP_PARAMS_FILE = "sweep_params.json"
PENDING_RUNS_FILE = "pending_runs.json"

LOCAL = os.getenv("LOCAL", "False").lower() == "true"
if LOCAL:
    base_cmd = ["poetry", "run", "python", "src/ajax/agents/DynaSAC/dyna_SAC_multi.py"]
else:
    # Insert the GPU device dynamically here:
    base_cmd = [
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        f"CUDA_VISIBLE_DEVICES={args.GPU}",
        "poetry",
        "run",
        "python",
        "src/ajax/agents/DynaSAC/dyna_SAC_multi.py",
    ]

def load_sweep_params():
    with open(SWEEP_PARAMS_FILE) as f:
        return json.load(f)

def generate_combinations(sweep_params):
    keys = list(sweep_params.keys())
    values_product = list(itertools.product(*sweep_params.values()))
    combos = []
    for vals in values_product:
        combos.append(dict(zip(keys, vals)))
    return combos

def load_pending_runs():
    if os.path.exists(PENDING_RUNS_FILE):
        with open(PENDING_RUNS_FILE) as f:
            return json.load(f)
    return None

def save_pending_runs(pending):
    with open(PENDING_RUNS_FILE, "w") as f:
        json.dump(pending, f, indent=2)

def build_cmd_from_config(config):
    flags = []
    for k, v in config.items():
        flags.append(f"--{k}")
        flags.append(str(v))
    return base_cmd + flags

def force_kill_all(procs):
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass

def main():
    sweep_params = load_sweep_params()
    all_combos = generate_combinations(sweep_params)

    pending_runs = load_pending_runs()
    if pending_runs is None:
        print("No pending_runs.json found, initializing with all combinations...")
        pending_runs = all_combos
        save_pending_runs(pending_runs)
    else:
        print(f"Resuming from pending_runs.json with {len(pending_runs)} runs left...")

    running_processes = []
    running_configs = []

    def reap_finished():
        nonlocal running_processes, running_configs
        new_procs = []
        new_configs = []
        for p, c in zip(running_processes, running_configs):
            ret = p.poll()
            if ret is not None:
                print(f"Run finished with exit code {ret}: {c}")
            else:
                new_procs.append(p)
                new_configs.append(c)
        running_processes = new_procs
        running_configs = new_configs

    def launch_next():
        nonlocal pending_runs, running_processes, running_configs
        while len(running_processes) < N_MAX_RUNS and pending_runs:
            config = pending_runs.pop(0)
            cmd = " ".join(build_cmd_from_config(config))
            print(f"Launching ({len(running_processes)+1}/{N_MAX_RUNS}): {cmd}")
            # Use shell=False to avoid issues with env vars split
            p = subprocess.Popen(cmd, shell =True)
            running_processes.append(p)
            running_configs.append(config)
            save_pending_runs(pending_runs)

    def signal_handler(sig, frame):
        print("\nReceived interrupt, stopping all subprocesses...")

        force_kill_all(running_processes)

        # Add currently running jobs back to pending runs so they can be retried later
        pending_runs.extend(running_configs)

        print(f"Saving {len(pending_runs)} pending runs before exit...")
        save_pending_runs(pending_runs)

        sys.exit(1)


    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    try:
        while running_processes or pending_runs:
            reap_finished()
            launch_next()
            sleep(1)  # avoid busy wait
    except KeyboardInterrupt:
        signal_handler(None, None)

    print("All runs completed!")

if __name__ == "__main__":
    
    main()
