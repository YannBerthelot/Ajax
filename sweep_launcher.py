import itertools
import os
import signal
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()  # Loads from .env file into environment variables
# Fixed part of the command
LOCAL = os.getenv("LOCAL", "False").lower() == "true"
if LOCAL:
    base_cmd = ["poetry", "run", "python", "src/ajax/agents/DynaSAC/dyna_SAC_multi.py"]
else:
    base_cmd = [
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "CUDA_VISIBLE_DEVICES=0",
        "poetry",
        "run",
        "python",
        "-m",
        "ajax.agents.DynaSAC.dyna_SAC",
    ]

# Define the hyperparameters to sweep
# sweep_params = {
#     "distillation_lr": [1e-5, 1e-4],
#     "n_avg_agents": [1, 4, 16, 64],
#     "n_envs_AVG":[1, 4, 16, 64],
#     "n_epochs_distillation": [1, 2, 5,10],
# }
sweep_params = {
    "distillation_lr": [1e-5, 1e-4],
}

# Generate all combinations of hyperparameters
keys = list(sweep_params.keys())
value_combinations = list(itertools.product(*sweep_params.values()))

print(f"Launching {len(value_combinations)} runs...")


processes = []

try:
    for i, values in enumerate(value_combinations):
        # Create CLI arguments from the combination
        flags = []
        for key, val in zip(keys, values):
            flags.append(f"--{key}")
            flags.append(str(val))

        # Join full command
        cmd = base_cmd + flags
        print(f"\n[{i+1}/{len(value_combinations)}] Launching: {' '.join(cmd)}")

        # Start the subprocess
        p = subprocess.Popen(cmd)
        processes.append(p)

    # Optionally wait for all to complete
    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nCaught KeyboardInterrupt. Terminating all running jobs...")
    for p in processes:
        try:
            p.send_signal(signal.SIGINT)  # or p.terminate()
        except Exception as e:
            print(f"Failed to kill process {p.pid}: {e}")
    sys.exit(1)


# Optionally, wait for all to finish
# for p in processes:
#     p.wait()
