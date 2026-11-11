"""Learning curves of one agent on Gymnasium's MuJoCo task and on Ajax's.

The same agent, settings and seeds train on "gymnasium:<task>" (Gymnasium on
MuJoCo's C library, stepped in Python) and on "mujoco:<task>" (Ajax's task on
MJX). If the tasks are the same, the curves agree within seed noise. Each
call trains one side and appends its evaluations to a JSON-lines file;
--plot draws both sides' mean and 95% band per task from that file.

    python benchmarks/mujoco_curves.py train --side mujoco --task Hopper-v5 \
        --seeds 0 1 2 3 4 --steps 1000000 --out curves.jsonl
    python benchmarks/mujoco_curves.py plot --out curves.jsonl --figure curves.png
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

EVALUATION_KEY = "Eval/episodic mean reward"


def train(arguments: argparse.Namespace) -> None:
    import ajax
    from ajax.logging.wandb_logging import LoggingConfig

    agent = getattr(ajax, arguments.agent)(f"{arguments.side}:{arguments.task}")
    start = time.time()
    _, evaluations = agent.train(
        seed=arguments.seeds,
        n_timesteps=arguments.steps,
        logging_config=LoggingConfig(log_frequency=arguments.eval_every, backends=()),
    )
    returns = np.asarray(evaluations[EVALUATION_KEY])
    steps = np.asarray(evaluations["timestep"])
    with open(arguments.out, "a") as file:
        for index, seed in enumerate(arguments.seeds):
            file.write(
                json.dumps(
                    {
                        "agent": arguments.agent,
                        "task": arguments.task,
                        "side": arguments.side,
                        "seed": seed,
                        "steps": steps[index].tolist(),
                        "returns": returns[index].tolist(),
                        "wall_s": time.time() - start,
                    }
                )
                + "\n"
            )


def plot(arguments: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    runs: dict = defaultdict(lambda: defaultdict(list))
    with open(arguments.out) as file:
        for line in file:
            row = json.loads(line)
            runs[row["task"]][row["side"]].append(row)
    tasks = sorted(runs)
    figure, axes = plt.subplots(
        1, len(tasks), figsize=(4.5 * len(tasks), 3.5), squeeze=False
    )
    labels = {"gymnasium": "Gymnasium (C MuJoCo)", "mujoco": "Ajax (MJX)"}
    for axis, task in zip(axes[0], tasks, strict=False):
        for side, rows in sorted(runs[task].items()):
            steps = np.asarray(rows[0]["steps"])
            returns = np.asarray([row["returns"] for row in rows])
            mean = returns.mean(axis=0)
            # A 95% band for the mean over seeds (normal approximation).
            band = 1.96 * returns.std(axis=0, ddof=1) / np.sqrt(len(rows))
            axis.plot(steps, mean, label=f"{labels[side]}, {len(rows)} seeds")
            axis.fill_between(steps, mean - band, mean + band, alpha=0.25)
        axis.set_title(task)
        axis.set_xlabel("environment steps")
        axis.set_ylabel("evaluation return")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(arguments.figure, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("train", "plot"))
    parser.add_argument("--side", choices=("gymnasium", "mujoco"))
    parser.add_argument("--task")
    parser.add_argument("--agent", default="SAC")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--out", default="curves.jsonl")
    parser.add_argument("--figure", default="curves.png")
    arguments = parser.parse_args()
    if arguments.action == "train":
        train(arguments)
    else:
        plot(arguments)


if __name__ == "__main__":
    main()
