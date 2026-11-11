"""How far Ajax's MuJoCo tasks drift from Gymnasium's over whole episodes.

Each task starts from Gymnasium's reset state and both sides take the same
random actions until Gymnasium's episode ends. With contacts, differences of
1e-15 grow along a trajectory, as any two MuJoCo builds do, so this is a
report, not a test (the tests check one step, and whole trajectories only
without contacts). It prints, per task and precision, over the episodes:
the largest observation gap at the end, the steps until the gap first
passes 1e-6 and 1e-3, the episode lengths on each side, and the largest
return gap.

    python benchmarks/mujoco_drift.py --episodes 10 --dtype float32
"""

import argparse

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from ajax.environments.mujoco.tasks import TASKS, make


def first_step_above(gaps: np.ndarray, threshold: float) -> float:
    above = np.flatnonzero(gaps > threshold)
    return float(above[0]) if above.size else float("nan")


def episode(task_id: str, task, step, template, seed: int) -> dict:
    reference = gymnasium.make(task_id)
    limit = reference.spec.max_episode_steps
    reference = reference.unwrapped
    reference.reset(seed=seed)
    data = task.set_state(
        jnp.asarray(reference.data.qpos), jnp.asarray(reference.data.qvel)
    )
    state = template.replace(data=data, obs=task.observe(data))
    random = np.random.default_rng(seed)
    gaps, gym_return, ajax_return, ajax_length = [], 0.0, 0.0, None
    for index in range(limit):
        action = random.uniform(reference.action_space.low, reference.action_space.high)
        observation, reward, terminated, _, _ = reference.step(action)
        state = step(state, jnp.asarray(action))
        gaps.append(float(np.max(np.abs(np.asarray(state.obs) - observation))))
        gym_return += reward
        if ajax_length is None:
            ajax_return += float(state.reward)
            if bool(state.done):
                ajax_length = index + 1
        if terminated:
            break
    gaps_array = np.asarray(gaps)
    return {
        "final_gap": gaps_array[-1],
        "steps_to_1e-6": first_step_above(gaps_array, 1e-6),
        "steps_to_1e-3": first_step_above(gaps_array, 1e-3),
        "gym_length": len(gaps),
        "ajax_length": ajax_length or len(gaps),
        "return_gap": abs(gym_return - ajax_return),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    arguments = parser.parse_args()
    with jax.enable_x64(arguments.dtype == "float64"):
        for task_id in arguments.tasks:
            task = make(task_id)
            step = jax.jit(task.step)
            template = task.reset(jax.random.key(0))
            rows = [
                episode(task_id, task, step, template, seed)
                for seed in range(arguments.episodes)
            ]
            summary = {key: [row[key] for row in rows] for key in rows[0]}
            print(
                f"{task_id:27s} {arguments.dtype}"
                f"  final gap max {max(summary['final_gap']):.1e}"
                f"  steps to 1e-6 median {np.nanmedian(summary['steps_to_1e-6']):.0f}"
                f"  steps to 1e-3 median {np.nanmedian(summary['steps_to_1e-3']):.0f}"
                f"  lengths gym/ajax {np.mean(summary['gym_length']):.0f}/"
                f"{np.mean(summary['ajax_length']):.0f}"
                f"  return gap max {max(summary['return_gap']):.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
