"""The same trained policy on Gymnasium's MuJoCo task and on Ajax's.

A policy reaches the states that matter, where random actions do not. Each
side runs the policy deterministically for --episodes episodes from its own
reset draws; the two return samples should come from the same distribution.
The test is an equivalence test (two one-sided tests at 5%): the 90%
confidence interval of the difference of mean returns must lie within
+-margin of Gymnasium's mean (normal approximation, fine at 100 episodes a
side). Episode lengths are reported beside it.

The policy is an RL Zoo SAC actor (sb3/sac-<task>-v3 on Hugging Face),
saved as plain arrays: two ReLU layers of 256, then tanh of the mean head,
rescaled to the action bounds. They were trained on v3 tasks; both sides
here run v5, so the policy only has to be the same on both, not optimal.

    python benchmarks/mujoco_same_policy.py --weights sac-Hopper-v3.npz \
        --task Hopper-v5 --episodes 100
"""

import argparse

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from ajax.environments.mujoco.tasks import make


def actor(weights: dict, low: np.ndarray, high: np.ndarray):
    def act(observation):
        hidden = jax.nn.relu(
            observation @ weights["latent_pi.0.weight"].T + weights["latent_pi.0.bias"]
        )
        hidden = jax.nn.relu(
            hidden @ weights["latent_pi.2.weight"].T + weights["latent_pi.2.bias"]
        )
        squashed = jnp.tanh(hidden @ weights["mu.weight"].T + weights["mu.bias"])
        return low + (squashed + 1.0) * 0.5 * (high - low)

    return act


def gymnasium_returns(
    task_id: str, act, episodes: int
) -> tuple[np.ndarray, np.ndarray]:
    env = gymnasium.make(task_id)
    act = jax.jit(act)
    returns, lengths = [], []
    for seed in range(episodes):
        observation, _ = env.reset(seed=seed)
        total, length, done = 0.0, 0, False
        while not done:
            observation, reward, terminated, truncated, _ = env.step(
                np.asarray(act(jnp.asarray(observation, dtype=jnp.float32)))
            )
            total += reward
            length += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(length)
    return np.asarray(returns), np.asarray(lengths)


def ajax_returns(task_id: str, act, episodes: int) -> tuple[np.ndarray, np.ndarray]:
    task = make(task_id)
    limit = task.episode_length

    def run(key):
        def step(carry, _):
            state, total, length, alive = carry
            state = task.step(state, act(state.obs))
            total = total + jnp.where(alive, state.reward, 0.0)
            length = length + alive
            alive = alive & (state.done == 0)
            return (state, total, length, alive), None

        start = (
            task.reset(key),
            jnp.zeros(()),
            jnp.zeros((), jnp.int32),
            jnp.array(True),
        )
        (_, total, length, _), _ = jax.lax.scan(step, start, None, length=limit)
        return total, length

    keys = jax.random.split(jax.random.key(0), episodes)
    totals, lengths = jax.jit(jax.vmap(run))(keys)
    return np.asarray(totals), np.asarray(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.02)
    arguments = parser.parse_args()

    weights = {
        key: jnp.asarray(value) for key, value in np.load(arguments.weights).items()
    }
    space = gymnasium.make(arguments.task).action_space
    act = actor(weights, jnp.asarray(space.low), jnp.asarray(space.high))
    gym_returns, gym_lengths = gymnasium_returns(
        arguments.task, act, arguments.episodes
    )
    ajax_totals, ajax_lengths = ajax_returns(arguments.task, act, arguments.episodes)

    difference = ajax_totals.mean() - gym_returns.mean()
    standard_error = np.sqrt(
        gym_returns.var(ddof=1) / gym_returns.size
        + ajax_totals.var(ddof=1) / ajax_totals.size
    )
    # 1.645: the one-sided 5% normal quantile, so the interval is 90%.
    low, high = difference - 1.645 * standard_error, difference + 1.645 * standard_error
    margin = arguments.margin * abs(gym_returns.mean())
    verdict = "EQUIVALENT" if -margin < low and high < margin else "NOT SHOWN"
    print(
        f"{arguments.task:12s} gymnasium {gym_returns.mean():9.1f} +- {gym_returns.std():7.1f}"
        f" (length {gym_lengths.mean():6.1f})   ajax {ajax_totals.mean():9.1f}"
        f" +- {ajax_totals.std():7.1f} (length {ajax_lengths.mean():6.1f})"
        f"   difference 90% CI [{low:.1f}, {high:.1f}], margin +-{margin:.1f}: {verdict}",
        flush=True,
    )


if __name__ == "__main__":
    main()
