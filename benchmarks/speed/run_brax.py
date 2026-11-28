"""Times Brax's SAC and PPO trainers, one seed per ``train`` call, as Brax
and MuJoCo Playground run them: ``brax.training.agents.<algorithm>.train``
on a Brax environment, HalfCheetah-v5 from gymnasium-mjx through Playground's
``wrap_for_brax_training``, Pendulum-v1 from gymnax.

usage: python run_brax.py ALGORITHM TASK SEED SEGMENT RECORD

Brax trains in epochs between evaluations and reports each epoch's steps per
second (``training/sps``), evaluation left out. A call runs two epochs of
SEGMENT steps: the first compiles and warms up; at its end the process waits
for the other processes of a 10-process run (``SPEED_PROCESSES``, a barrier
like speed_clock's), then the second is timed and appended to RECORD.
Imports nothing from Ajax.
"""

import functools
import json
import os
import sys
import time

import gymnax
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from speed_specs import settings


class GymnaxAsBrax(Env):
    """A gymnax environment as a Brax ``Env``; actions in ``[-1, 1]`` map
    onto its bounds, as SB3's and CleanRL's do; ``done`` is gymnax's
    termination, Brax's EpisodeWrapper cutting the episode at its limit."""

    def __init__(self, env_id: str):
        self._env, self._params = gymnax.make(env_id)
        self._space = self._env.action_space(self._params)

    def reset(self, rng):
        obs, state = self._env.reset_env(rng, self._params)
        zero = jnp.zeros(())
        return State(pipeline_state=state, obs=obs, reward=zero, done=zero, info={})

    def step(self, state, action):
        low, high = self._space.low, self._space.high
        u = low + (action + 1.0) / 2.0 * (high - low)
        obs, inner, reward, done, _ = self._env.step_env(
            jax.random.PRNGKey(0), state.pipeline_state, u, self._params
        )
        done = jnp.asarray(done, jnp.float32)
        return state.replace(pipeline_state=inner, obs=obs, reward=reward, done=done)

    @property
    def observation_size(self):
        return self._env.observation_space(self._params).shape[0]

    @property
    def action_size(self):
        return self._space.shape[0]

    @property
    def backend(self):
        return "gymnax"


def environment(task: str) -> tuple:
    """The env, its episode length and the wrapper ``train`` applies."""
    if task == "Pendulum-v1":
        return GymnaxAsBrax(task), 200, None
    import gymnasium_mjx
    import gymnasium_mjx.playground
    from mujoco_playground import wrapper

    class KeepMetrics(gymnasium_mjx.playground.PlaygroundTask):
        """gymnasium-mjx's step returns only its reward terms as metrics,
        dropping the "reward" Brax's evaluator adds at reset, and Brax's
        scans need the same keys on both sides; this keeps every key."""

        def step(self, state, action):
            stepped = super().step(state, action)
            return stepped.replace(metrics={**state.metrics, **stepped.metrics})

    env = KeepMetrics(gymnasium_mjx.make(task, physics="jax"))
    return env, 1000, wrapper.wrap_for_brax_training


def keywords(algorithm: str, task: str, segment: int) -> dict:
    s = settings(algorithm, task)
    if algorithm == "SAC":
        return {
            "num_timesteps": s["learning_starts"] + 2 * segment,
            "num_envs": s["n_envs"],
            "batch_size": s["batch_size"],
            "grad_updates_per_step": s["updates_per_step"],
            "learning_rate": s["learning_rate"],
            "discounting": s["gamma"],
            "tau": s["tau"],
            "min_replay_size": s["learning_starts"],
            "max_replay_size": s["buffer_size"],
            "num_eval_envs": 1,
        }  # Its networks are 256-256 ReLU by default.
    from brax.training.agents.ppo import networks

    # Brax's PPO collects batch_size x num_minibatches unrolls per update and
    # makes each minibatch of whole unrolls: 32 unrolls of 64 steps.
    unroll = s["n_steps"] // s["n_minibatches"]
    return {
        "num_timesteps": 2 * segment,
        "num_envs": s["n_envs"],
        "unroll_length": unroll,
        "batch_size": 1,
        "num_minibatches": s["n_minibatches"],
        "num_updates_per_batch": s["n_epochs"],
        "learning_rate": s["learning_rate"],
        "entropy_cost": s["ent_coef"],
        "discounting": s["gamma"],
        "gae_lambda": s["gae_lambda"],
        "clipping_epsilon": s["clip_range"],
        "vf_loss_coefficient": s["vf_coef"],
        "max_grad_norm": s["max_grad_norm"],
        "normalize_observations": s["normalize"],
        "run_evals": False,
        "network_factory": functools.partial(
            networks.make_ppo_networks,
            policy_hidden_layer_sizes=s["hidden"],
            value_hidden_layer_sizes=s["hidden"],
            activation=jax.nn.tanh,
            distribution_type="normal",
        ),
    }


def wait_for_the_others(folder: str, expected: int) -> None:
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, str(os.getpid())), "w").close()
    while len(os.listdir(folder)) < expected:
        time.sleep(0.05)


def train_one(
    algorithm: str, task: str, seed: int, segment: int, record: str, expected: int
) -> None:
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.sac import train as sac

    env, episode_length, wrap = environment(task)
    marks: dict = {}

    def progress(_step, metrics):
        if "training/sps" not in metrics:  # SAC's evaluation before training
            return
        if not marks:
            wait_for_the_others(os.environ["SPEED_BARRIER"], expected)
            marks["start"] = (time.time(), time.process_time())
            return
        start, cpu = marks["start"]
        seconds = segment / float(metrics["training/sps"])
        line = {
            "pid": os.getpid(),
            "seed": seed,
            "steps": segment,
            "start": start,
            "end": start + seconds,
            "cpu_s": time.process_time() - cpu,
        }
        with open(record, "a") as file:
            file.write(json.dumps(line) + "\n")

    train = {"SAC": sac.train, "PPO": ppo.train}[algorithm]
    train(
        environment=env,
        episode_length=episode_length,
        wrap_env_fn=wrap,
        num_evals=3,
        seed=seed,
        progress_fn=progress,
        **keywords(algorithm, task, segment),
    )


def main() -> None:
    algorithm, task = sys.argv[1:3]
    seed, segment = map(int, sys.argv[3:5])
    expected = int(os.environ.get("SPEED_PROCESSES", "1"))
    train_one(algorithm, task, seed, segment, sys.argv[5], expected)


if __name__ == "__main__":
    main()
