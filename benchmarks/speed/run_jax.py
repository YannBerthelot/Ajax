"""Times PureJaxRL's and purejaxql's agents as their READMEs run them:
``jax.jit(jax.vmap(make_train(config)))(keys)``, every seed in one program.

usage: python run_jax.py {PureJaxRL,purejaxql} ALGORITHM TASK N_SEEDS WARMUP SEGMENT

A run is one compiled program that cannot be stopped midway, so two are
timed: one of WARMUP steps per seed and one of WARMUP + SEGMENT. The second
repeats the first's steps exactly, so the difference of their run times is
the segment's. Each compiles first, outside the timing (jit's lower and
compile). Prints one JSON line. Imports nothing from Ajax.
"""

import json
import pathlib
import sys
import time
import types

import jax
import jax.numpy as jnp
from speed_specs import settings

REFERENCE = pathlib.Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(REFERENCE))
from refcmp_jax import _FiveValueStep, load_purejaxql  # noqa: E402


class _OneDoneStep:
    """PureJaxRL's own wrappers expect gymnax 0.0's step, one ``done`` and an
    info of ``discount``; gymnax 1.0 returns ``terminated, truncated``."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, key, state, action, params=None):
        obs, state, reward, term, trunc, info = self._env.step(
            key, state, action, params
        )
        return obs, state, reward, term | trunc, {"discount": info["discount"]}


def _mjx_wrapper(base):
    """PureJaxRL's BraxGymnaxWrapper around gymnasium-mjx's HalfCheetah-v5,
    the MJX task Ajax and Brax train on, instead of a Brax environment. Brax's
    AutoResetWrapper, which PureJaxRL's uses, restarts an episode from its
    first state; Playground's state keeps the physics in ``data``, so the
    restart is written here."""
    import gymnasium_mjx
    import gymnasium_mjx.playground
    from brax.envs.wrappers.training import EpisodeWrapper, Wrapper

    class FirstStateReset(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            state.info["first_data"], state.info["first_obs"] = state.data, state.obs
            return state

        def step(self, state, action):
            steps = state.info["steps"]
            state.info.update(steps=jnp.where(state.done, jnp.zeros_like(steps), steps))
            state = self.env.step(
                state.replace(done=jnp.zeros_like(state.done)), action
            )

            def pick(first, now):
                return jnp.where(state.done.astype(bool), first, now)

            data = jax.tree.map(pick, state.info["first_data"], state.data)
            return state.replace(
                data=data, obs=pick(state.info["first_obs"], state.obs)
            )

    class MjxGymnaxWrapper(base):
        def __init__(self, env_name, backend=None):  # noqa: ARG002
            task = gymnasium_mjx.make(env_name, physics="jax")
            env = gymnasium_mjx.playground.PlaygroundTask(task)
            env = FirstStateReset(EpisodeWrapper(env, 1000, 1))
            self._env = env
            self.action_size = env.action_size
            self.observation_size = (env.observation_size,)

    return MjxGymnaxWrapper


def _purejaxrl(algorithm: str, task: str):
    """The PureJaxRL module and its config for ``steps`` steps per seed."""
    for name in ("wandb", "navix"):  # imported, used only by other entry points
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    sys.path.insert(0, str(REFERENCE / "vendor" / "purejaxrl" / "purejaxrl"))
    s = settings(algorithm, task)
    if algorithm == "DQN":
        # dqn.py calls jax.tree_map, an alias of jax.tree.map that JAX 0.6
        # removed; restoring it runs the file unchanged on Ajax's JAX.
        if not hasattr(jax, "tree_map"):
            jax.tree_map = jax.tree.map
        import dqn as module
        from gymnax.wrappers.purerl import LogWrapper

        module.LogWrapper = lambda env: _FiveValueStep(LogWrapper(env))
        start, end, fraction = s["epsilon"]
        return module, lambda steps: {
            "NUM_ENVS": s["n_envs"],
            "BUFFER_SIZE": s["buffer_size"],
            "BUFFER_BATCH_SIZE": s["batch_size"],
            "TOTAL_TIMESTEPS": steps,
            "EPSILON_START": start,
            "EPSILON_FINISH": end,
            "EPSILON_ANNEAL_TIME": fraction * steps,
            "TARGET_UPDATE_INTERVAL": s["target_update"],
            "LR": s["learning_rate"],
            "LEARNING_STARTS": s["learning_starts"],
            "TRAINING_INTERVAL": s["train_frequency"],
            "LR_LINEAR_DECAY": False,
            "GAMMA": s["gamma"],
            "TAU": 1.0,
            "ENV_NAME": task,
            "WANDB_MODE": "disabled",
        }
    if task == "CartPole-v1":
        import gymnax
        import ppo as module

        def make(name):
            env, params = gymnax.make(name)
            return _OneDoneStep(env), params

        module.gymnax = types.SimpleNamespace(make=make)
    else:
        import ppo_continuous_action as module

        module.BraxGymnaxWrapper = _mjx_wrapper(module.BraxGymnaxWrapper)
    return module, lambda steps: {
        "LR": s["learning_rate"],
        "NUM_ENVS": s["n_envs"],
        "NUM_STEPS": s["n_steps"],
        "TOTAL_TIMESTEPS": steps,
        "UPDATE_EPOCHS": s["n_epochs"],
        "NUM_MINIBATCHES": s["n_minibatches"],
        "GAMMA": s["gamma"],
        "GAE_LAMBDA": s["gae_lambda"],
        "CLIP_EPS": s["clip_range"],
        "ENT_COEF": s["ent_coef"],
        "VF_COEF": s["vf_coef"],
        "MAX_GRAD_NORM": s["max_grad_norm"],
        "ACTIVATION": s["activation"],
        "ENV_NAME": task,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": s["normalize"],
        "DEBUG": False,
    }


def _purejaxql(task: str, n_seeds: int):
    """purejaxql's pqn_cartpole.yaml (47af6d7) with the comparison's settings,
    its tests during training off, as every library's evaluations are."""
    s = settings("PQN", task)
    start, end, fraction = s["epsilon"]
    return load_purejaxql(), lambda steps: {
        "TOTAL_TIMESTEPS": steps,
        "TOTAL_TIMESTEPS_DECAY": steps,
        "NUM_ENVS": s["n_envs"],
        "NUM_STEPS": s["n_steps"],
        "EPS_START": start,
        "EPS_FINISH": end,
        "EPS_DECAY": fraction,
        "NUM_MINIBATCHES": s["n_minibatches"],
        "NUM_EPOCHS": s["n_epochs"],
        "NORM_INPUT": False,
        "HIDDEN_SIZE": s["hidden"][0],
        "NUM_LAYERS": len(s["hidden"]),
        "NORM_TYPE": "layer_norm",
        "LR": s["learning_rate"],
        "MAX_GRAD_NORM": s["max_grad_norm"],
        "LR_LINEAR_DECAY": True,
        "REW_SCALE": 0.1,
        "GAMMA": s["gamma"],
        "LAMBDA": s["q_lambda"],
        "ENV_NAME": task,
        "ENV_KWARGS": {},
        "TEST_DURING_TRAINING": False,
        "NUM_SEEDS": n_seeds,
        "WANDB_MODE": "disabled",
    }


def time_program(module, config, n_seeds: int, steps: int) -> dict:
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
    began = time.perf_counter()
    train = jax.jit(jax.vmap(module.make_train(config(steps))))
    compiled = train.lower(keys).compile()
    compile_s = time.perf_counter() - began
    start, cpu = time.time(), time.process_time()
    jax.block_until_ready(compiled(keys))
    end = time.time()
    return {
        "start": start,
        "end": end,
        "cpu_s": time.process_time() - cpu,
        "compile_s": compile_s,
    }


def main() -> None:
    library, algorithm, task = sys.argv[1:4]
    n_seeds, warmup, segment = map(int, sys.argv[4:7])
    if library == "purejaxql":
        module, config = _purejaxql(task, n_seeds)
    else:
        module, config = _purejaxrl(algorithm, task)
    first = time_program(module, config, n_seeds, warmup)
    second = time_program(module, config, n_seeds, warmup + segment)
    seconds = (second["end"] - second["start"]) - (first["end"] - first["start"])
    record = {
        "steps": n_seeds * segment,
        "seconds": seconds,
        "windows": [first, second],
        "device_kind": jax.devices()[0].device_kind,
    }
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
