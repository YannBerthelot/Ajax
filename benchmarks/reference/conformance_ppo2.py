"""OpenAI Baselines' ppo2 (vendored at ea25b9e), Ajax PPO's reference, recorded for the conformance
check (tests/conformance/test_ppo2_original.py).

usage: python conformance_ppo2.py CASE OUT_DIR   (in envs/tf115: TensorFlow 1.15.5 on Python 3.6)

Imports nothing from Ajax. ppo2.learn trains as baselines' run.py starts it, with the settings of
ppo2/defaults.py for the case:

* ``mlp``: Pendulum-v0 under the mujoco() defaults (a classic-control task, which needs no MuJoCo),
  whose policy and value MLPs are separate (value_network='copy'): one env, 2048 steps, 32
  minibatches, 10 epochs, the learning rate annealed from 3e-4.
* ``cnn``: BreakoutNoFrameskip-v4 behind the Atari wrappers run.py puts on it (make_atari,
  wrap_deepmind, a 4-frame VecFrameStack) under the atari() defaults, whose nature_cnn both heads
  read: 2 envs of 64 steps instead of run.py's one env per CPU of 128, which would make records of
  tens of megabytes; 4 minibatches, 4 epochs.

Before the first rollout every variable takes its conformance_start.start_value, which the test
recomputes, so that no record holds the start of a network of millions of parameters. At each
update in the case's calls, ``CASE_CALL.npz`` keeps the state before the update (``in.*``, but for
leaves larger than conformance_start.SAMPLED, the start, which the test recomputes), the
batch model.train reads (``batch.*``, env-major rows b T + t), the rewards and episode ends the
envs returned and the observations after the last step (``env.*``), each epoch's shuffle
(``perm``), the learning rate and clip range, and the state after the update in float32, the run's
own (``f32.*``), and in float64 (``f64.*``): the same update rebuilt in a float64 graph from the
same state, batch and shuffles. A leaf larger than SAMPLED is kept at its
conformance_start.sampled_positions (``at.*``), with its shape (``shape.*``); a case holding one
records its first update only. Variables keep TensorFlow's names.

Runs on Python 3.6. TensorFlow 1.15 has no wheel for Python 3.8 or later, so CI cannot run this;
its records are committed (tests/conformance/records/openai_baselines).
"""

import contextlib
import json
import pathlib
import platform
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE / "vendor" / "openai_baselines")]
import gym  # noqa: E402
import tensorflow as tf  # noqa: E402
from baselines.common.atari_wrappers import make_atari, wrap_deepmind  # noqa: E402
from baselines.common.policies import build_policy  # noqa: E402
from baselines.common.vec_env import VecEnvWrapper  # noqa: E402
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv  # noqa: E402
from baselines.common.vec_env.vec_frame_stack import VecFrameStack  # noqa: E402
from baselines.ppo2 import defaults, ppo2  # noqa: E402
from baselines.ppo2.model import Model  # noqa: E402
from baselines.ppo2.runner import Runner  # noqa: E402
from conformance_start import SAMPLED, sampled_positions, start_value  # noqa: E402

SEED = 0
#: run.py's default num_timesteps, which sets how far the learning rate has annealed.
TOTAL_TIMESTEPS = int(1e6)
CASES = {
    "mlp": {
        "env_id": "Pendulum-v0",
        "network": "mlp",
        "nenvs": 1,
        "calls": (1, 3),
        "settings": defaults.mujoco(),
    },
    "cnn": {
        "env_id": "BreakoutNoFrameskip-v4",
        "network": "cnn",
        "nenvs": 2,
        "calls": (1,),
        "settings": dict(defaults.atari(), nsteps=64),
    },
}
BATCH = ("obs", "returns", "masks", "actions", "values", "neglogpacs")
#: One thread: the recorded update's sums in a fixed order, and bob's CPUs left to others.
ONE_THREAD = tf.ConfigProto(
    inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
)


class Logged(VecEnvWrapper):
    """The envs, their rewards and episode ends logged step by step."""

    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.rewards, self.dones = [], []

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.rewards.append(np.asarray(rewards, np.float32))
        self.dones.append(np.asarray(dones, bool))
        return obs, rewards, dones, infos


def make_envs(case):
    """The case's envs as run.py builds them (cmd_util.make_env, seeded SEED + rank)."""
    env_id, nenvs = case["env_id"], case["nenvs"]

    def make(rank):
        if case["network"] == "cnn":
            env = make_atari(env_id)
            env.seed(SEED + rank)
            return wrap_deepmind(env)
        env = gym.make(env_id)
        env.seed(SEED + rank)
        return env

    venv = DummyVecEnv([lambda rank=rank: make(rank) for rank in range(nenvs)])
    if case["network"] == "cnn":
        venv = VecFrameStack(venv, 4)
    return Logged(venv)


def variables_state(sess):
    return {v.name[: -len(":0")]: sess.run(v) for v in tf.global_variables()}


def load(sess, state):
    for v in tf.global_variables():
        v.load(state[v.name[: -len(":0")]].astype(v.dtype.as_numpy_dtype), sess)


def kept(prefix, state, arrays):
    """``state`` into ``arrays`` under ``prefix``: a leaf larger than SAMPLED with its shape
    (``shape.*``), at its sampled positions (``at.*``), and left out of the state before the
    update, its start."""
    for name, value in state.items():
        value = np.asarray(value)
        if value.size <= SAMPLED:
            arrays[prefix + name] = value
            continue
        arrays["shape." + name] = np.asarray(value.shape)
        if prefix != "in.":
            at = sampled_positions(name, value.size)
            arrays["at." + name] = at
            arrays[prefix + name] = value.reshape(-1)[at]


@contextlib.contextmanager
def float64_graph():
    """A new graph whose building reads float64 where ppo2 names float32: its placeholders and casts
    name tf.float32 and tf.to_float, its Gaussian samples tf.random_normal's default dtype, and its
    variables take their scope's dtype. Adam's constants are float64 too: tf.train.AdamOptimizer
    makes each a float32 tensor that it casts to its variables' dtype, which in float64 would keep
    float32's 1 - beta2 (1.3e-5 off), a rounding of the float32 run the null must measure."""
    saved = tf.float32, tf.to_float, tf.random_normal, tf.train.AdamOptimizer
    random_normal, adam = tf.random_normal, tf.train.AdamOptimizer

    def normal(shape, mean=0.0, stddev=1.0, dtype=tf.float64, seed=None, name=None):
        return random_normal(shape, mean, stddev, dtype, seed, name)

    def exact_adam(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        constants = (tf.constant(x, tf.float64) for x in (beta1, beta2, epsilon))
        return adam(learning_rate, *constants, **kwargs)

    tf.float32 = tf.float64
    tf.to_float = lambda x, name="ToFloat": tf.cast(x, tf.float64, name=name)
    tf.random_normal = normal
    tf.train.AdamOptimizer = exact_adam
    graph = tf.Graph()
    # One context at a time: Python 3.6 has no parenthesised with-items.
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(graph.as_default())
            sess = stack.enter_context(tf.Session(graph=graph, config=ONE_THREAD))
            stack.enter_context(sess.as_default())
            scope = tf.get_variable_scope()
            stack.enter_context(tf.variable_scope(scope, dtype=tf.float64))
            yield sess
    finally:
        tf.float32, tf.to_float, tf.random_normal, tf.train.AdamOptimizer = saved


def float64_update(case, venv, state, batch, perms, lr, cliprange):
    """The update ppo2.learn made, from ``state`` on ``batch`` with the shuffles ``perms``, in a
    float64 graph: its Model, and its epochs as learn runs them (ppo2.py:157-166)."""
    settings = dict(case["settings"])
    nsteps, nminibatches = settings["nsteps"], settings["nminibatches"]
    network_kwargs = (
        {"value_network": settings["value_network"]}
        if "value_network" in settings
        else {}
    )
    space = venv.observation_space
    if space.dtype == np.float32:
        space = gym.spaces.Box(space.low, space.high, dtype=np.float64)
    env = types.SimpleNamespace(observation_space=space, action_space=venv.action_space)
    nbatch = case["nenvs"] * nsteps
    nbatch_train = nbatch // nminibatches
    batch = [x.astype(np.float64) if x.dtype == np.float32 else x for x in batch]
    with float64_graph() as sess:
        policy = build_policy(env, case["network"], **network_kwargs)
        model = Model(
            policy=policy,
            ob_space=space,
            ac_space=venv.action_space,
            nbatch_act=case["nenvs"],
            nbatch_train=nbatch_train,
            nsteps=nsteps,
            ent_coef=settings["ent_coef"],
            vf_coef=0.5,  # learn's defaults
            max_grad_norm=0.5,
        )
        # Adam's beta powers exactly, from its step: the float32 run's are rounded products.
        steps = round(np.log(float(state["beta2_power"])) / np.log(0.999)) - 1
        powers = {
            "beta1_power": 0.9 ** (steps + 1),
            "beta2_power": 0.999 ** (steps + 1),
        }
        load(sess, dict(state, **{k: np.float64(v) for k, v in powers.items()}))
        for inds in perms:
            for start in range(0, nbatch, nbatch_train):
                mbinds = inds[start : start + nbatch_train]
                model.train(lr, cliprange, *(arr[mbinds] for arr in batch))
        return variables_state(sess)


class Finished(Exception):
    pass


def record(name, out):
    case = CASES[name]
    out.mkdir(parents=True, exist_ok=True)
    venv = make_envs(case)
    settings = dict(case["settings"])
    nsteps = settings["nsteps"]
    seen = {"update": 0, "perms": []}
    run, shuffle = Runner.run, np.random.shuffle

    def logged_shuffle(x):
        shuffle(x)
        seen["perms"].append(np.array(x))

    def recorded_run(runner):
        sess = tf.get_default_session()
        if seen["update"] == 0:
            large = [k for k, v in variables_state(sess).items() if v.size > SAMPLED]
            # The records leave out the large leaves before the update: they must be the start.
            assert case["calls"] == (1,) or not large, large
            load(
                sess,
                {
                    k: start_value(k, v.shape)
                    if k.endswith(("/w", "/b", "logstd"))
                    else v
                    for k, v in variables_state(sess).items()
                },
            )
        first = len(venv.rewards)
        batch = run(runner)
        seen["update"] += 1
        seen.update(
            state=variables_state(sess),
            batch=batch[: len(BATCH)],
            rewards=np.stack(venv.rewards[first:]),
            dones=np.stack(venv.dones[first:]),
            last_obs=np.array(runner.obs),
            perms=[],
        )
        return batch

    def after_update(update):
        if update not in case["calls"]:
            return
        lr = settings["lr"](
            1.0 - (update - 1.0) / (TOTAL_TIMESTEPS // (case["nenvs"] * nsteps))
        )
        cliprange = settings["cliprange"]
        f32 = variables_state(tf.get_default_session())
        f64 = float64_update(
            case, venv, seen["state"], seen["batch"], seen["perms"], lr, cliprange
        )
        arrays = {
            "perm": np.stack(seen["perms"]).astype(np.int32),
            "lr": np.float64(lr),
            "cliprange": np.float64(cliprange),
        }
        kept("in.", seen["state"], arrays)
        # Indexed, not zipped: ruff asks for zip's strict=, which Python 3.6 lacks.
        for i, key in enumerate(BATCH):
            arrays["batch." + key] = seen["batch"][i]
        arrays.update(
            {
                "env.rewards": seen["rewards"],
                "env.dones": seen["dones"],
                "env.last_obs": seen["last_obs"],
            }
        )
        kept("f32.", f32, arrays)
        kept("f64.", f64, arrays)
        np.savez_compressed(str(out / "{}_{}.npz".format(name, update)), **arrays)
        if update == max(case["calls"]):
            raise Finished

    Runner.run, np.random.shuffle = recorded_run, logged_shuffle
    try:
        with tf.Session(config=ONE_THREAD).as_default():
            ppo2.learn(
                network=case["network"],
                env=venv,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
                update_fn=after_update,
                **settings,
            )
    except Finished:
        pass
    finally:
        Runner.run, np.random.shuffle = run, shuffle
    versions = {
        "openai_baselines": "ea25b9e",
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "gym": gym.__version__,
    }
    (out / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
