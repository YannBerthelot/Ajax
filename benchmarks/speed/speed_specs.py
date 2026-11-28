"""What the speed comparison runs: tasks, libraries and settings.

Pure data, no library import: each library's runner imports it from that
library's own environment (envs/<library>).

Each algorithm has one set of settings per task, in neutral names; every
library takes them under its own names, and EXCEPTIONS lists what a library
cannot take and keeps of its own. The settings are CleanRL's defaults where
its scripts and PureJaxRL's agree (PPO on CartPole, DQN), the defaults SB3 and
CleanRL share for SAC, TD3 and PPO on MuJoCo, and the configuration of
purejaxql, the PQN authors' code, for PQN on CartPole. They were fixed before
any timing. ``learning_starts`` only moves when the timed segment can begin:
the segment starts once updates run.
"""

#: The figure's panels.
FAMILIES = {
    "SAC and TD3": ("SAC", "TD3"),
    "PPO": ("PPO",),
    "DQN and PQN": ("DQN", "PQN"),
}

#: Algorithm -> its tasks.
TASKS = {
    "SAC": ("Pendulum-v1", "HalfCheetah-v5"),
    "TD3": ("Pendulum-v1", "HalfCheetah-v5"),
    "PPO": ("CartPole-v1", "HalfCheetah-v5"),
    "DQN": ("CartPole-v1",),
    "PQN": ("CartPole-v1",),
}

#: Algorithm -> the libraries that have it. Brax's PPO has no discrete
#: actions, so it skips CartPole (UNSUPPORTED).
LIBRARIES = {
    "SAC": ("Ajax", "SB3", "SBX", "CleanRL", "Brax"),
    "TD3": ("Ajax", "SB3", "SBX", "CleanRL"),
    "PPO": ("Ajax", "SB3", "SBX", "CleanRL", "PureJaxRL", "Brax"),
    "DQN": ("Ajax", "SB3", "SBX", "CleanRL", "PureJaxRL"),
    "PQN": ("Ajax", "CleanRL", "purejaxql"),
}
UNSUPPORTED = {("PPO", "CartPole-v1", "Brax")}

#: Libraries that train every seed in one compiled program; the others run
#: one seed per process.
ONE_PROGRAM = ("Ajax", "PureJaxRL", "purejaxql")
#: Libraries timed on the GPU too: Ajax and the JAX libraries.
ON_GPU = ("Ajax", "Brax", "PureJaxRL", "purejaxql")

#: CleanRL's script for each algorithm: its JAX version where its docs say
#: that one is faster (TD3, DQN), else its only one.
CLEANRL_SCRIPTS = {
    "SAC": "sac_continuous_action.py",
    "TD3": "td3_continuous_action_jax.py",
    "PPO-CartPole-v1": "ppo.py",
    "PPO-HalfCheetah-v5": "ppo_continuous_action.py",
    "DQN": "dqn_jax.py",
    "PQN": "pqn.py",
}

_OFF_POLICY = {
    "n_envs": 1,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "updates_per_step": 1,
    "hidden": (256, 256),
    "activation": "relu",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "learning_starts": 1_000,
}
_PPO = {
    "hidden": (64, 64),
    "activation": "tanh",
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

#: (algorithm, task) -> settings; task None holds for every task.
SETTINGS = {
    ("SAC", None): {**_OFF_POLICY, "entropy": "learned"},
    ("TD3", None): {
        **_OFF_POLICY,
        "policy_delay": 2,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
        "exploration_noise": 0.1,
    },
    ("PPO", "CartPole-v1"): {
        **_PPO,
        "n_envs": 4,
        "n_steps": 128,
        "n_epochs": 4,
        "n_minibatches": 4,
        "learning_rate": 2.5e-4,
        "ent_coef": 0.01,
        "normalize": False,
    },
    ("PPO", "HalfCheetah-v5"): {
        **_PPO,
        "n_envs": 1,
        "n_steps": 2048,
        "n_epochs": 10,
        "n_minibatches": 32,
        "learning_rate": 3e-4,
        "ent_coef": 0.0,
        "normalize": True,
    },
    ("DQN", "CartPole-v1"): {
        "n_envs": 1,
        "buffer_size": 10_000,
        "batch_size": 128,
        "train_frequency": 10,
        "target_update": 500,
        "hidden": (120, 84),
        "activation": "relu",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "epsilon": (1.0, 0.05, 0.5),
        "learning_starts": 1_000,
    },
    ("PQN", "CartPole-v1"): {
        "n_envs": 32,
        "n_steps": 64,
        "n_epochs": 4,
        "n_minibatches": 16,
        "hidden": (256, 256),
        "activation": "relu",
        "layer_norm": True,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "q_lambda": 0.95,
        "max_grad_norm": 10.0,
        "epsilon": (1.0, 0.2, 0.2),
    },
}

#: (algorithm, library) -> what the library keeps of its own.
EXCEPTIONS = {
    ("PQN", "CleanRL"): "120-84 layers with layer norm (fixed in pqn.py)",
    ("PPO", "Brax"): "minibatches of whole 64-step unrolls, GAE within each,"
    " observations normalised but not rewards (Brax has no reward normaliser)",
}

#: Environment ids. The Gymnasium libraries step C MuJoCo; CleanRL's scripts
#: need Gymnasium 0.29's vector API, which has HalfCheetah-v4, the same model.
#: Ajax, Brax and PureJaxRL step gymnasium-mjx's HalfCheetah-v5 on MJX.
GYMNASIUM_IDS = {"CleanRL": {"HalfCheetah-v5": "HalfCheetah-v4"}}
AJAX_IDS = {"HalfCheetah-v5": "mujoco:HalfCheetah-v5"}


def settings(algorithm: str, task: str) -> dict:
    return dict(SETTINGS.get((algorithm, task)) or SETTINGS[(algorithm, None)])


def cells() -> list[tuple[str, str, str]]:
    """Every (algorithm, task, library) the comparison times."""
    return [
        (algorithm, task, library)
        for algorithm, tasks in TASKS.items()
        for task in tasks
        for library in LIBRARIES[algorithm]
        if (algorithm, task, library) not in UNSUPPORTED
    ]


def physics(task: str, library: str) -> str:
    if not task.startswith("HalfCheetah"):
        return ""
    return "MJX" if library in ("Ajax", "Brax", "PureJaxRL") else "C MuJoCo"


def gymnasium_id(task: str, library: str) -> str:
    return GYMNASIUM_IDS.get(library, {}).get(task, task)


def cleanrl_script(algorithm: str, task: str) -> str:
    return CLEANRL_SCRIPTS.get(f"{algorithm}-{task}") or CLEANRL_SCRIPTS[algorithm]


#: Libraries that step their environments in Python, one process per seed.
PYTHON_LOOP = ("SB3", "SBX", "CleanRL")
#: Steps per seed before the timed stretch: past learning_starts and the
#: first updates (and their compilation), or two to four rollouts.
WARMUP = {
    "PPO-CartPole-v1": 2_048,
    "PPO-HalfCheetah-v5": 4_096,
    "PQN-CartPole-v1": 4_096,
}
#: (algorithm, task, "python" | "compiled", seeds, device) -> the timed
#: stretch, in steps per seed: about a minute for the slowest library of its
#: kind on four of bob's cores, half a minute on its L40S, from a
#: calibration pass of shorter stretches.
SEGMENTS = {
    ("SAC", "Pendulum-v1", "python", 1, "cpu"): 8_000,
    ("SAC", "Pendulum-v1", "python", 10, "cpu"): 2_000,
    ("SAC", "Pendulum-v1", "compiled", 1, "cpu"): 18_000,
    ("SAC", "Pendulum-v1", "compiled", 10, "cpu"): 3_000,
    ("SAC", "HalfCheetah-v5", "python", 1, "cpu"): 5_000,
    ("SAC", "HalfCheetah-v5", "python", 10, "cpu"): 2_000,
    ("SAC", "HalfCheetah-v5", "compiled", 1, "cpu"): 11_000,
    ("SAC", "HalfCheetah-v5", "compiled", 10, "cpu"): 2_000,
    ("TD3", "Pendulum-v1", "python", 1, "cpu"): 14_000,
    ("TD3", "Pendulum-v1", "python", 10, "cpu"): 3_000,
    ("TD3", "Pendulum-v1", "compiled", 1, "cpu"): 25_000,
    ("TD3", "Pendulum-v1", "compiled", 10, "cpu"): 4_000,
    ("TD3", "HalfCheetah-v5", "python", 1, "cpu"): 10_000,
    ("TD3", "HalfCheetah-v5", "python", 10, "cpu"): 3_000,
    ("TD3", "HalfCheetah-v5", "compiled", 1, "cpu"): 14_000,
    ("TD3", "HalfCheetah-v5", "compiled", 10, "cpu"): 3_000,
    ("PPO", "CartPole-v1", "python", 1, "cpu"): 311_296,
    ("PPO", "CartPole-v1", "python", 10, "cpu"): 135_168,
    ("PPO", "CartPole-v1", "compiled", 1, "cpu"): 7_917_568,
    ("PPO", "CartPole-v1", "compiled", 10, "cpu"): 1_777_152,
    ("PPO", "HalfCheetah-v5", "python", 1, "cpu"): 86_016,
    ("PPO", "HalfCheetah-v5", "python", 10, "cpu"): 28_672,
    ("PPO", "HalfCheetah-v5", "compiled", 1, "cpu"): 38_912,
    ("PPO", "HalfCheetah-v5", "compiled", 10, "cpu"): 6_144,
    ("DQN", "CartPole-v1", "python", 1, "cpu"): 194_000,
    ("DQN", "CartPole-v1", "python", 10, "cpu"): 55_000,
    ("DQN", "CartPole-v1", "compiled", 1, "cpu"): 1_562_000,
    ("DQN", "CartPole-v1", "compiled", 10, "cpu"): 495_000,
    ("PQN", "CartPole-v1", "python", 1, "cpu"): 841_728,
    ("PQN", "CartPole-v1", "python", 10, "cpu"): 311_296,
    ("PQN", "CartPole-v1", "compiled", 1, "cpu"): 1_249_280,
    ("PQN", "CartPole-v1", "compiled", 10, "cpu"): 428_032,
}


def segment(
    algorithm: str, task: str, library: str, seeds: int, device: str
) -> tuple[int, int]:
    """The warm-up and the timed stretch, in steps per seed."""
    kind = "python" if library in PYTHON_LOOP else "compiled"
    warmup = WARMUP.get(f"{algorithm}-{task}", 2_000)
    return warmup, SEGMENTS[(algorithm, task, kind, seeds, device)]
