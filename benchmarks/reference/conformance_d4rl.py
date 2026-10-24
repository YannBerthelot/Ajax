"""D4RL-format datasets for the offline originals' recorders (conformance_rebrac.py, conformance_calql.py,
conformance_cql.py).

Uniform random actions in [-1, 1]^A on a gymnasium task, stored as D4RL stores a dataset
(observations, actions, rewards, terminals, timeouts), for the originals' own loaders to turn into
rows. MountainCarEnds is MountainCarContinuous terminating every ENDS_EVERY-th step, so that rows end
episodes. The tasks' action boxes are [-1, 1], as D4RL's are: the originals compare their tanh actions
with the dataset's directly. Runs on Python 3.8 (envs/torch14). Imports nothing from Ajax.
"""

import gymnasium
import numpy as np

ENDS_EVERY = 20
BASE = {"MountainCarEnds": "MountainCarContinuous-v0"}


def dataset(env_id: str, steps: int = 5_000, seed: int = 0) -> dict:
    """``steps`` transitions of uniform random actions on ``env_id``, in D4RL's format."""
    env = gymnasium.make(BASE.get(env_id, env_id))
    rng = np.random.default_rng(seed)
    keys = ("observations", "actions", "rewards", "terminals", "timeouts")
    data: dict = {k: [] for k in keys}
    (obs, _), t = env.reset(seed=seed), 0
    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
        nxt, reward, terminated, truncated, _ = env.step(action * env.action_space.high)
        t += 1
        terminal = terminated or (env_id in BASE and t % ENDS_EVERY == 0)
        timeout = truncated and not terminal
        for k, v in zip(keys, (obs, action, reward, terminal, timeout), strict=False):
            data[k].append(v)
        obs = nxt
        if terminal or timeout:
            (obs, _), t = env.reset(), 0
    return {k: np.asarray(v) for k, v in data.items()}


def qlearning_dataset(data: dict) -> dict:
    """D4RL's qlearning_dataset on ``data`` (terminate_on_end False), as ReBRAC copies it from d4rl
    d842aa1 (vendor/tinkoff_rebrac/src/utils/buffer.py:16-91) without the next actions ReBRAC adds:
    each time-limit row and the last row are dropped, and next_observations[i] is observations[i + 1]."""
    keys = ("observations", "actions", "next_observations", "rewards", "terminals")
    out: dict = {k: [] for k in keys}
    for i in range(data["rewards"].shape[0] - 1):
        if data["timeouts"][i]:
            continue
        values = (
            data["observations"][i].astype(np.float32),
            data["actions"][i].astype(np.float32),
            data["observations"][i + 1].astype(np.float32),
            data["rewards"][i].astype(np.float32),
            bool(data["terminals"][i]),
        )
        for k, v in zip(keys, values, strict=False):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}
