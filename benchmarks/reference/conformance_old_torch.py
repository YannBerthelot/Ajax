"""What the recorders in envs/torch14 share (conformance_tqc.py, conformance_cql.py): a gymnasium task
behind the old gym API their loops use, and a record's library versions. That env has torch 1.4 on
Python 3.8, and no jax, so the tasks are gymnasium's rather than gymnax's (conformance_torch.OldGym).
Runs on Python 3.8. Imports nothing from Ajax.
"""

import importlib.metadata
import json
import pathlib
import platform

import gymnasium
import numpy as np


class OldGym:
    """A gymnasium task with the gym API the originals' loops use: reset() -> obs, step(a) -> (obs,
    reward, done, info), ``terminated`` (a true termination, not a time limit) and the action box."""

    def __init__(self, env_id: str, seed: int):
        self.env = gymnasium.make(env_id)
        space = self.env.action_space
        self.shape = space.shape
        self.low = np.broadcast_to(np.asarray(space.low, np.float32), self.shape)
        self.high = np.broadcast_to(np.asarray(space.high, np.float32), self.shape)
        self.obs_dim = self.env.observation_space.shape[0]
        self.seed = seed

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None  # seeded once, as a gymnasium run is
        return np.asarray(obs, np.float32)

    def step(self, action):
        obs, reward, term, trunc, _ = self.env.step(np.asarray(action, np.float32))
        self.terminated = bool(term)
        return np.asarray(obs, np.float32), float(reward), bool(term or trunc), {}


def write_versions(out: pathlib.Path, **original: str) -> None:
    versions = {
        lib: importlib.metadata.version(lib) for lib in ("torch", "numpy", "gymnasium")
    }
    versions.update(python=platform.python_version(), **original)
    (out / "versions.json").write_text(json.dumps(versions))
