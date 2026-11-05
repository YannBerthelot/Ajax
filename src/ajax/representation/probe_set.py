"""A fixed set of states to read a network's activations on.

The same states at every checkpoint of every seed make activations
comparable: what changes between two probes is the network, not its input.
They come from a random policy, so that no agent's own training shapes them,
and with each one what the env knows and an agent would not be told, to
colour a manifold with.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax.environments.interaction import reset, step
from ajax.environments.utils import (
    agent_action_to_env,
    check_if_environment_has_continuous_actions,
    env_keys,
    get_action_dim,
)
from ajax.random_keys import derive_key, split_by_index
from ajax.state import EnvironmentConfig

#: The file of a probe set in a run directory.
PROBE_SET_FILE = "probe_set.npz"
#: The steps per env a rollout runs at least: enough to leave the episodes'
#: first states whatever the number of envs.
MIN_ROLLOUT_STEPS = 1024
#: A state's env fields wider than this stay out of its ground truth.
MAX_TRUTH_COLUMNS = 32
TRUTH_PREFIX = "truth/"


@dataclasses.dataclass(frozen=True, eq=False)
class ProbeSet:
    """States, with the seed that generated them.

    ``observations`` are as the env emits them; ``actions`` (in the agent's
    units, ``[-1, 1]`` for continuous ones) are the random policy's there,
    what a Q-critic reads them with, so that its input too stays fixed.
    ``ground_truth`` holds the env's own fields at each state.
    """

    observations: np.ndarray
    actions: np.ndarray
    ground_truth: dict[str, np.ndarray]
    seed: int

    def save(self, path: str | os.PathLike) -> None:
        """Writes the file whole or not at all: another run may read it."""
        arrays: dict[str, Any] = {
            "observations": self.observations,
            "actions": self.actions,
            "seed": self.seed,
        }
        arrays.update({f"{TRUTH_PREFIX}{k}": v for k, v in self.ground_truth.items()})
        temporary = Path(f"{path}.{os.getpid()}.tmp")
        with open(temporary, "wb") as file:
            np.savez(file, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ProbeSet":
        with np.load(path) as saved:
            truth = {
                name.removeprefix(TRUTH_PREFIX): saved[name]
                for name in saved.files
                if name.startswith(TRUTH_PREFIX)
            }
            return cls(
                saved["observations"], saved["actions"], truth, int(saved["seed"])
            )


def _ground_truth_leaves(env_state: Any, reward: jax.Array) -> dict[str, jax.Array]:
    """``reward`` and the env state's numeric fields, per env, narrow enough
    to keep (an env's random keys are unsigned, and stay out)."""
    n_envs = reward.shape[0]
    truth = {"reward": reward}
    for path, leaf in jax.tree_util.tree_leaves_with_path(env_state):
        numeric = jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(
            leaf.dtype, jnp.signedinteger
        )
        per_env = leaf.ndim > 0 and leaf.shape[0] == n_envs
        if numeric and per_env and leaf.size / n_envs <= MAX_TRUTH_COLUMNS:
            truth[jax.tree_util.keystr(path, simple=True, separator="/")] = leaf
    return truth


def random_rollouts(env_args: EnvironmentConfig, n_steps: int, key: jax.Array) -> dict:
    """``n_steps`` of uniform random actions in every env: the observations,
    the actions and the ground truth, leaves ``(n_steps, n_envs, ...)``."""
    env, env_params, mode = env_args.env, env_args.env_params, env_args.mode
    n_envs = env_args.n_envs
    continuous = check_if_environment_has_continuous_actions(env, env_params)
    action_dim = get_action_dim(env, env_params)

    def random_action(action_key: jax.Array) -> jax.Array:
        if continuous:
            return jax.random.uniform(
                action_key, (n_envs, action_dim), minval=-1.0, maxval=1.0
            )
        drawn = jax.random.randint(action_key, (n_envs, 1), 0, action_dim)
        return drawn.astype(jnp.float32)

    def one_step(carry: tuple, step_key: jax.Array) -> tuple[tuple, dict]:
        observations, env_state = carry
        actions = random_action(derive_key(step_key, "rollout_action"))
        played = agent_action_to_env(actions, env, env_params)
        if not continuous:
            played = played.squeeze(-1)
        env_key = derive_key(step_key, "rollout_env_step")
        next_observations, next_state, reward, *_ = step(
            env_keys(env_key, mode, n_envs), env_state, played, env, mode, env_params
        )
        row = {
            "observations": observations,
            "actions": actions,
            "truth": _ground_truth_leaves(env_state, reward),
        }
        return (next_observations, next_state), row

    def rollouts(key: jax.Array) -> dict:
        reset_key = derive_key(key, "rollout_reset")
        start = reset(env_keys(reset_key, mode, n_envs), env, mode, env_params)
        return jax.lax.scan(one_step, start, split_by_index(key, n_steps))[1]

    return jax.jit(rollouts)(key)


def make_probe_set(
    env_args: EnvironmentConfig, n_states: int = 2048, seed: int = 0
) -> ProbeSet:
    """``n_states`` drawn without replacement from random rollouts holding
    at least twice as many, on ``seed`` alone: the same call gives the same
    states."""
    n_envs = env_args.n_envs
    n_steps = max(2 * n_states // n_envs, MIN_ROLLOUT_STEPS)
    rows = random_rollouts(env_args, n_steps, jax.random.PRNGKey(seed))
    rows = jax.device_get(rows)
    chosen = np.random.default_rng(seed).choice(
        n_steps * n_envs, n_states, replace=False
    )
    chosen.sort()

    def kept(leaf: np.ndarray) -> np.ndarray:
        return leaf.reshape(n_steps * n_envs, *leaf.shape[2:])[chosen]

    return ProbeSet(
        kept(rows["observations"]),
        kept(rows["actions"]),
        {name: kept(leaf) for name, leaf in rows["truth"].items()},
        seed,
    )


def probe_set_for(
    directory: str | os.PathLike,
    env_args: EnvironmentConfig,
    n_states: int = 2048,
    seed: int = 0,
) -> ProbeSet:
    """The probe set saved in ``directory``, or a new one saved there, so
    that every run pointed at that directory reads the same states.

    Raises ``ValueError`` when the one saved there has another seed or size.
    """
    path = Path(directory) / PROBE_SET_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        made = make_probe_set(env_args, n_states, seed)
        made.save(path)
        return made
    saved = ProbeSet.load(path)
    if saved.seed != seed or len(saved.observations) != n_states:
        raise ValueError(
            f"{path} holds {len(saved.observations)} states of seed {saved.seed},"
            f" not {n_states} of seed {seed}: a run compares its probes on one"
            " set. Give it another directory, or the same settings."
        )
    return saved
