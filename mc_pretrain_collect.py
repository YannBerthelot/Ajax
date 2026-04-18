"""
mc_pretrain_collect.py — standalone MC data collection for the ablation suite.

Collects expert trajectories once (for the maximum episode count needed),
saves to a CSV file, and provides a loader that slices the right number of
rows for each experiment.

Usage (called automatically by ablation_study.py):

    python mc_pretrain_collect.py  # run manually to pre-populate the cache
"""

import os
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.environments.utils import check_env_is_gymnax
from ajax.state import EnvironmentConfig

MC_DATA_PATH = os.path.abspath("mc_pretrain_data.csv")
_META_COMMENT = "# mc_pretrain_data"


# ---------------------------------------------------------------------------
# Collection + save
# ---------------------------------------------------------------------------


def collect_and_save_mc_data(
    expert_policy: Callable,
    env_args: EnvironmentConfig,
    n_mc_steps: int,
    n_mc_episodes: int,
    gamma: float,
    reward_scale: float,
    rng_seed: int = 0,
    path: str = MC_DATA_PATH,
) -> None:
    """Collect expert MC trajectories and save raw (obs, action, mc_return) to CSV.

    The CSV contains raw environment observations (no train_frac or expert-action
    augmentation — those are experiment-specific and applied later inside the JIT).

    The file stores `n_mc_steps * n_mc_episodes` rows regardless of n_envs;
    use `load_mc_data_for_experiment` to slice the right number for each run.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    rng = jax.random.PRNGKey(rng_seed)

    n_total_steps = max(1, (n_mc_steps * n_mc_episodes) // env_args.n_envs)
    print(
        f"[MC collect] {n_mc_episodes} episodes × {n_mc_steps} steps "
        f"= {n_total_steps * env_args.n_envs:,} transitions  "
        f"(n_envs={env_args.n_envs})"
    )

    all_transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_total_steps,
    )

    # Compute MC returns (backward scan)
    rewards = all_transitions.reward * reward_scale  # (T, n_envs, 1)
    dones = jnp.logical_or(
        all_transitions.terminated, all_transitions.truncated
    ).astype(jnp.float32)

    def _mc_scan(carry, x):
        reward, done = x
        mc_return = reward + gamma * carry * (1.0 - done)
        return mc_return, mc_return

    _, mc_returns = jax.lax.scan(
        _mc_scan,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], dones[::-1]),
    )
    mc_returns = mc_returns[::-1]  # (T, n_envs, 1)

    T, n_envs = all_transitions.obs.shape[:2]
    obs_np    = np.array(all_transitions.obs.reshape(T * n_envs, -1))
    action_np = np.array(all_transitions.action.reshape(T * n_envs, -1))
    mc_np     = np.array(mc_returns.reshape(T * n_envs, 1))

    obs_dim    = obs_np.shape[1]
    action_dim = action_np.shape[1]
    n_rows     = obs_np.shape[0]

    obs_cols    = [f"obs_{i}"    for i in range(obs_dim)]
    action_cols = [f"action_{i}" for i in range(action_dim)]
    cols        = obs_cols + action_cols + ["mc_return"]

    df = pd.DataFrame(
        np.concatenate([obs_np, action_np, mc_np], axis=1),
        columns=cols,
    )
    # Store metadata as the file's first comment line so loaders know the schema.
    meta = (
        f"{_META_COMMENT}: "
        f"obs_dim={obs_dim} action_dim={action_dim} n_envs={n_envs} "
        f"n_mc_steps={n_mc_steps} n_mc_episodes={n_mc_episodes} "
        f"gamma={gamma} reward_scale={reward_scale} n_rows={n_rows}"
    )
    with open(path, "w") as f:
        f.write(meta + "\n")
        df.to_csv(f, index=False)

    print(
        f"[MC collect] Saved {n_rows:,} transitions "
        f"(obs_dim={obs_dim}, action_dim={action_dim}) → {path}"
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def read_mc_meta(path: str) -> dict:
    """Parse the metadata comment written by collect_and_save_mc_data."""
    with open(path) as f:
        first = f.readline().strip()
    if not first.startswith(_META_COMMENT):
        raise ValueError(f"MC data file has unexpected format: {path}")
    meta_str = first.split(":", 1)[1].strip()
    meta = {}
    for token in meta_str.split():
        k, v = token.split("=")
        try:
            meta[k] = int(v)
        except ValueError:
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v
    return meta


def load_mc_data_for_experiment(
    n_mc_steps: int,
    n_mc_episodes: int,
    path: str = MC_DATA_PATH,
) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Return (obs, action, mc_return) JAX arrays for a specific episode count.

    Slices the first `n_mc_steps * n_mc_episodes` rows from the cached CSV.
    Returns None if the file does not exist (caller falls back to in-run collection).

    Args:
        n_mc_steps:    steps-per-episode used during collection (must match the
                       value that was used when the CSV was generated).
        n_mc_episodes: number of episodes this experiment needs.
        path:          path to the cached CSV file.
    """
    if not os.path.exists(path):
        return None

    meta = read_mc_meta(path)
    obs_dim    = meta["obs_dim"]
    action_dim = meta["action_dim"]
    n_rows_available = meta["n_rows"]

    n_rows_needed = n_mc_steps * n_mc_episodes
    if n_rows_needed > n_rows_available:
        raise ValueError(
            f"Experiment needs {n_rows_needed:,} MC rows "
            f"({n_mc_episodes} episodes × {n_mc_steps} steps) "
            f"but the cache only has {n_rows_available:,}. "
            f"Re-run mc_pretrain_collect.py with n_mc_episodes≥{n_mc_episodes}."
        )

    df = pd.read_csv(path, comment="#", nrows=n_rows_needed)

    obs_cols    = [f"obs_{i}"    for i in range(obs_dim)]
    action_cols = [f"action_{i}" for i in range(action_dim)]

    obs    = jnp.array(df[obs_cols].values,    dtype=jnp.float32)
    action = jnp.array(df[action_cols].values, dtype=jnp.float32)
    mc     = jnp.array(df[["mc_return"]].values, dtype=jnp.float32)

    return obs, action, mc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import dill as pickle

    import jax
    import jax.numpy as jnp
    from target_gym import Plane, PlaneParams

    from ajax.plane.plane_exps_utils import load_hyperparams
    from ajax.environments.create import prepare_env
    from ajax.state import EnvironmentConfig

    parser = argparse.ArgumentParser(description="Pre-collect MC data for ablation study")
    parser.add_argument("--n-mc-steps",    type=int, default=10_000)
    parser.add_argument("--n-mc-episodes", type=int, default=500,
                        help="Max episodes across all ablation experiments")
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--reward-scale",  type=float, default=1.0)
    parser.add_argument("--rng-seed",      type=int,   default=0)
    parser.add_argument("--n-envs",        type=int,   default=64,
                        help="Parallel environments for faster collection")
    parser.add_argument("--output",        type=str,   default=MC_DATA_PATH)
    args = parser.parse_args()

    env_raw = Plane()
    env_params = PlaneParams(
        target_altitude_range=(3_000, 8_000),
        initial_altitude_range=(3_000, 8_000),
        max_steps_in_episode=10_000,
    )

    if "expert_policy.pkl" in os.listdir():
        with open("expert_policy.pkl", "rb") as f:
            expert_policy = pickle.load(f)
    else:
        expert_policy = env_raw.expert_policy
        with open("expert_policy.pkl", "wb") as f:
            pickle.dump(expert_policy, f)

    env, env_params_prepared, _, continuous = prepare_env(env_raw, env_params=env_params, n_envs=args.n_envs)
    env_args = EnvironmentConfig(env=env, env_params=env_params_prepared, n_envs=args.n_envs, continuous=continuous)

    collect_and_save_mc_data(
        expert_policy=expert_policy,
        env_args=env_args,
        n_mc_steps=args.n_mc_steps,
        n_mc_episodes=args.n_mc_episodes,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        rng_seed=args.rng_seed,
        path=args.output,
    )
