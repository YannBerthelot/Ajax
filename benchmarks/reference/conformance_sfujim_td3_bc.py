"""sfujim/TD3_BC's update, recorded for the conformance check (tests/conformance/test_td3bc_sfujim.py).

usage: python conformance_sfujim_td3_bc.py ENV_ID OUT_DIR   (in envs/torch: CPU-only PyTorch, gymnax, jax)

Imports nothing from Ajax. sfujim's own TD3_BC.py and ReplayBuffer (vendor/sfujim_td3_bc) train offline
by main.py's loop, written out below: ``convert_D4RL`` of a dataset, ``normalize_states``, one
``train()`` per step. There is no D4RL here: the dataset holds DATASET transitions of the gymnax task
under uniform random actions, its terminals true terminations only, as D4RL's ``qlearning_dataset``
gives. Before the updates that bring total_it to SNAPSHOTS + 1 the update is recorded as
conformance_sfujim_td3.py records TD3's (conformance_torch.sfujim_snapshot), with the dataset's mean
and std.
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "sfujim_td3_bc"))
import TD3_BC
import utils
from conformance_torch import OldGym, sfujim_snapshot, write_versions

SNAPSHOTS = (9, 10, 499, 500)
DATASET, BATCH, SEED = 5_000, 256, 0
COMMIT = "8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f"
FIELDS = ("observations", "actions", "next_observations", "rewards", "terminals")


def dataset(env: OldGym) -> dict:
    rows: dict = {k: [] for k in FIELDS}
    obs = env.reset()
    for _ in range(DATASET):
        action = env.sample()
        next_obs, reward, done, _ = env.step(action)
        for k, v in zip(
            FIELDS, (obs, action, next_obs, reward, float(env.terminated)), strict=False
        ):
            rows[k].append(v)
        obs = env.reset() if done else next_obs
    return {k: np.asarray(v, np.float32) for k, v in rows.items()}


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = OldGym(env_id, SEED)
    max_action = float(env.high[0])
    policy = TD3_BC.TD3_BC(
        state_dim=env.obs_dim,
        action_dim=env.shape[0],
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        policy_freq=2,
        alpha=2.5,
    )  # main.py's kwargs
    rb = utils.ReplayBuffer(env.obs_dim, env.shape[0])
    rb.convert_D4RL(dataset(env))
    mean, std = rb.normalize_states()
    out.mkdir(parents=True, exist_ok=True)
    while policy.total_it <= max(SNAPSHOTS):
        if policy.total_it in SNAPSHOTS:
            arrays = sfujim_snapshot(policy, rb, BATCH, (BATCH, env.shape[0]))
            arrays |= {"max_action": np.asarray(max_action), "mean": mean, "std": std}
            np.savez(out / f"{env_id}_{policy.total_it}.npz", **arrays)
        policy.train(rb, BATCH)
    write_versions(out, sfujim_td3_bc=COMMIT)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
