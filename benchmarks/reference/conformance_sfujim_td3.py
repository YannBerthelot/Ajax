"""sfujim/TD3's update, recorded for the conformance check (tests/conformance/test_td3_sfujim.py).

usage: python conformance_sfujim_td3.py ENV_ID OUT_DIR   (in envs/torch: CPU-only PyTorch, gymnax, jax)

Imports nothing from Ajax. sfujim's own TD3.py and ReplayBuffer (vendor/sfujim_td3) train on a gymnax
task by main.py's loop, written out below: random actions for START steps, then the actor's action plus
N(0, 0.1 max_action), done_bool 0 at a time limit, one train() per step. Before the updates that bring
total_it to SNAPSHOTS + 1, the state is recorded, with the update's own draws (its batch indices, then
its target noise), and the update is run from it twice: as is, and in float64, whose difference is the
float32 rounding the check tolerates (conformance_torch.sfujim_snapshot). One .npz per snapshot:
``in.*`` (the state), ``batch.*``, ``noise``, ``max_action``, ``f32.*`` and ``f64.*`` (the states after
the update), and ``versions.json``.
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "sfujim_td3"))
import TD3
import utils
from conformance_torch import OldGym, sfujim_snapshot, write_versions

SNAPSHOTS = (9, 10, 499, 500)
START, BATCH, SEED = 1_000, 256, 0
COMMIT = "34770ccdcb51df6f6c7d85c1ba7e40d71b940d16"


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = OldGym(env_id, SEED)
    max_action = float(env.high[0])
    policy = TD3.TD3(
        state_dim=env.obs_dim,
        action_dim=env.shape[0],
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        policy_freq=2,
    )  # main.py's kwargs
    rb = utils.ReplayBuffer(env.obs_dim, env.shape[0])
    out.mkdir(parents=True, exist_ok=True)
    state, done, episode_timesteps, t = env.reset(), False, 0, 0
    while policy.total_it <= max(SNAPSHOTS):
        episode_timesteps += 1
        if t < START:
            action = env.sample()
        else:
            noise = np.random.normal(0, max_action * 0.1, size=env.shape[0])
            action = (policy.select_action(np.array(state)) + noise).clip(
                -max_action, max_action
            )
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        rb.add(state, action, next_state, reward, done_bool)
        state = next_state
        if done:
            state, done, episode_timesteps = env.reset(), False, 0
        if t >= START:
            if policy.total_it in SNAPSHOTS:
                arrays = sfujim_snapshot(policy, rb, BATCH, (BATCH, env.shape[0]))
                arrays["max_action"] = np.asarray(max_action)
                np.savez(out / f"{env_id}_{policy.total_it}.npz", **arrays)
            policy.train(rb, BATCH)
        t += 1
    write_versions(out, sfujim_td3=COMMIT)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
