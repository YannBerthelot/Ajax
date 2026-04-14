"""
partial_expert_train.py — Train vanilla SAC checkpoints at different training budgets.

Used for "Degradation 3: Value-based sub-optimality".  Each checkpoint is a SAC
actor stopped at a known number of steps; the resulting callable can be passed as
expert_policy in any EDGE experiment to sweep over expert quality.

Usage (called automatically by ablation_study.py):
    from partial_expert_train import setup_partial_expert_checkpoints, load_partial_expert_policy

    setup_partial_expert_checkpoints(env, env_params)   # trains & caches all checkpoints
    policy = load_partial_expert_policy(steps=300_000)  # returns obs → actions callable
"""
import os
from functools import partial
from typing import Optional

import dill as pickle
import jax
import jax.numpy as jnp

from ajax import SAC
from ajax.plane.plane_exps_utils import load_hyperparams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARTIAL_EXPERT_DIR = os.path.abspath("partial_expert_checkpoints")

# Training budgets to checkpoint at.  Chosen to span the SAC learning curve
# from early plateau (~100k) through convergence (~1M).
PARTIAL_EXPERT_STEPS = [100_000, 200_000, 300_000, 500_000, 1_000_000]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(steps: int) -> str:
    label = f"{steps // 1_000}k"
    return os.path.join(PARTIAL_EXPERT_DIR, f"sac_{label}_actor.pkl")


def _partial_sac_action(obs, apply_fn, params):
    """Deterministic (mode) action from a frozen SAC actor.

    Defined at module level so it is picklable when used inside functools.partial.
    """
    return jnp.tanh(apply_fn(params, obs).distribution.mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save_partial_expert(
    steps: int,
    env,
    env_params,
    seed: int = 0,
) -> str:
    """Train a vanilla SAC agent for `steps` timesteps and save the actor to disk.

    Returns the checkpoint path.  If the checkpoint already exists the function
    is a no-op and returns the path immediately.
    """
    path = _checkpoint_path(steps)
    if os.path.exists(path):
        print(f"[partial_expert] Cache hit ({steps:,} steps): {path}")
        return path

    os.makedirs(PARTIAL_EXPERT_DIR, exist_ok=True)
    print(f"[partial_expert] Training SAC for {steps:,} steps (seed={seed}) ...")

    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")

    agent = SAC(
        env_id=env,
        env_params=env_params,
        actor_architecture=architecture,
        critic_architecture=architecture,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        policy_update_start=10_000,
        alpha_update_start=10_000,
        critic_warmup_frac=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        num_critic_updates=1,
        **hp,
    )

    # Train with a single seed to keep things simple; we only need one set of params.
    result = agent.train(
        seed=[seed],
        n_timesteps=steps,
        num_episode_test=25,
        logging_config=None,
    )

    # result is (vmapped_SACState, vmapped_out).  Params have a leading dim=1.
    agent_state = result[0]
    params = jax.tree.map(lambda x: x[0], agent_state.actor_state.params)
    apply_fn = agent_state.actor_state.apply_fn  # pure function, same across seeds

    checkpoint = {"params": params, "apply_fn": apply_fn, "steps": steps}
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"[partial_expert] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_partial_expert_policy(steps: int):
    """Load a saved partial expert and return a JIT-able callable: obs → actions.

    The returned callable produces deterministic (mode) actions.
    """
    path = _checkpoint_path(steps)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Partial expert checkpoint not found: {path}\n"
            "Run setup_partial_expert_checkpoints(env, env_params) first."
        )
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return partial(_partial_sac_action, apply_fn=ckpt["apply_fn"], params=ckpt["params"])


# ---------------------------------------------------------------------------
# One-shot setup
# ---------------------------------------------------------------------------

def setup_partial_expert_checkpoints(env, env_params) -> None:
    """Train and cache all PARTIAL_EXPERT_STEPS checkpoints (no-op if already cached)."""
    print(
        f"\n[partial_expert] ── Checkpoint setup ────────────────────────────────\n"
        f"[partial_expert] Budgets: {[f'{s:,}' for s in PARTIAL_EXPERT_STEPS]}\n"
        f"[partial_expert] Output dir: {PARTIAL_EXPERT_DIR}"
    )
    for steps in PARTIAL_EXPERT_STEPS:
        train_and_save_partial_expert(steps, env, env_params)
    print("[partial_expert] ── All checkpoints ready ──────────────────────────────\n")
