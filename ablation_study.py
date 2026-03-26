"""
ablation_study.py

5-tier ablation suite for SAC + AWBC on the Plane environment.

Run via gpu_launcher (recommended — waits for free GPUs system-wide):
    python gpu_launcher.py --script ablation_study

Run a single experiment by index (used internally by the launcher):
    python ablation_study.py --exp-index 5

List all experiments:
    python ablation_study.py --list
"""
import os
import sys
from pathlib import Path

# Detect if we are a child process immediately
if "spawn" in sys.argv or (hasattr(sys, "frozen") is False and "parent_pid" in str(sys.argv)):
    # This is a worker. Force JAX to CPU before it even thinks about CUDA.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# NOW you can import multiprocessing and your libraries
import multiprocessing as mp

import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections.abc import Callable

import dill as pickle
import jax
import jax.numpy as jnp
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState
from tqdm import tqdm

from ajax import SAC
from ajax.early_termination_wrapper import EarlyTerminationWrapper
from ajax.logging.wandb_logging import upload_tensorboard_to_wandb
from ajax.plane.plane_exps_utils import (
    get_log_config,
    get_mode,
    get_policy_score,
)
from ajax.stable_utils import get_expert_policy

# ---------------------------------------------------------------------------
# YAML hyperparameter defaults
# ---------------------------------------------------------------------------

_HP_YAML = Path("hyperparams/ajax_sac.yml")

# Hard-coded fallbacks used only when the YAML file doesn't exist yet
# (i.e. before the hyperparameter search has run).
_HP_DEFAULTS = {
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "alpha_learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 5e-3,
    "alpha_init": 1.0,
    "target_entropy_per_dim": -1.0,
    "max_grad_norm": 0.5,
    "batch_size": 256,
    "arch_width": 256,
}


def load_plane_hyperparams() -> dict:
    """Return Plane hyperparameters from hyperparams/ajax_sac.yml, or hard-coded defaults."""
    if _HP_YAML.exists():
        import yaml
        with open(_HP_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return {**_HP_DEFAULTS, **cfg.get("Plane", {})}
    return dict(_HP_DEFAULTS)


# ---------------------------------------------------------------------------
# Experiment configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    One experiment in the ablation suite.
    Extends plane_exps.ExperimentConfig with ablation-specific flags.
    All expert-guidance flags default to off → safe baseline when not set.
    """
    name: str

    # Environment
    use_box: bool = False

    # Expert warmup
    use_expert_warmup: bool = False

    # Expert guidance flags
    use_expert_guidance: bool = False
    use_mc_critic_pretrain: bool = False
    use_bellman_critic_pretrain: bool = False
    value_constraint_coef: float = 0.0
    augment_obs_with_expert_action: bool = False

    # AWBC parameters
    num_critic_updates: int = 1
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1
    box_threshold: float = 500.0
    proximity_scale: Optional[float] = None
    altitude_obs_idx: int = 1
    target_obs_idx: int = 6

    # MC pretraining parameters
    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_mc_episodes: int = 100
    mc_pretrain_n_steps: int = 5_000

    # Standard SAC hyperparameters
    # None means "use the value from hyperparams/ajax_sac.yml" (set by hyperparam search).
    # Set explicitly in an ExperimentConfig to override for a specific ablation.
    num_critics: int = 2
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    alpha_lr: Optional[float] = None
    gamma: Optional[float] = None
    tau: Optional[float] = None
    alpha_init: Optional[float] = None
    target_entropy_per_dim: Optional[float] = None
    max_grad_norm: Optional[float] = None
    batch_size: Optional[int] = None
    arch_width: Optional[int] = None

    # AWBC ablation flags (new)
    awbc_normalize: bool = True          # False → remove |L_actor| denominator
    awbc_use_relu: bool = True           # False → raw (Q*-Qπ) difference, no self-annealing
    fixed_awbc_lambda: Optional[float] = None   # float → bypass adaptive λ with a constant
    detach_obs_aug_action: bool = False  # True → stop-grad through a_expert dims in actor obs


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_experiments() -> List[ExperimentConfig]:
    """
    Ablation suite across five tiers:

    Tier 1 — Core ablations: establish contribution of each component in isolation.
    Tier 2 — Obs augmentation deep dive.
    Tier 3 — Critic architecture and update ratio.
    Tier 4 — MC pretrain hyperparameters.
    Tier 5 — Expert mix fraction and buffer strategy.
    """
    # Shared config blocks
    MC = dict(
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_steps=5_000,
    )
    BUF = dict(expert_buffer_n_steps=20_000, expert_mix_fraction=0.1)
    # Best anchor: mc_pretrain_awbc uses 4 critic updates
    AWBC = dict(use_expert_guidance=True, num_critic_updates=4)

    exps = []

    # ==================================================================
    # Tier 1 — Core ablations
    # ==================================================================

    # Pure SAC floor — no expert at all
    exps.append(ExperimentConfig(
        name="sac_baseline",
        expert_buffer_n_steps=0, expert_mix_fraction=0.0,
    ))

    # AWBC only, no MC pretrain — reference, previously run
    exps.append(ExperimentConfig(
        name="awbc_only",
        use_expert_warmup=True,
        **AWBC, **BUF,
    ))

    # MC pretrain only — critic init alone, no AWBC gradient at training time
    exps.append(ExperimentConfig(
        name="mc_pretrain_only",
        use_expert_warmup=True,
        **MC, **BUF,
    ))

    # Main reference — previously run, anchor for all comparisons
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc",
        use_expert_warmup=True,
        **AWBC, **MC, **BUF,
    ))

    # Remove |L_actor| normalization from λ(s): λ = relu(Q* - Qπ) only
    exps.append(ExperimentConfig(
        name="awbc_no_normalization",
        use_expert_warmup=True,
        awbc_normalize=False,
        **AWBC, **MC, **BUF,
    ))

    # Fixed λ = 0.1 — no state-dependent Q-gating
    exps.append(ExperimentConfig(
        name="awbc_fixed_lambda_0.1",
        use_expert_warmup=True,
        fixed_awbc_lambda=0.1,
        **AWBC, **MC, **BUF,
    ))

    # Fixed λ = 1.0
    exps.append(ExperimentConfig(
        name="awbc_fixed_lambda_1.0",
        use_expert_warmup=True,
        fixed_awbc_lambda=1.0,
        **AWBC, **MC, **BUF,
    ))

    # Raw (Q* - Qπ) instead of relu — no self-annealing when policy ≥ expert
    exps.append(ExperimentConfig(
        name="awbc_no_relu",
        use_expert_warmup=True,
        awbc_use_relu=False,
        **AWBC, **MC, **BUF,
    ))

    # ==================================================================
    # Tier 2 — Obs augmentation deep dive
    # ==================================================================

    # obs aug + AWBC, no MC pretrain — is obs aug safe without a good critic init?
    exps.append(ExperimentConfig(
        name="obs_aug_awbc_only",
        use_expert_warmup=True,
        augment_obs_with_expert_action=True,
        **AWBC, **BUF,
    ))

    # obs aug + MC pretrain + AWBC — best combo, previously run
    exps.append(ExperimentConfig(
        name="obs_aug_mc_pretrain_awbc",
        use_expert_warmup=True,
        augment_obs_with_expert_action=True,
        **AWBC, **MC, **BUF,
    ))

    # obs aug + MC pretrain, no AWBC — confirms obs aug alone is dangerous, previously run
    exps.append(ExperimentConfig(
        name="obs_aug_mc_pretrain_no_awbc",
        use_expert_warmup=True,
        augment_obs_with_expert_action=True,
        **MC, **BUF,
    ))

    # obs aug with detached expert-action dims in actor gradient.
    # Tests whether the actor is "gaming" the augmentation: if this matches
    # obs_aug_mc_pretrain_awbc, the actor wasn't exploiting the hint at all.
    exps.append(ExperimentConfig(
        name="obs_aug_detached_action",
        use_expert_warmup=True,
        augment_obs_with_expert_action=True,
        detach_obs_aug_action=True,
        **AWBC, **MC, **BUF,
    ))

    # ==================================================================
    # Tier 3 — Critic architecture and update ratio
    # ==================================================================

    # 1 critic — less than default (2), lower variance targets
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_1critic",
        use_expert_warmup=True,
        num_critics=1,
        **AWBC, **MC, **BUF,
    ))

    # 4 critics — more than default (2), stronger pessimism
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_4critics",
        use_expert_warmup=True,
        num_critics=4,
        **AWBC, **MC, **BUF,
    ))

    # 2 critics (default), 2 updates/step — is the x2 update cost justified?
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_x2updates",
        use_expert_warmup=True,
        num_critic_updates=2,
        **MC, **BUF,
        use_expert_guidance=True,
    ))

    # 1 critic, 2 updates/step — cross both dimensions
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_1critic_x2updates",
        use_expert_warmup=True,
        num_critics=1,
        num_critic_updates=2,
        **MC, **BUF,
        use_expert_guidance=True,
    ))

    # ==================================================================
    # Tier 4 — MC pretrain hyperparameters
    # ==================================================================

    # Data quantity sweep: how many MC episodes are needed?
    for n_ep in [10, 50, 500]:  # 100 is the default (mc_pretrain_awbc above)
        exps.append(ExperimentConfig(
            name=f"mc_pretrain_episodes_{n_ep}",
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_mc_episodes=n_ep,
            mc_pretrain_n_steps=5_000,
            **AWBC, **BUF,
        ))

    # Regression duration sweep: how long to train the critic on MC returns?
    for n_steps in [1_000, 20_000]:  # 5_000 is the default (mc_pretrain_awbc above)
        exps.append(ExperimentConfig(
            name=f"mc_pretrain_steps_{n_steps // 1000}k",
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_mc_episodes=100,
            mc_pretrain_n_steps=n_steps,
            **AWBC, **BUF,
        ))

    # Bellman pretrain as alternative to MC — confirms MC > Bellman
    exps.append(ExperimentConfig(
        name="bellman_pretrain_awbc",
        use_expert_warmup=True,
        use_bellman_critic_pretrain=True,
        mc_pretrain_n_steps=5_000,   # reused as n_steps for Bellman pretrain
        **AWBC, **BUF,
    ))

    # ==================================================================
    # Tier 5 — Expert mix fraction and buffer strategy
    # ==================================================================

    # No expert oversampling, no expert buffer seeding — AWBC gradient signal only
    exps.append(ExperimentConfig(
        name="expert_mix_0",
        use_expert_warmup=True,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        **AWBC, **MC,
    ))

    # Higher expert fractions in replay mix
    for frac in [0.3, 0.5]:
        exps.append(ExperimentConfig(
            name=f"expert_mix_{frac}",
            use_expert_warmup=True,
            expert_mix_fraction=frac,
            expert_buffer_n_steps=20_000,
            **AWBC, **MC,
        ))

    # Expert data seeds the buffer from warmup but no oversampling after learning_starts
    exps.append(ExperimentConfig(
        name="expert_buffer_only_pretrain",
        use_expert_warmup=True,
        expert_buffer_n_steps=20_000,
        expert_mix_fraction=0.0,    # no explicit expert oversampling during training
        **AWBC, **MC,
    ))

    return exps


# ---------------------------------------------------------------------------
# Shared setup (runs in every process)
# ---------------------------------------------------------------------------


def setup():
    env = Plane()
    env_params = PlaneParams(
        target_altitude_range=(3_000, 8_000),
        initial_altitude_range=(3_000, 8_000),
        max_steps_in_episode=10_000,
    )

    if "expert_policy.pkl" in os.listdir():
        with open("expert_policy.pkl", "rb") as f:
            expert_policy = pickle.load(f)
    else:
        expert_policy = get_expert_policy(env, env_params)
        with open("expert_policy.pkl", "wb") as f:
            pickle.dump(expert_policy, f)

    for test_alt in [3_000, 8_000]:
        test_obs = jnp.array([0, 0, 0, 0, 0, 0, test_alt], dtype=jnp.float32)
        if jnp.isnan(expert_policy(test_obs)).any():
            raise ValueError(f"Expert policy produces NaN at altitude {test_alt}")

    return env, env_params, expert_policy


def make_trunc_condition(box_threshold: float):
    def trunc_condition(state: PlaneState, params: PlaneParams) -> bool:
        return jnp.abs(state.target_altitude - state.z) < box_threshold
    return trunc_condition


def run_single_experiment(
    exp: ExperimentConfig,
    env,
    env_params,
    expert_policy,
    n_seeds: int,
    n_timesteps: int,
    num_episode_test: int,
    log_frequency: int,
    project_name: str,
    sweep_mode: bool,
):
    """Run one ExperimentConfig to completion."""
    mode = get_mode()

    # Resolve SAC hyperparameters: experiment value → YAML → hard-coded fallback
    hp = load_plane_hyperparams()
    actor_lr            = exp.actor_lr            if exp.actor_lr            is not None else hp["actor_learning_rate"]
    critic_lr           = exp.critic_lr           if exp.critic_lr           is not None else hp["critic_learning_rate"]
    alpha_lr            = exp.alpha_lr            if exp.alpha_lr            is not None else hp["alpha_learning_rate"]
    gamma               = exp.gamma               if exp.gamma               is not None else hp["gamma"]
    tau                 = exp.tau                 if exp.tau                 is not None else hp["tau"]
    alpha_init          = exp.alpha_init          if exp.alpha_init          is not None else hp["alpha_init"]
    target_entropy_pd   = exp.target_entropy_per_dim if exp.target_entropy_per_dim is not None else hp["target_entropy_per_dim"]
    max_grad_norm       = exp.max_grad_norm       if exp.max_grad_norm       is not None else hp["max_grad_norm"]
    batch_size          = exp.batch_size          if exp.batch_size          is not None else hp["batch_size"]
    arch_width          = exp.arch_width          if exp.arch_width          is not None else hp["arch_width"]
    arch                = (str(arch_width), "relu", str(arch_width), "relu")

    trunc_condition = make_trunc_condition(exp.box_threshold)
    wrapped_env = EarlyTerminationWrapper(
        env, trunc_condition=trunc_condition, expert_policy=expert_policy,
    )

    policy_score = get_policy_score(expert_policy, env, env_params)
    print(f"[{exp.name}] Expert score: {policy_score:.1f}")
    print(
        f"[{exp.name}] warmup={exp.use_expert_warmup}  awbc={exp.use_expert_guidance}  "
        f"mc={exp.use_mc_critic_pretrain}  bellman={exp.use_bellman_critic_pretrain}  "
        f"obs_aug={exp.augment_obs_with_expert_action}  "
        f"normalize={exp.awbc_normalize}  relu={exp.awbc_use_relu}  "
        f"fixed_λ={exp.fixed_awbc_lambda}  detach={exp.detach_obs_aug_action}  "
        f"critics={exp.num_critics}×{exp.num_critic_updates}  "
        f"mix={exp.expert_mix_fraction}"
    )

    agent_expert_policy = expert_policy if exp.use_expert_warmup else None

    logging_config = get_log_config(
        project_name=project_name,
        agent_name=exp.name,
        log_frequency=log_frequency,
        use_wandb=True,
        sweep=sweep_mode,
        use_box=exp.use_box,
        use_expert_warmup=exp.use_expert_warmup,
        use_expert_guidance=exp.use_expert_guidance,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        use_bellman_critic_pretrain=exp.use_bellman_critic_pretrain,
        value_constraint_coef=exp.value_constraint_coef,
        augment_obs_with_expert_action=exp.augment_obs_with_expert_action,
        num_critics=exp.num_critics,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        box_threshold=exp.box_threshold,
        proximity_scale=exp.proximity_scale,
        tau=tau,
        target_entropy_per_dim=target_entropy_pd,
        awbc_normalize=exp.awbc_normalize,
        awbc_use_relu=exp.awbc_use_relu,
        fixed_awbc_lambda=exp.fixed_awbc_lambda,
        detach_obs_aug_action=exp.detach_obs_aug_action,
    )

    _agent = SAC(
        env_id=wrapped_env if exp.use_box else env,
        env_params=env_params,
        expert_policy=agent_expert_policy,
        eval_expert_policy=expert_policy,
        action_scale=1.0,
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        alpha_learning_rate=alpha_lr,
        gamma=gamma,
        batch_size=batch_size,
        actor_architecture=arch,
        critic_architecture=arch,
        max_grad_norm=max_grad_norm,
        early_termination_condition=trunc_condition if exp.use_box else None,
        num_critics=exp.num_critics,
        tau=tau,
        alpha_init=alpha_init,
        target_entropy_per_dim=target_entropy_pd,
        residual=False,
        fixed_alpha=False,
        use_expert_guidance=exp.use_expert_guidance,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        box_threshold=exp.box_threshold,
        proximity_scale=exp.proximity_scale,
        altitude_obs_idx=exp.altitude_obs_idx,
        target_obs_idx=exp.target_obs_idx,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        mc_pretrain_n_mc_steps=exp.mc_pretrain_n_mc_steps,
        mc_pretrain_n_mc_episodes=exp.mc_pretrain_n_mc_episodes,
        mc_pretrain_n_steps=exp.mc_pretrain_n_steps,
        use_bellman_critic_pretrain=exp.use_bellman_critic_pretrain,
        value_constraint_coef=exp.value_constraint_coef,
        augment_obs_with_expert_action=exp.augment_obs_with_expert_action,
        awbc_normalize=exp.awbc_normalize,
        awbc_use_relu=exp.awbc_use_relu,
        fixed_awbc_lambda=exp.fixed_awbc_lambda,
        detach_obs_aug_action=exp.detach_obs_aug_action,
    )

    if mode == "CPU":
        for seed in tqdm(range(n_seeds), desc=exp.name):
            t0 = time.time()
            _agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
            )
            print(f"[{exp.name}] seed {seed} done in {time.time() - t0:.1f}s")
            if sweep_mode:
                upload_tensorboard_to_wandb(_agent.run_ids, logging_config)
    else:
        t0 = time.time()
        _agent.train(
            seed=list(range(n_seeds)),
            logging_config=logging_config,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
        )
        elapsed = time.time() - t0
        print(f"[{exp.name}] {n_seeds} seeds done in {elapsed:.1f}s "
              f"({elapsed/n_seeds:.1f}s/seed)")
        if sweep_mode:
            upload_tensorboard_to_wandb(_agent.run_ids, logging_config)


# ---------------------------------------------------------------------------
# Main — supports both full sweep and single-experiment mode
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-index",
        type=int,
        default=None,
        help=(
            "Run a single experiment by index (0-based). "
            "Used by gpu_launcher.py to run one experiment per GPU process. "
            "Omit to run all experiments sequentially."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all experiment names and exit.",
    )
    args = parser.parse_args()

    # --- Shared config ---
    project_name = "ablation_study_awbc_2"
    n_timesteps = int(1e6)
    n_seeds = 25
    num_episode_test = 25
    log_frequency = 10_000
    sweep_mode = False

    experiments = build_experiments()

    if args.list:
        tiers = {
            "Tier 1 — Core ablations": slice(0, 8),
            "Tier 2 — Obs augmentation": slice(8, 12),
            "Tier 3 — Critic arch / update ratio": slice(12, 16),
            "Tier 4 — MC pretrain hyperparams": slice(16, 22),
            "Tier 5 — Expert mix / buffer": slice(22, 26),
        }
        for tier, s in tiers.items():
            print(f"\n{tier}:")
            for i, exp in enumerate(experiments[s], start=s.start):
                print(f"  [{i:2d}] {exp.name}")
        raise SystemExit(0)

    env, env_params, expert_policy = setup()

    if args.exp_index is not None:
        if args.exp_index >= len(experiments):
            raise ValueError(
                f"--exp-index {args.exp_index} out of range "
                f"(only {len(experiments)} experiments defined)"
            )
        exp = experiments[args.exp_index]
        print(f"Single-experiment mode: [{args.exp_index}] {exp.name}")
        run_single_experiment(
            exp=exp,
            env=env,
            env_params=env_params,
            expert_policy=expert_policy,
            n_seeds=n_seeds,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            log_frequency=log_frequency,
            project_name=project_name,
            sweep_mode=sweep_mode,
        )
    else:
        print(f"Sequential sweep: {len(experiments)} experiments × {n_seeds} seeds\n")
        for exp in experiments:
            run_single_experiment(
                exp=exp,
                env=env,
                env_params=env_params,
                expert_policy=expert_policy,
                n_seeds=n_seeds,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
                log_frequency=log_frequency,
                project_name=project_name,
                sweep_mode=sweep_mode,
            )
