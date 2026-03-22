import os
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
# Experiment configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    One experiment in the sweep.
    All expert-guidance flags default to off → safe baseline when not set.
    """

    name: str

    # Environment
    use_box: bool = False  # EarlyTerminationWrapper (expert takes over near target)

    # Expert warmup
    use_expert_warmup: bool = False  # expert/uniform mix during warmup phase

    # Expert guidance flags (all independent, all off by default)
    use_expert_guidance: bool = False  # AWBC gradient term
    use_mc_critic_pretrain: bool = False  # MC return critic pretraining
    value_constraint_coef: float = 0.0  # value floor penalty (0 = off)
    augment_obs_with_expert_action: bool = False  # append a_expert to obs

    # AWBC parameters
    num_critic_updates: int = 1
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1
    box_threshold: float = 500.0
    proximity_scale: Optional[float] = None  # None = no decay
    altitude_obs_idx: int = 1
    target_obs_idx: int = 6

    # MC pretraining parameters
    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_steps: int = 5_000

    # Standard SAC hyperparameters
    num_critics: int = 4
    tau: float = 5e-4
    alpha_init: float = 1.0
    target_entropy_per_dim: float = -1.0
    max_grad_norm: Optional[float] = 0.5


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_experiments() -> List[ExperimentConfig]:
    """
    New experiment table focused on the three new expert-guidance approaches:
    1. MC critic pretraining    — unbiased Q-value initialisation from expert returns
    2. Value constraint         — Q_π >= Q_expert floor, action-agnostic
    3. Obs augmentation         — a_expert appended to obs as a hint

    Baselines:
      baseline_sac_vanilla      — true SAC (2 critics, tau=0.005, no expert)
      baseline_sac_tuned        — our tuned SAC (4 critics, tau=5e-4, no expert)
      baseline_sac_expert_warmup — tuned SAC + expert warmup only

    Previous AWBC experiments are kept as reference points but are not the
    primary focus of this sweep.
    """
    exps = []

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="baseline_sac_vanilla",
            num_critics=2,
            tau=0.005,
            max_grad_norm=None,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )
    exps.append(
        ExperimentConfig(
            name="baseline_sac_tuned",
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )
    exps.append(
        ExperimentConfig(
            name="baseline_sac_expert_warmup",
            use_expert_warmup=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )

    # ------------------------------------------------------------------
    # New approach 1: MC critic pretraining
    # Pre-trains critic to Q(s,a_expert) ≈ MC return before any RL.
    # Critic starts with accurate near-target values → policy gradient
    # immediately favours expert-valued states, no imitation term needed.
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="mc_pretrain",
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_steps=5_000,
            expert_buffer_n_steps=20_000,
            expert_mix_fraction=0.1,
        )
    )
    # MC pretrain + AWBC: does accurate Q-init make AWBC work better?
    exps.append(
        ExperimentConfig(
            name="mc_pretrain_awbc",
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            use_expert_guidance=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_steps=5_000,
            expert_buffer_n_steps=20_000,
            expert_mix_fraction=0.1,
            num_critic_updates=4,
        )
    )

    # ------------------------------------------------------------------
    # New approach 2: Value constraint
    # Policy penalised when Q_π < Q_expert. Action-agnostic: allows the
    # agent to approach aggressively as long as value is >= expert.
    # ------------------------------------------------------------------
    for coef in [0.1, 0.5, 1.0]:
        exps.append(
            ExperimentConfig(
                name=f"value_constraint_{coef}",
                use_expert_warmup=True,
                value_constraint_coef=coef,
                expert_buffer_n_steps=20_000,
                expert_mix_fraction=0.1,
                num_critic_updates=4,
            )
        )

    # ------------------------------------------------------------------
    # New approach 3: Observation augmentation
    # a_expert(target) appended to obs. Network learns to use this hint
    # near target (trivially) and override it far away.
    # Note: requires network initialised with extra_obs_dim=2.
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="obs_augment",
            use_expert_warmup=True,
            augment_obs_with_expert_action=True,
            expert_buffer_n_steps=20_000,
            expert_mix_fraction=0.1,
        )
    )

    # ------------------------------------------------------------------
    # Combination: MC pretrain + value constraint (best of both)
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="mc_pretrain_vc_0.5",
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            value_constraint_coef=0.5,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_steps=5_000,
            expert_buffer_n_steps=20_000,
            expert_mix_fraction=0.1,
            num_critic_updates=4,
        )
    )

    # ------------------------------------------------------------------
    # Legacy AWBC reference (best previous result: no BC, 4 critics)
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="awbc_4critic_reference",
            use_box=True,
            use_expert_warmup=True,
            use_expert_guidance=True,
            num_critic_updates=4,
            expert_buffer_n_steps=20_000,
            expert_mix_fraction=0.1,
        )
    )

    return exps


# ---------------------------------------------------------------------------
# Shared setup (runs in every process)
# ---------------------------------------------------------------------------


def setup():
    """
    Load environment and expert policy.
    box_threshold is per-experiment so trunc_condition and wrapped_env
    are built inside run_single_experiment, not here.
    """
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
    """
    Build a truncation condition from box_threshold.
    Single source of truth: the same threshold is used by both the
    EarlyTerminationWrapper and the AWBC proximity weight normalization.
    """

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
    """Run one ExperimentConfig to completion. Called per-process by the launcher."""
    mode = get_mode()

    # Build per-experiment box condition from exp.box_threshold — single source of truth
    trunc_condition = make_trunc_condition(exp.box_threshold)
    wrapped_env = EarlyTerminationWrapper(
        env,
        trunc_condition=trunc_condition,
        expert_policy=expert_policy,
    )

    policy_score = get_policy_score(expert_policy, env, env_params)
    print(f"[{exp.name}] Expert score: {policy_score:.1f}")
    print(
        f"[{exp.name}] box={exp.use_box}  warmup={exp.use_expert_warmup}  "
        f"awbc={exp.use_expert_guidance}  mc_pretrain={exp.use_mc_critic_pretrain}  "
        f"vc={exp.value_constraint_coef}  obs_aug={exp.augment_obs_with_expert_action}  "
        f"critics={exp.num_critic_updates}"
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
        value_constraint_coef=exp.value_constraint_coef,
        augment_obs_with_expert_action=exp.augment_obs_with_expert_action,
        num_critics=exp.num_critics,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        box_threshold=exp.box_threshold,
        proximity_scale=exp.proximity_scale,
        tau=exp.tau,
        target_entropy_per_dim=exp.target_entropy_per_dim,
    )

    _agent = SAC(
        env_id=wrapped_env if exp.use_box else env,
        env_params=env_params,
        expert_policy=agent_expert_policy,
        eval_expert_policy=expert_policy,
        action_scale=1.0,
        max_grad_norm=exp.max_grad_norm,
        early_termination_condition=trunc_condition if exp.use_box else None,
        num_critics=exp.num_critics,
        tau=exp.tau,
        alpha_init=exp.alpha_init,
        target_entropy_per_dim=exp.target_entropy_per_dim,
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
        mc_pretrain_n_steps=exp.mc_pretrain_n_steps,
        value_constraint_coef=exp.value_constraint_coef,
        augment_obs_with_expert_action=exp.augment_obs_with_expert_action,
    )

    if mode == "CPU":
        for seed in tqdm(range(n_seeds), desc=exp.name):
            _agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
            )
            if sweep_mode:
                upload_tensorboard_to_wandb(_agent.run_ids, logging_config)
    else:
        _agent.train(
            seed=list(range(n_seeds)),
            logging_config=logging_config,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
        )
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
            "Omit to run all experiments sequentially (original behaviour)."
        ),
    )
    args = parser.parse_args()

    # --- Shared config ---
    project_name = "tests_SAC_plane_awbc_sweep"
    n_timesteps = int(1e6)
    n_seeds = 3
    num_episode_test = 25
    log_frequency = 5_000
    sweep_mode = False

    experiments = build_experiments()
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
