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
    One experiment in the sweep. Each field maps directly to a SAC constructor
    argument. Add rows to build_experiments() to extend the sweep.
    """

    name: str  # W&B run label

    # Environment
    use_box: bool = True  # wrap with EarlyTerminationWrapper

    # Expert involvement — three independent flags:
    # use_expert_warmup:  expert actions used during buffer warmup phase
    # use_expert_guidance: AWBC gradient term active during training
    # Setting both False → true vanilla SAC with no expert involvement whatsoever
    use_expert_warmup: bool = True  # expert/uniform mix during warmup
    use_expert_guidance: bool = False  # AWBC gradient term active

    # AWBC core parameters
    num_critic_updates: int = 1  # critic gradient steps per env step
    actor_pretrain_steps: int = 0  # BC pre-training steps (0 = disabled)
    critic_pretrain_steps: int = 5_000  # critic pre-training on expert buffer
    expert_buffer_n_steps: int = 20_000  # expert transitions pre-loaded into buffer
    expert_mix_fraction: float = 0.1  # fraction of each batch from expert buffer

    # Asymmetric AWBC (disabled by default → symmetric, set True to activate)
    use_asymmetric_awbc: bool = False
    above_expert_coef: float = 0.01  # only used when use_asymmetric_awbc=True

    # Proximity-weighted AWBC
    # box_threshold is the single source of truth: it sets both the
    # EarlyTerminationWrapper boundary AND normalizes the proximity weight.
    # proximity_scale=None → no decay (uniform expert trust everywhere)
    # proximity_scale=1.0  → weight=0.37 at box boundary, ~0 at 5× boundary
    # proximity_scale=2.0  → gentler: still 8% trust at 5× boundary
    box_threshold: float = 500.0
    proximity_scale: Optional[float] = None
    altitude_obs_idx: int = 1  # raw_obs index for current altitude
    target_obs_idx: int = 6  # raw_obs index for target altitude

    # Standard SAC hyperparameters
    num_critics: int = 4
    tau: float = 5e-4
    alpha_init: float = 1.0
    target_entropy_per_dim: float = -1.0


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_experiments() -> List[ExperimentConfig]:
    """
    Full sweep table. Modify here to add/remove/change experiments.

    Current experiments:
      1. Baseline SAC              — no expert, no box
      2. Baseline SAC + box        — box only, no AWBC (isolates box benefit)
      3. AWBC no BC, 1 critic      — replicates original working run
      4. AWBC no BC, 4 critics     — best known result so far
      5. AWBC + BC, 1 critic       — BC without enough critic updates (expect instability)
      6. AWBC + BC, 4 critics      — currently running, hypothesis: best of both worlds
      7. Asymmetric AWBC no BC     — tests saturation hypothesis without BC
      8. Asymmetric AWBC + BC      — tests saturation hypothesis with BC
    """
    exps = []

    # ------------------------------------------------------------------
    # 1. True vanilla SAC — no expert involvement whatsoever
    # Uses standard SAC hyperparameters: 2 critics, tau=0.005.
    # Our tuned settings (4 critics, tau=5e-4) are deliberate departures
    # from vanilla SAC — this baseline uses neither.
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="baseline_sac_vanilla",
            use_box=False,
            use_expert_warmup=False,
            use_expert_guidance=False,
            num_critics=2,  # standard SAC (not our tuned 4)
            tau=0.005,  # standard SAC (not our tuned 5e-4)
            num_critic_updates=1,
            actor_pretrain_steps=0,
            critic_pretrain_steps=0,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )

    # ------------------------------------------------------------------
    # 2. SAC with our tuned settings but no expert involvement
    # Separates the effect of our hyperparameter choices from the expert.
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="baseline_sac_tuned",
            use_box=False,
            use_expert_warmup=False,
            use_expert_guidance=False,
            num_critics=4,  # our tuned setting
            tau=5e-4,  # our tuned setting
            num_critic_updates=1,
            actor_pretrain_steps=0,
            critic_pretrain_steps=0,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )

    # ------------------------------------------------------------------
    # 3. SAC with expert warmup, no AWBC, no box
    # Isolates the effect of expert buffer seeding alone.
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="baseline_sac_expert_warmup",
            use_box=False,
            use_expert_warmup=True,
            use_expert_guidance=False,
            num_critics=4,
            tau=5e-4,
            num_critic_updates=1,
            actor_pretrain_steps=0,
            critic_pretrain_steps=0,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )

    # ------------------------------------------------------------------
    # 4. Baseline SAC + box — no AWBC, but box mechanism active
    # ------------------------------------------------------------------
    exps.append(
        ExperimentConfig(
            name="baseline_sac_box",
            use_box=True,
            use_expert_warmup=True,
            use_expert_guidance=False,
            num_critics=4,
            tau=5e-4,
            num_critic_updates=1,
            actor_pretrain_steps=0,
            critic_pretrain_steps=0,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    )

    # ------------------------------------------------------------------
    # 4 & 5. AWBC, no BC — symmetric, varying critic updates
    # ------------------------------------------------------------------
    for n in [1, 4]:
        exps.append(
            ExperimentConfig(
                name=f"awbc_no_bc_{n}critic",
                use_box=True,
                use_expert_warmup=True,
                use_expert_guidance=True,
                num_critic_updates=n,
                actor_pretrain_steps=0,
                critic_pretrain_steps=5_000,
                expert_buffer_n_steps=20_000,
                use_asymmetric_awbc=False,
            )
        )

    # ------------------------------------------------------------------
    # 6 & 7. AWBC + BC — symmetric, varying critic updates
    # ------------------------------------------------------------------
    for n in [1, 4]:
        exps.append(
            ExperimentConfig(
                name=f"awbc_bc_{n}critic",
                use_box=True,
                use_expert_warmup=True,
                use_expert_guidance=True,
                num_critic_updates=n,
                actor_pretrain_steps=2_000,
                critic_pretrain_steps=5_000,
                expert_buffer_n_steps=20_000,
                use_asymmetric_awbc=False,
            )
        )

    # ------------------------------------------------------------------
    # 8 & 9. Asymmetric AWBC — with and without BC, 4 critics
    # ------------------------------------------------------------------
    for use_bc in [False, True]:
        exps.append(
            ExperimentConfig(
                name=f"awbc_asymmetric_{'bc' if use_bc else 'no_bc'}_4critic",
                use_box=True,
                use_expert_warmup=True,
                use_expert_guidance=True,
                num_critic_updates=4,
                actor_pretrain_steps=2_000 if use_bc else 0,
                critic_pretrain_steps=5_000,
                expert_buffer_n_steps=20_000,
                use_asymmetric_awbc=True,
                above_expert_coef=0.01,
            )
        )

    # ------------------------------------------------------------------
    # 10–12. Proximity-weighted AWBC — varying decay scale, no BC, 4 critics
    # ------------------------------------------------------------------
    for scale in [0.5, 1.0, 2.0]:
        exps.append(
            ExperimentConfig(
                name=f"awbc_proximity_{scale}_4critic",
                use_box=True,
                use_expert_warmup=True,
                use_expert_guidance=True,
                num_critic_updates=4,
                actor_pretrain_steps=0,
                critic_pretrain_steps=5_000,
                expert_buffer_n_steps=20_000,
                box_threshold=500.0,
                proximity_scale=scale,
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
        f"[{exp.name}] box={exp.use_box}  threshold={exp.box_threshold}  "
        f"expert_warmup={exp.use_expert_warmup}  guidance={exp.use_expert_guidance}  "
        f"critics={exp.num_critic_updates}  bc={exp.actor_pretrain_steps > 0}  "
        f"proximity_scale={exp.proximity_scale}"
    )

    # When use_expert_warmup=False, pass None to the agent so collect_experience
    # uses pure uniform random actions during warmup — true vanilla SAC behaviour.
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
        num_critics=exp.num_critics,
        num_critic_updates=exp.num_critic_updates,
        actor_pretrain_steps=exp.actor_pretrain_steps,
        critic_pretrain_steps=exp.critic_pretrain_steps,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        use_asymmetric_awbc=exp.use_asymmetric_awbc,
        above_expert_coef=exp.above_expert_coef,
        box_threshold=exp.box_threshold,
        proximity_scale=exp.proximity_scale,
        tau=exp.tau,
        target_entropy_per_dim=exp.target_entropy_per_dim,
    )

    _agent = SAC(
        env_id=wrapped_env if exp.use_box else env,
        env_params=env_params,
        expert_policy=agent_expert_policy,  # None → pure uniform warmup
        eval_expert_policy=expert_policy,  # always set → expert bias logged for all runs
        action_scale=1.0,
        early_termination_condition=trunc_condition if exp.use_box else None,
        num_critics=exp.num_critics,
        tau=exp.tau,
        alpha_init=exp.alpha_init,
        target_entropy_per_dim=exp.target_entropy_per_dim,
        residual=False,
        fixed_alpha=False,
        use_expert_guidance=exp.use_expert_guidance,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=(
            exp.expert_buffer_n_steps if exp.use_expert_guidance else 0
        ),
        critic_pretrain_steps=(
            exp.critic_pretrain_steps if exp.use_expert_guidance else 0
        ),
        actor_pretrain_steps=(
            exp.actor_pretrain_steps if exp.use_expert_guidance else 0
        ),
        expert_mix_fraction=(
            exp.expert_mix_fraction if exp.use_expert_guidance else 0.0
        ),
        use_asymmetric_awbc=exp.use_asymmetric_awbc,
        above_expert_coef=exp.above_expert_coef,
        box_threshold=exp.box_threshold,
        proximity_scale=exp.proximity_scale,
        altitude_obs_idx=exp.altitude_obs_idx,
        target_obs_idx=exp.target_obs_idx,
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
