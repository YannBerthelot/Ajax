import os
import time
from dataclasses import dataclass
from typing import List, Optional

import dill as pickle
import jax
import jax.numpy as jnp

# Persist compiled XLA modules across runs — avoids recompilation on re-runs
# with the same static args. Safe to share across experiments on the same machine.
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_xla"))
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
    load_hyperparams,
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

    # Expert warmup
    use_expert_warmup: bool = False

    # Expert guidance flags
    use_expert_guidance: bool = False
    use_mc_critic_pretrain: bool = False
    use_critic_blend: bool = False
    critic_warmup_frac: float = 0.15

    # Value-threshold box (v_min/v_max inferred from MC pretraining)
    use_box: bool = False

    # Update start thresholds
    policy_update_start: int = 2_000
    alpha_update_start: int = 2_000

    # AWBC parameters
    num_critic_updates: int = 1
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1

    # MC pretraining parameters
    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_mc_episodes: int = 500
    mc_pretrain_n_steps: int = 5_000

    # Standard SAC hyperparameters (defaults; overridden by yml at runtime)
    num_critics: int = 2


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_experiments() -> List[ExperimentConfig]:
    """
    4-experiment sweep.

    Phase 1: Establish the main result.

    E1: Vanilla SAC (tuned) — baseline.
    E2: SAC + MC pretrain + AWBC — full expert method without box.
    E3: SAC + MC pretrain + AWBC + Shaping — adds potential-based shaping.
    E4: SAC + MC pretrain + Box — box without AWBC.

    E2 vs E1: expert guidance helps.
    E3 vs E2: shaping on top of E2.
    E4 vs E1: box independently.
    E3 vs E4: AWBC+shaping vs box.
    """
    MC = dict(
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_steps=5_000,
    )
    BUF = dict(expert_buffer_n_steps=20_000, expert_mix_fraction=0.1)

    return [
        # E1: Vanilla SAC (tuned) — baseline (no expert; higher update starts)
        ExperimentConfig(
            name="e1_vanilla_sac",
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            policy_update_start=10_000,
            alpha_update_start=10_000,
        ),
        # E2: MC pretrain + AWBC — full expert method without box
        ExperimentConfig(
            name="e2_mc_pretrain_awbc",
            use_expert_warmup=True,
            use_expert_guidance=True,
            num_critic_updates=4,
            **MC, **BUF,
        ),
        # E3: MC pretrain + AWBC + blended Bellman target
        ExperimentConfig(
            name="e3_mc_pretrain_awbc_blend",
            use_expert_warmup=True,
            use_expert_guidance=True,
            use_critic_blend=True,
            num_critic_updates=4,
            **MC, **BUF,
        ),
        # E4: MC pretrain + Box (value threshold)
        ExperimentConfig(
            name="e4_mc_pretrain_box",
            use_expert_warmup=True,
            use_box=True,
            **MC, **BUF,
        ),
    ]


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
    use_wandb: bool = True,
):
    """Run one ExperimentConfig to completion. Called per-process by the launcher."""
    mode = get_mode()

    # Load tuned hyperparameters from yml — single source of truth
    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")

    agent_expert_policy = expert_policy if exp.use_expert_warmup else None

    policy_score = get_policy_score(expert_policy, env, env_params)
    print(f"[{exp.name}] Expert score: {policy_score:.1f}")
    print(
        f"[{exp.name}] warmup={exp.use_expert_warmup}  "
        f"awbc={exp.use_expert_guidance}  mc_pretrain={exp.use_mc_critic_pretrain}  "
        f"blend={exp.use_critic_blend}  box={exp.use_box}  "
        f"critics={exp.num_critic_updates}"
    )

    logging_config = get_log_config(
        project_name=project_name,
        agent_name=exp.name,
        log_frequency=log_frequency,
        use_wandb=use_wandb,
        sweep=sweep_mode,
        use_expert_warmup=exp.use_expert_warmup,
        use_expert_guidance=exp.use_expert_guidance,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        use_critic_blend=exp.use_critic_blend,
        critic_warmup_frac=exp.critic_warmup_frac,
        use_box=exp.use_box,
        num_critics=exp.num_critics,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
    )

    _agent = SAC(
        env_id=env,
        env_params=env_params,
        expert_policy=agent_expert_policy,
        eval_expert_policy=expert_policy,
        actor_architecture=architecture,
        critic_architecture=architecture,
        num_critics=exp.num_critics,
        use_expert_guidance=exp.use_expert_guidance,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        mc_pretrain_n_mc_steps=exp.mc_pretrain_n_mc_steps,
        mc_pretrain_n_mc_episodes=exp.mc_pretrain_n_mc_episodes,
        mc_pretrain_n_steps=exp.mc_pretrain_n_steps,
        use_critic_blend=exp.use_critic_blend,
        critic_warmup_frac=exp.critic_warmup_frac,
        use_box=exp.use_box,
        policy_update_start=exp.policy_update_start,
        alpha_update_start=exp.alpha_update_start,
        use_train_frac=exp.use_expert_warmup,
        # Tuned hyperparameters from yml
        **hp,
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
        print(
            f"[{exp.name}] {n_seeds} seeds done in {elapsed:.1f}s "
            f"({elapsed/n_seeds:.1f}s/seed)"
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help=(
            "Disable W&B logging (TensorBoard still written). "
            "Saves ~1-3s × n_seeds of blocking network calls before training starts. "
            "Use for large seed sweeps where you only need post-hoc plots."
        ),
    )
    args = parser.parse_args()

    # --- Shared config ---
    project_name = "tests_SAC_plane_phase1"
    n_timesteps = int(1e6)
    n_seeds = 25
    num_episode_test = 25
    log_frequency = 10_000
    sweep_mode = False
    use_wandb = not args.no_wandb

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
            use_wandb=use_wandb,
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
                use_wandb=use_wandb,
            )
