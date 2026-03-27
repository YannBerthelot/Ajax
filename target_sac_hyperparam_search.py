"""
target_sac_hyperparam_search.py

Bayesian hyperparameter optimisation for SAC + AWBC + MC-pretrain on the Plane
environment.  Tunes both the base SAC hyperparameters and the AWBC / MC-pretrain
specific ones jointly.

Optimisation method
─────────────────────────────────────────────────────────────────────────────
Same TPE-based approach as sac_hyperparam_search.py.  See that file for the
full rationale.

Search space (new vs. vanilla SAC search)
─────────────────────────────────────────────────────────────────────────────
  Base SAC     : actor_lr, critic_lr, alpha_lr, gamma, tau, alpha_init,
                 target_entropy_per_dim, batch_size, arch_width, max_grad_norm
  AWBC-specific: num_critics, num_critic_updates, expert_mix_fraction
  MC-pretrain  : mc_pretrain_n_mc_episodes, mc_pretrain_n_steps

Fixed (structural choices confirmed by ablation study):
  use_expert_guidance=True, use_mc_critic_pretrain=True,
  use_expert_warmup=True, expert_buffer_n_steps=20_000,
  awbc_normalize=True, awbc_use_relu=True

Multi-GPU architecture
─────────────────────────────────────────────────────────────────────────────
• One Optuna study, persisted in SQLite (target_hp_results/optuna.db).
• Master process owns the GPU pool and the study.
• Each trial is a subprocess with CUDA_VISIBLE_DEVICES set by the master.
• On restart: stale RUNNING trials are marked FAILED so TPE can re-explore.

Usage
─────────────────────────────────────────────────────────────────────────────
  python target_sac_hyperparam_search.py                     # all GPUs, 40 trials
  python target_sac_hyperparam_search.py --gpus 0 1          # restrict to GPUs 0-1
  python target_sac_hyperparam_search.py --trials 60         # more trials
  python target_sac_hyperparam_search.py --results           # ranked table
  python target_sac_hyperparam_search.py --dry-run           # show search space only
  python target_sac_hyperparam_search.py --phase2 --top-k 5  # confirm top-5 configs

Do NOT run via gpu_launcher — this script manages GPUs internally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("target_hp_results")
DB_PATH = RESULTS_DIR / "optuna.db"
STUDY_NAME = "target_sac_hp_search_plane"
EVAL_TAG = "Eval/episodic_mean_reward"

# Phase 1: short budget for fast exploration
PHASE1_TIMESTEPS = int(3e5)
PHASE1_SEEDS = 5

# Phase 2: full budget for statistical confirmation
PHASE2_TIMESTEPS = int(1e6)
PHASE2_SEEDS = 15
# Average eval metric over the last N steps for Phase 2 (mirrors final experiment eval)
PHASE2_EVAL_WINDOW = 200_000

LOG_FREQUENCY = 10_000
NUM_EPISODE_TEST = 25
PROJECT_NAME = "target_sac_hyperparam_search_plane"

# Fixed structural choices (confirmed by ablation study)
EXPERT_BUFFER_N_STEPS = 20_000
MC_PRETRAIN_N_MC_STEPS = 10_000   # total env steps for MC rollout collection


# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class HPParams:
    """One set of SAC + AWBC + MC-pretrain hyperparameters — JSON serialisable."""
    trial_number: int

    # Base SAC
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    gamma: float
    tau: float
    alpha_init: float
    target_entropy_per_dim: float
    batch_size: int
    arch_width: int                 # 256 or 512; expands to full arch tuple
    max_grad_norm: Optional[float]  # None = no clipping

    # AWBC-specific
    num_critics: int
    num_critic_updates: int
    expert_mix_fraction: float

    # MC-pretrain
    mc_pretrain_n_mc_episodes: int
    mc_pretrain_n_steps: int

    @property
    def actor_architecture(self) -> tuple:
        w = str(self.arch_width)
        return (w, "relu", w, "relu")

    @property
    def critic_architecture(self) -> tuple:
        w = str(self.arch_width)
        return (w, "relu", w, "relu")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> HPParams:
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _params_path(n: int) -> Path:
    return RESULTS_DIR / f"trial_{n:03d}_params.json"


def _result_path(n: int, phase2: bool = False) -> Path:
    suffix = "_phase2" if phase2 else ""
    return RESULTS_DIR / f"trial_{n:03d}{suffix}_result.json"


def _log_path(n: int, phase2: bool = False) -> Path:
    suffix = "_phase2" if phase2 else ""
    return RESULTS_DIR / f"trial_{n:03d}{suffix}.log"


def _tb_folder(n: int, phase2: bool = False) -> str:
    suffix = "_phase2" if phase2 else ""
    return str(RESULTS_DIR / f"trial_{n:03d}{suffix}_tb")


# ---------------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------------


def suggest_params(trial) -> HPParams:
    """Sample one configuration from the search space."""
    # Base SAC hyperparameters
    actor_lr = trial.suggest_float("actor_lr", 1e-4, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-4, 1e-3, log=True)
    alpha_lr  = trial.suggest_float("alpha_lr",  1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    tau = trial.suggest_float("tau", 5e-4, 5e-2, log=True)
    alpha_init = trial.suggest_float("alpha_init", 0.05, 2.0, log=True)
    target_entropy_per_dim = trial.suggest_float("target_entropy_per_dim", -4.0, -0.2)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    arch_width = trial.suggest_categorical("arch_width", [256, 512])
    # None is not hashable in Optuna categoricals; use 0 to represent "no clip"
    grad_clip = trial.suggest_categorical("grad_clip", [0, 0.5, 1.0])
    max_grad_norm = None if grad_clip == 0 else float(grad_clip)

    # AWBC-specific hyperparameters
    num_critics = trial.suggest_categorical("num_critics", [2, 4])
    num_critic_updates = trial.suggest_categorical("num_critic_updates", [1, 2, 4, 8])
    expert_mix_fraction = trial.suggest_categorical(
        "expert_mix_fraction", [0.0, 0.1, 0.3, 0.5]
    )

    # MC-pretrain hyperparameters
    mc_pretrain_n_mc_episodes = trial.suggest_categorical(
        "mc_pretrain_n_mc_episodes", [50, 100, 500]
    )
    mc_pretrain_n_steps = trial.suggest_categorical(
        "mc_pretrain_n_steps", [1_000, 5_000, 10_000]
    )

    return HPParams(
        trial_number=trial.number,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        alpha_init=alpha_init,
        target_entropy_per_dim=target_entropy_per_dim,
        batch_size=batch_size,
        arch_width=arch_width,
        max_grad_norm=max_grad_norm,
        num_critics=num_critics,
        num_critic_updates=num_critic_updates,
        expert_mix_fraction=float(expert_mix_fraction),
        mc_pretrain_n_mc_episodes=mc_pretrain_n_mc_episodes,
        mc_pretrain_n_steps=mc_pretrain_n_steps,
    )


# ---------------------------------------------------------------------------
# Metric extraction from TensorBoard
# ---------------------------------------------------------------------------


def read_final_metric(
    tb_folder: str, run_ids: list, window_steps: Optional[int] = None
) -> float:
    """
    Compute the eval metric averaged across all seed run_ids.

    window_steps=None  → use the last logged value only (Phase 1, fast).
    window_steps=N     → average over the last N timesteps (Phase 2, robust).

    Returns -inf if no data is found (trial treated as failed by Optuna).
    """
    from ajax.logging.wandb_logging import load_scalars_from_tfevents

    seed_scores = []
    for run_id in run_ids:
        log_dir = os.path.join(tb_folder, "tensorboard", run_id)
        if not os.path.exists(log_dir):
            continue
        try:
            scalars = load_scalars_from_tfevents(log_dir)
        except Exception:
            continue
        if EVAL_TAG not in scalars or not scalars[EVAL_TAG]:
            continue
        entries = scalars[EVAL_TAG]  # sorted list of (step, value)
        if window_steps is None:
            seed_scores.append(entries[-1][1])
        else:
            max_step = entries[-1][0]
            window = [v for step, v in entries if step >= max_step - window_steps]
            if window:
                seed_scores.append(sum(window) / len(window))

    if not seed_scores:
        return float("-inf")
    return float(sum(seed_scores) / len(seed_scores))


# ---------------------------------------------------------------------------
# Worker — entry point for each trial subprocess
# ---------------------------------------------------------------------------


def run_worker(
    params: HPParams,
    n_timesteps: int,
    n_seeds: int,
    use_wandb: bool,
    phase2: bool = False,
) -> None:
    """
    Runs inside a subprocess.  JAX (and GPU allocation) is initialised here,
    not in the master, so each worker gets a clean CUDA context.

    CUDA_VISIBLE_DEVICES is set in the environment by the master before
    launching this subprocess.
    """
    import dill as pickle
    import jax.numpy as jnp
    from target_gym import Plane, PlaneParams

    from ajax import SAC
    from ajax.early_termination_wrapper import EarlyTerminationWrapper
    from ajax.logging.wandb_logging import LoggingConfig
    from ajax.plane.plane_exps_utils import get_mode
    from ajax.stable_utils import get_expert_policy

    tb = _tb_folder(params.trial_number, phase2=phase2)

    env = Plane()
    env_params = PlaneParams(
        target_altitude_range=(3_000, 8_000),
        initial_altitude_range=(3_000, 8_000),
        max_steps_in_episode=10_000,
    )

    if os.path.exists("expert_policy.pkl"):
        with open("expert_policy.pkl", "rb") as f:
            expert_policy = pickle.load(f)
    else:
        expert_policy = get_expert_policy(env, env_params)
        with open("expert_policy.pkl", "wb") as f:
            pickle.dump(expert_policy, f)

    # Validate expert policy
    for test_alt in [3_000, 8_000]:
        test_obs = jnp.array([0, 0, 0, 0, 0, 0, test_alt], dtype=jnp.float32)
        if jnp.isnan(expert_policy(test_obs)).any():
            raise ValueError(f"Expert policy produces NaN at altitude {test_alt}")

    from target_gym.plane.env import PlaneState
    def trunc_condition(state: PlaneState, p: PlaneParams) -> bool:
        return jnp.abs(state.target_altitude - state.z) < 500.0

    wrapped_env = EarlyTerminationWrapper(
        env, trunc_condition=trunc_condition, expert_policy=expert_policy,
    )

    phase_tag = "_phase2" if phase2 else ""
    logging_config = LoggingConfig(
        project_name=PROJECT_NAME,
        run_name=f"trial_{params.trial_number:03d}{phase_tag}",
        config={
            "debug": False,
            "log_frequency": LOG_FREQUENCY,
            # Base SAC
            "actor_lr": params.actor_lr,
            "critic_lr": params.critic_lr,
            "alpha_lr": params.alpha_lr,
            "gamma": params.gamma,
            "tau": params.tau,
            "alpha_init": params.alpha_init,
            "target_entropy_per_dim": params.target_entropy_per_dim,
            "batch_size": params.batch_size,
            "arch_width": params.arch_width,
            "max_grad_norm": params.max_grad_norm,
            # AWBC
            "num_critics": params.num_critics,
            "num_critic_updates": params.num_critic_updates,
            "expert_mix_fraction": params.expert_mix_fraction,
            "expert_buffer_n_steps": EXPERT_BUFFER_N_STEPS,
            # MC pretrain
            "mc_pretrain_n_mc_episodes": params.mc_pretrain_n_mc_episodes,
            "mc_pretrain_n_steps": params.mc_pretrain_n_steps,
            "mc_pretrain_n_mc_steps": MC_PRETRAIN_N_MC_STEPS,
            "phase": 2 if phase2 else 1,
        },
        log_frequency=LOG_FREQUENCY,
        horizon=10_000,
        folder=tb,
        use_tensorboard=True,
        use_wandb=use_wandb,
        sweep=False,
    )

    agent = SAC(
        env_id=wrapped_env,
        env_params=env_params,
        action_scale=1.0,
        expert_policy=expert_policy,
        eval_expert_policy=expert_policy,
        actor_learning_rate=params.actor_lr,
        critic_learning_rate=params.critic_lr,
        alpha_learning_rate=params.alpha_lr,
        gamma=params.gamma,
        tau=params.tau,
        batch_size=params.batch_size,
        alpha_init=params.alpha_init,
        target_entropy_per_dim=params.target_entropy_per_dim,
        max_grad_norm=params.max_grad_norm,
        actor_architecture=params.actor_architecture,
        critic_architecture=params.critic_architecture,
        residual=False,
        fixed_alpha=False,
        num_critics=params.num_critics,
        # AWBC
        use_expert_guidance=True,
        num_critic_updates=params.num_critic_updates,
        expert_buffer_n_steps=EXPERT_BUFFER_N_STEPS,
        expert_mix_fraction=params.expert_mix_fraction,
        awbc_normalize=True,
        awbc_use_relu=True,
        fixed_awbc_lambda=None,
        detach_obs_aug_action=False,
        use_train_frac=True,
        # MC pretrain
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=MC_PRETRAIN_N_MC_STEPS,
        mc_pretrain_n_mc_episodes=params.mc_pretrain_n_mc_episodes,
        mc_pretrain_n_steps=params.mc_pretrain_n_steps,
        # Disabled alternatives
        use_bellman_critic_pretrain=False,
        value_constraint_coef=0.0,
        augment_obs_with_expert_action=False,
        early_termination_condition=trunc_condition,
        box_threshold=500.0,
    )

    mode = get_mode()
    print(
        f"[trial {params.trial_number:03d}] device={mode}  "
        f"actor_lr={params.actor_lr:.1e}  critic_lr={params.critic_lr:.1e}  "
        f"alpha_lr={params.alpha_lr:.1e}  γ={params.gamma:.4f}  τ={params.tau:.1e}  "
        f"tep={params.target_entropy_per_dim:.2f}  α₀={params.alpha_init:.2f}  "
        f"batch={params.batch_size}  arch={params.arch_width}  "
        f"clip={params.max_grad_norm}  "
        f"critics={params.num_critics}×{params.num_critic_updates}  "
        f"mix={params.expert_mix_fraction}  "
        f"mc_ep={params.mc_pretrain_n_mc_episodes}  mc_steps={params.mc_pretrain_n_steps}"
    )

    t0 = time.time()
    if mode == "CPU":
        from tqdm import tqdm
        for seed in tqdm(range(n_seeds), desc=f"trial_{params.trial_number:03d}"):
            agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=NUM_EPISODE_TEST,
            )
    else:
        agent.train(
            seed=list(range(n_seeds)),
            logging_config=logging_config,
            n_timesteps=n_timesteps,
            num_episode_test=NUM_EPISODE_TEST,
        )

    elapsed = time.time() - t0
    metric = read_final_metric(
        tb, agent.run_ids,
        window_steps=PHASE2_EVAL_WINDOW if phase2 else None,
    )
    print(f"[trial {params.trial_number:03d}] done  metric={metric:.2f}  "
          f"elapsed={elapsed:.0f}s")

    result = {
        "trial_number": params.trial_number,
        "metric": metric,
        "elapsed_s": elapsed,
        "run_ids": list(agent.run_ids),
        "tb_folder": tb,
        "phase": 2 if phase2 else 1,
    }
    with open(_result_path(params.trial_number, phase2=phase2), "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Master — manages the Optuna study and GPU pool
# ---------------------------------------------------------------------------


def _get_gpu_ids(requested: Optional[List[int]] = None) -> List[int]:
    if requested:
        return requested
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed — is CUDA available?")
    return [int(ln.strip()) for ln in result.stdout.strip().splitlines() if ln.strip()]


def _create_or_load_study():
    import optuna
    from optuna.samplers import TPESampler

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(
        n_startup_trials=10,    # first 10 trials are purely random (cold start)
        seed=42,
        constant_liar=True,     # parallel-safe: pretend in-flight trials return
                                # the current best so next suggestions aren't clones
        multivariate=True,      # model cross-parameter correlations
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    return study


def _fix_stale_running_trials(study) -> int:
    import optuna
    fixed = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t, state=optuna.trial.TrialState.FAIL)
            fixed += 1
    if fixed:
        print(f"[master] Marked {fixed} stale RUNNING trial(s) as FAILED.")
    return fixed


@dataclass
class RunningJob:
    trial_number: int
    gpu_id: int
    process: subprocess.Popen
    optuna_trial: object  # optuna.Trial
    log_file: object


def _spawn_worker(
    params: HPParams,
    gpu_id: int,
    n_timesteps: int,
    n_seeds: int,
    use_wandb: bool,
    phase2: bool = False,
) -> subprocess.Popen:
    """Launch one trial as a subprocess on gpu_id."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("JAX_PLATFORMS", None)

    cmd = [
        sys.executable, __file__,
        "--worker",
        "--trial-number", str(params.trial_number),
        "--n-timesteps", str(n_timesteps),
        "--n-seeds", str(n_seeds),
    ]
    if phase2:
        cmd.append("--phase2-worker")
    if not use_wandb:
        cmd.append("--no-wandb")

    log = open(_log_path(params.trial_number, phase2=phase2), "w")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log


def run_master(
    gpu_ids: List[int],
    n_trials: int,
    n_timesteps: int,
    n_seeds: int,
    poll_interval: int,
    use_wandb: bool,
) -> None:
    import optuna

    study = _create_or_load_study()
    _fix_stale_running_trials(study)

    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    pending = n_trials - n_complete
    if pending <= 0:
        print(f"All {n_trials} trials already complete.  Run --results to view.")
        return

    print(f"\nPhase 1 — exploration: {pending} trials remaining  |  GPUs: {gpu_ids}")
    print(f"Budget: {n_timesteps:,} steps × {n_seeds} seeds per trial")
    print(f"Study DB: {DB_PATH}   (kill/relaunch safely at any time)\n")

    running: Dict[int, Optional[RunningJob]] = {g: None for g in gpu_ids}

    from tqdm import tqdm
    progress = tqdm(total=n_trials, initial=n_complete, desc="Phase 1 trials", unit="trial")

    try:
        while True:
            # ── Collect finished workers ──────────────────────────────────
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is None:
                    continue

                job.log_file.close()
                rp = _result_path(job.trial_number)

                if ret == 0 and rp.exists():
                    with open(rp) as f:
                        metric = json.load(f)["metric"]
                    study.tell(job.optuna_trial, metric)
                    n_done = sum(
                        1 for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    )
                    best = study.best_trial
                    progress.update(1)
                    progress.set_postfix({"best": f"{best.value:.2f}", "trial": best.number})
                    tqdm.write(f"  [GPU {gpu_id}] ✓ trial {job.trial_number:03d}  "
                               f"score={metric:.2f}  ({n_done}/{n_trials} done)")
                else:
                    study.tell(job.optuna_trial, state=optuna.trial.TrialState.FAIL)
                    tqdm.write(f"  [GPU {gpu_id}] ✗ trial {job.trial_number:03d}  "
                               f"exit={ret} — marked FAILED  (log: "
                               f"{_log_path(job.trial_number)})")

                running[gpu_id] = None

            # ── Check completion ──────────────────────────────────────────
            n_done = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            if n_done >= n_trials and all(j is None for j in running.values()):
                break

            # ── Fill free GPU slots ───────────────────────────────────────
            for gpu_id in gpu_ids:
                if running[gpu_id] is not None:
                    continue

                in_flight = sum(1 for j in running.values() if j is not None)
                n_done = sum(
                    1 for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                )
                if n_done + in_flight >= n_trials:
                    break

                optuna_trial = study.ask()
                params = suggest_params(optuna_trial)
                params.trial_number = optuna_trial.number
                params.to_json(_params_path(params.trial_number))

                proc, log = _spawn_worker(
                    params=params,
                    gpu_id=gpu_id,
                    n_timesteps=n_timesteps,
                    n_seeds=n_seeds,
                    use_wandb=use_wandb,
                )
                running[gpu_id] = RunningJob(
                    trial_number=params.trial_number,
                    gpu_id=gpu_id,
                    process=proc,
                    optuna_trial=optuna_trial,
                    log_file=log,
                )
                print(
                    f"  [GPU {gpu_id}] → trial {params.trial_number:03d}  "
                    f"actor_lr={params.actor_lr:.1e}  critic_lr={params.critic_lr:.1e}  "
                    f"alpha_lr={params.alpha_lr:.1e}  γ={params.gamma:.4f}  "
                    f"τ={params.tau:.1e}  tep={params.target_entropy_per_dim:.2f}  "
                    f"α₀={params.alpha_init:.2f}  batch={params.batch_size}  "
                    f"arch={params.arch_width}  clip={params.max_grad_norm}  "
                    f"critics={params.num_critics}×{params.num_critic_updates}  "
                    f"mix={params.expert_mix_fraction}  "
                    f"mc_ep={params.mc_pretrain_n_mc_episodes}  "
                    f"mc_steps={params.mc_pretrain_n_steps}"
                )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                os.killpg(os.getpgid(job.process.pid), 15)
                job.log_file.close()
                study.tell(job.optuna_trial, state=optuna.trial.TrialState.FAIL)
                tqdm.write(f"  [GPU {gpu_id}] terminated trial {job.trial_number:03d}")
        progress.close()
        sys.exit(1)

    progress.close()
    print_results(study)


# ---------------------------------------------------------------------------
# Phase 2: confirm top-K at full budget
# ---------------------------------------------------------------------------


def run_phase2(top_k: int, gpu_ids: List[int], use_wandb: bool) -> None:
    """Run top-K Phase 1 configs at full budget in parallel across GPUs."""
    import optuna
    from tqdm import tqdm

    study = _create_or_load_study()
    complete = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )
    if not complete:
        print("No completed Phase 1 trials found.  Run Phase 1 first.")
        return

    top = complete[:top_k]
    print(f"\nPhase 2 — confirmation: top-{len(top)} configs × "
          f"{PHASE2_TIMESTEPS:,} steps × {PHASE2_SEEDS} seeds  |  GPUs: {gpu_ids}\n")

    queue: deque = deque()
    for rank, trial in enumerate(top, 1):
        pp = _params_path(trial.number)
        if not pp.exists():
            tqdm.write(f"  [skip] trial {trial.number:03d} — params file missing")
            continue
        rp = _result_path(trial.number, phase2=True)
        if rp.exists():
            with open(rp) as f:
                cached = json.load(f)
            tqdm.write(f"  [{rank}/{len(top)}] trial {trial.number:03d}  "
                       f"Phase1={trial.value:.2f}  Phase2={cached['metric']:.2f}  (cached)")
            continue
        queue.append((rank, trial))

    if not queue:
        print("All Phase 2 trials already cached.")
        _print_phase2_summary(top)
        return

    running: Dict[int, Optional[RunningJob]] = {g: None for g in gpu_ids}
    progress = tqdm(total=len(top), initial=len(top) - len(queue),
                    desc="Phase 2 trials", unit="trial")

    try:
        while queue or any(j is not None for j in running.values()):
            # ── Collect finished workers ──────────────────────────────────
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is None:
                    continue
                job.log_file.close()
                rp = _result_path(job.trial_number, phase2=True)
                if ret == 0 and rp.exists():
                    with open(rp) as f:
                        r = json.load(f)
                    progress.update(1)
                    progress.set_postfix({"last": f"{r['metric']:.2f}", "trial": job.trial_number})
                    tqdm.write(f"  [GPU {gpu_id}] ✓ trial {job.trial_number:03d}  "
                               f"Phase2={r['metric']:.2f}  ({r['elapsed_s']:.0f}s)")
                else:
                    tqdm.write(f"  [GPU {gpu_id}] ✗ trial {job.trial_number:03d}  "
                               f"exit={ret} — see {_log_path(job.trial_number, phase2=True)}")
                running[gpu_id] = None

            # ── Fill free GPU slots ───────────────────────────────────────
            for gpu_id in gpu_ids:
                if running[gpu_id] is not None or not queue:
                    continue
                rank, trial = queue.popleft()
                params = HPParams.from_json(_params_path(trial.number))
                tqdm.write(f"  [GPU {gpu_id}] → trial {trial.number:03d}  "
                           f"Phase1={trial.value:.2f}")
                proc, log = _spawn_worker(
                    params=params,
                    gpu_id=gpu_id,
                    n_timesteps=PHASE2_TIMESTEPS,
                    n_seeds=PHASE2_SEEDS,
                    use_wandb=use_wandb,
                    phase2=True,
                )
                running[gpu_id] = RunningJob(
                    trial_number=trial.number,
                    gpu_id=gpu_id,
                    process=proc,
                    optuna_trial=trial,
                    log_file=log,
                )

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating Phase 2 workers...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                os.killpg(os.getpgid(job.process.pid), 15)
                job.log_file.close()
                tqdm.write(f"  [GPU {gpu_id}] terminated trial {job.trial_number:03d}")
        progress.close()
        sys.exit(1)

    progress.close()
    _print_phase2_summary(top)


def _print_phase2_summary(top) -> None:
    """Print a ranked table of Phase 2 results."""
    print(f"\n{'='*60}")
    print("Phase 2 summary")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'Trial':>5}  {'Phase1':>8}  {'Phase2':>8}")
    print("-" * 60)
    results = []
    for rank, trial in enumerate(top, 1):
        rp = _result_path(trial.number, phase2=True)
        if rp.exists():
            with open(rp) as f:
                r = json.load(f)
            results.append((r["metric"], trial.number, trial.value))
        else:
            results.append((float("-inf"), trial.number, trial.value))
    results.sort(reverse=True)
    for rank, (p2, tnum, p1) in enumerate(results, 1):
        p2_s = f"{p2:.2f}" if p2 != float("-inf") else "FAILED"
        print(f"{rank:>4}  {tnum:>5}  {p1:>8.2f}  {p2_s:>8}")
    best = next((r for r in results if r[0] != float("-inf")), None)
    if best:
        print(f"\nBest Phase 2: trial #{best[1]}  score={best[0]:.2f}")
        _save_best_to_yaml(best[1])


def _save_best_to_yaml(trial_number: int) -> None:
    """Write the best trial's hyperparameters into hyperparams/ajax_sac_target.yml."""
    import yaml

    pp = _params_path(trial_number)
    if not pp.exists():
        print(f"[save] params file missing for trial {trial_number:03d}, skipping YAML save.")
        return

    params = HPParams.from_json(pp)
    yaml_path = Path("hyperparams/ajax_sac_target.yml")

    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg["Plane"] = {
        # Base SAC
        "actor_learning_rate": params.actor_lr,
        "critic_learning_rate": params.critic_lr,
        "alpha_learning_rate": params.alpha_lr,
        "gamma": params.gamma,
        "tau": params.tau,
        "alpha_init": params.alpha_init,
        "target_entropy_per_dim": params.target_entropy_per_dim,
        "batch_size": params.batch_size,
        "arch_width": params.arch_width,
        "max_grad_norm": params.max_grad_norm,
        # AWBC
        "num_critics": params.num_critics,
        "num_critic_updates": params.num_critic_updates,
        "expert_mix_fraction": params.expert_mix_fraction,
        "expert_buffer_n_steps": EXPERT_BUFFER_N_STEPS,
        # MC pretrain
        "mc_pretrain_n_mc_episodes": params.mc_pretrain_n_mc_episodes,
        "mc_pretrain_n_steps": params.mc_pretrain_n_steps,
        "mc_pretrain_n_mc_steps": MC_PRETRAIN_N_MC_STEPS,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[save] Best hyperparameters written to {yaml_path}  (trial #{trial_number})")


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(study=None) -> None:
    import optuna

    if study is None:
        study = _create_or_load_study()

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed   = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if not complete:
        print("No completed trials yet.")
        return

    ranked = sorted(complete, key=lambda t: t.value, reverse=True)

    print(f"\n{'='*130}")
    print(f"Phase 1 results: {len(complete)} completed, {len(failed)} failed")
    print(f"{'='*130}")
    hdr = (f"{'Rank':>4}  {'#':>5}  {'Score':>8}  "
           f"{'actor_lr':>8}  {'critic_lr':>9}  {'alpha_lr':>8}  "
           f"{'γ':>6}  {'τ':>8}  {'tep':>5}  {'α₀':>5}  "
           f"{'batch':>5}  {'arch':>4}  {'clip':>5}  "
           f"{'crit':>4}  {'upd':>3}  {'mix':>4}  "
           f"{'mc_ep':>5}  {'mc_st':>5}")
    print(hdr)
    print("-" * 130)
    for rank, t in enumerate(ranked, 1):
        p = t.params
        clip = p.get("grad_clip", 0)
        clip_s = "None" if clip == 0 else f"{float(clip):.1f}"
        print(
            f"{rank:>4}  {t.number:>5}  {t.value:>8.2f}  "
            f"{p.get('actor_lr', 0):>8.2e}  "
            f"{p.get('critic_lr', 0):>9.2e}  "
            f"{p.get('alpha_lr', 0):>8.2e}  "
            f"{p.get('gamma', 0):>6.4f}  "
            f"{p.get('tau', 0):>8.2e}  "
            f"{p.get('target_entropy_per_dim', 0):>5.2f}  "
            f"{p.get('alpha_init', 0):>5.2f}  "
            f"{int(p.get('batch_size', 0)):>5}  "
            f"{int(p.get('arch_width', 0)):>4}  "
            f"{clip_s:>5}  "
            f"{int(p.get('num_critics', 0)):>4}  "
            f"{int(p.get('num_critic_updates', 0)):>3}  "
            f"{p.get('expert_mix_fraction', 0):>4.2f}  "
            f"{int(p.get('mc_pretrain_n_mc_episodes', 0)):>5}  "
            f"{int(p.get('mc_pretrain_n_steps', 0)):>5}"
        )

    best = ranked[0]
    print(f"\nBest: trial #{best.number}  score={best.value:.2f}")
    print(json.dumps(best.params, indent=4))
    print(f"\nTo confirm top-5 at full budget:")
    print(f"  python target_sac_hyperparam_search.py --phase2 --top-k 5")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SAC+AWBC+MC-pretrain on the Plane environment."
    )

    # Worker flags (hidden — called internally by master)
    parser.add_argument("--worker",        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase2-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial-number",  type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-timesteps",   type=int, default=PHASE1_TIMESTEPS, help=argparse.SUPPRESS)
    parser.add_argument("--n-seeds",       type=int, default=PHASE1_SEEDS, help=argparse.SUPPRESS)

    # Master / user flags
    parser.add_argument("--gpus",    nargs="+", type=int, default=None,
                        help="GPU IDs to use (default: all detected by nvidia-smi).")
    parser.add_argument("--trials",  type=int, default=50,
                        help="Total Phase 1 trials (default: 40).")
    parser.add_argument("--poll",    type=int, default=30,
                        help="Master poll interval in seconds (default: 30).")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--results",  action="store_true",
                        help="Print ranked results table and exit.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show search space info and exit.")
    parser.add_argument("--phase2",   action="store_true",
                        help="Run Phase 2 confirmation for top-K configs.")
    parser.add_argument("--top-k",    type=int, default=5,
                        help="Number of top configs to confirm in Phase 2 (default: 5).")

    args = parser.parse_args()
    use_wandb = not args.no_wandb

    # ── Worker subprocess ─────────────────────────────────────────────────
    if args.worker:
        if args.trial_number is None:
            sys.exit("--trial-number required in --worker mode")
        pp = _params_path(args.trial_number)
        if not pp.exists():
            sys.exit(f"Params file not found: {pp}")
        params = HPParams.from_json(pp)
        run_worker(
            params=params,
            n_timesteps=args.n_timesteps,
            n_seeds=args.n_seeds,
            use_wandb=use_wandb,
            phase2=args.phase2_worker,
        )
        return

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        import optuna
        _study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            direction="maximize",
        )
        print("\nSearch space (10 random samples from TPE cold start):\n")
        hdr = (f"{'#':>2}  {'actor_lr':>8}  {'critic_lr':>9}  {'alpha_lr':>8}  "
               f"{'γ':>6}  {'τ':>8}  {'tep':>5}  {'α₀':>5}  "
               f"{'batch':>5}  {'arch':>4}  {'clip':>5}  "
               f"{'crit':>4}  {'upd':>3}  {'mix':>4}  "
               f"{'mc_ep':>5}  {'mc_st':>5}")
        print(hdr)
        print("-" * 105)
        for i in range(10):
            t = _study.ask()
            p = suggest_params(t)
            clip_s = "None" if p.max_grad_norm is None else f"{p.max_grad_norm:.1f}"
            print(
                f"{i:>2}  {p.actor_lr:>8.2e}  {p.critic_lr:>9.2e}  {p.alpha_lr:>8.2e}  "
                f"{p.gamma:>6.4f}  {p.tau:>8.2e}  "
                f"{p.target_entropy_per_dim:>5.2f}  {p.alpha_init:>5.2f}  "
                f"{p.batch_size:>5}  {p.arch_width:>4}  {clip_s:>5}  "
                f"{p.num_critics:>4}  {p.num_critic_updates:>3}  "
                f"{p.expert_mix_fraction:>4.2f}  "
                f"{p.mc_pretrain_n_mc_episodes:>5}  {p.mc_pretrain_n_steps:>5}"
            )
        print(f"\nFixed: use_expert_guidance=True, use_mc_critic_pretrain=True, "
              f"expert_buffer_n_steps={EXPERT_BUFFER_N_STEPS}, "
              f"awbc_normalize=True, awbc_use_relu=True")
        print(f"\nPhase 1: {args.trials} trials × {PHASE1_SEEDS} seeds "
              f"× {PHASE1_TIMESTEPS:,} steps")
        print(f"Phase 2: top-{args.top_k} × {PHASE2_SEEDS} seeds "
              f"× {PHASE2_TIMESTEPS:,} steps")
        print(f"Study DB: {DB_PATH}")
        return

    # ── Results ───────────────────────────────────────────────────────────
    if args.results:
        print_results()
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if args.phase2:
        gpu_ids = _get_gpu_ids(args.gpus)
        run_phase2(top_k=args.top_k, gpu_ids=gpu_ids, use_wandb=use_wandb)
        return

    # ── Phase 1 master ────────────────────────────────────────────────────
    gpu_ids = _get_gpu_ids(args.gpus)
    print(f"Candidate GPUs: {gpu_ids}")
    run_master(
        gpu_ids=gpu_ids,
        n_trials=args.trials,
        n_timesteps=PHASE1_TIMESTEPS,
        n_seeds=PHASE1_SEEDS,
        poll_interval=args.poll,
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    main()
