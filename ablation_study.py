"""
ablation_study.py — SAC + AWBC / actor-pretrain / shaping ablation suite for Plane

Ordering: most-promising experiments (actor pretrain tier) run first.

Run via gpu_launcher (recommended — waits for free GPUs system-wide):
    python gpu_launcher.py --script ablation_study

Run a single experiment by index (used internally by the launcher):
    python ablation_study.py --exp-index 5

List all experiments:
    python ablation_study.py --list

Manually inject baseline into a W&B project (after sac_baseline has run):
    python ablation_study.py --inject-baseline PROJECT_NAME
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import dill as pickle
import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_xla"))

from target_gym import Plane, PlaneParams
from tqdm import tqdm

from ajax import SAC
from ajax.logging.wandb_logging import (
    merge_and_upload_tensorboard_to_wandb,
    upload_tensorboard_to_wandb,
)
from ajax.plane.plane_exps_utils import (
    get_log_config,
    get_mode,
    get_policy_score,
    load_hyperparams,
)
from ajax.stable_utils import get_expert_policy


# ---------------------------------------------------------------------------
# W&B project names — one project per thematic tier.
# Edit these to match your W&B workspace.
# ---------------------------------------------------------------------------

GROUP_PROJECTS = {
    "actor_pretrain":       "ablation_actor_pretrain_plane",
    "component_isolation":  "ablation_component_isolation_plane",
    "awbc":                 "ablation_awbc_plane",
    "critics":              "ablation_critics_plane",
    "mc_pretrain":          "ablation_mc_pretrain_plane",
    "expert_mix":           "ablation_expert_mix_plane",
}

# Path where baseline run IDs are cached after sac_baseline completes.
BASELINE_CACHE_FILE = os.path.abspath("ablation_baseline_cache.json")
# Path that tracks which projects have already received the baseline.
SEEDED_PROJECTS_FILE = os.path.abspath("ablation_seeded_projects.json")


# ---------------------------------------------------------------------------
# Experiment configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    One experiment in the ablation suite.
    All expert-guidance flags default to off — safe baseline when not set.
    """
    name: str
    wandb_project: str = ""  # empty → use the global project passed at runtime

    # Expert warmup (seed replay buffer with expert rollouts)
    use_expert_warmup: bool = False

    # Expert guidance mechanisms
    use_expert_guidance: bool = False       # AWBC actor regularisation
    use_mc_critic_pretrain: bool = False    # MC-return critic pretraining
    use_actor_pretrain: bool = False        # Value-weighted BC actor pretrain (requires MC)
    use_critic_blend: bool = False          # Blended Bellman target (replaces potential shaping)
    critic_warmup_frac: float = 0.15        # Fraction of training over which blend decays 1→0
    use_box: bool = False                   # Value-threshold curriculum box

    # Update start thresholds.
    # 2k for expert runs (actor pretrained → no need for long critic warm-up).
    # sac_baseline overrides to 10k (no expert → critic needs time to stabilise).
    policy_update_start: int = 5_000
    alpha_update_start: int = 5_000

    # AWBC parameters
    num_critic_updates: int = 1
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1

    # AWBC mechanism ablation flags
    awbc_normalize: bool = True
    awbc_use_relu: bool = True
    fixed_awbc_lambda: Optional[float] = None

    # MC / actor pretrain parameters
    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_mc_episodes: int = 1000
    mc_pretrain_n_steps: int = 5_000
    actor_pretrain_n_steps: int = 5_000

    # Standard SAC hyperparameters (overridable per experiment)
    num_critics: int = 2


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_experiments() -> List[ExperimentConfig]:
    """
    Ablation suite ordered by expected impact (highest first).

    Tier 0 — Actor pretrain (new, most promising):
        Exercises the full MC-pretrain → value-weighted BC → AWBC / shaping / box stack.
    Tier 1 — Component isolation: establishes contribution of each legacy mechanism.
    Tier 2 — AWBC mechanism: normalisation, ReLU gating, fixed λ.
    Tier 3 — Critic count and update ratio.
    Tier 4 — MC pretrain hyperparameters.
    Tier 5 — Expert mix fraction.
    """
    P = GROUP_PROJECTS      # shorthand

    MC = dict(
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_steps=5_000,
    )
    BUF = dict(expert_buffer_n_steps=20_000, expert_mix_fraction=0.1)
    AWBC = dict(use_expert_guidance=True, num_critic_updates=4)
    AP = dict(use_actor_pretrain=True, actor_pretrain_n_steps=5_000)

    exps = []

    # ==================================================================
    # sac_baseline — anchor, runs first so the cache is written before
    # any other experiment needs to inject it into a new project.
    # index 0
    # ==================================================================

    exps.append(ExperimentConfig(
        name="sac_baseline",
        wandb_project=P["actor_pretrain"],   # lives in the first project
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        policy_update_start=10_000,          # no expert → critic needs longer warm-up
        alpha_update_start=10_000,
    ))

    # ==================================================================
    # Tier 0 — Actor pretrain  (indices 1–8)
    # Most promising: warm-started actor + expert critic + AWBC / shaping / box.
    # All use policy_update_start=2_000 (actor is already pre-trained).
    # ==================================================================

    # Full stack: MC pretrain + actor pretrain + AWBC (most complete expert guidance)
    exps.append(ExperimentConfig(
        name="mc_pretrain_actor_pretrain_awbc",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        **AWBC, **MC, **AP, **BUF,
    ))

    # Actor pretrain alone — isolates warm-start effect without ongoing AWBC
    exps.append(ExperimentConfig(
        name="mc_pretrain_actor_pretrain",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        num_critic_updates=1,
        **MC, **AP, **BUF,
    ))

    # MC + actor pretrain + blended Bellman target (critic_warmup_frac=0.15)
    exps.append(ExperimentConfig(
        name="mc_pretrain_actor_pretrain_blend",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        use_critic_blend=True,
        num_critic_updates=1,
        **MC, **AP, **BUF,
    ))

    # AWBC + blend — tests whether AWBC and blended target are additive
    exps.append(ExperimentConfig(
        name="mc_pretrain_actor_pretrain_awbc_blend",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        use_critic_blend=True,
        **AWBC, **MC, **AP, **BUF,
    ))

    # Actor pretrain + box curriculum
    exps.append(ExperimentConfig(
        name="mc_pretrain_actor_pretrain_box",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        use_box=True,
        num_critic_updates=1,
        **MC, **AP, **BUF,
    ))

    # Actor pretrain n_steps ablation — fewer steps (1k)
    exps.append(ExperimentConfig(
        name="actor_pretrain_n_steps_1k",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        actor_pretrain_n_steps=1_000,
        **AWBC, **MC, **dict(use_actor_pretrain=True), **BUF,
    ))

    # Actor pretrain n_steps ablation — more steps (20k)
    exps.append(ExperimentConfig(
        name="actor_pretrain_n_steps_20k",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        actor_pretrain_n_steps=20_000,
        **AWBC, **MC, **dict(use_actor_pretrain=True), **BUF,
    ))

    # Blend only (no actor pretrain) — isolates blended target contribution
    exps.append(ExperimentConfig(
        name="mc_pretrain_blend",
        wandb_project=P["actor_pretrain"],
        use_expert_warmup=True,
        use_critic_blend=True,
        num_critic_updates=1,
        **MC, **BUF,
    ))

    # ==================================================================
    # Tier 1 — Component isolation  (indices 9–13)
    # ==================================================================

    # KEY DIAGNOSTIC: seeding only — isolates whether expert buffer pre-seeding helps
    exps.append(ExperimentConfig(
        name="sac_expert_seeding",
        wandb_project=P["component_isolation"],
        use_expert_warmup=True,
        expert_buffer_n_steps=20_000,
        expert_mix_fraction=0.0,
    ))

    # AWBC only — no MC pretrain (tests AWBC with an un-pretrained critic)
    exps.append(ExperimentConfig(
        name="awbc_only",
        wandb_project=P["component_isolation"],
        use_expert_warmup=True,
        **AWBC, **BUF,
    ))

    # MC pretrain only — critic init alone, no AWBC at training time
    exps.append(ExperimentConfig(
        name="mc_pretrain_only",
        wandb_project=P["component_isolation"],
        use_expert_warmup=True,
        num_critic_updates=1,
        **MC, **BUF,
    ))

    # MC pretrain + AWBC (legacy anchor E2)
    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc",
        wandb_project=P["component_isolation"],
        use_expert_warmup=True,
        **AWBC, **MC, **BUF,
    ))

    # MC pretrain + Box (legacy anchor E4)
    exps.append(ExperimentConfig(
        name="mc_pretrain_box",
        wandb_project=P["component_isolation"],
        use_expert_warmup=True,
        use_box=True,
        **MC, **BUF,
    ))

    # ==================================================================
    # Tier 2 — AWBC mechanism  (indices 14–17)
    # ==================================================================

    exps.append(ExperimentConfig(
        name="awbc_no_normalize",
        wandb_project=P["awbc"],
        use_expert_warmup=True,
        awbc_normalize=False,
        **AWBC, **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="awbc_no_relu",
        wandb_project=P["awbc"],
        use_expert_warmup=True,
        awbc_use_relu=False,
        **AWBC, **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="awbc_fixed_lambda_0.1",
        wandb_project=P["awbc"],
        use_expert_warmup=True,
        fixed_awbc_lambda=0.1,
        **AWBC, **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="awbc_fixed_lambda_1.0",
        wandb_project=P["awbc"],
        use_expert_warmup=True,
        fixed_awbc_lambda=1.0,
        **AWBC, **MC, **BUF,
    ))

    # ==================================================================
    # Tier 3 — Critic count and update ratio  (indices 18–21)
    # ==================================================================

    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_x1update",
        wandb_project=P["critics"],
        use_expert_warmup=True,
        num_critic_updates=1,
        use_expert_guidance=True,
        **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_x2updates",
        wandb_project=P["critics"],
        use_expert_warmup=True,
        num_critic_updates=2,
        use_expert_guidance=True,
        **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_1critic",
        wandb_project=P["critics"],
        use_expert_warmup=True,
        num_critics=1,
        **AWBC, **MC, **BUF,
    ))

    exps.append(ExperimentConfig(
        name="mc_pretrain_awbc_4critics",
        wandb_project=P["critics"],
        use_expert_warmup=True,
        num_critics=4,
        **AWBC, **MC, **BUF,
    ))

    # ==================================================================
    # Tier 4 — MC pretrain hyperparameters  (indices 22–26)
    # ==================================================================

    for n_ep in [10, 50, 500]:
        exps.append(ExperimentConfig(
            name=f"mc_pretrain_episodes_{n_ep}",
            wandb_project=P["mc_pretrain"],
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_mc_episodes=n_ep,
            mc_pretrain_n_steps=5_000,
            **AWBC, **BUF,
        ))

    for n_steps in [1_000, 20_000]:
        exps.append(ExperimentConfig(
            name=f"mc_pretrain_steps_{n_steps // 1000}k",
            wandb_project=P["mc_pretrain"],
            use_expert_warmup=True,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_mc_episodes=100,
            mc_pretrain_n_steps=n_steps,
            **AWBC, **BUF,
        ))

    # ==================================================================
    # Tier 5 — Expert mix fraction  (indices 27–30)
    # ==================================================================

    exps.append(ExperimentConfig(
        name="expert_mix_0",
        wandb_project=P["expert_mix"],
        use_expert_warmup=True,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        **AWBC, **MC,
    ))

    for frac in [0.3, 0.5]:
        exps.append(ExperimentConfig(
            name=f"expert_mix_{frac}",
            wandb_project=P["expert_mix"],
            use_expert_warmup=True,
            expert_mix_fraction=frac,
            expert_buffer_n_steps=20_000,
            **AWBC, **MC,
        ))

    exps.append(ExperimentConfig(
        name="expert_buffer_pretrain_only",
        wandb_project=P["expert_mix"],
        use_expert_warmup=True,
        expert_buffer_n_steps=20_000,
        expert_mix_fraction=0.0,
        **AWBC, **MC,
    ))

    return exps


# ---------------------------------------------------------------------------
# Baseline sharing helpers
# ---------------------------------------------------------------------------


def _save_baseline_cache(run_ids: list, project: str) -> None:
    """Persist baseline run IDs and the TensorBoard base folder to disk."""
    cache = {
        "run_ids": run_ids,
        "project": project,
        "tb_folder": os.path.abspath("."),
    }
    with open(BASELINE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[baseline] Cached {len(run_ids)} run IDs → {BASELINE_CACHE_FILE}")


def _copy_baseline_to_project(target_project: str) -> None:
    """
    Upload baseline TensorBoard logs as fresh runs in target_project.
    Creates one new W&B run per seed — does NOT resume existing runs.
    """
    if not os.path.exists(BASELINE_CACHE_FILE):
        print(f"[baseline] Cache not found; skipping injection into {target_project}")
        return

    with open(BASELINE_CACHE_FILE) as f:
        cache = json.load(f)

    import wandb as _wandb

    for run_id in cache["run_ids"]:
        log_dir = os.path.join(cache["tb_folder"], "tensorboard", run_id)
        if not os.path.exists(log_dir):
            print(f"[baseline] TB dir missing for {run_id}: {log_dir}")
            continue
        try:
            _wandb.init(
                project=target_project,
                name=f"sac_baseline_{run_id[:8]}",
                config={
                    "experiment": "sac_baseline",
                    "is_baseline_copy": True,
                    "source_project": cache["project"],
                },
            )
            merge_and_upload_tensorboard_to_wandb(log_dir)  # calls wandb.finish()
            print(f"[baseline] Injected {run_id[:8]} → {target_project}")
        except Exception as exc:
            print(f"[baseline] Failed to inject {run_id[:8]} → {target_project}: {exc}")


def ensure_baseline_in_project(target_project: str, baseline_project: str) -> None:
    """
    Upload the baseline to target_project if it hasn't been done yet (this process).

    Uses a JSON file to persist which projects have been seeded so the check
    survives across processes (each gpu_launcher slot is a separate process).
    A small race window exists when two processes try to seed the same project
    simultaneously — the result is a duplicate baseline run, which is harmless.
    """
    # Skip if target is the same project the baseline already lives in
    if target_project == baseline_project:
        return

    seeded: dict = {}
    if os.path.exists(SEEDED_PROJECTS_FILE):
        try:
            with open(SEEDED_PROJECTS_FILE) as f:
                seeded = json.load(f)
        except Exception:
            pass

    if target_project in seeded:
        return

    _copy_baseline_to_project(target_project)

    seeded[target_project] = True
    try:
        with open(SEEDED_PROJECTS_FILE, "w") as f:
            json.dump(seeded, f, indent=2)
    except Exception:
        pass


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


def run_single_experiment(
    exp: ExperimentConfig,
    env,
    env_params,
    expert_policy,
    n_seeds: int,
    n_timesteps: int,
    num_episode_test: int,
    log_frequency: int,
    global_project_name: str,
    sweep_mode: bool,
    use_wandb: bool = True,
):
    """Run one ExperimentConfig to completion. Called per-process by the launcher."""
    effective_project = exp.wandb_project or global_project_name
    baseline_project = GROUP_PROJECTS["actor_pretrain"]

    # Inject baseline into this project before running (skipped when already present
    # or when this IS the baseline, or when W&B is disabled).
    if use_wandb and exp.name != "sac_baseline":
        ensure_baseline_in_project(effective_project, baseline_project)

    mode = get_mode()
    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")

    agent_expert_policy = expert_policy if exp.use_expert_warmup else None

    policy_score = get_policy_score(expert_policy, env, env_params)
    print(
        f"[{exp.name}] project={effective_project}  expert_score={policy_score:.1f}\n"
        f"  warmup={exp.use_expert_warmup}  awbc={exp.use_expert_guidance}  "
        f"mc_pretrain={exp.use_mc_critic_pretrain}  actor_pretrain={exp.use_actor_pretrain}  "
        f"blend={exp.use_critic_blend}  box={exp.use_box}\n"
        f"  critics={exp.num_critics}×{exp.num_critic_updates}  "
        f"normalize={exp.awbc_normalize}  relu={exp.awbc_use_relu}  "
        f"fixed_λ={exp.fixed_awbc_lambda}  mix={exp.expert_mix_fraction}  "
        f"policy_start={exp.policy_update_start}"
    )

    logging_config = get_log_config(
        project_name=effective_project,
        agent_name=exp.name,
        group_name=exp.name,
        log_frequency=log_frequency,
        use_wandb=use_wandb,
        sweep=sweep_mode,
        use_expert_warmup=exp.use_expert_warmup,
        use_expert_guidance=exp.use_expert_guidance,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        use_actor_pretrain=exp.use_actor_pretrain,
        use_critic_blend=exp.use_critic_blend,
        critic_warmup_frac=exp.critic_warmup_frac,
        use_box=exp.use_box,
        num_critics=exp.num_critics,
        num_critic_updates=exp.num_critic_updates,
        expert_buffer_n_steps=exp.expert_buffer_n_steps,
        expert_mix_fraction=exp.expert_mix_fraction,
        awbc_normalize=exp.awbc_normalize,
        awbc_use_relu=exp.awbc_use_relu,
        fixed_awbc_lambda=exp.fixed_awbc_lambda,
        actor_pretrain_n_steps=exp.actor_pretrain_n_steps,
        policy_update_start=exp.policy_update_start,
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
        use_actor_pretrain=exp.use_actor_pretrain,
        actor_pretrain_n_steps=exp.actor_pretrain_n_steps,
        use_critic_blend=exp.use_critic_blend,
        critic_warmup_frac=exp.critic_warmup_frac,
        use_box=exp.use_box,
        policy_update_start=exp.policy_update_start,
        alpha_update_start=exp.alpha_update_start,
        use_train_frac=exp.use_expert_warmup,  # enable train_frac obs for all expert runs
        awbc_normalize=exp.awbc_normalize,
        awbc_use_relu=exp.awbc_use_relu,
        fixed_awbc_lambda=exp.fixed_awbc_lambda,
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

    # After the baseline runs: cache run IDs so other projects can share it.
    if use_wandb and exp.name == "sac_baseline" and getattr(_agent, "run_ids", None):
        _save_baseline_cache(_agent.run_ids, effective_project)


# ---------------------------------------------------------------------------
# Main — supports full sweep, single-experiment, and baseline injection modes
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help=(
            "Disable W&B logging (TensorBoard still written). "
            "Use for large seed sweeps where you only need post-hoc plots."
        ),
    )
    parser.add_argument(
        "--inject-baseline",
        type=str,
        default=None,
        metavar="PROJECT",
        help=(
            "Upload the cached baseline runs to PROJECT and exit. "
            "Run sac_baseline (index 0) first so the cache exists."
        ),
    )
    args = parser.parse_args()

    # --- Shared config ---
    global_project_name = GROUP_PROJECTS["actor_pretrain"]  # fallback for exps without wandb_project
    n_timesteps = int(1e6)
    n_seeds = 25
    num_episode_test = 25
    log_frequency = 10_000
    sweep_mode = False
    use_wandb = not args.no_wandb

    experiments = build_experiments()

    # --inject-baseline mode: upload cached baseline to a specific project and exit
    if args.inject_baseline:
        print(f"Injecting baseline into project: {args.inject_baseline}")
        _copy_baseline_to_project(args.inject_baseline)
        raise SystemExit(0)

    if args.list:
        tiers = {
            "Tier 0 — Actor pretrain (most promising)": slice(0, 9),
            "Tier 1 — Component isolation":             slice(9, 14),
            "Tier 2 — AWBC mechanism":                  slice(14, 18),
            "Tier 3 — Critics and update ratio":        slice(18, 22),
            "Tier 4 — MC pretrain hyperparams":         slice(22, 27),
            "Tier 5 — Expert mix fraction":             slice(27, 31),
        }
        for tier, s in tiers.items():
            print(f"\n{tier}:")
            for i, exp in enumerate(experiments[s], start=s.start):
                proj = exp.wandb_project or global_project_name
                print(f"  [{i:2d}] {exp.name:<45}  project={proj}")
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
            global_project_name=global_project_name,
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
                global_project_name=global_project_name,
                sweep_mode=sweep_mode,
                use_wandb=use_wandb,
            )
