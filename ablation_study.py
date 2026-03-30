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
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import dill as pickle
import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_xla"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
os.environ["WANDB_SILENT"] = "true"

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
from ajax.environments.create import prepare_env
from ajax.environments.utils import get_action_dim
from ajax.stable_utils import get_expert_policy
from ajax.state import EnvironmentConfig
from mc_pretrain_collect import (
    MC_DATA_PATH,
    read_mc_meta,
    collect_and_save_mc_data,
    load_mc_data_for_experiment,
)


# ---------------------------------------------------------------------------
# Single W&B project — all experiments land here.
# Edit to match your W&B workspace.
# ---------------------------------------------------------------------------

WANDB_PROJECT = "ablation_plane_final_clean_2"

# W&B group names — one group per thematic tier (used as the `group` field).
WANDB_GROUPS = {
    "baselines":            "ablation_baselines_plane_debug",
    "best_variants":        "ablation_best_variants_plane_debug",
    "component_isolation":  "ablation_component_isolation_plane_debug",
    "alpha_deferral":       "ablation_alpha_deferral_plane_debug",
    "critic_architecture":  "ablation_critic_architecture_plane_debug",
    "actor_pretrain":       "ablation_actor_pretrain_plane_debug",
    "awbc":                 "ablation_awbc_plane_debug",
    "critics":              "ablation_critics_plane_debug",
    "mc_pretrain":          "ablation_mc_pretrain_plane_debug",
    "expert_mix":           "ablation_expert_mix_plane_debug",
}

# Path where baseline run IDs are cached after sac_baseline completes.
BASELINE_CACHE_FILE = os.path.abspath("ablation_baseline_cache.json")
# Path that tracks which projects have already received the baseline.
SEEDED_PROJECTS_FILE = os.path.abspath("ablation_seeded_projects.json")
# Global run registry: maps run_id → {exp_name, wandb_group, config} for offline plotting.
RUN_REGISTRY_FILE = os.path.abspath("ablation_run_registry.json")


# ---------------------------------------------------------------------------
# Experiment configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str
    wandb_group: str = ""

    use_expert_warmup: bool = False
    use_expert_guidance: bool = False
    use_mc_critic_pretrain: bool = False
    use_online_bc: bool = False
    use_online_critic_light_pretrain: bool = False
    use_critic_blend: bool = False
    critic_warmup_frac: float = 0.3
    use_box: bool = False

    policy_update_start: int = 5_000
    alpha_update_start: int = 5_000

    num_critic_updates: int = 1
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1

    awbc_normalize: bool = True
    awbc_use_relu: bool = True
    fixed_awbc_lambda: Optional[float] = None

    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_mc_episodes: int = 1000
    mc_pretrain_n_steps: int = 5_000

    online_critic_pretrain_steps: int = 100
    online_critic_pretrain_lr_scale: float = 0.1

    alpha_learning_rate_scale: float = 1.0  # fixed: was 0.5 for non-baselines, now 1.0 everywhere
    bc_coef: float = 1.0

    use_expert_guided_exploration: bool = False
    exploration_decay_frac: float = 0.15
    exploration_tau: float = 1.0
    exploration_boltzmann: bool = True       # new: True = adaptive gate, False = fixed epsilon
    fixed_exploration_prob: float = 0.5     # new: used when exploration_boltzmann=False

    use_distance_entropy: bool = False
    target_entropy_far_scale: float = 0.5

    use_mc_correction: bool = False
    variance_threshold: float = 1.0

    use_phi_refresh: bool = False
    phi_refresh_interval: int = 10_000
    phi_refresh_steps: int = 200

    num_critics: int = 2


def build_experiments() -> List[ExperimentConfig]:
    P = WANDB_GROUPS

    MC = dict(
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_mc_episodes=1000,
        mc_pretrain_n_steps=5_000,
    )

    EGE_BASE = dict(
        use_expert_guided_exploration=True,
        # exploration_decay_frac, exploration_tau, exploration_boltzmann omitted
        # — dataclass defaults (0.15, 1.0, True) apply; override per-experiment as needed
        expert_buffer_n_steps=0,            # no pre-population: EGE fills organically
        expert_mix_fraction=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        use_expert_guidance=False,
        num_critic_updates=1,
        alpha_learning_rate_scale=1.0,      # fixed: match baseline, no silent asymmetry
    )

    exps = []

    # ==================================================================
    # Anchors
    # ==================================================================

    # Vanilla SAC — clean baseline, no expert data, no modifications
    exps.append(ExperimentConfig(
        name="sac_baseline",
        wandb_group=P["baselines"],
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        policy_update_start=10_000,
        alpha_update_start=10_000,
        alpha_learning_rate_scale=1.0,
        critic_warmup_frac=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        num_critic_updates=1,
    ))

    # EGE only — primary winning method, main anchor for all ablations
    # exploration_decay_frac=0.15 from EGE_BASE
    exps.append(ExperimentConfig(
        name="ege_only",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        use_distance_entropy=False,
    ))

    # ==================================================================
    # Q1: Does distance entropy add to or hurt EGE?
    # Compare ege_dist_entropy vs ege_only (anchor).
    # From partial data: dist entropy alone did nothing, combined hurt EGE.
    # Run to 1M to confirm.
    # ==================================================================

    # Already run — kept as anchor for completeness
    exps.append(ExperimentConfig(
        name="ege_dist_entropy",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        use_distance_entropy=True,
        target_entropy_far_scale=0.5,
    ))

    # Dist entropy alone — no EGE, no expert buffer, no MC pretrain
    # Note: MC removed because φ* is unused without EGE or BC
    exps.append(ExperimentConfig(
        name="dist_entropy_only",
        wandb_group=P["best_variants"],
        use_mc_critic_pretrain=False,       # fixed: φ* unused without EGE
        use_expert_guided_exploration=False,
        use_distance_entropy=True,
        target_entropy_far_scale=0.5,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        use_expert_guidance=False,
        num_critic_updates=1,
        alpha_learning_rate_scale=1.0,
    ))

    # ==================================================================
    # Q2: Is MC pretrain (φ*) load-bearing for EGE?
    # Without MC, EGE gates on the live critic alone — no stable φ* reference.
    # Compare ege_only_no_mc vs ege_only (anchor).
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_only_no_mc",
        wandb_group=P["best_variants"],
        use_mc_critic_pretrain=False,       # no φ* — EGE uses live critic for value gap
        use_expert_guided_exploration=True,
        exploration_decay_frac=0.15,
        exploration_tau=1.0,
        exploration_boltzmann=True,
        use_distance_entropy=False,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        use_expert_guidance=False,
        num_critic_updates=1,
        alpha_learning_rate_scale=1.0,
    ))

    # ==================================================================
    # Q3: What is the right EGE decay horizon?
    # Ablation on ege_only (no dist entropy — cleaner isolation).
    # Three-way: 0.15 (anchor) vs 0.30 (A5) vs 0.50 (A6).
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_decay_030",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        exploration_decay_frac=0.30,        # override EGE_BASE default of 0.15
        use_distance_entropy=False,
    ))

    exps.append(ExperimentConfig(
        name="ege_decay_050",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        exploration_decay_frac=0.50,
        use_distance_entropy=False,
    ))

    # ==================================================================
    # Q4: Does the adaptive Boltzmann gate matter vs fixed epsilon-greedy?
    # This is the key mechanistic question: is the value gap doing real work,
    # or is EGE just epsilon-greedy data augmentation?
    # If fixed_epsilon matches ege_only, the value gap contributes nothing.
    # If ege_only wins, adaptive gating is the mechanism.
    # Same global decay (0.15), same expert data fraction on average,
    # only the per-state gating logic differs.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_fixed_epsilon",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        exploration_boltzmann=False,        # fixed probability instead of value-gap gate
        fixed_exploration_prob=0.5,         # 50% expert, same global decay applied on top
        use_distance_entropy=False,
    ))

    # ==================================================================
    # Q5: Does periodic φ* refresh improve upon static frozen φ*?
    # EGE tags transitions where expert acted (was_expert_action=True).
    # Every phi_refresh_interval steps, short MC regression on those transitions.
    # No Bellman bootstrapping — only valid expert-action transitions used.
    # Compare ege_phi_refresh vs ege_only (anchor).
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_phi_refresh",
        wandb_group=P["best_variants"],
        **MC, **EGE_BASE,
        use_distance_entropy=False,
        use_phi_refresh=True,
        phi_refresh_interval=10_000,
        phi_refresh_steps=200,
    ))

    return exps


# ---------------------------------------------------------------------------
# Baseline sharing helpers
# ---------------------------------------------------------------------------


def _save_run_configs(exp: ExperimentConfig, run_ids: list) -> None:
    """
    For each run_id:
      1. Write tensorboard/{run_id}/config.json  — per-run, self-contained.
      2. Append an entry to run_registry.json    — global index for plotting.

    Safe to call even if run_ids is empty.  Uses append-then-write so a crash
    mid-run does not corrupt the existing registry.
    """
    if not run_ids:
        return

    config_dict = asdict(exp)
    tb_base = os.path.abspath("tensorboard")

    # --- per-run config files ---
    for run_id in run_ids:
        run_dir = os.path.join(tb_base, run_id)
        os.makedirs(run_dir, exist_ok=True)
        payload = {
            "run_id": run_id,
            "exp_name": exp.name,
            "group": exp.wandb_group,
            "project": WANDB_PROJECT,
            "config": config_dict,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(payload, f, indent=2)

    # --- global registry (read-modify-write, atomic write) ---
    registry: list = []
    if os.path.exists(RUN_REGISTRY_FILE):
        try:
            with open(RUN_REGISTRY_FILE) as f:
                registry = json.load(f)
        except Exception:
            # Partial write / corruption — try to recover whatever is valid
            try:
                with open(RUN_REGISTRY_FILE) as f:
                    raw = f.read()
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(raw)
                registry = obj if isinstance(obj, list) else []
                print(f"[registry] Recovered {len(registry)} entries from corrupt file")
            except Exception:
                registry = []
                print("[registry] Could not recover registry; starting fresh")

    existing_ids = {entry["run_id"] for entry in registry}
    for run_id in run_ids:
        if run_id not in existing_ids:
            registry.append({
                "run_id": run_id,
                "exp_name": exp.name,
                "group": exp.wandb_group,
                "project": WANDB_PROJECT,
                "tb_path": os.path.join("tensorboard", run_id),
                "config": config_dict,
            })

    # Atomic write: write to temp file then rename so a killed process never
    # leaves a partial/corrupt registry file.
    tmp_path = RUN_REGISTRY_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_path, RUN_REGISTRY_FILE)

    print(
        f"[registry] Saved config for {len(run_ids)} run(s) "
        f"({exp.name}) → {RUN_REGISTRY_FILE}"
    )


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


def setup_mc_data(env, env_params, expert_policy, experiments: List["ExperimentConfig"]) -> None:
    """Collect MC expert trajectories once for all experiments that need them.

    Runs for the maximum episode count across all MC-pretrain experiments and
    saves results to MC_DATA_PATH (a CSV).  Subsequent calls are no-ops when a
    sufficient cache already exists.
    """
    mc_exps = [e for e in experiments if e.use_mc_critic_pretrain]
    if not mc_exps:
        return

    max_episodes = max(e.mc_pretrain_n_mc_episodes for e in mc_exps)
    n_mc_steps   = max(e.mc_pretrain_n_mc_steps   for e in mc_exps)
    n_rows_needed = n_mc_steps * max_episodes

    if os.path.exists(MC_DATA_PATH):
        try:
            meta = read_mc_meta(MC_DATA_PATH)
            if meta.get("n_rows", 0) >= n_rows_needed:
                print(
                    f"[MC collect] Cache hit: {meta['n_rows']:,} rows available "
                    f"(need {n_rows_needed:,}) — skipping collection."
                )
                return
        except Exception:
            pass  # corrupt file → re-collect

    print(
        f"\n[MC collect] ── Starting collection ──────────────────────────────────\n"
        f"[MC collect] {max_episodes} episodes × {n_mc_steps} steps "
        f"= {n_rows_needed:,} transitions  →  {MC_DATA_PATH}"
    )

    mc_n_envs = 64  # parallel envs for MC collection — increase for faster collection
    env_prepared, env_params_prepared, _, continuous = prepare_env(env, env_params=env_params, n_envs=mc_n_envs)
    env_args = EnvironmentConfig(env=env_prepared, env_params=env_params_prepared, n_envs=mc_n_envs, continuous=continuous)

    hp = load_hyperparams("SAC", "Plane")
    gamma        = hp.get("gamma",        0.99)
    reward_scale = hp.get("reward_scale", 1.0)

    collect_and_save_mc_data(
        expert_policy=expert_policy,
        env_args=env_args,
        n_mc_steps=n_mc_steps,
        n_mc_episodes=max_episodes,
        gamma=gamma,
        reward_scale=reward_scale,
    )
    print("[MC collect] ── Collection complete ──────────────────────────────────\n")


def run_single_experiment(
    exp: ExperimentConfig,
    env,
    env_params,
    expert_policy,
    n_seeds: int,
    n_timesteps: int,
    num_episode_test: int,
    log_frequency: int,
    sweep_mode: bool,
    use_wandb: bool = True,
    upload_after: bool = False,
):
    """Run one ExperimentConfig to completion. Called per-process by the launcher."""
    mode = get_mode()
    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")
    if "alpha_learning_rate" in hp and exp.alpha_learning_rate_scale != 1.0:
        hp["alpha_learning_rate"] = hp["alpha_learning_rate"] * exp.alpha_learning_rate_scale

    # Pass expert_policy to any config that uses expert-based features.
    # Previously gated on use_expert_warmup, which silently disabled EGE and MC pretrain.
    # Buffer pre-seeding is controlled by expert_buffer_n_steps (0 = no seeding).
    # Baseline gets None to avoid per-step expert_action overhead.
    needs_expert_policy = (
        exp.use_expert_warmup
        or exp.use_expert_guided_exploration
        or exp.use_mc_critic_pretrain
        or exp.use_expert_guidance
        or exp.use_critic_blend
        or exp.use_box
    )
    agent_expert_policy = expert_policy if needs_expert_policy else None

    # Compute absolute target_entropy_far from per-dim scale × action_dim
    target_entropy_far: Optional[float] = None
    if exp.use_distance_entropy:
        tep = hp.get("target_entropy_per_dim", -1.0)
        action_dim = get_action_dim(env, env_params)
        target_entropy_far = tep * exp.target_entropy_far_scale * action_dim

    mc_variance_threshold: Optional[float] = exp.variance_threshold if exp.use_mc_correction else None

    policy_score = get_policy_score(expert_policy, env, env_params)
    print(
        f"[{exp.name}] group={exp.wandb_group}  expert_score={policy_score:.1f}\n"
        f"  warmup={exp.use_expert_warmup}  awbc={exp.use_expert_guidance}  "
        f"mc_pretrain={exp.use_mc_critic_pretrain}  light_critic_pretrain={exp.use_online_critic_light_pretrain}  "
        f"blend={exp.use_critic_blend}  box={exp.use_box}\n"
        f"  critics={exp.num_critics}×{exp.num_critic_updates}  "
        f"normalize={exp.awbc_normalize}  relu={exp.awbc_use_relu}  "
        f"fixed_λ={exp.fixed_awbc_lambda}  mix={exp.expert_mix_fraction}  "
        f"policy_start={exp.policy_update_start}"
    )

    logging_config = get_log_config(
        project_name=WANDB_PROJECT,
        agent_name=exp.name,
        group_name=exp.wandb_group,
        log_frequency=log_frequency,
        use_wandb=use_wandb,
        sweep=sweep_mode,
        exp_name=exp.name,
        use_expert_warmup=exp.use_expert_warmup,
        use_expert_guidance=exp.use_expert_guidance,
        use_mc_critic_pretrain=exp.use_mc_critic_pretrain,
        use_online_critic_light_pretrain=exp.use_online_critic_light_pretrain,
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
        online_critic_pretrain_steps=exp.online_critic_pretrain_steps,
        online_critic_pretrain_lr_scale=exp.online_critic_pretrain_lr_scale,
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
        use_online_critic_light_pretrain=exp.use_online_critic_light_pretrain,
        online_critic_pretrain_steps=exp.online_critic_pretrain_steps,
        online_critic_pretrain_lr_scale=exp.online_critic_pretrain_lr_scale,
        use_critic_blend=exp.use_critic_blend,
        critic_warmup_frac=exp.critic_warmup_frac,
        use_box=exp.use_box,
        use_online_bc=exp.use_online_bc,
        bc_coef=exp.bc_coef,
        use_expert_guided_exploration=exp.use_expert_guided_exploration,
        exploration_decay_frac=exp.exploration_decay_frac,
        exploration_tau=exp.exploration_tau,
        target_entropy_far=target_entropy_far,
        mc_variance_threshold=mc_variance_threshold,
        use_phi_refresh=exp.use_phi_refresh,
        phi_refresh_interval=exp.phi_refresh_interval,
        phi_refresh_steps=exp.phi_refresh_steps,
        policy_update_start=exp.policy_update_start,
        alpha_update_start=exp.alpha_update_start,
        use_train_frac=exp.use_expert_warmup,  # enable train_frac obs for all expert runs
        awbc_normalize=exp.awbc_normalize,
        awbc_use_relu=exp.awbc_use_relu,
        fixed_awbc_lambda=exp.fixed_awbc_lambda,
        **hp,
    )

    # Load pre-collected MC data for this experiment (None → in-run collection)
    mc_preloaded_data = None
    if exp.use_mc_critic_pretrain:
        print(
            f"[{exp.name}] Loading MC data "
            f"({exp.mc_pretrain_n_mc_episodes} episodes × {exp.mc_pretrain_n_mc_steps} steps) ..."
        )
        mc_preloaded_data = load_mc_data_for_experiment(
            n_mc_steps=exp.mc_pretrain_n_mc_steps,
            n_mc_episodes=exp.mc_pretrain_n_mc_episodes,
        )
        if mc_preloaded_data is not None:
            print(
                f"[{exp.name}] Loaded {mc_preloaded_data[0].shape[0]:,} MC rows from cache."
            )
        else:
            print(f"[{exp.name}] MC cache miss — data will be collected in-run.")

    def _on_ids_ready(run_ids):
        """Called immediately after run IDs are assigned — before JIT compilation."""
        _save_run_configs(exp, run_ids)

    if mode == "CPU":
        for seed in tqdm(range(n_seeds), desc=exp.name):
            t0 = time.time()
            _agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
                on_ids_ready=_on_ids_ready,
                mc_preloaded_data=mc_preloaded_data,
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
            on_ids_ready=_on_ids_ready,
            mc_preloaded_data=mc_preloaded_data,
        )
        elapsed = time.time() - t0
        print(
            f"[{exp.name}] {n_seeds} seeds done in {elapsed:.1f}s "
            f"({elapsed/n_seeds:.1f}s/seed)"
        )
        if sweep_mode:
            upload_tensorboard_to_wandb(_agent.run_ids, logging_config)

    if upload_after and getattr(_agent, "run_ids", None):
        print(f"[{exp.name}] Uploading {len(_agent.run_ids)} run(s) from TensorBoard to W&B...")
        upload_tensorboard_to_wandb(_agent.run_ids, logging_config)

    # After the baseline runs: cache run IDs so other projects can share it.
    if (use_wandb or upload_after) and exp.name == "sac_baseline" and getattr(_agent, "run_ids", None):
        _save_baseline_cache(_agent.run_ids, WANDB_PROJECT)


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
        "--upload-after",
        action="store_true",
        help=(
            "After each experiment completes, upload its TensorBoard data to W&B. "
            "Combine with --no-wandb for fully local training with deferred upload."
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
    n_timesteps = int(1e6)
    n_seeds = 100
    num_episode_test = 25
    log_frequency = 5_000
    sweep_mode = False
    use_wandb = not args.no_wandb
    upload_after = args.upload_after

    experiments = build_experiments()

    # --inject-baseline mode: upload cached baseline to a specific project and exit
    if args.inject_baseline:
        print(f"Injecting baseline into project: {args.inject_baseline}")
        _copy_baseline_to_project(args.inject_baseline)
        raise SystemExit(0)

    if args.list:
        print(f"\n{len(experiments)} experiments:\n")
        for i, exp in enumerate(experiments):
            print(f"  [{i:2d}] {exp.name:<45}  group={exp.wandb_group}")
        raise SystemExit(0)

    env, env_params, expert_policy = setup()

    # Collect MC expert trajectories once (max episodes across all configs).
    # Saves to MC_DATA_PATH; each experiment loads the slice it needs.
    setup_mc_data(env, env_params, expert_policy, experiments)

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
            sweep_mode=sweep_mode,
            use_wandb=use_wandb,
            upload_after=upload_after,
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
                sweep_mode=sweep_mode,
                use_wandb=use_wandb,
                upload_after=upload_after,
            )
