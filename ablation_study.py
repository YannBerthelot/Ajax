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

WANDB_PROJECT = "ablation_plane_clean"

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
    use_online_bc: bool = False             # decoupled from AWBC, explicit default off
    use_online_critic_light_pretrain: bool = False  # explicit default off, must opt in
    use_critic_blend: bool = False
    critic_warmup_frac: float = 0.3
    use_box: bool = False

    policy_update_start: int = 5_000
    alpha_update_start: int = 5_000        # default deferred to prevent early collapse

    num_critic_updates: int = 1             # default 1, UTD=4 must be explicit opt-in
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1

    awbc_normalize: bool = True
    awbc_use_relu: bool = True
    fixed_awbc_lambda: Optional[float] = None

    mc_pretrain_n_mc_steps: int = 10_000
    mc_pretrain_n_mc_episodes: int = 1000   # explicit, matches winning run
    mc_pretrain_n_steps: int = 5_000

    online_critic_pretrain_steps: int = 100  # default matches winning run
    online_critic_pretrain_lr_scale: float = 0.1

    alpha_learning_rate_scale: float = 0.5   # multiplier on SAC-tuned alpha_lr (0.5 for non-baselines)
    bc_coef: float = 1.0                     # BC term strength multiplier

    use_expert_guided_exploration: bool = False
    exploration_decay_frac: float = 0.30
    exploration_tau: float = 1.0

    # Distance-modulated entropy (None = disabled; scale applied to target_entropy_per_dim)
    use_distance_entropy: bool = False
    target_entropy_far_scale: float = 0.5

    # Online MC correction for high-variance states
    use_mc_correction: bool = False
    variance_threshold: float = 1.0

    num_critics: int = 2


def build_experiments() -> List[ExperimentConfig]:
    P = WANDB_GROUPS

    # ------------------------------------------------------------------
    # Shared dicts — explicit, no silent inheritance
    # ------------------------------------------------------------------
    MC = dict(
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_mc_episodes=1000,     # explicit
        mc_pretrain_n_steps=5_000,
    )
    BUF = dict(
        expert_buffer_n_steps=20_000,
        expert_mix_fraction=0.1,
    )
    AWBC = dict(
        use_expert_guidance=True,
        num_critic_updates=1,               # confirmed: AWBC + UTD=4 is harmful
    )
    UTD4 = dict(
        num_critic_updates=4,               # only for non-AWBC runs
    )
    LIGHT = dict(
        use_online_critic_light_pretrain=True,
        online_critic_pretrain_steps=100,
    )
    BC = dict(
        use_online_bc=True,
        critic_warmup_frac=0.3,
        bc_coef=5.0,
    )
    BC_OFF = dict(
        use_online_bc=False,
        critic_warmup_frac=0.0,
    )
    EGE = dict(
        use_expert_guided_exploration=True,
        exploration_decay_frac=0.30,
        exploration_tau=1.0,
        expert_buffer_n_steps=0,        # no pre-population needed
        expert_mix_fraction=0.0,        # no static mixing needed
    )
    BLEND = dict(
        use_critic_blend=True,
    )
    EGE_BASE = dict(
        use_expert_guided_exploration=True,
        exploration_decay_frac=0.30,
        exploration_tau=1.0,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        use_expert_guidance=False,
        num_critic_updates=1,
    )
    # alpha_update_start=50_000 is now the default in ExperimentConfig.
    # SAC baseline overrides back to 10_000 since it has no expert critic.

    exps = []

    # ==================================================================
    # Tier -1 — EGE solution ablations
    # Each activates exactly one add-on on top of EGE_BASE + MC pretrain.
    # Tier -1 runs first: shortest iteration time, most diagnostic value.
    # ==================================================================

    # 1. EGE + online MC correction only
    # Cleanest test of Solution 1. Does replacing Bellman targets with
    # short expert rollouts in high-variance states improve Q accuracy?
    exps.append(ExperimentConfig(
        name="ege_mc_correction",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE_BASE,
        use_mc_correction=True,
        variance_threshold=1.0,
    ))

    # 2. EGE + distance-modulated entropy only
    # Tests whether higher entropy far from setpoint improves reach-phase
    # exploration without interfering with near-setpoint precision.
    # target_entropy_far_scale=0.5 → target_entropy_per_dim * 0.5 (less negative = more entropy far away)
    exps.append(ExperimentConfig(
        name="ege_entropy_distance",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE_BASE,
        use_distance_entropy=True,
        target_entropy_far_scale=0.5,
    ))

    # 3. EGE + MC correction + distance entropy combined
    exps.append(ExperimentConfig(
        name="ege_mc_entropy",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE_BASE,
        use_mc_correction=True,
        variance_threshold=1.0,
        use_distance_entropy=True,
        target_entropy_far_scale=0.5,
    ))

    # ==================================================================
    # Tier 0 — Baseline anchors
    # Must run first. No expert components, alpha deferred at 10k (no MC).
    # ==================================================================

    # exps.append(ExperimentConfig(
    #     name="sac_baseline",
    #     wandb_group=P["baselines"],
    #     expert_buffer_n_steps=0,
    #     expert_mix_fraction=0.0,
    #     policy_update_start=10_000,
    #     alpha_update_start=10_000,          # no MC pretrain → no early collapse risk
    #     use_online_bc=False,
    #     use_online_critic_light_pretrain=False,
    # ))

    # exps.append(ExperimentConfig(
    #     name="sac_expert_seeding",
    #     wandb_group=P["baselines"],
    #     use_expert_warmup=True,
    #     expert_buffer_n_steps=20_000,
    #     expert_mix_fraction=0.0,
    #     policy_update_start=10_000,
    #     alpha_update_start=10_000,          # no MC pretrain → no early collapse risk
    #     use_online_bc=False,
    #     use_online_critic_light_pretrain=False,
    # ))

    # ==================================================================
    # Tier 1 — Current best and direct variants
    # Ranked by expected impact. Alpha deferred (50k) is the default.
    # Key question: is BC alone sufficient? is UTD=4 safe without AWBC?
    # ==================================================================

    # EGE alone, UTD=1 — cleanest test of the mechanism, no other expert components
    exps.append(ExperimentConfig(
        name="ege_utd1",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,    # explicit: no pre-population
        **MC, **EGE,
        use_expert_guidance=False,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        num_critic_updates=1,
    ))

    # EGE + UTD=4 — tests whether faster critic learning helps EGE
    # UTD=4 is now safe: no BC term, no gradient conflict, no actor-side guidance
    exps.append(ExperimentConfig(
        name="ege_utd4",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE, **UTD4,
        use_expert_guidance=False,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
    ))

    # EGE + blend — expert guides both exploration and critic targets
    # blend provides expert signal to critic, EGE provides expert data to buffer
    # no gradient conflict since neither touches the actor loss directly
    exps.append(ExperimentConfig(
        name="ege_blend_utd1",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE, **BLEND,
        use_expert_guidance=False,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        num_critic_updates=1,
    ))

    # EGE + blend + UTD=4 — full stack without any actor-side constraints
    exps.append(ExperimentConfig(
        name="ege_blend_utd4",
        wandb_group=P["best_variants"],
        use_expert_warmup=False,
        **MC, **EGE, **BLEND, **UTD4,
        use_expert_guidance=False,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
    ))

    # No light pretrain — is the 100-step nudge load-bearing?
    exps.append(ExperimentConfig(
        name="mc_bc_utd4_no_light",
        wandb_group=P["best_variants"],
        use_expert_warmup=True,
        **MC, **BUF, **BC, **UTD4,
        use_online_critic_light_pretrain=False,
        use_expert_guidance=False,
    ))

    # ==================================================================
    # Tier 2 — Component isolation
    # Each experiment activates exactly one component on top of MC pretrain.
    # BC_OFF is explicit to prevent silent activation via critic_warmup_frac default.
    # ==================================================================

    # MC alone — frozen phi* exists but nothing uses it during training
    exps.append(ExperimentConfig(
        name="mc_only",
        wandb_group=P["component_isolation"],
        use_expert_warmup=True,
        **MC, **BUF, **BC_OFF,
        use_online_critic_light_pretrain=False,
        use_expert_guidance=False,
    ))

    # MC + light pretrain only — nudge without BC or UTD
    exps.append(ExperimentConfig(
        name="mc_light_only",
        wandb_group=P["component_isolation"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC_OFF,
        use_expert_guidance=False,
    ))

    # MC + BC only, no light pretrain, no UTD — pure BC contribution
    exps.append(ExperimentConfig(
        name="mc_bc_only",
        wandb_group=P["component_isolation"],
        use_expert_warmup=True,
        **MC, **BUF, **BC,
        use_online_critic_light_pretrain=False,
        use_expert_guidance=False,
    ))

    # MC + AWBC only, UTD=1, no BC — legacy E2 anchor, clean AWBC test
    exps.append(ExperimentConfig(
        name="mc_awbc_utd1_no_bc",
        wandb_group=P["component_isolation"],
        use_expert_warmup=True,
        **MC, **AWBC, **BUF, **BC_OFF,
        use_online_critic_light_pretrain=False,
    ))

    # MC + UTD=4 only, no BC, no AWBC — does UTD alone help?
    exps.append(ExperimentConfig(
        name="mc_utd4_no_bc_no_awbc",
        wandb_group=P["component_isolation"],
        use_expert_warmup=True,
        **MC, **BUF, **BC_OFF, **UTD4,
        use_online_critic_light_pretrain=False,
        use_expert_guidance=False,
    ))

    # ==================================================================
    # Tier 3 — Alpha deferral ablation
    # Confirms that alpha_update_start=50k is the right fix.
    # Runs against mc_light_bc_utd4 (Tier 1 best) as the reference.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd4_alpha5k",   # alpha NOT deferred — reproduces collapse
        wandb_group=P["alpha_deferral"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC, **UTD4,
        use_expert_guidance=False,
        alpha_update_start=5_000,           # explicit: no deferral
    ))

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd4_alpha25k",  # partial deferral
        wandb_group=P["alpha_deferral"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC, **UTD4,
        use_expert_guidance=False,
        alpha_update_start=25_000,
    ))

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd4_alpha100k", # overcautious deferral
        wandb_group=P["alpha_deferral"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC, **UTD4,
        use_expert_guidance=False,
        alpha_update_start=100_000,
    ))

    # ==================================================================
    # Tier 4 — Critic architecture
    # Tests UTD ratio and ensemble size independently.
    # Base config: mc_light_bc (UTD=1) to isolate each change.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd2",
        wandb_group=P["critic_architecture"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC,
        num_critic_updates=2,
        use_expert_guidance=False,
    ))

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd8",
        wandb_group=P["critic_architecture"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC,
        num_critic_updates=8,
        use_expert_guidance=False,
    ))

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd4_4critics",
        wandb_group=P["critic_architecture"],
        use_expert_warmup=True,
        **MC, **LIGHT, **BUF, **BC, **UTD4,
        num_critics=4,
        use_expert_guidance=False,
    ))

    # ==================================================================
    # Tier 5 — MC pretrain data quality
    # Tests whether phi* quality affects BC term and light pretrain.
    # Base: mc_light_bc_utd4 (Tier 1 best).
    # ==================================================================

    for n_ep in [100, 500]:
        exps.append(ExperimentConfig(
            name=f"mc_light_bc_utd4_{n_ep}ep",
            wandb_group=P["mc_pretrain"],
            use_expert_warmup=True,
            **LIGHT, **BUF, **BC, **UTD4,
            use_mc_critic_pretrain=True,
            mc_pretrain_n_mc_steps=10_000,
            mc_pretrain_n_mc_episodes=n_ep,  # below default of 1000
            mc_pretrain_n_steps=5_000,
            use_expert_guidance=False,
        ))

    exps.append(ExperimentConfig(
        name="mc_light_bc_utd4_20ksteps",
        wandb_group=P["mc_pretrain"],
        use_expert_warmup=True,
        **LIGHT, **BUF, **BC, **UTD4,
        use_mc_critic_pretrain=True,
        mc_pretrain_n_mc_steps=10_000,
        mc_pretrain_n_mc_episodes=1000,
        mc_pretrain_n_steps=20_000,          # more regression steps
        use_expert_guidance=False,
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
):
    """Run one ExperimentConfig to completion. Called per-process by the launcher."""
    mode = get_mode()
    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")
    if "alpha_learning_rate" in hp and exp.alpha_learning_rate_scale != 1.0:
        hp["alpha_learning_rate"] = hp["alpha_learning_rate"] * exp.alpha_learning_rate_scale

    agent_expert_policy = expert_policy if exp.use_expert_warmup else None

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

    if mode == "CPU":
        for seed in tqdm(range(n_seeds), desc=exp.name):
            t0 = time.time()
            _agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
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
            mc_preloaded_data=mc_preloaded_data,
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
            )
