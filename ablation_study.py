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
import fcntl
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import dill as pickle
import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_xla"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
os.environ["WANDB_SILENT"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from partial_expert_train import (
    PARTIAL_EXPERT_STEPS,
    setup_partial_expert_checkpoints,
    load_partial_expert_policy,
)


# ---------------------------------------------------------------------------
# Single W&B project — all experiments land here.
# Edit to match your W&B workspace.
# ---------------------------------------------------------------------------

WANDB_PROJECT = "ablation_plane_final_clean_3"

# W&B group names — one group per question tier (used as the `group` field).
# Q1: Is the value-gap gate doing real work, or is EDGE just epsilon-greedy?
# Q2: Is MC pretrain (φ*) necessary for the gate to work?
# Q3: What is the right decay horizon?
WANDB_GROUPS = {
    "baselines":    "ablation_baselines_plane",
    "q1_decay":     "ablation_q1_decay_plane",      # Q1: decay horizon sweep
    "q2_epsilon":   "ablation_q2_epsilon_plane",    # Q2: epsilon sensitivity (uses best decay from Q1)
    "q3_gating":    "ablation_q3_gating_plane",     # Q3: gating style (uses best decay+epsilon from Q1/Q2)
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

    use_residual_rl: bool = False
    use_expert_guided_exploration: bool = False
    exploration_decay_frac: float = 0.50
    exploration_tau: float = 1.0
    exploration_boltzmann: bool = False      # True = adaptive value-gap gate, False = fixed epsilon
    fixed_exploration_prob: float = 0.5     # used when exploration_boltzmann=False
    exploration_argmax: bool = False         # True = IBRL-style argmax gating (deterministic)

    use_mc_correction: bool = False
    variance_threshold: float = 1.0

    use_phi_refresh: bool = False
    phi_refresh_interval: int = 10_000
    phi_refresh_steps: int = 200

    num_critics: int = 2

    # Degradation 3: Value-based sub-optimality.
    # When set, the experiment uses a SAC actor trained for this many steps
    # (loaded from partial_expert_checkpoints/) as the expert_policy instead
    # of the full PID controller.  None = use the full PID expert (default).
    partial_expert_steps: Optional[int] = None

    # Noise expert ablation: replace the expert with a uniform-random policy
    # to verify that the actual structure of the expert is necessary, not just
    # the presence of any non-policy action signal.
    use_noise_expert: bool = False

    # IBRL bootstrap: modify the TD target to use max(Q_policy, Q_expert) at
    # next states, consistent with the argmax action-selection during rollout.
    # When True + exploration_argmax=True  → full IBRL applied to SAC.
    # When True + EDGE gating             → test whether this addition helps or hurts.
    use_ibrl_bootstrap: bool = False


def build_experiments() -> List[ExperimentConfig]:
    P = WANDB_GROUPS

    # Shared EDGE defaults — all EDGE experiments inherit from this.
    # MC pre-train is disabled for all experiments.
    # Boltzmann gating is disabled everywhere; epsilon-greedy is the default.
    EGE = dict(
        use_expert_guided_exploration=True,
        use_mc_critic_pretrain=False,
        exploration_tau=1.0,
        exploration_boltzmann=False,
        expert_buffer_n_steps=0,            # no pre-population: EDGE fills organically
        expert_mix_fraction=0.0,
        use_online_bc=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        use_expert_guidance=False,
        num_critic_updates=1,
        alpha_learning_rate_scale=1.0,
    )

    exps = []

    # ==================================================================
    # Baselines — run first (sac_baseline must be index 0 for cache)
    # ==================================================================

    # Vanilla SAC — no expert, no modifications.  Lower bound.
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

    # Residual RL baseline (Johannink et al.):
    # Vanilla SAC where the executed action is clip(a_expert + a_policy, -1, 1).
    exps.append(ExperimentConfig(
        name="residual_rl",
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
        use_residual_rl=True,
    ))

    # IBRL-style argmax gating, no decay — true IBRL baseline.
    # In baselines group so it appears alongside SAC in expert-bias overview plots.
    exps.append(ExperimentConfig(
        name="ibrl_style",
        wandb_group=P["baselines"],
        **EGE,
        exploration_argmax=True,
        exploration_decay_frac=0.0,
    ))

    # ==================================================================
    # Q1: What is the right decay horizon?
    # Best decay from this study feeds into Q2 and Q3.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_no_decay",
        wandb_group=P["q1_decay"],
        **EGE,
        exploration_decay_frac=0.0,
        fixed_exploration_prob=0.5,
    ))

    exps.append(ExperimentConfig(
        name="ege_decay_005",
        wandb_group=P["q1_decay"],
        **EGE,
        exploration_decay_frac=0.05,
        fixed_exploration_prob=0.5,
    ))

    exps.append(ExperimentConfig(
        name="ege_simple",
        wandb_group=P["q1_decay"],
        **EGE,
        exploration_decay_frac=0.15,
        fixed_exploration_prob=0.5,
    ))

    # This experiment also serves as the main result (update MAIN_EXP in plot_sweep.py
    # and move wandb_group=P["baselines"] to whichever decay wins).
    exps.append(ExperimentConfig(
        name="ege_decay_050",
        wandb_group=P["baselines"],
        **EGE,
        exploration_decay_frac=0.50,
        fixed_exploration_prob=0.5,
    ))

    exps.append(ExperimentConfig(
        name="ege_decay_075",
        wandb_group=P["q1_decay"],
        **EGE,
        exploration_decay_frac=0.75,
        fixed_exploration_prob=0.5,
    ))

    # ==================================================================
    # Q2: How sensitive is EDGE to the fixed-epsilon value?
    # Uses the best decay found in Q1.  ε=0.5 is covered by ege_decay_050.
    # ==================================================================

    for epsilon in [0.1, 0.25, 0.75, 0.9, 0.95, 0.99]:
        exps.append(ExperimentConfig(
            name=f"ege_eps_{epsilon}",
            wandb_group=P["q2_epsilon"],
            **EGE,
            fixed_exploration_prob=epsilon,
            exploration_decay_frac=0.50,  # best decay from Q1
        ))

    # ==================================================================
    # Q3: Does gating style matter?  Uses best decay from Q1.
    # IBRL with decay uses the Q1-optimal decay value.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ibrl_style_decay",
        wandb_group=P["q3_gating"],
        **EGE,
        exploration_argmax=True,
        exploration_decay_frac=0.50,  # best decay from Q1
    ))

    # ==================================================================
    # Q4: Does adding the IBRL bootstrap modification help or hurt?
    # Full IBRL = argmax action-selection (ibrl_style_decay) + IBRL bootstrap.
    # EDGE + IBRL bootstrap = test whether this addition is compatible with EGE.
    # ==================================================================

    # True IBRL applied to SAC: argmax gating (decay=0.5) + IBRL bootstrap.
    exps.append(ExperimentConfig(
        name="ibrl_true",
        wandb_group=P["q3_gating"],
        **EGE,
        exploration_argmax=True,
        exploration_decay_frac=0.50,
        use_ibrl_bootstrap=True,
    ))

    # EDGE (decay=0.5, ε=0.5) + IBRL bootstrap — tests compatibility.
    exps.append(ExperimentConfig(
        name="ege_ibrl_bootstrap",
        wandb_group=P["q3_gating"],
        **EGE,
        exploration_decay_frac=0.50,
        fixed_exploration_prob=0.5,
        use_ibrl_bootstrap=True,
    ))

    # ==================================================================
    # Noise-expert control: uniform-random "expert" — same EDGE config as
    # ege_decay_050 but with a structureless noise policy instead of PID.
    # Should perform no better than sac_baseline if expert structure matters.
    # ==================================================================

    exps.append(ExperimentConfig(
        name="ege_noise_expert",
        wandb_group=P["baselines"],
        **EGE,
        exploration_decay_frac=0.50,
        fixed_exploration_prob=0.5,
        use_noise_expert=True,
    ))

    return exps


# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------


def _last_step_in_run(run_id: str) -> Optional[int]:
    """Return the last logged step in a run's TFEvents file, or None if unreadable.

    Parses the raw TFRecord binary format to avoid a hard tensorboard dependency.
    Each TFRecord: uint64 length | uint32 crc_len | byte[length] data | uint32 crc_data
    Each Event proto field 2 (step) is a varint at wire type 0.
    """
    import struct

    tb_dir = os.path.join(os.path.abspath("tensorboard"), run_id)
    if not os.path.isdir(tb_dir):
        return None

    def _read_varint(data: bytes, idx: int):
        val = 0; shift = 0
        while idx < len(data):
            b = data[idx]; idx += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        return val, idx

    def _step_from_record(data: bytes) -> Optional[int]:
        """Extract the step field (proto field 2, varint) from a serialised Event."""
        idx = 0
        while idx < len(data):
            b = data[idx]; idx += 1
            field_num = b >> 3
            wire_type = b & 0x7
            if field_num == 2 and wire_type == 0:
                val, _ = _read_varint(data, idx)
                return val
            # Skip over this field
            if wire_type == 0:
                _, idx = _read_varint(data, idx)
            elif wire_type == 1:
                idx += 8
            elif wire_type == 2:
                length, idx = _read_varint(data, idx)
                idx += length
            elif wire_type == 5:
                idx += 4
            else:
                break
        return None

    last_step = None
    for fname in os.listdir(tb_dir):
        if "tfevents" not in fname:
            continue
        fpath = os.path.join(tb_dir, fname)
        try:
            with open(fpath, "rb") as fh:
                while True:
                    header = fh.read(8)
                    if len(header) < 8:
                        break
                    data_len = struct.unpack("<Q", header)[0]
                    fh.read(4)                  # crc of length
                    record = fh.read(data_len)
                    fh.read(4)                  # crc of data
                    if len(record) < data_len:
                        break
                    step = _step_from_record(record)
                    if step is not None:
                        last_step = step
        except Exception:
            pass

    return last_step


# Fields excluded from config-matching: pure metadata, not training behaviour.
_CONFIG_METADATA_KEYS = {"name", "wandb_group"}

# Dataclass defaults used as fallback when a stored registry entry pre-dates a field.
_EXPERIMENT_DEFAULTS: dict = asdict(ExperimentConfig(name="_defaults_"))


def _configs_match(exp: ExperimentConfig, stored: dict) -> bool:
    """Return True if stored config is equivalent to exp for all training-relevant fields.

    Missing keys in stored (e.g. fields added after the run was saved) are treated
    as the dataclass default.  If the default matches the current experiment value the
    run is still considered valid; if it differs the run is stale and is rejected.
    """
    current = asdict(exp)
    for key, value in current.items():
        if key in _CONFIG_METADATA_KEYS:
            continue
        stored_value = stored.get(key, _EXPERIMENT_DEFAULTS[key])
        if stored_value != value:
            return False
    return True


def is_experiment_complete(exp: ExperimentConfig, n_seeds: int, n_timesteps: int) -> bool:
    """Return True if exp has n_seeds distinct, fully-finished runs with a matching config.

    Rules:
    - Only runs from WANDB_PROJECT are considered (ignores old / other projects).
    - A run is "finished" when its last logged TFEvents step is ≥ 99% of n_timesteps.
    - The stored run config must match the current experiment config on all
      training-relevant fields (excluding name/wandb_group).  Runs saved before a
      field existed are compared against the dataclass default for that field.
    - Duplicate run_ids are counted only once.
    """
    if not os.path.exists(RUN_REGISTRY_FILE):
        return False
    try:
        with open(RUN_REGISTRY_FILE) as f:
            registry = json.load(f)
    except Exception:
        return False

    threshold = int(n_timesteps * 0.99)

    # Filter: current project + matching exp_name + matching training config,
    # deduplicated by run_id.
    seen = set()
    candidates = []
    for entry in registry:
        rid = entry.get("run_id")
        if (
            entry.get("exp_name") == exp.name
            and entry.get("project") == WANDB_PROJECT
            and rid not in seen
            and _configs_match(exp, entry.get("config", {}))
        ):
            seen.add(rid)
            candidates.append(rid)

    finished = sum(
        1 for rid in candidates
        if (_last_step_in_run(rid) or 0) >= threshold
    )
    return finished >= n_seeds


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
    # Use a lock file to serialise concurrent writes from parallel GPU processes.
    lock_path = RUN_REGISTRY_FILE + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
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

            # Atomic write: write to a per-process temp file then rename so a
            # killed process never leaves a partial/corrupt registry file.
            tmp_path = f"{RUN_REGISTRY_FILE}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp_path, RUN_REGISTRY_FILE)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

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


def make_noise_expert_policy(action_dim: int):
    """Return a uniform-noise expert policy with the given action dimension.

    The returned callable has the same signature as the PID expert (obs → action)
    and is safe to use inside JIT-compiled JAX code.  Actions are sampled
    uniformly from [-1, 1]^action_dim using a key derived from the observation,
    so repeated calls with the same obs return the same "random" action.
    """
    def _noise_policy(obs):
        # Fold the sum of obs values into a base key to get obs-dependent randomness.
        key = jax.random.fold_in(
            jax.random.PRNGKey(0),
            jnp.int32(jnp.sum(obs * 1e4)),
        )
        return jax.random.uniform(key, shape=obs.shape[:-1] + (action_dim,), minval=-1.0, maxval=1.0)

    return _noise_policy


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
    """Run one ExperimentConfig to completion. Called per-process by the launcher.

    Skips the experiment if n_seeds finished runs already exist in the registry.
    """
    if is_experiment_complete(exp, n_seeds, n_timesteps):
        print(
            f"[{exp.name}] Already complete ({n_seeds} seeds × {n_timesteps:,} steps) — skipping."
        )
        return

    mode = get_mode()
    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")
    if "alpha_learning_rate" in hp and exp.alpha_learning_rate_scale != 1.0:
        hp["alpha_learning_rate"] = hp["alpha_learning_rate"] * exp.alpha_learning_rate_scale

    # Pass expert_policy to any config that uses expert-based features.
    # Previously gated on use_expert_warmup, which silently disabled EDGE and MC pretrain.
    # Buffer pre-seeding is controlled by expert_buffer_n_steps (0 = no seeding).
    # Baseline gets None to avoid per-step expert_action overhead.
    needs_expert_policy = (
        exp.use_expert_warmup
        or exp.use_expert_guided_exploration
        or exp.use_mc_critic_pretrain
        or exp.use_expert_guidance
        or exp.use_critic_blend
        or exp.use_box
        or exp.use_residual_rl
    )

    if exp.use_noise_expert:
        # Noise-expert control: replace the PID expert with a uniform-random policy.
        # eval_expert_policy stays as the full PID controller for fair comparison.
        action_dim = get_action_dim(env, env_params)
        agent_expert_policy = make_noise_expert_policy(action_dim)
    elif exp.partial_expert_steps is not None:
        # Degradation 3: swap in a partial SAC actor as the expert.
        # eval_expert_policy stays as the full PID controller for fair comparison.
        agent_expert_policy = load_partial_expert_policy(exp.partial_expert_steps)
    else:
        agent_expert_policy = expert_policy if needs_expert_policy else None

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
        residual=exp.use_residual_rl,
        use_expert_guided_exploration=exp.use_expert_guided_exploration,
        exploration_decay_frac=exp.exploration_decay_frac,
        exploration_tau=exp.exploration_tau,
        exploration_boltzmann=exp.exploration_boltzmann,
        fixed_exploration_prob=exp.fixed_exploration_prob,
        exploration_argmax=exp.exploration_argmax,
        ibrl_bootstrap=exp.use_ibrl_bootstrap,
        target_entropy_far=None,
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
            done = is_experiment_complete(exp, n_seeds, n_timesteps)
            status = "DONE" if done else "    "
            print(f"  [{i:2d}] {status}  {exp.name:<45}  group={exp.wandb_group}")
        raise SystemExit(0)

    env, env_params, expert_policy = setup()

    # Collect MC expert trajectories once (max episodes across all configs).
    # Saves to MC_DATA_PATH; each experiment loads the slice it needs.
    setup_mc_data(env, env_params, expert_policy, experiments)

    # Train partial SAC experts at different budgets (no-op if already cached).
    # Only runs when the experiment list contains degradation-3 experiments.
    if any(e.partial_expert_steps is not None for e in experiments):
        setup_partial_expert_checkpoints(env, env_params)

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
        pending = [e for e in experiments if not is_experiment_complete(e, n_seeds, n_timesteps)]
        done_count = len(experiments) - len(pending)
        print(
            f"Sequential sweep: {len(experiments)} experiments × {n_seeds} seeds  "
            f"({done_count} already complete, {len(pending)} to run)\n"
        )
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
