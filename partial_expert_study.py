"""
partial_expert_study.py — Degradation 3: Value-based sub-optimality.

Replaces the full PID expert with SAC policies trained to different budgets
({100k, 200k, 300k, 500k, 1M} steps) and measures how expert quality affects
EDGE performance.  No MC pretrain — the value gap uses the live critic, so
expert quality is the only variable.

Ordering:
  [0]  ege_pid_expert          — reference: EDGE with PID, no MC (mirrors ablation ege_no_mc)
  [1]  ege_partial_expert_100k
  [2]  ege_partial_expert_200k
  [3]  ege_partial_expert_300k
  [4]  ege_partial_expert_500k
  [5]  ege_partial_expert_1000k

Each partial-expert experiment automatically trains its SAC checkpoint before
running (no-op when the checkpoint already exists in partial_expert_checkpoints/).

Run via gpu_launcher (recommended):
    python gpu_launcher.py --script partial_expert_study

Run a single experiment by index:
    python partial_expert_study.py --exp-index 2

List all experiments (shows completion status):
    python partial_expert_study.py --list
"""
import json
import os
import time
from dataclasses import asdict, dataclass
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
from ajax.logging.wandb_logging import upload_tensorboard_to_wandb
from ajax.plane.plane_exps_utils import (
    get_log_config,
    get_mode,
    get_policy_score,
    load_hyperparams,
)
from ajax.environments.utils import get_action_dim
from ajax.stable_utils import get_expert_policy
from partial_expert_train import (
    PARTIAL_EXPERT_STEPS,
    train_and_save_partial_expert,
    load_partial_expert_policy,
)

# ---------------------------------------------------------------------------
# Project / group config
# ---------------------------------------------------------------------------

WANDB_PROJECT = "partial_expert_degradation_plane"

WANDB_GROUPS = {
    "reference":        "partial_expert_reference",
    "degradation":      "partial_expert_degradation",
}

RUN_REGISTRY_FILE = os.path.abspath("partial_expert_run_registry.json")

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    name: str
    wandb_group: str
    # None → use PID controller; int → use SAC checkpoint at that many steps
    partial_expert_steps: Optional[int] = None

    exploration_decay_frac: float = 0.50
    exploration_tau: float = 1.0
    fixed_exploration_prob: float = 0.5


def build_experiments() -> List[ExpConfig]:
    P = WANDB_GROUPS
    exps = []

    # Reference: EDGE with the full PID expert, no MC pretrain.
    # Equivalent to ablation_study's ege_no_mc — useful as a local upper bound.
    exps.append(ExpConfig(
        name="ege_pid_expert",
        wandb_group=P["reference"],
        partial_expert_steps=None,
    ))

    for steps in PARTIAL_EXPERT_STEPS:
        label = f"{steps // 1_000}k"
        exps.append(ExpConfig(
            name=f"ege_partial_expert_{label}",
            wandb_group=P["degradation"],
            partial_expert_steps=steps,
        ))

    return exps


# ---------------------------------------------------------------------------
# Completion check (mirrors ablation_study.py)
# ---------------------------------------------------------------------------

def _last_step_in_run(run_id: str) -> Optional[int]:
    """Return the last logged step in a run's TFEvents file, or None."""
    import struct

    tb_dir = os.path.join(os.path.abspath("tensorboard"), run_id)
    if not os.path.isdir(tb_dir):
        return None

    def _read_varint(data, idx):
        val = 0; shift = 0
        while idx < len(data):
            b = data[idx]; idx += 1
            val |= (b & 0x7F) << shift; shift += 7
            if not (b & 0x80): break
        return val, idx

    def _step_from_record(data):
        idx = 0
        while idx < len(data):
            b = data[idx]; idx += 1
            fn = b >> 3; wt = b & 0x7
            if fn == 2 and wt == 0:
                val, _ = _read_varint(data, idx)
                return val
            if wt == 0: _, idx = _read_varint(data, idx)
            elif wt == 1: idx += 8
            elif wt == 2:
                l, idx = _read_varint(data, idx); idx += l
            elif wt == 5: idx += 4
            else: break
        return None

    last_step = None
    for fname in os.listdir(tb_dir):
        if "tfevents" not in fname:
            continue
        try:
            with open(os.path.join(tb_dir, fname), "rb") as fh:
                while True:
                    header = fh.read(8)
                    if len(header) < 8: break
                    data_len = struct.unpack("<Q", header)[0]
                    fh.read(4)
                    record = fh.read(data_len)
                    fh.read(4)
                    if len(record) < data_len: break
                    step = _step_from_record(record)
                    if step is not None:
                        last_step = step
        except Exception:
            pass
    return last_step


def is_experiment_complete(exp_name: str, n_seeds: int, n_timesteps: int) -> bool:
    """True if exp_name has n_seeds runs that each logged ≥99% of n_timesteps."""
    if not os.path.exists(RUN_REGISTRY_FILE):
        return False
    try:
        with open(RUN_REGISTRY_FILE) as f:
            registry = json.load(f)
    except Exception:
        return False

    threshold = int(n_timesteps * 0.99)
    seen = set()
    candidates = []
    for entry in registry:
        rid = entry.get("run_id")
        if (
            entry.get("exp_name") == exp_name
            and entry.get("project") == WANDB_PROJECT
            and rid not in seen
        ):
            seen.add(rid)
            candidates.append(rid)

    finished = sum(1 for rid in candidates if (_last_step_in_run(rid) or 0) >= threshold)
    return finished >= n_seeds


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _save_run_configs(exp: ExpConfig, run_ids: list) -> None:
    if not run_ids:
        return

    config_dict = asdict(exp)
    tb_base = os.path.abspath("tensorboard")

    for run_id in run_ids:
        run_dir = os.path.join(tb_base, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({
                "run_id": run_id,
                "exp_name": exp.name,
                "group": exp.wandb_group,
                "project": WANDB_PROJECT,
                "config": config_dict,
            }, f, indent=2)

    registry: list = []
    if os.path.exists(RUN_REGISTRY_FILE):
        try:
            with open(RUN_REGISTRY_FILE) as f:
                registry = json.load(f)
        except Exception:
            pass

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

    tmp = RUN_REGISTRY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, RUN_REGISTRY_FILE)


# ---------------------------------------------------------------------------
# Environment setup
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
            pid_policy = pickle.load(f)
    else:
        pid_policy = get_expert_policy(env, env_params)
        with open("expert_policy.pkl", "wb") as f:
            pickle.dump(pid_policy, f)

    for test_alt in [3_000, 8_000]:
        test_obs = jnp.array([0, 0, 0, 0, 0, 0, test_alt], dtype=jnp.float32)
        if jnp.isnan(pid_policy(test_obs)).any():
            raise ValueError(f"PID policy produces NaN at altitude {test_alt}")

    return env, env_params, pid_policy


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_experiment(
    exp: ExpConfig,
    env,
    env_params,
    pid_policy,
    n_seeds: int,
    n_timesteps: int,
    num_episode_test: int,
    log_frequency: int,
    use_wandb: bool = True,
    upload_after: bool = False,
) -> None:
    """Train and evaluate one experiment.  Skips if already complete."""
    if is_experiment_complete(exp.name, n_seeds, n_timesteps):
        print(f"[{exp.name}] Already complete ({n_seeds} seeds × {n_timesteps:,} steps) — skipping.")
        return

    # Ensure the SAC checkpoint exists before loading it.
    if exp.partial_expert_steps is not None:
        train_and_save_partial_expert(exp.partial_expert_steps, env, env_params)
        expert_policy = load_partial_expert_policy(exp.partial_expert_steps)
        expert_label = f"SAC@{exp.partial_expert_steps // 1_000}k"
    else:
        expert_policy = pid_policy
        expert_label = "PID"

    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")

    policy_score = get_policy_score(pid_policy, env, env_params)
    print(
        f"[{exp.name}] expert={expert_label}  pid_score={policy_score:.1f}  "
        f"decay={exp.exploration_decay_frac}"
    )

    logging_config = get_log_config(
        project_name=WANDB_PROJECT,
        agent_name=exp.name,
        group_name=exp.wandb_group,
        log_frequency=log_frequency,
        use_wandb=use_wandb,
        sweep=False,
        exp_name=exp.name,
        use_expert_warmup=False,
        use_expert_guidance=False,
        use_mc_critic_pretrain=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        critic_warmup_frac=0.0,
        use_box=False,
        num_critics=2,
        num_critic_updates=1,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        awbc_normalize=True,
        awbc_use_relu=True,
        fixed_awbc_lambda=None,
        online_critic_pretrain_steps=100,
        online_critic_pretrain_lr_scale=0.1,
        policy_update_start=5_000,
    )

    _agent = SAC(
        env_id=env,
        env_params=env_params,
        expert_policy=expert_policy,
        eval_expert_policy=pid_policy,        # always evaluate against PID for fair comparison
        actor_architecture=architecture,
        critic_architecture=architecture,
        num_critics=2,
        use_expert_guidance=False,
        num_critic_updates=1,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=False,
        use_online_critic_light_pretrain=False,
        use_critic_blend=False,
        critic_warmup_frac=0.0,
        use_box=False,
        use_online_bc=False,
        bc_coef=1.0,
        residual=False,
        use_expert_guided_exploration=True,
        exploration_decay_frac=exp.exploration_decay_frac,
        exploration_tau=exp.exploration_tau,
        exploration_boltzmann=False,
        fixed_exploration_prob=exp.fixed_exploration_prob,
        exploration_argmax=False,
        awbc_normalize=True,
        awbc_use_relu=True,
        fixed_awbc_lambda=None,
        mc_variance_threshold=None,
        use_phi_refresh=False,
        target_entropy_far=None,
        policy_update_start=5_000,
        alpha_update_start=5_000,
        use_train_frac=False,
        **hp,
    )

    def _on_ids_ready(run_ids):
        _save_run_configs(exp, run_ids)

    mode = get_mode()
    if mode == "CPU":
        for seed in tqdm(range(n_seeds), desc=exp.name):
            t0 = time.time()
            _agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
                on_ids_ready=_on_ids_ready,
            )
            print(f"[{exp.name}] seed {seed} done in {time.time() - t0:.1f}s")
    else:
        t0 = time.time()
        _agent.train(
            seed=list(range(n_seeds)),
            logging_config=logging_config,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            on_ids_ready=_on_ids_ready,
        )
        elapsed = time.time() - t0
        print(f"[{exp.name}] {n_seeds} seeds done in {elapsed:.1f}s ({elapsed/n_seeds:.1f}s/seed)")

    if upload_after and getattr(_agent, "run_ids", None):
        upload_tensorboard_to_wandb(_agent.run_ids, logging_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-index", type=int, default=None,
                        help="Run a single experiment by index (used by gpu_launcher).")
    parser.add_argument("--list", action="store_true",
                        help="Print all experiments and exit.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--upload-after", action="store_true",
                        help="Upload TensorBoard logs to W&B after each experiment.")
    args = parser.parse_args()

    n_timesteps     = int(1e6)
    n_seeds         = 100
    num_episode_test = 25
    log_frequency   = 5_000
    use_wandb       = not args.no_wandb
    upload_after    = args.upload_after

    experiments = build_experiments()

    if args.list:
        print(f"\n{len(experiments)} experiments:\n")
        for i, exp in enumerate(experiments):
            done = is_experiment_complete(exp.name, n_seeds, n_timesteps)
            status = "DONE" if done else "    "
            ckpt = f"SAC@{exp.partial_expert_steps // 1_000}k" if exp.partial_expert_steps else "PID"
            print(f"  [{i:2d}] {status}  {exp.name:<35}  expert={ckpt}  group={exp.wandb_group}")
        raise SystemExit(0)

    env, env_params, pid_policy = setup()

    if args.exp_index is not None:
        if args.exp_index >= len(experiments):
            raise ValueError(
                f"--exp-index {args.exp_index} out of range "
                f"(only {len(experiments)} experiments defined)"
            )
        exp = experiments[args.exp_index]
        print(f"Single-experiment mode: [{args.exp_index}] {exp.name}")
        run_experiment(
            exp=exp,
            env=env,
            env_params=env_params,
            pid_policy=pid_policy,
            n_seeds=n_seeds,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            log_frequency=log_frequency,
            use_wandb=use_wandb,
            upload_after=upload_after,
        )
    else:
        pending = [e for e in experiments if not is_experiment_complete(e.name, n_seeds, n_timesteps)]
        done_count = len(experiments) - len(pending)
        print(f"Sequential sweep: {len(experiments)} experiments × {n_seeds} seeds  "
              f"({done_count} already complete, {len(pending)} to run)\n")
        for exp in experiments:
            run_experiment(
                exp=exp,
                env=env,
                env_params=env_params,
                pid_policy=pid_policy,
                n_seeds=n_seeds,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
                log_frequency=log_frequency,
                use_wandb=use_wandb,
                upload_after=upload_after,
            )
