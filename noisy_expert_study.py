"""
noisy_expert_study.py — Degradation study: PID expert with corrupted target altitude.

Degrades the expert by adding Gaussian noise to the target altitude the PID sees
(obs[6]), while the SAC agent always receives the true target altitude.
This keeps the expert's control law (PID structure) fully intact — only its
goal perception is corrupted — making it a cleaner ablation than training
partial SAC checkpoints.

Noise is derived from obs[6] via jax.random.fold_in so the wrapped expert
remains a pure JAX callable compatible with jit/vmap (expert_policy is a
static_argname in collect_experience).

Experiments (index order):
  [0]  ege_noise_0pct    — perfect PID (reference, equivalent to ablation ege_decay_050)
  [1]  ege_noise_2pct    — σ = 100 m
  [2]  ege_noise_5pct    — σ = 250 m
  [3]  ege_noise_10pct   — σ = 500 m
  [4]  ege_noise_20pct   — σ = 1000 m
  [5]  ege_noise_40pct   — σ = 2000 m
  [6]  ege_noise_80pct   — σ = 4000 m  (near-random)

Run via gpu_launcher (recommended):
    python gpu_launcher.py --script noisy_expert_study

Run a single experiment by index:
    python noisy_expert_study.py --exp-index 2

List all experiments (shows completion status):
    python noisy_expert_study.py --list
"""
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from dataclasses import asdict, dataclass
from functools import partial as functools_partial
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

# ---------------------------------------------------------------------------
# Project / group config
# ---------------------------------------------------------------------------

WANDB_PROJECT = "noisy_expert_degradation_plane"

WANDB_GROUPS = {
    "reference":          "noisy_expert_reference",
    "degradation":        "noisy_expert_degradation",
    "ibrl_reference":     "noisy_ibrl_reference",
    "ibrl_degradation":   "noisy_ibrl_degradation",
    "residual_reference": "noisy_residual_reference",
    "residual_degradation": "noisy_residual_degradation",
}

RUN_REGISTRY_FILE = os.path.abspath("noisy_expert_run_registry.json")

# Altitude range of the Plane environment (used to convert % → metres).
ALTITUDE_RANGE_M = 5_000.0  # 8000 - 3000 m

# Noise levels as % of the altitude range.
# σ_m = noise_pct / 100 * ALTITUDE_RANGE_M
# e.g. 10% → σ = 500 m
NOISE_LEVELS_PCT = [0, 2, 5, 10, 20, 40, 80]


# ---------------------------------------------------------------------------
# Noisy expert factory
# ---------------------------------------------------------------------------

def _noisy_expert_fn(obs, interpolator, noise_std_m: float):
    """PID expert with Gaussian noise on the perceived target altitude.

    obs[..., 6] = target altitude (metres).  A pseudo-random key is derived
    from the integer altitude via fold_in so the function is pure and
    jit-compatible.  jnp.vectorize handles both the un-vmapped (batched obs)
    and vmapped (single obs) call sites in interaction.py.
    """
    def _perturb(t):
        key = jax.random.fold_in(jax.random.PRNGKey(0), jnp.int32(t))
        return jnp.clip(t + jax.random.normal(key) * noise_std_m, 3_000.0, 8_000.0)

    noisy_target = jnp.vectorize(_perturb)(obs[..., 6])
    power = interpolator(noisy_target)[..., None]
    return jnp.concatenate([power, jnp.zeros_like(power)], axis=-1)


def make_noisy_expert_policy(env, env_params, noise_pct: float):
    """Return a PID expert whose perceived target altitude is corrupted by N(0, σ²).

    noise_pct  — noise std as % of the altitude range (ALTITUDE_RANGE_M).
    noise_pct = 0  →  identical to the clean PID (same object, no overhead).
    """
    base_policy = env.expert_policy
    if noise_pct == 0.0:
        return base_policy

    noise_std_m = noise_pct / 100.0 * ALTITUDE_RANGE_M
    from target_gym.interpolator import get_interpolator
    interpolator = jax.jit(get_interpolator(env.__class__, env_params))
    return functools_partial(_noisy_expert_fn, interpolator=interpolator, noise_std_m=noise_std_m)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    name: str
    wandb_group: str
    noise_pct: float = 0.0              # noise std as % of altitude range (0–100)

    # Algorithm selector — controls which baseline is run at each noise level.
    # "ege"      → EDGE with ε-greedy decay (matches ablation_study.py's ege_decay_050)
    # "ibrl"     → IBRL-style argmax gating, no decay (matches ablation_study.py's ibrl_style)
    # "residual" → Residual RL (matches ablation_study.py's residual_rl)
    algorithm: str = "ege"

    use_expert_guided_exploration: bool = True
    exploration_argmax: bool = False
    use_residual_rl: bool = False

    exploration_decay_frac: float = 0.50
    exploration_tau: float = 1.0
    fixed_exploration_prob: float = 0.5

    policy_update_start: int = 5_000
    alpha_update_start: int = 5_000


def build_experiments() -> List[ExpConfig]:
    P = WANDB_GROUPS
    exps = []

    # EDGE — ε-greedy with decay=0.50 (main result, matches ablation_study ege_decay_050)
    for pct in NOISE_LEVELS_PCT:
        label = f"{pct}pct"
        group = P["reference"] if pct == 0 else P["degradation"]
        exps.append(ExpConfig(
            name=f"ege_noise_{label}",
            wandb_group=group,
            noise_pct=float(pct),
            algorithm="ege",
            use_expert_guided_exploration=True,
            exploration_argmax=False,
            use_residual_rl=False,
            exploration_decay_frac=0.50,
            fixed_exploration_prob=0.5,
        ))

    # IBRL — argmax gating, no decay (matches ablation_study ibrl_style)
    for pct in NOISE_LEVELS_PCT:
        label = f"{pct}pct"
        group = P["ibrl_reference"] if pct == 0 else P["ibrl_degradation"]
        exps.append(ExpConfig(
            name=f"ibrl_noise_{label}",
            wandb_group=group,
            noise_pct=float(pct),
            algorithm="ibrl",
            use_expert_guided_exploration=True,
            exploration_argmax=True,
            use_residual_rl=False,
            exploration_decay_frac=0.0,
            fixed_exploration_prob=0.5,
        ))

    # Residual RL — clip(a_expert + a_policy, -1, 1) (matches ablation_study residual_rl)
    for pct in NOISE_LEVELS_PCT:
        label = f"{pct}pct"
        group = P["residual_reference"] if pct == 0 else P["residual_degradation"]
        exps.append(ExpConfig(
            name=f"residual_noise_{label}",
            wandb_group=group,
            noise_pct=float(pct),
            algorithm="residual",
            use_expert_guided_exploration=False,
            exploration_argmax=False,
            use_residual_rl=True,
            exploration_decay_frac=0.0,
            fixed_exploration_prob=0.5,
            policy_update_start=10_000,
            alpha_update_start=10_000,
        ))

    return exps


# ---------------------------------------------------------------------------
# Completion check (mirrors partial_expert_study.py)
# ---------------------------------------------------------------------------

def _last_step_in_run(run_id: str) -> Optional[int]:
    """Return the last logged step of Eval/episodic_mean_reward, or None.

    Using the policy reward tag rather than max-across-all-tags because
    scheduled/value metrics log throughout even when the policy has gone NaN,
    which would otherwise make a crashed run appear complete.
    """
    from ajax.logging.wandb_logging import load_scalars_from_tfevents

    tb_dir = os.path.join(os.path.abspath("tensorboard"), run_id)
    if not os.path.isdir(tb_dir):
        return None
    try:
        scalars = load_scalars_from_tfevents(tb_dir)
        entries = scalars.get("Eval/episodic_mean_reward", [])
        return entries[-1][0] if entries else None
    except Exception:
        return None


def is_experiment_complete(exp_name: str, n_seeds: int, n_timesteps: int) -> bool:
    """True if exp_name has n_seeds runs that each logged ≥99% of n_timesteps.

    For 0pct noise references, also accepts the corresponding ablation_study.py
    experiment as equivalent — no need to re-run an identical experiment.
      ege_noise_0pct      ↔  ablation ege_decay_050
      ibrl_noise_0pct     ↔  ablation ibrl_style
      residual_noise_0pct ↔  ablation residual_rl
    """
    # Mapping: this study's 0-noise reference → ablation_study.py equivalent name.
    _ABLATION_EQUIV = {
        "ege_noise_0pct":      "ege_decay_050",
        "ibrl_noise_0pct":     "ibrl_style",
        "residual_noise_0pct": "residual_rl",
    }
    if exp_name in _ABLATION_EQUIV:
        ablation_registry = os.path.abspath("ablation_run_registry.json")
        if os.path.exists(ablation_registry):
            try:
                with open(ablation_registry) as f:
                    registry = json.load(f)
                threshold = int(n_timesteps * 0.99)
                ablation_name = _ABLATION_EQUIV[exp_name]
                seen = set()
                candidates = []
                for entry in registry:
                    rid = entry.get("run_id")
                    if entry.get("exp_name") == ablation_name and rid not in seen:
                        seen.add(rid)
                        candidates.append(rid)
                finished = sum(1 for rid in candidates if (_last_step_in_run(rid) or 0) >= threshold)
                if finished >= n_seeds:
                    return True
            except Exception:
                pass

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
        pid_policy = env.expert_policy
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

    expert_policy = make_noisy_expert_policy(env, env_params, exp.noise_pct)

    hp = load_hyperparams("SAC", "Plane")
    arch_width = hp.pop("arch_width", 256)
    architecture = (str(arch_width), "relu", str(arch_width), "relu")

    noise_std_m = exp.noise_pct / 100.0 * ALTITUDE_RANGE_M
    # Score the noisy expert against the env (always uses true target altitude).
    expert_score = get_policy_score(expert_policy, env, env_params)
    pid_score = get_policy_score(pid_policy, env, env_params)
    print(
        f"[{exp.name}] algo={exp.algorithm}  noise={exp.noise_pct:.0f}% ({noise_std_m:.0f}m)  "
        f"expert_score={expert_score:.1f}  pid_score={pid_score:.1f}  "
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
        policy_update_start=exp.policy_update_start,
    )

    _agent = SAC(
        env_id=env,
        env_params=env_params,
        expert_policy=expert_policy,
        eval_expert_policy=pid_policy,        # always evaluate against clean PID for fair comparison
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
        residual=exp.use_residual_rl,
        use_expert_guided_exploration=exp.use_expert_guided_exploration,
        exploration_decay_frac=exp.exploration_decay_frac,
        exploration_tau=exp.exploration_tau,
        exploration_boltzmann=False,
        fixed_exploration_prob=exp.fixed_exploration_prob,
        exploration_argmax=exp.exploration_argmax,
        awbc_normalize=True,
        awbc_use_relu=True,
        fixed_awbc_lambda=None,
        mc_variance_threshold=None,
        use_phi_refresh=False,
        target_entropy_far=None,
        policy_update_start=exp.policy_update_start,
        alpha_update_start=exp.alpha_update_start,
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

    n_timesteps      = int(1e6)
    n_seeds          = 100
    num_episode_test = 25
    log_frequency    = 5_000
    use_wandb        = not args.no_wandb
    upload_after     = args.upload_after

    experiments = build_experiments()

    if args.list:
        print(f"\n{len(experiments)} experiments:\n")
        for i, exp in enumerate(experiments):
            noise_std_m = exp.noise_pct / 100.0 * ALTITUDE_RANGE_M
            done = is_experiment_complete(exp.name, n_seeds, n_timesteps)
            status = "DONE" if done else "    "
            print(
                f"  [{i:2d}] {status}  {exp.name:<38}  "
                f"algo={exp.algorithm:<8}  noise={exp.noise_pct:.0f}% ({noise_std_m:.0f}m)  group={exp.wandb_group}"
            )
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
        print(
            f"Sequential sweep: {len(experiments)} experiments × {n_seeds} seeds  "
            f"({done_count} already complete, {len(pending)} to run)\n"
        )
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
