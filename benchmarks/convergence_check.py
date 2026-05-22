"""Convergence-parity check: fp32 vs bf16 vs bf16_full.

Trains the same SAC config with matched seeds in each precision mode,
then evaluates each final policy. Per-seed final returns are reported
so the caller can compare distributions across modes (any large gap
indicates bf16 broke convergence).

The mode is selected via env vars before this script imports ajax:
  (none)              -> fp32 (default)
  AJAX_BF16_NETS=1    -> bf16 compute (params fp32)
  AJAX_BF16_NETS=1
    AJAX_BF16_PARAMS=1 -> full bf16

Usage (run separately for each mode so env vars apply at import time):

    python benchmarks/convergence_check.py \
        --tag fp32 \
        --env Pendulum-v1 \
        --n-timesteps 30000 \
        --seeds 0,1,2,3,4,5,6,7 \
        --eval-episodes 10 \
        --out benchmarks/convergence.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List


def _eval_returns(agent, final_state, eval_seed: int, num_episodes: int) -> List[float]:
    """Evaluate each seed's final actor on `num_episodes` rollouts.

    Uses Ajax's jitted ``evaluate`` directly (vmaps over the seed axis
    of the trained actor states). Returns the per-seed mean return.
    """
    import jax
    import jax.numpy as jnp
    from ajax.evaluate import evaluate

    actor_state = final_state.actor_state if hasattr(final_state, "actor_state") else final_state[0].actor_state

    if isinstance(final_state, tuple):
        actor_state = final_state[0].actor_state
    else:
        actor_state = final_state.actor_state

    rng = jax.random.PRNGKey(eval_seed)

    def _eval_one_seed(actor_state_one_seed, key):
        return evaluate(
            env=agent.env_args.env,
            actor_state=actor_state_one_seed,
            num_episodes=num_episodes,
            rng=key,
            env_params=agent.env_args.env_params,
            recurrent=False,
            lstm_hidden_size=None,
            gamma=getattr(agent, "gamma", 0.99),
        )

    n_seeds = jax.tree_util.tree_leaves(actor_state.params)[0].shape[0]
    keys = jax.random.split(rng, n_seeds)
    rewards, *_ = jax.vmap(_eval_one_seed, in_axes=(0, 0))(actor_state, keys)
    rewards = jax.device_get(rewards)
    return [float(r) for r in rewards]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Mode label (fp32 / bf16_nets / bf16_full)")
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--n-timesteps", type=int, default=30_000)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated seed list; same seeds across modes for matched comparison.")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--num-critics", type=int, default=2)
    p.add_argument("--arch-width", type=int, default=64)
    p.add_argument("--learning-starts", type=int, default=None,
                   help="Override SAC learning_starts (default: min(2000, n_timesteps//10)).")
    p.add_argument("--params-json", default=None,
                   help="Load HPO-tuned SAC hyperparams from a trial_NNN_params.json. "
                        "Overrides --arch-width / --num-critics if those are set in the file.")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "convergence.jsonl"))
    p.add_argument("--match-hpo-metric", action="store_true",
                   help="Wire a LoggingConfig + tensorboard during training "
                        "and report the last logged Eval/episodic_mean_reward "
                        "scalar (the metric AjaxExperiments' "
                        "`read_final_metric` reads), in addition to the "
                        "fresh post-training eval.")
    p.add_argument("--tb-folder", default=None,
                   help="Tensorboard folder for --match-hpo-metric. "
                        "Default: a per-tag temp dir.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Echo the bf16 env vars so the JSON record carries provenance.
    bf16_nets = os.environ.get("AJAX_BF16_NETS", "")
    bf16_params = os.environ.get("AJAX_BF16_PARAMS", "")
    bf16_critic = os.environ.get("AJAX_BF16_CRITIC", "")

    # Import after env vars are read.
    import jax
    from ajax.agents.SAC.SAC import SAC

    # Resolve env. For AjaxExperiments / target_gym envs we delegate
    # to the same `make_env_by_name` the HPO harness uses, so episode
    # length and env params match the HPO config exactly. Otherwise
    # fall through to the env_id string (gymnax / brax /
    # mujoco_playground resolved by Ajax's `build_env_from_id`).
    env_id_arg = args.env
    env_params_override = None
    try:
        import sys as _sys
        if "/home/yberthel/AjaxExperiments" not in _sys.path:
            _sys.path.insert(0, "/home/yberthel/AjaxExperiments")
        from envs import make_env_by_name as _make_env_by_name

        env_obj, env_params = _make_env_by_name(args.env)
        env_id_arg = env_obj
        env_params_override = env_params
    except Exception:
        # Env not in AjaxExperiments registry; fall back to the SAC
        # default resolver. (Pendulum-v1 etc. land here.)
        pass

    # Optionally load HPO-tuned hyperparams (from
    # AjaxExperiments/hp_results/<env>/<method>/trial_NNN_params.json)
    # so the parity test runs at a config that fp32 actually converges
    # at. Default is a generic SAC config; for "real" parity on
    # bandwidth-bound envs prefer passing --params-json.
    hp = {}
    if args.params_json:
        with open(args.params_json) as f:
            hp = json.load(f)

    arch_width = int(hp.get("arch_width") or args.arch_width)
    num_critics = int(hp.get("num_critics") or args.num_critics)
    arch = (str(arch_width), "relu", str(arch_width), "relu")
    learning_starts = (
        args.learning_starts
        if args.learning_starts is not None
        else min(2000, args.n_timesteps // 10)
    )

    sac_kwargs = dict(
        env_id=env_id_arg,
        n_envs=args.n_envs,
        learning_starts=learning_starts,
        actor_architecture=arch,
        critic_architecture=arch,
        num_critics=num_critics,
        batch_size=int(hp.get("batch_size") or 256),
        buffer_size=int(min(1e5, args.n_timesteps)),
        # Match HPO `_build_method_kwargs("sac")`: agent-side running
        # obs normalisation is on by default for the HPO study.
        normalize_obs_running=True,
    )
    if env_params_override is not None:
        sac_kwargs["env_params"] = env_params_override
    # Whitelist of HPO keys SAC accepts directly. Skip keys that are
    # None or method-specific (exploration_*, lcb_*, residual_scale,
    # etc.) since this harness only varies precision, not method.
    SAC_KEYS = {
        "actor_learning_rate": "actor_lr",
        "critic_learning_rate": "critic_lr",
        "alpha_learning_rate": "alpha_lr",
        "gamma": "gamma",
        "tau": "tau",
        "alpha_init": "alpha_init",
        "target_entropy_per_dim": "target_entropy_per_dim",
        "max_grad_norm": "max_grad_norm",
    }
    for sac_key, hp_key in SAC_KEYS.items():
        if hp.get(hp_key) is not None:
            sac_kwargs[sac_key] = hp[hp_key]
    agent = SAC(**sac_kwargs)

    # Optionally set up an HPO-style LoggingConfig so the run writes the
    # same Eval/episodic_mean_reward tensorboard scalar that
    # AjaxExperiments' read_final_metric loads. This lets us report
    # exactly what HPO would have reported (Phase-1 metric =
    # last logged value) for the same training run.
    logging_config = None
    tb_folder = None
    captured_run_ids: list = []
    if args.match_hpo_metric:
        import tempfile
        from ajax.logging.wandb_logging import LoggingConfig

        tb_folder = args.tb_folder or tempfile.mkdtemp(
            prefix=f"ajax_conv_{args.tag}_"
        )
        logging_config = LoggingConfig(
            project_name=f"convergence_{args.tag}",
            run_name=f"{args.tag}_{args.env}",
            config={"tag": args.tag, **{k: hp.get(k) for k in hp}},
            log_frequency=max(1000, args.n_timesteps // 30),
            horizon=10_000,
            folder=tb_folder,
            use_tensorboard=True,
            use_wandb=False,
            sweep=False,
        )

        def _capture_ids(ids):
            captured_run_ids.extend(ids)

        train_kwargs = {"on_ids_ready": _capture_ids,
                        "logging_config": logging_config}
    else:
        train_kwargs = {}

    t0 = time.perf_counter()
    final = agent.train(seed=seeds, n_timesteps=args.n_timesteps,
                        **train_kwargs)
    jax.block_until_ready(final)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_seed_returns = _eval_returns(
        agent, final, eval_seed=12345, num_episodes=args.eval_episodes,
    )
    eval_s = time.perf_counter() - t1

    n = len(per_seed_returns)
    mean = sum(per_seed_returns) / n
    var = sum((r - mean) ** 2 for r in per_seed_returns) / max(n - 1, 1)
    std = var ** 0.5

    # Re-implement HPO's read_final_metric: load the last logged
    # Eval/episodic_mean_reward per run_id, average across seeds.
    hpo_metric = None
    hpo_per_seed = None
    if logging_config is not None and captured_run_ids:
        try:
            from ajax.logging.wandb_logging import load_scalars_from_tfevents

            EVAL_TAG = "Eval/episodic_mean_reward"
            seed_scores = []
            for rid in captured_run_ids:
                log_dir = os.path.join(tb_folder, "tensorboard", rid)
                if not os.path.exists(log_dir):
                    continue
                try:
                    scalars = load_scalars_from_tfevents(log_dir)
                except Exception:
                    continue
                if EVAL_TAG not in scalars or not scalars[EVAL_TAG]:
                    continue
                seed_scores.append(scalars[EVAL_TAG][-1][1])
            if seed_scores:
                hpo_metric = float(sum(seed_scores) / len(seed_scores))
                hpo_per_seed = [float(s) for s in seed_scores]
        except Exception as exc:
            print(f"[hpo-metric load failed: {exc}]", flush=True)

    record = {
        "tag": args.tag,
        "env": args.env,
        "n_timesteps": args.n_timesteps,
        "seeds": seeds,
        "n_envs": args.n_envs,
        "num_critics": num_critics,
        "arch_width": arch_width,
        "params_json": args.params_json,
        "eval_episodes": args.eval_episodes,
        "per_seed_returns": per_seed_returns,
        "mean_return": mean,
        "std_return": std,
        "min_return": min(per_seed_returns),
        "max_return": max(per_seed_returns),
        "train_s": train_s,
        "eval_s": eval_s,
        "AJAX_BF16_NETS": bf16_nets,
        "AJAX_BF16_PARAMS": bf16_params,
        "AJAX_BF16_CRITIC": bf16_critic,
        "hpo_metric_last_eval_tag": hpo_metric,
        "hpo_metric_per_seed": hpo_per_seed,
        "tb_folder": tb_folder,
    }
    print(json.dumps(record))
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
