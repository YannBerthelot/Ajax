"""SVGD-style kernel repulsion vs vanilla REDQ on Hopper.

Two runs on Hopper at 1M timesteps with 25 seeds each:
  1. vanilla REDQ  (repulsion_coef = 0.0)
  2. SVGD-REDQ     (repulsion_coef = <coef>)

The repulsion term is a function-space RBF kernel on the per-critic Q-value
outputs, with bandwidth set by the median heuristic (see
`ajax.agents.REDQ.train_REDQ.q_kernel_repulsion`). Minimising the mean
kernel value pushes ensemble members apart in prediction space; this is a
simple add-on to the Bellman loss (SVPG-style).

WandB grouping: each variant logs under the same project with `run_name`
differing so the dashboard can group on it.
"""

from __future__ import annotations

import argparse

from ajax.agents.REDQ.REDQ import REDQ
from ajax.logging.wandb_logging import LoggingConfig


def run_variant(
    env_id: str,
    repulsion_coef: float,
    run_name: str,
    n_seeds: int,
    n_timesteps: int,
    log_frequency: int,
    project_name: str,
    use_wandb: bool,
    use_tensorboard: bool,
) -> None:
    logging_config = LoggingConfig(
        project_name=project_name,
        run_name=run_name,
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "repulsion_coef": repulsion_coef,
            "variant": run_name,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
    )
    agent = REDQ(
        env_id=env_id,
        learning_starts=int(1e4),
        n_envs=1,
        repulsion_coef=repulsion_coef,
    )
    agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=n_timesteps,
        logging_config=logging_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="hopper")
    parser.add_argument("--n_seeds", type=int, default=25)
    parser.add_argument("--n_timesteps", type=int, default=int(1e6))
    parser.add_argument("--log_frequency", type=int, default=5_000)
    parser.add_argument("--repulsion_coef", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="REDQ_svgd_benchmark")
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["both", "vanilla", "svgd"],
        help="Which variant(s) to run.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=True,
        help="Disable wandb logging (default: disabled; monitor via tensorboard).",
    )
    parser.add_argument(
        "--wandb",
        dest="no_wandb",
        action="store_false",
        help="Re-enable wandb logging.",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable tensorboard logging (on by default).",
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    use_tensorboard = not args.no_tensorboard

    if args.variant in ("vanilla", "both"):
        run_variant(
            env_id=args.env_id,
            repulsion_coef=0.0,
            run_name="vanilla_REDQ",
            n_seeds=args.n_seeds,
            n_timesteps=args.n_timesteps,
            log_frequency=args.log_frequency,
            project_name=args.project_name,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
        )

    if args.variant in ("svgd", "both"):
        run_variant(
            env_id=args.env_id,
            repulsion_coef=args.repulsion_coef,
            run_name=f"svgd_REDQ_coef{args.repulsion_coef}",
            n_seeds=args.n_seeds,
            n_timesteps=args.n_timesteps,
            log_frequency=args.log_frequency,
            project_name=args.project_name,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
        )


if __name__ == "__main__":
    main()
