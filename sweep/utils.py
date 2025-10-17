import argparse

import optuna

from ajax import APO, ASAC, PPO, REDQ, SAC
from ajax.logging.wandb_logging import LoggingConfig

# Map string names to agent classes
AGENT_MAP = {"PPO": PPO, "SAC": SAC, "APO": APO, "ASAC": ASAC, "REDQ": REDQ}


def get_log_config_for_sweep(n_seeds: int, agent_name, short=True) -> LoggingConfig:
    return LoggingConfig(
        project_name=f"{agent_name}_optuna_sweep_{'short' if short else 'long'}",
        run_name=agent_name,
        config={
            "debug": False,
            "log_frequency": 20_000,
            "n_seeds": n_seeds,
        },
        log_frequency=20_000,
        horizon=10_000,
        use_tensorboard=False,
        use_wandb=not (short),
        sweep=short,
    )


def get_study(args):
    return optuna.load_study(
        study_name=f"{args.agent}-{args.env_id}-study",
        storage=f"sqlite:///sweep/{args.agent}/study.db",
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Run optuna hyperparam search with a chosen agent."
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=AGENT_MAP.keys(),
        required=True,
        help="Which agent to run (PPO, SAC, APO)",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Which env to run",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        required=False,
        default=1,
        help="Number of parallel workers to use for hyperparam search",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        required=False,
        default=10,
        help="Number of seeds to run per combination",
    )
    parser.add_argument(
        "--n_seeds_long",
        type=int,
        required=False,
        default=25,
        help="Number of seeds to run per combination",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        required=False,
        default=300_000,
        help="Number of timesteps for hyperparam search",
    )
    parser.add_argument(
        "--n_timesteps_long",
        type=int,
        required=False,
        default=1_000_000,
        help="Number of timesteps for long run",
    )
    parser.add_argument(
        "--n_episode_test",
        type=int,
        required=False,
        default=25,
        help="Number test episodes to run per seed",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        required=False,
        default=100,
        help="Number of combination per worker",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        required=False,
        default=10,
        help="Number of agents to select for long run",
    )
    parser.add_argument(
        "--new_study",
        type=bool,
        required=False,
        default=False,
        help="Start a new study (deletes old one if exists)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=False,
        default=None,
        help="Config to use for agent (without .yaml extension)",
    )
    parser.add_argument(
        "--worker-trial",
        type=int,
        required=False,
        default=1,
        help="Config to use for agent (without .yaml extension)",
    )
    args = parser.parse_args()

    if args.config_name is None:
        args.config_name = args.agent
    return args
