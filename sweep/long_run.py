import json
from time import time

from utils import get_args, get_study


def optuna_config_to_actual_config(config: dict) -> dict:
    N_NEURONS = config.pop("n_neurons", 128)
    activation = config.pop("activation", "relu")
    config["actor_architecture"] = (
        (f"{N_NEURONS}", activation, f"{N_NEURONS}", activation),
    )
    config["critic_architecture"] = (
        f"{N_NEURONS}",
        activation,
        f"{N_NEURONS}",
        activation,
    )
    return config


def get_best_n_configs(study, n=5):
    """
    Return the best N configurations from an Optuna study.

    Args:
        study (optuna.Study): Completed Optuna study object.
        n (int): Number of best configurations to return.
        include_values (bool): Whether to include the objective values.

    Returns:
        list: A list of dicts containing parameters (and optionally their values).
    """
    # Sort trials by their objective value (descending for maximize, ascending for minimize)
    reverse = study.direction.name == "MAXIMIZE"
    valid_trials = [t for t in study.trials if t.value is not None]

    best_trials = sorted(valid_trials, key=lambda t: t.value, reverse=reverse)[:n]

    results = []
    for t in best_trials:
        config = t.params.copy()
        results.append({"config": config, "value": t.value})

    return results


def main(args):
    study = get_study(args)

    optuna_best_n_configs = get_best_n_configs(study, n=args.subset_size)
    print(f"Best {args.subset_size} configs from study: {optuna_best_n_configs}")
    with open(f"sweep/{args.agent}/best_n_configs.json", "w") as f:
        json.dump(optuna_best_n_configs, f, indent=4)


if __name__ == "__main__":
    start = time()
    args = get_args()
    main(args)


#     from joblib import Parallel, delayed
#     results = Parallel(n_jobs=4)(
#     delayed(train)(
#         cfg.copy(),
#         agent,
#         env_id,
#         n_seeds_long,
#         n_timesteps_long,
#         num_episode_test=25,
#         env_params=None,
#         logging_config=logging_config,
#     )
#     for cfg in optuna_best_n_configs
# )
#     print(
#         "best config:",
#         optuna_best_n_configs[np.argmax(results)],
#         "with reward:",
#         np.max(results),
#     )

#     # Load existing JSON if it exists
#     path = f"sweep/{agent.name}/best_config.json"
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             data = json.load(f)
#     else:
#         data = {}
#     data[env_id] = optuna_best_n_configs[np.argmax(results)]
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)
#     print(f"Long runs finished in {time() - start:.2f} seconds")
