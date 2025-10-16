import json
import os
from functools import partial
import numpy as np

import optuna
import wandb
from ajax.agents.PPO.PPO import PPO
from ajax.logging.wandb_logging import LoggingConfig
from train import train
from time import time

def objective(
    trial: optuna.trial.Trial,
    agent,
    env_id,
    n_seeds,
    n_timesteps,
    num_episode_test,
    env_params,
    logging_config,
):
    #wandb.init(project="rl-optuna", config=trial.params)
    actor_learning_rate = trial.suggest_float("actor_learning_rate", 1e-5, 1e-3, log=True)
    critic_learning_rate = trial.suggest_float("critic_learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    n_neurons = trial.suggest_categorical("n_neurons", [64, 128, 256])
    config = {
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "gamma": gamma,
        "n_neurons": n_neurons,
        "n_steps": n_steps,
        "n_epochs": 10,
        "batch_size": 64,
        "activation": "relu",
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": ent_coef,
        "normalize_observations": False,
        "normalize_rewards": False,
        "max_grad_norm": 0.5,
    }
    reward = train(
        config,
        agent,
        env_id,
        n_seeds,
        n_timesteps,
        num_episode_test,
        env_params,
        logging_config,
    )
    # wandb.log({"reward": reward})
    # wandb.finish()
    return reward

def optuna_config_to_actual_config(config:dict)->dict:
    N_NEURONS = config.pop("n_neurons", 128)
    activation = config.pop("activation", "relu")
    config["actor_architecture"]=(f"{N_NEURONS}", activation, f"{N_NEURONS}", activation),
    config["critic_architecture"]=(
                f"{N_NEURONS}",
                activation,
                f"{N_NEURONS}",
                activation,
            )
    return config


def get_best_n_configs(study, n=5, include_values=True):
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
    for i, t in enumerate(best_trials, 1):
        config = t.params.copy()
        results.append(config)

    return results

def analyse_study(study: optuna.study.Study, agent):
    N = 5
    best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:N]

    df = study.trials_dataframe()
    df_sorted = df.sort_values(by="value", ascending=False)
    best_n = df_sorted.head(N)
    os.makedirs(f"sweep/{agent.name}", exist_ok=True)
    best_n.to_csv(f"sweep/{agent.name}/best_trials.csv", index=False)

    for i, trial in enumerate(best_trials, 1):
        print(f"Rank {i}")
        print(f"  Value: {trial.value}")
        print(f"  Params: {trial.params}\n")

if __name__ == "__main__":
    agent = PPO

    n_combinations = 100
    n_seeds = 5
    n_timesteps = 100_000


    subset_size =  10
    n_seeds_long = 5
    n_timesteps_long = 300_000

    n_workers = 1

    log_frequency = 20_000
    logging_config = LoggingConfig(
        project_name="mission_debug_PPO_Ant_3",
        run_name="PPO",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=False,
        use_wandb=False,
        sweep=True,
    )
    env_id = "CartPole-v1"


    start_time = time()
    objective_short = partial(
        objective,
        agent=agent,
        env_id=env_id,
        n_seeds=n_seeds,
        n_timesteps=n_timesteps,
        num_episode_test=25,
        env_params=None,
        logging_config=logging_config,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_short, n_trials=n_combinations, n_jobs=n_workers, show_progress_bar=True)

    print(f"Optimization finished in {time() - start_time:.2f} seconds")

    ############
    # Analysis #
    ############
    start = time()
    analyse_study(study, agent)

    optuna_best_n_configs = get_best_n_configs(study, n=subset_size, include_values=False)
    results = [train(
        config.copy(),
        agent,
        env_id,
        n_seeds_long,
        n_timesteps_long,
        num_episode_test=25,
        env_params=None,
        logging_config=logging_config,
    ) for config in optuna_best_n_configs]
    print("best config:", optuna_best_n_configs[np.argmax(results)], "with reward:", np.max(results))

    # Load existing JSON if it exists
    path = f"sweep/{agent.name}/best_config.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[env_id] = optuna_best_n_configs[np.argmax(results)]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Long runs finished in {time() - start:.2f} seconds")
