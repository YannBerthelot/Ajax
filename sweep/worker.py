from functools import partial

import optuna
import yaml
from train import train
from utils import AGENT_MAP, get_args, get_log_config_for_sweep, get_study


def load_search_space(yaml_path, env_id):
    """Load hyperparameter definitions from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config[env_id]


def suggest_from_yaml(trial: optuna.trial.Trial, search_space):
    """Use Optuna trial to suggest values based on the YAML spec."""
    config = {}

    for name, params in search_space.items():
        ptype = params["type"]

        if ptype == "float":
            low = float(params["low"])
            high = float(params["high"])
            log = params.get("log", False)
            config[name] = trial.suggest_float(name, low, high, log=log)

        elif ptype == "int":
            low = int(params["low"])
            high = int(params["high"])
            log = params.get("log", False)
            config[name] = trial.suggest_int(name, low, high, log=log)

        elif ptype == "categorical":
            choices = params["choices"]
            config[name] = trial.suggest_categorical(name, choices)

        else:
            raise ValueError(f"Unknown parameter type: {ptype}")

    return config


def objective(
    trial: optuna.trial.Trial,
    agent,
    env_id,
    n_seeds,
    n_timesteps,
    num_episode_test,
    env_params,
    logging_config,
    config_name: str,
):
    # # wandb.init(project="rl-optuna", config=trial.params)
    # actor_learning_rate = trial.suggest_float(
    #     "actor_learning_rate", 1e-5, 1e-3, log=True
    # )
    # critic_learning_rate = trial.suggest_float(
    #     "critic_learning_rate", 1e-5, 1e-3, log=True
    # )
    # n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    # ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-3, log=True)
    # gamma = trial.suggest_float("gamma", 0.9, 0.999)
    # n_neurons = trial.suggest_categorical("n_neurons", [64, 128, 256])
    # config = {
    #     "actor_learning_rate": actor_learning_rate,
    #     "critic_learning_rate": critic_learning_rate,
    #     "gamma": gamma,
    #     "n_neurons": n_neurons,
    #     "n_steps": n_steps,
    #     "ent_coef": ent_coef,

    # }
    config = suggest_from_yaml(
        trial,
        load_search_space(f"sweep/configs/{agent.name}/{config_name}.yaml", env_id),
    )
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


def main(args):
    if args.n_workers > 1:
        launch_workers(args)
    else:
        worker_main(args)


def launch_workers(args):
    import multiprocessing

    processes = []
    for _ in range(args.n_workers):
        p = multiprocessing.Process(target=partial(worker_main, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def worker_main(args):
    study = get_study(args)

    objective_fn = partial(
        objective,
        agent=AGENT_MAP[args.agent],
        env_id=args.env_id,
        n_seeds=args.n_seeds,
        n_timesteps=args.n_timesteps,
        num_episode_test=args.n_episode_test,
        env_params=None,
        logging_config=get_log_config_for_sweep(args.n_seeds),
        config_name=args.config_name,
    )
    study.optimize(objective_fn, n_trials=args.n_trials)


if __name__ == "__main__":
    args = get_args()
    main(args)
