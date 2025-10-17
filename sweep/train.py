import numpy as np


def train(
    config: dict,
    agent,
    env_id,
    n_seeds,
    n_timesteps,
    num_episode_test,
    env_params,
    logging_config,
):
    N_NEURONS = config.pop("n_neurons", 128)
    activation = config.pop("activation", "relu")

    normalize_observations = config.pop("normalize_observations", False)
    normalize_rewards = config.pop("normalize_rewards", False)

    if "n_steps" in config.keys():
        _logging_config = logging_config.replace(log_frequency=config["n_steps"])
    else:
        _logging_config = logging_config

    if "critic_architecture" not in config:
        config["critic_architecture"] = (
            f"{N_NEURONS}",
            activation,
            f"{N_NEURONS}",
            activation,
        )
    if "actor_architecture" not in config:
        config["actor_architecture"] = (
            f"{N_NEURONS}",
            activation,
            f"{N_NEURONS}",
            activation,
        )

    _agent = agent(
        env_id=env_id,
        **config,
        env_params=env_params,
        normalize_observations=normalize_observations,
        normalize_rewards=normalize_rewards,
    )
    _, out = _agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=n_timesteps,
        logging_config=_logging_config,
        num_episode_test=num_episode_test,
    )
    score = out["Eval/episodic mean reward"]
    # print(score, len(score[0]))
    # print(score[out["timestep"] > 0.8 * n_timesteps])
    score = np.nanmean(score[out["timestep"] > 0.8 * n_timesteps])
    return score
