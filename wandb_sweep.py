import multiprocessing
import signal
import sys
from typing import List
import wandb
from ajax.agents import SAC, PPO
from ajax.logging.wandb_logging import LoggingConfig
from target_gym import Plane, PlaneParams
import numpy as np

processes: List = []

agent = PPO
project_name = f"Plane_{agent.name}_sweep_clip"


def main():
    import wandb

    run = wandb.init(project=project_name)
    n_seeds = 10
    n_timesteps = int(1e6) if agent is SAC else int(1e6)
    log_frequency = 10_000
    logging_config = LoggingConfig(
        project_name=project_name,
        run_name="run",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=False,
        use_wandb=False,
    )
    env_id = Plane(integration_method="rk4_1")
    env_params = PlaneParams(
        target_altitude_range=(5000.0, 5000.0),
    )

    def init_and_train(config):
        config = config.as_dict()
        N_NEURONS = config.pop("n_neurons")
        activation = config.pop("activation")
        if "n_steps" in config.keys():
            _logging_config = logging_config.replace(
                log_frequency=config["n_steps"]
            )  # TODO : make sure its a good fit
        else:
            _logging_config = logging_config

        _agent = agent(
            env_id=env_id,
            **config,
            actor_architecture=(f"{N_NEURONS}", activation, f"{N_NEURONS}", activation),
            critic_architecture=(
                f"{N_NEURONS}",
                activation,
                f"{N_NEURONS}",
                activation,
            ),
            env_params=env_params,
        )
        _, out = _agent.train(
            seed=list(range(n_seeds)),
            n_timesteps=n_timesteps,
            logging_config=_logging_config,
        )
        score = out["Eval/episodic mean reward"]
        print(score, len(score[0]))
        print(score[out["timestep"] > 0.9 * n_timesteps])
        score = np.nanmean(score[out["timestep"] > 0.9 * n_timesteps])
        return score

    score = init_and_train(wandb.config)
    run.log({"score": score})


def run_wandb_agent(sweep_id):
    import wandb

    try:
        wandb.agent(sweep_id, function=main, project=project_name)
    except Exception as e:
        print(f"Agent for sweep {sweep_id} exited with error: {e}")


def create_sweep(queue, sweep_config, project_name):
    import wandb

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    queue.put(sweep_id)


def terminate_all_processes():
    print("\nTerminating all wandb agents...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()
    print("All agents terminated.")


def signal_handler(sig, frame):
    terminate_all_processes()
    sys.exit(0)


def launch_agents(sweep_id, num_agents):
    for _ in range(num_agents):
        p = multiprocessing.Process(target=run_wandb_agent, args=(sweep_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, signal_handler)
    method = "bayes"
    if agent is PPO:
        sweep_configuration = {
            "method": method,
            "metric": {"goal": "maximize", "name": "score"},
            "parameters": {
                "actor_learning_rate": {"max": 1e-2, "min": 1e-5},
                "critic_learning_rate": {"max": 1e-2, "min": 1e-5},
                "n_envs": {"values": [1, 4, 8]},
                "activation": {"values": ["relu", "tanh"]},
                "n_neurons": {"values": [32, 64, 128, 256]},
                "gamma": {"max": 0.999, "min": 0.9},
                "ent_coef": {"max": 1e-2, "min": 0.0},
                "clip_range": {"max": 0.3, "min": 0.0},
                "n_steps": {"values": [1024, 2048, 4096]},
            },
        }
    elif agent is SAC:
        sweep_configuration = {
            "method": method,
            "metric": {"goal": "maximize", "name": "score"},
            "parameters": {
                "actor_learning_rate": {"max": 1e-2, "min": 1e-5},
                "critic_learning_rate": {"max": 1e-2, "min": 1e-5},
                "alpha_learning_rate": {"max": 1e-2, "min": 1e-5},
                "activation": {"values": ["relu", "tanh"]},
                "n_neurons": {"values": [64, 128, 256]},
                "gamma": {"max": 0.999, "min": 0.9},
                "target_entropy_per_dim": {"min": -1.0, "max": 1.0},
                "batch_size": {"values": [128, 256, 512]},
                "tau": {"min": 1e-3, "max": 1e-2},
            },
        }

    try:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=main, project=project_name, count=500)
    except KeyboardInterrupt:
        pass
