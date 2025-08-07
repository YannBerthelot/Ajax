import multiprocessing
import signal
import sys
from typing import List

from ajax.agents import DynaSACMulti
from ajax.logging.wandb_logging import LoggingConfig

processes: List = []

agent = DynaSACMulti
project_name = "dyna_sac_tests_sweep"


def main():
    import wandb

    run = wandb.init(project=project_name)
    n_seeds = 10
    log_frequency = 20_000
    logging_config = None
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
        use_wandb=True,
    )
    env_id = "hopper"

    N_NEURONS = 32

    def init_and_train(config):
        sac_agent = agent(
            env_id=env_id,
            learning_starts=0,
            sac_length=1,
            # transition_mix_fraction=0.5,
            **config,
            actor_architecture=(f"{N_NEURONS}", "relu", f"{N_NEURONS}", "relu"),
            critic_architecture=(f"{N_NEURONS}", "relu", f"{N_NEURONS}", "relu"),
            model_noise=1.0
        )
        _, score = sac_agent.train(
            seed=list(range(n_seeds)),
            n_timesteps=int(1e5),
            logging_config=logging_config,
            sweep=True,
        )
        print(score, score.mean())
        return score.mean()

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
    multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGINT, signal_handler)

    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            # "actor_distillation_lr": {"max": 1e-3, "min": 1e-5},
            # "critic_distillation_lr": {"max": 1e-3, "min": 1e-5},
            # "n_avg_agents": {"values": [1]},
            "num_envs_AVG": {"values": [1, 8, 32, 128]},
            "avg_length": {"values": [1]},
            # "num_epochs_distillation": {"values": [3]},
            # "n_distillation_samples": {"values": [256]},
            # "alpha_polyak_primary_to_secondary": {"max": 1e-1, "min": 1e-3},
            # "initial_alpha_polyak_secondary_to_primary": {"max": 1e-3, "min": 1e-5},
            # "final_alpha_polyak_secondary_to_primary": {"max": 1e-1, "min": 1e-3},
            "transition_mix_fraction": {"max": 0.99, "min": 0.8},
        },
    }

    # 🟩 Create sweep in subprocess and retrieve the ID safely
    sweep_id_queue: multiprocessing.Queue = multiprocessing.Queue()
    sweep_proc = multiprocessing.Process(
        target=create_sweep,
        args=(sweep_id_queue, sweep_configuration, project_name),
    )
    sweep_proc.start()
    sweep_proc.join()
    sweep_id = sweep_id_queue.get()

    num_agents = 1
    try:
        launch_agents(sweep_id, num_agents)
    except KeyboardInterrupt:
        terminate_all_processes()
