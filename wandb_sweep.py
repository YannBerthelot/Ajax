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

    n_seeds = 5
    log_frequency = 5000
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
    env_id = "halfcheetah"

    def init_and_train(config):
        sac_agent = agent(env_id=env_id, **config)
        _, score = sac_agent.train(
            seed=list(range(n_seeds)),
            n_timesteps=int(2e5),
            logging_config=logging_config,
        )
        return score.mean()

    wandb.init(project=project_name)
    score = init_and_train(wandb.config)
    wandb.log({"score": score})
    wandb.finish()


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
            "distillation_lr": {"max": 1e-3, "min": 1e-5},
            "n_avg_agents": {"values": [1, 2, 4]},
            "num_envs_AVG": {"values": [1, 2, 4]},
            "num_epochs_distillation": {"values": [1, 2, 5]},
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

    num_agents = 2
    try:
        launch_agents(sweep_id, num_agents)
    except KeyboardInterrupt:
        terminate_all_processes()
