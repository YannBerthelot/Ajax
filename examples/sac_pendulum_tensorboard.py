"""Train SAC on Pendulum and log the runs to TensorBoard.

Afterwards: ``tensorboard --logdir runs``.
"""

from ajax import SAC
from ajax.logging import LoggingConfig, TensorBoardBackend


def main(n_timesteps: int = 30_000, folder: str = "runs") -> None:
    # SAC waits learning_starts steps (10,000 by default) before its first
    # update; Pendulum needs fewer.
    agent = SAC("Pendulum-v1", learning_starts=1_000)
    logging = LoggingConfig(
        run_name="sac_pendulum",
        log_frequency=1_000,
        backends=(TensorBoardBackend(folder),),  # <folder>/tensorboard/<run id>/
    )
    _, evaluations = agent.train(
        seed=[1, 2], n_timesteps=n_timesteps, logging_config=logging
    )
    print("last evaluation, per seed:", evaluations.returns[:, -1])
    print("runs logged:", agent.run_ids)


if __name__ == "__main__":
    main()
