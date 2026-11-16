"""Train a grid of learning rates with every seed in one program."""

import numpy as np

from ajax import SAC
from ajax.logging import LoggingConfig


def main(n_timesteps: int = 20_000) -> None:
    agent = SAC("Pendulum-v1", learning_starts=1_000)
    logging = LoggingConfig(log_frequency=n_timesteps // 4, backends=())
    # Three configurations over two seeds: six members, compiled once.
    _, evaluations = agent.train(
        seed=[1, 2],
        n_timesteps=n_timesteps,
        sweep={"actor_learning_rate": [1e-4, 3e-4, 1e-3]},
        logging_config=logging,
    )

    last = np.asarray(evaluations["Eval/episodic mean reward"])[:, -1]
    # One row per member, in the order agent.members lists them.
    for member, final_return in zip(agent.members, last, strict=True):
        print(member, f"final return {float(final_return):8.1f}")


if __name__ == "__main__":
    main()
