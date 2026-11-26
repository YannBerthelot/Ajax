"""Train a grid of learning rates with every seed in one program."""

from ajax import SAC


def main(n_timesteps: int = 20_000) -> None:
    agent = SAC("Pendulum-v1", learning_starts=1_000)
    # Three configurations over two seeds: six members, compiled once.
    _, evaluations = agent.train(
        seed=[1, 2],
        n_timesteps=n_timesteps,
        sweep={"actor_learning_rate": [1e-4, 3e-4, 1e-3]},
    )

    # Each configuration with its returns, a row per seed.
    for configuration, returns in agent.by_configuration(evaluations.returns):
        print(configuration, f"final return {float(returns[:, -1].mean()):8.1f}")


if __name__ == "__main__":
    main()
