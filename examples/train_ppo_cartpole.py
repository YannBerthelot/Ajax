"""Train PPO on CartPole with three seeds and read the evaluations."""

from ajax import PPO


def main(n_timesteps: int = 40_960) -> None:
    agent = PPO("CartPole-v1", n_envs=8, n_steps=128)  # 1,024 steps per rollout
    # Without a logging configuration, train evaluates about ten times and
    # prints a line for each.
    state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=n_timesteps)

    returns = evaluations.returns  # seeds x evaluations
    print("evaluated at steps:", evaluations.timesteps)
    print("last evaluation, per seed:", returns[:, -1])
    # Fresh episodes of each seed's trained policy: seeds x episodes.
    print("20 new episodes, per seed:", agent.evaluate(state, episodes=20).mean(axis=1))


if __name__ == "__main__":
    main()
