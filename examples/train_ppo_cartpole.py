"""Train PPO on CartPole with three seeds and read the evaluations."""

from ajax import PPO
from ajax.logging import LoggingConfig


def main(n_timesteps: int = 32_768) -> None:
    agent = PPO("CartPole-v1", n_envs=8, n_steps=128)  # 1,024 steps per rollout
    # Evaluate every 4,096 steps; backends=() keeps the results in memory only.
    logging = LoggingConfig(log_frequency=4_096, backends=())
    _, evaluations = agent.train(
        seed=[1, 2, 3], n_timesteps=n_timesteps, logging_config=logging
    )

    steps = evaluations["timestep"][0]
    returns = evaluations["Eval/episodic mean reward"]  # seeds x evaluations
    for step, mean in zip(steps, returns.mean(axis=0), strict=True):
        print(f"step {int(step):6d}: mean return over seeds {float(mean):6.1f}")


if __name__ == "__main__":
    main()
