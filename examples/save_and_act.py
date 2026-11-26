"""Save a trained agent, load it and play an episode."""

import os
import tempfile

import gymnax
import jax

import ajax
from ajax import PPO


def main(n_timesteps: int = 32_768) -> float:
    agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
    state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
    # The file holds the state with the agent's class and arguments.
    agent.save(state, path)

    agent, state = ajax.load(path)

    env, params = gymnax.make("CartPole-v1")
    key = jax.random.PRNGKey(0)
    obs, env_state = env.reset(key, params)
    episode_return, done = 0.0, False
    while not done:
        action = agent.predict(state, obs)  # acts as the evaluation does
        key, step_key = jax.random.split(key)
        obs, env_state, reward, terminated, truncated, _ = env.step(
            step_key, env_state, action, params
        )
        episode_return += float(reward)
        done = bool(terminated or truncated)
    print("episode return:", episode_return)
    return episode_return


if __name__ == "__main__":
    main()
