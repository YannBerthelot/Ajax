"""Save a trained state, load it into a new agent and play an episode."""

import os
import tempfile

import gymnax
import jax

from ajax import PPO
from ajax.checkpoint import load_checkpoint, save_checkpoint


def make_agent() -> PPO:
    # A checkpoint holds the state, not the agent's arguments: the agent
    # that loads it is built with the same ones.
    return PPO("CartPole-v1", n_envs=8, n_steps=128)


def main(n_timesteps: int = 32_768) -> float:
    state, _ = make_agent().train(seed=0, n_timesteps=n_timesteps)
    path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
    save_checkpoint(state, path)

    agent = make_agent()
    state = load_checkpoint(agent, path)

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
