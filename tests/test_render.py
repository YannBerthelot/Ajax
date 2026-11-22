"""Saving a trained policy's episode as an animation.

Contract (``DeclaredAgent.render``, ``ajax.evaluate.policy_episode``,
docs/guide/deploy.md, "Watch the policy"):
  - ``policy_episode`` runs one episode step by step: the state at the
    reset and after every step but the ending one, and every step's reward,
    stopping at the episode's end or after ``max_steps``;
  - ``agent.render(state, path)`` writes the animation at ``path`` and
    returns it: a GIF with a frame per state on a gymnax task gymnax
    draws, an HTML page on Brax;
  - a path with another ending is refused before anything runs, as are a
    gymnax task gymnax does not draw and a Gymnasium environment;
  - a classic-control drawing without pygame names what to install.
"""

import importlib.util
import os

import jax
import pytest
from PIL import Image

from ajax import PPO, PQN
from ajax.evaluate import policy_episode


@pytest.fixture(scope="module")
def cartpole():
    agent = PPO("CartPole-v1", n_envs=2, n_steps=16)
    state, _ = agent.train(seed=0, n_timesteps=32, logging_config=None)
    return agent, state


def episode(agent, state, max_steps=None):
    policy = agent.policy(state)
    parts = policy.build(policy.agent_state)
    return policy_episode(agent.env_args, parts, jax.random.PRNGKey(0), max_steps)


def test_an_episode_keeps_a_state_per_step_until_its_end(cartpole):
    """CartPole pays 1 per step and ends when the pole falls, which a policy
    trained for 32 steps lets happen long before the 500-step limit. The
    states are the reset's and each step's but the last: one per reward."""
    _, states, rewards = episode(*cartpole)
    assert 1 <= len(rewards) < 500
    assert rewards == [1.0] * len(rewards)
    assert len(states) == len(rewards)


def test_max_steps_cuts_an_episode_short(cartpole):
    """The pole starts within 0.05 rad of upright and falls past 0.21 rad:
    not within 3 steps. Three steps leave the reset's state and three more."""
    _, states, rewards = episode(*cartpole, max_steps=3)
    assert len(rewards) == 3
    assert len(states) == 4


def test_render_writes_a_gif_with_a_frame_per_state(tmp_path):
    """Breakout loses a life no sooner than the ball crosses the 10 rows, so
    4 steps do not end the episode: 5 states, 5 frames."""
    agent = PQN("Breakout-MinAtar", cnn_image_shape=(10, 10, 4), n_envs=4, n_steps=8)
    state, _ = agent.train(seed=0, n_timesteps=32, logging_config=None)
    path = tmp_path / "breakout.gif"
    assert agent.render(state, path, max_steps=4) == str(path)
    with Image.open(path) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames == 5


def test_render_writes_a_page_on_brax(tmp_path):
    agent = PPO("inverted_pendulum", n_envs=2, n_steps=8)
    state, _ = agent.train(seed=0, n_timesteps=16, logging_config=None)
    path = agent.render(state, tmp_path / "pendulum.html", max_steps=3)
    with open(path) as page:
        assert "brax-viewer" in page.read()


def test_a_path_with_another_ending_is_refused_before_anything_runs(cartpole):
    agent, _ = cartpole
    with pytest.raises(ValueError, match=r"ending in \.gif"):
        agent.render(None, "cartpole.mp4")


def test_a_gymnax_task_gymnax_does_not_draw_is_refused():
    agent = PPO("Reacher-misc", n_envs=2, n_steps=8)
    with pytest.raises(NotImplementedError, match="CartPole-v1"):
        agent.render(None, "reacher.gif")


def test_a_gymnasium_environment_is_refused():
    with pytest.warns(UserWarning):
        agent = PPO("gymnasium:CartPole-v1", n_envs=2, n_steps=8)
    with pytest.raises(NotImplementedError, match="Gymnasium"):
        agent.render(None, "cartpole.gif")


def test_a_classic_control_drawing_needs_pygame_and_says_so(cartpole, tmp_path):
    agent, state = cartpole
    path = tmp_path / "cartpole.gif"
    if importlib.util.find_spec("pygame") is None:
        with pytest.raises(ImportError, match=r"gymnasium\[classic-control\]"):
            agent.render(state, path, max_steps=3)
    else:
        agent.render(state, path, max_steps=3)
        assert os.path.getsize(path) > 0


@pytest.mark.skipif(
    "MUJOCO_GL" not in os.environ, reason="MuJoCo's renderer needs a GL backend"
)
def test_render_writes_a_gif_on_a_mujoco_task(tmp_path):
    """InvertedPendulum falls past 0.2 rad no sooner than a few steps in."""
    from ajax import SAC

    agent = SAC("mujoco:InvertedPendulum-v5", learning_starts=8)
    state, _ = agent.train(seed=0, n_timesteps=16, logging_config=None)
    path = agent.render(state, tmp_path / "pendulum.gif", max_steps=2)
    with Image.open(path) as animation:
        assert animation.n_frames == 3
