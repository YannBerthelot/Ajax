"""Saving a trained policy's episode as an animation.

Contract (``DeclaredAgent.render``, ``ajax.evaluate.policy_episode``,
docs/guide/deploy.md, "Watch the policy"):
  - ``policy_episode`` runs one episode step by step: the state at the
    reset and after every step but the ending one, and every step's reward,
    stopping at the episode's end or after ``max_steps``;
  - ``agent.render(state, path)`` writes the animation at ``path`` and
    returns it: a GIF with a frame per state on a gymnax task gymnax
    draws, an HTML page on Brax;
  - a path with another ending is refused before anything runs, as is a
    gymnax task gymnax does not draw, naming ``draw``;
  - a Gymnasium environment is drawn with its own pictures, a frame per
    state (owner's decision of 2026-11-22: before, it was refused);
  - with ``draw``, any environment is saved as a GIF of what ``draw``
    returns for each state, one environment's, in order; a picture is
    ``(height, width, 3)``, 8-bit or floats in [0, 1], and another shape is
    refused;
  - a classic-control drawing is written with Ajax's render extra, and
    without pygame names that extra.
"""

import importlib.util
import os

import jax
import numpy as np
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
    with pytest.raises(NotImplementedError, match="CartPole-v1") as refusal:
        agent.render(None, "reacher.gif")
    assert "draw=" in str(refusal.value)


@pytest.mark.skipif(
    importlib.util.find_spec("pygame") is None,
    reason="Gymnasium draws CartPole with pygame",
)
def test_a_gymnasium_environment_is_drawn_with_its_own_pictures(tmp_path):
    """Gymnasium draws CartPole on a 600 by 400 screen (its cartpole.py);
    three steps cannot drop the pole, so four states are drawn."""
    with pytest.warns(UserWarning):
        agent = PPO("gymnasium:CartPole-v1", n_envs=2, n_steps=8)
    state, _ = agent.train(seed=0, n_timesteps=16, logging_config=None)
    path = agent.render(state, tmp_path / "cartpole.gif", max_steps=3)
    with Image.open(path) as animation:
        assert animation.n_frames == 4
        assert animation.size == (600, 400)


@pytest.fixture(scope="module")
def reacher():
    """A gymnax task gymnax does not draw."""
    agent = PPO("Reacher-misc", n_envs=2, n_steps=8)
    state, _ = agent.train(seed=0, n_timesteps=16, logging_config=None)
    return agent, state


def test_a_draw_function_draws_each_state_of_one_environment_in_order(
    reacher, tmp_path
):
    """gymnax counts an episode's steps in its state's ``time``: the reset
    and three steps are times 0 to 3, each a scalar (one environment). The
    pictures are white, 8 rows by 12 columns."""
    agent, state = reacher
    times = []

    def draw(env_state):
        times.append(np.asarray(env_state.time).tolist())
        return np.ones((8, 12, 3), np.float32)

    path = agent.render(state, tmp_path / "reacher.gif", max_steps=3, draw=draw)
    assert times == [0, 1, 2, 3]
    with Image.open(path) as animation:
        assert animation.n_frames == 4
        assert animation.size == (12, 8)
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_a_picture_of_another_shape_is_refused(reacher, tmp_path):
    agent, state = reacher
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        agent.render(
            state,
            tmp_path / "reacher.gif",
            max_steps=1,
            draw=lambda env_state: np.zeros((8, 12)),
        )


def test_a_classic_control_drawing_needs_pygame_and_says_so(cartpole, tmp_path):
    agent, state = cartpole
    path = tmp_path / "cartpole.gif"
    if importlib.util.find_spec("pygame") is None:
        with pytest.raises(ImportError, match=r"ajax-rl\[render\]"):
            agent.render(state, path, max_steps=3)
    else:
        agent.render(state, path, max_steps=3)
        with Image.open(path) as animation:
            assert animation.n_frames == 4


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
