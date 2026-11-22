"""Saving an episode as an animation (``agent.render``): gymnax's drawings
for the tasks it draws, Brax's interactive page, MuJoCo's renderer for
MuJoCo Playground and the ``mujoco:`` tasks. The gymnax drawings and the
GIF writer need packages Ajax does not require, which its ``render`` extra
installs; the message says so when one is missing."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from ajax.environments.utils import (
    check_env_is_playground,
    get_env_type,
    get_raw_env,
    steps_in_python,
)

#: The gymnax tasks gymnax.visualize draws with Gymnasium's own drawings,
#: which need pygame.
CLASSIC_CONTROL = frozenset(
    {
        "Acrobot-v1",
        "CartPole-v1",
        "Pendulum-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
    }
)
#: Every gymnax task gymnax.visualize draws (its Visualizer lists them inline).
GYMNAX_DRAWN = CLASSIC_CONTROL.union(
    {
        "Catch-bsuite",
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "Seaquest-MinAtar",
        "SpaceInvaders-MinAtar",
        "Pong-misc",
        "PointRobot-misc",
        "MetaMaze-misc",
        "FourRooms-misc",
    }
)
#: The frame size of a MuJoCo animation, (height, width) in pixels.
MUJOCO_FRAME = (240, 320)


def animation_suffix(env: Any) -> str:
    """The file an episode of ``env`` is saved as: ``".gif"``, or ``".html"``
    on Brax, whose page shows the episode in three dimensions. Raises for
    an env nothing here draws."""
    if steps_in_python(env):
        raise NotImplementedError(
            "agent.render does not draw a Gymnasium environment: build it"
            " with render_mode='rgb_array' and act with agent.predict."
        )
    if get_env_type(env) == "brax":
        return ".gif" if check_env_is_playground(env) else ".html"
    name = getattr(get_raw_env(env), "name", None)
    if name not in GYMNAX_DRAWN:
        raise NotImplementedError(
            f"agent.render draws the gymnax tasks gymnax draws, and {name!r} is"
            f" not one: {', '.join(sorted(GYMNAX_DRAWN))}. For another"
            " environment, act with agent.predict or agent.policy and draw"
            " its states yourself."
        )
    return ".gif"


def _one_env(tree: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf[0], tree)


def _missing(package: str, error: Exception) -> ImportError:
    return ImportError(
        f"agent.render needs {package} here, which Ajax's render extra"
        f" installs: pip install 'ajax-rl[render]' ({error})"
    )


def save_episode(
    env: Any, states: list, rewards: list[float], path: str, action_repeat: int = 1
) -> None:
    """Saves an episode at ``path``: ``states`` are ``env``'s, each with a
    leading axis of one env, ``rewards`` the steps' (a gymnax drawing titles
    each frame with the return so far)."""
    if get_env_type(env) != "brax":
        _save_gymnax(
            get_raw_env(env), [_one_env(state) for state in states], rewards, path
        )
    elif check_env_is_playground(env):
        _save_mujoco(get_raw_env(env), states, path, action_repeat)
    else:
        _save_brax(get_raw_env(env), states, path)


def _save_gymnax(env: Any, states: list, rewards: list[float], path: str) -> None:
    # Checked first: matplotlib's writer hides a drawing's failure behind
    # an IndexError on its empty list of frames.
    if env.name in CLASSIC_CONTROL and importlib.util.find_spec("pygame") is None:
        raise ImportError(
            f"gymnax draws {env.name} with Gymnasium's drawing, which needs"
            " pygame. Ajax's render extra installs it: pip install"
            " 'ajax-rl[render]'"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")  # draws to the file, with or without a display
        from gymnax.visualize import Visualizer
    except ModuleNotFoundError as error:
        raise _missing("matplotlib", error) from error
    returns = np.concatenate([[0.0], np.cumsum(rewards)])[: len(states)]
    Visualizer(env, env.default_params, states, returns).animate(path)


def _save_brax(env: Any, states: list, path: str) -> None:
    from brax.io import html

    system = env.sys.tree_replace({"opt.timestep": env.dt})
    html.save(path, system, [_one_env(state.pipeline_state) for state in states])


def _save_mujoco(env: Any, states: list, path: str, action_repeat: int) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise _missing("pillow", error) from error
    height, width = MUJOCO_FRAME
    # MuJoCo's renderer reads each state's data alone.
    trajectory = [SimpleNamespace(data=_one_env(state.data)) for state in states]
    frames = [Image.fromarray(frame) for frame in env.render(trajectory, height, width)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 * float(env.dt) * action_repeat,
        loop=0,
    )
