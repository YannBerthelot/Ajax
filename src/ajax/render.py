"""Saving an episode as an animation (``agent.render``): gymnax's drawings
for the tasks it draws, Brax's interactive page, MuJoCo's renderer for
MuJoCo Playground and the ``mujoco:`` tasks, a Gymnasium environment's own
pictures, and for any other environment a ``draw`` function of the
caller's. The gymnax drawings and the GIF writer need packages Ajax does
not require, which its ``render`` extra installs; the message says so when
one is missing."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Callable, Optional

import jax
import numpy as np

from ajax.environments.utils import (
    check_env_is_playground,
    get_env_type,
    get_raw_env,
    steps_in_python,
)
from ajax.evaluate import policy_episode

#: The speed of an animation drawn by a ``draw`` function, and of a
#: Gymnasium environment that names none, in frames per second.
DEFAULT_FPS = 30.0

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


def animation_suffix(env: Any, draw: Optional[Callable] = None) -> str:
    """The file an episode of ``env`` is saved as: ``".gif"``, or ``".html"``
    on Brax, whose page shows the episode in three dimensions. With
    ``draw``, a function from a state to a picture, any env is saved as a
    ``".gif"``. Raises for an env nothing here draws."""
    if draw is not None or steps_in_python(env):
        return ".gif"
    if get_env_type(env) == "brax":
        return ".gif" if check_env_is_playground(env) else ".html"
    name = getattr(get_raw_env(env), "name", None)
    if name not in GYMNAX_DRAWN:
        raise NotImplementedError(
            f"agent.render draws the gymnax tasks gymnax draws, and {name!r} is"
            f" not one: {', '.join(sorted(GYMNAX_DRAWN))}. For another"
            " environment, pass a function drawing one state:"
            " agent.render(state, path, draw=my_draw), where my_draw(env_state)"
            " returns a (height, width, 3) picture."
        )
    return ".gif"


def render_episode(
    env_args: Any,
    parts: Any,
    key: jax.Array,
    path: str,
    max_steps: Optional[int] = None,
    draw: Optional[Callable] = None,
) -> None:
    """Runs one episode of a trained policy's ``parts`` on ``env_args``' env
    (:func:`ajax.evaluate.policy_episode`) and saves it at ``path``.
    ``draw(env_state)``, given, draws each state, one env's, as a
    ``(height, width, 3)`` picture; else a Gymnasium env draws its own, and
    the others are drawn as :func:`save_episode` does."""
    with _without_sound():
        _render_episode(env_args, parts, key, path, max_steps, draw)


@contextlib.contextmanager
def _without_sound() -> Iterator[None]:
    """Gymnasium's classic-control drawings start pygame, whose SDL opens an
    audio device every time (once a frame under gymnax's drawing): on a
    machine without one, such as a CI runner or a server, ALSA prints eight
    error lines each time. Drawing needs no sound, so SDL gets its dummy
    driver while the episode is drawn, unless the caller chose one."""
    if "SDL_AUDIODRIVER" in os.environ:
        yield
        return
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        yield
    finally:
        del os.environ["SDL_AUDIODRIVER"]


def _render_episode(
    env_args: Any,
    parts: Any,
    key: jax.Array,
    path: str,
    max_steps: Optional[int],
    draw: Optional[Callable],
) -> None:
    if draw is not None:
        _, states, _ = policy_episode(env_args, parts, key, max_steps)
        frames = [np.asarray(draw(_one_env(state))) for state in states]
        _write_gif(frames, path, 1.0 / DEFAULT_FPS)
    elif steps_in_python(env_args.env):
        from ajax.environments.gymnasium import (
            frames_per_second,
            rendered_frame,
            with_own_copies,
        )

        drawn = env_args.replace(
            env=with_own_copies(env_args.env, render_mode="rgb_array")
        )
        env, frames, _ = policy_episode(
            drawn,
            parts,
            key,
            max_steps,
            lambda env, _: rendered_frame(get_raw_env(env)),
        )
        _write_gif(frames, path, 1.0 / frames_per_second(get_raw_env(env), DEFAULT_FPS))
    else:
        env, states, rewards = policy_episode(env_args, parts, key, max_steps)
        save_episode(env, states, rewards, path, env_args.action_repeat)


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


def _write_gif(frames: list, path: str, seconds_per_frame: float) -> None:
    """``frames``, ``(height, width, 3)`` pictures of 8-bit colours, as a
    looping GIF."""
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise _missing("pillow", error) from error
    pictures = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                "A frame is a (height, width, 3) picture; got an array of"
                f" shape {frame.shape}."
            )
        if frame.dtype != np.uint8:
            # Floats are colours in [0, 1], as matplotlib and JAX code draw.
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        pictures.append(Image.fromarray(frame))
    pictures[0].save(
        path,
        save_all=True,
        append_images=pictures[1:],
        duration=1000 * seconds_per_frame,
        loop=0,
    )


def _save_mujoco(env: Any, states: list, path: str, action_repeat: int) -> None:
    height, width = MUJOCO_FRAME
    # MuJoCo's renderer reads each state's data alone.
    trajectory = [SimpleNamespace(data=_one_env(state.data)) for state in states]
    frames = env.render(trajectory, height, width)
    _write_gif(frames, path, float(env.dt) * action_repeat)
