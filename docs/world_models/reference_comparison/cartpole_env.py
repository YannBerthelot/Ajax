"""Numpy float32 port of gymnax 1.0.0 CartPole-v1 as an ``embodied.Env``.

PROTOCOL.md section 2.2.

Source ported (gymnax by Robert Tjarko Lange, Apache License 2.0; its
CartPole follows OpenAI Gym's): ``gymnax/environments/classic_control/cartpole.py``
of the gymnax that Ajax runs (1.0.0, ``YannBerthelot/gymnax@61ff068``):
``step_env`` lines 52-99, ``reset_env`` 101-113, ``is_terminated`` 119-131,
and ``environment.py`` 63-100 / 150-152 (six-value step, truncation
``time >= max_steps_in_episode``).
Modified: re-implemented in numpy float32 as an ``embodied.Env`` (the jax
functions and the gymnax environment class are replaced by numpy arithmetic
and a stateful class; reset and step follow the row conventions below). The
gymnax license (Apache License 2.0) is in LICENSE.gymnax in this directory.

Rows emitted (identical to Ajax's row collector, PROTOCOL.md 2.1):
* a step with ``action['reset']`` true returns the reset row: obs drawn
  ``uniform(-0.05, 0.05, 4)``, reward 0, ``is_first``;
* any other step applies the gymnax arithmetic in float32, reward
  ``1 - is_terminated(previous state)`` (1.0 in a live episode),
  ``is_last = terminated or truncated``, ``is_terminal = terminated`` (a time
  limit end at step 500 has ``is_terminal = False``).

The module imports without ``embodied`` (so that the self-test can run in the
Ajax venv, which has gymnax but not the reference's ``embodied``); the
``obs_space`` / ``act_space`` properties need ``embodied``.
"""

from __future__ import annotations

import numpy as np

try:  # the reference venv (PYTHONPATH contains the reference checkout)
    import embodied

    _EnvBase = embodied.Env
except ImportError:  # the Ajax venv, for the port self-test only
    embodied = None
    _EnvBase = object

F32 = np.float32

# gymnax EnvParams defaults (cartpole.py:21-33), as float32 constants: under
# jit gymnax's Python-float params are weakly typed and combine with the
# float32 state in float32.
GRAVITY = F32(9.8)
MASSCART = F32(1.0)
MASSPOLE = F32(0.1)
TOTAL_MASS = F32(1.0 + 0.1)
LENGTH = F32(0.5)
POLEMASS_LENGTH = F32(0.05)
FORCE_MAG = F32(10.0)
TAU = F32(0.02)
THETA_THRESHOLD = F32(12 * 2 * np.pi / 360)
X_THRESHOLD = F32(2.4)
FOUR_THIRDS = F32(4.0 / 3.0)
ONE = F32(1.0)
MAX_STEPS = 500


def terminated(state: np.ndarray) -> bool:
    """gymnax ``is_terminated`` (cartpole.py:119-131), strict inequalities."""
    x, _, theta, _ = state
    return bool(
        (x < -X_THRESHOLD)
        or (x > X_THRESHOLD)
        or (theta < -THETA_THRESHOLD)
        or (theta > THETA_THRESHOLD)
    )


def step_state(state: np.ndarray, action: int) -> np.ndarray:
    """gymnax ``step_env`` arithmetic (cartpole.py:61-78), same operation order."""
    x, x_dot, theta, theta_dot = (F32(v) for v in state)
    a = F32(action)
    force = FORCE_MAG * a - FORCE_MAG * (ONE - a)
    costheta = F32(np.cos(theta))
    sintheta = F32(np.sin(theta))
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (FOUR_THIRDS - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc
    out = np.array([x, x_dot, theta, theta_dot], np.float32)
    assert out.dtype == np.float32
    return out


class CartPolePort(_EnvBase):  # type: ignore[valid-type]  # embodied.Env or object
    """gymnax CartPole-v1 as an ``embodied.Env`` (PROTOCOL.md 2.2).

    Args:
        entropy: seed material of this env's reset generator,
            ``np.random.default_rng(entropy)``; training env ``w`` of seed
            ``s`` uses ``[s, w]``, evaluation env ``i`` uses
            ``[s, 7_000_001, i]`` (PROTOCOL.md 4.2).
        index: the worker index written into the episode records.
        recorder: list that finished-episode records are appended to
            (training envs), or None.
    """

    def __init__(self, entropy, index: int = 0, recorder: list | None = None):
        self._rng = np.random.default_rng(entropy)
        self._index = int(index)
        self._recorder = recorder
        self._row = -1  # 0-based per-env index of the last returned row
        self._state: np.ndarray | None = None
        self._time = 0
        self._score = 0.0
        self._length = 0
        self._done = False

    # ------------------------------------------------------------- spaces
    @property
    def obs_space(self):
        return {
            "vector": embodied.Space(np.float32, (4,)),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "action": embodied.Space(np.int32, (), 0, 2),
            "reset": embodied.Space(bool),
        }

    # ------------------------------------------------------------- dynamics
    def reset_state(self) -> np.ndarray:
        """gymnax ``reset_env``: ``uniform(-0.05, 0.05)`` on the 4 variables."""
        return self._rng.uniform(-0.05, 0.05, 4).astype(np.float32)

    def set_state(self, state, time: int = 0) -> None:
        """Test hook: place the env in ``state`` at ``time`` (live episode)."""
        self._state = np.asarray(state, np.float32).copy()
        self._time = int(time)
        self._done = False

    def step(self, action):
        self._row += 1
        if action["reset"]:
            self._state = self.reset_state()
            self._time = 0
            self._score = 0.0
            self._length = 0
            self._done = False
            return self._obs(F32(0.0), True, False, False)
        assert (
            self._state is not None and not self._done
        ), "stepped past is_last without a reset"
        a = int(action["action"])
        assert a in (0, 1), a
        prev_terminal = terminated(self._state)
        self._state = step_state(self._state, a)
        reward = F32(1.0 - float(prev_terminal))
        self._time += 1
        term = terminated(self._state)
        trunc = self._time >= MAX_STEPS
        is_last = term or trunc
        self._score += float(reward)
        self._length += 1
        self._done = is_last
        if is_last and self._recorder is not None:
            self._recorder.append(
                {
                    "worker": self._index,
                    "row_last": self._row,
                    "length": self._length,
                    "score": self._score,
                    "terminal": bool(term),
                }
            )
        return self._obs(reward, False, is_last, term)

    def _obs(self, reward, is_first, is_last, is_terminal):
        return {
            "vector": self._state.copy(),
            "reward": F32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }
