"""Times a segment of training from inside a Gymnasium environment.

SB3, SBX and CleanRL make their environments with ``gymnasium.make``;
``gymnasium.make("speed_clock:speed/<id>")`` imports this module and makes
``<id>`` with a clock around it, so each library runs unchanged. The clock
counts the process's environment steps over all its copies. At step
``SPEED_START`` it waits until ``SPEED_PROCESSES`` processes have reached
theirs (files in the folder ``SPEED_BARRIER``), so the processes of a 10-seed
run time the same stretch; at step ``SPEED_END`` it appends the segment's
wall-clock and CPU seconds to ``SPEED_RECORD``. The segment holds
``SPEED_END - SPEED_START`` steps and every update made between them.

The passive environment checker is off: it only checks the first step, and
Gymnasium 0.29's (CleanRL's) calls ``np.bool8``, which NumPy 2 removed.
"""

import json
import os
import pathlib
import time

import gymnasium as gym

IDS = ("CartPole-v1", "Pendulum-v1", "HalfCheetah-v4", "HalfCheetah-v5")
_steps = 0
_start: tuple[float, float] = (0.0, 0.0)


def _wait_for_the_others() -> None:
    folder = pathlib.Path(os.environ["SPEED_BARRIER"])
    expected = int(os.environ.get("SPEED_PROCESSES", "1"))
    (folder / str(os.getpid())).touch()
    deadline = time.monotonic() + 3600
    while sum(1 for _ in folder.iterdir()) < expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"fewer than {expected} processes reached the start")
        time.sleep(0.05)


class SegmentClock(gym.Wrapper):
    def step(self, action):
        global _steps, _start
        _steps += 1
        if _steps == int(os.environ["SPEED_START"]):
            _wait_for_the_others()
            _start = (time.time(), time.process_time())
        elif _steps == int(os.environ["SPEED_END"]):
            record = {
                "pid": os.getpid(),
                "steps": _steps - int(os.environ["SPEED_START"]),
                "start": _start[0],
                "end": time.time(),
                "cpu_s": time.process_time() - _start[1],
            }
            with open(os.environ["SPEED_RECORD"], "a") as file:
                file.write(json.dumps(record) + "\n")
        return self.env.step(action)


def _make(env_id: str, **kwargs) -> gym.Env:
    return SegmentClock(gym.make(env_id, disable_env_checker=True, **kwargs))


for _id in IDS:
    if _id in gym.registry:  # HalfCheetah-v5 from Gymnasium 1.0 only
        gym.register(
            f"speed/{_id}",
            entry_point=_make,
            kwargs={"env_id": _id},
            disable_env_checker=True,
        )
