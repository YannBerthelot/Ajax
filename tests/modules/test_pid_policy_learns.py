"""The fixed-gain PID policy learns the set-point task's deadbeat gains.

Calibration (2026-10-11, macOS CPU, seeds 0-9, these settings): the worst
error over Kp, Ki, Kd was 0.011 for PPO (50k steps) and 0.012 for SAC (40k
steps); the initial zero gains are 0.25 off. The 0.03 tolerance leaves 2.5
times the worst error and still rejects SAC at 10k steps, whose Kp of about
0.285 compensates the shrinkage its wide exploration noise causes through
``tanh``.

Recalibrated 2026-10-19 once SAC's critic started as its reference code's
(xavier_uniform, zero biases): at 40k steps SAC's Kp reads 0.034 off on
seeds 0-4; at 80k the worst error is 0.0117 (Kp), so 0.03 leaves 2.6 times
it again (60k: 0.017 on Ki, 1.8 times).
"""

import numpy as np
import pytest

from ajax import PPO, SAC
from ajax.modules.pid_actor import PIDPolicy, pid_gains

from . import setpoint_env

TOLERANCE = 0.03
SEEDS = [0, 1, 2, 3, 4]
COMMON = {
    "actor_architecture": PIDPolicy(),
    "critic_architecture": ("64", "relu", "64", "relu"),
    "actor_learning_rate": 3e-3,
    "critic_learning_rate": 1e-3,
    "gamma": 0.9,
}
RUNS = {
    "SAC": (
        SAC,
        {"n_envs": 4, "learning_starts": 500, "buffer_size": 20_000, "batch_size": 128},
        80_000,
    ),
    "PPO": (
        PPO,
        {"n_envs": 8, "n_steps": 64, "batch_size": 64, "n_epochs": 10},
        50_000,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(RUNS))
def test_the_fixed_gains_learn_the_deadbeat_pid(name):
    agent_cls, kwargs, steps = RUNS[name]
    env, params = setpoint_env.make()
    agent = agent_cls(
        env, env_params=params, extensions=(setpoint_env.TERMS,), **COMMON, **kwargs
    )
    state, _ = agent.train(seed=SEEDS, n_timesteps=steps, num_episode_test=4)
    gains = pid_gains(state.actor_state.params)
    for gain, best in setpoint_env.BEST_GAINS.items():
        learned = np.asarray(gains[gain]).reshape(len(SEEDS))
        assert np.all(np.abs(learned - best) < TOLERANCE), (gain, learned)
