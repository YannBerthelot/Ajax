"""The entropy temperature every soft actor-critic takes from its family
(EntropyTemperature): a target entropy per action dimension that may be a
schedule of the timestep, which each temperature step reads at its
timestep."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ajax import ASAC, REDQ, SAC, TQC, CrossQ
from ajax.logging.wandb_logging import LoggingConfig
from ajax.schedule import linear

TINY: dict[str, Any] = {"env_id": "Reacher-misc", "n_envs": 2, "buffer_size": 64}
TINY.update({"batch_size": 8, "learning_starts": 8})
SOFT: dict[type, dict[str, Any]] = {
    CrossQ: {"actor_width": 8, "critic_width": 8, "policy_delay": 1},
    SAC: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    TQC: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    # One critic step per update: the fast tier's budget of gradient steps.
    REDQ: {
        "actor_architecture": ("8", "relu"),
        "critic_architecture": ("8", "relu"),
        "num_critic_updates": 1,
    },
    # Its temperature learned (ASAC fixes it by default).
    ASAC: {
        "actor_architecture": ("8", "relu"),
        "critic_architecture": ("8", "relu"),
        "fixed_alpha": False,
    },
}


@pytest.mark.parametrize("cls", list(SOFT), ids=lambda c: c.__name__)
def test_the_temperature_steps_towards_the_scheduled_target_entropy(cls: Any) -> None:
    """At each evaluation the last temperature step logged the schedule's
    value at its timestep, times the action's two dimensions."""
    ramp = linear(-3.0, -1.0, 32)
    agent = cls(**TINY, **SOFT[cls], target_entropy_per_dim=ramp)
    logging = LoggingConfig(log_frequency=8, backends=())
    logs = agent.train(seed=0, n_timesteps=32, logging_config=logging)[1]
    timesteps = np.asarray(logs["timestep"])[0]
    logged = np.asarray(logs["temperature/effective_target_entropy"])[0]
    trained = timesteps >= 8
    np.testing.assert_allclose(
        logged[trained], [2 * float(ramp(t)) for t in timesteps[trained]], rtol=1e-6
    )
    assert agent.target_entropy(16) == pytest.approx(2 * float(ramp(16)))


@pytest.mark.parametrize(
    "target_entropy_per_dim, direction",
    [(-10.0, -1.0), (10.0, 1.0)],
    ids=["target-far-below", "target-far-above"],
)
@pytest.mark.parametrize("cls", list(SOFT), ids=lambda c: c.__name__)
def test_the_temperature_moves_towards_its_target_entropy(
    cls: Any, target_entropy_per_dim: float, direction: float
) -> None:
    """Catches a temperature step that ignores target_entropy_per_dim (ASAC's
    and REDQ's once used -action_dim, the default -1 per dimension's value)
    or is never kept: log alpha's gradient is the policy's entropy less the
    target, so a target far below the entropy lowers alpha and one far above
    raises it. The temperature steps from the first update (SAC's waits
    for 2,000 steps by default)."""
    agent = cls(
        **TINY,
        **SOFT[cls],
        target_entropy_per_dim=target_entropy_per_dim,
        alpha_update_start=0,
    )
    state = agent.train(seed=0, n_timesteps=16)[0]
    log_alpha = float(np.ravel(state.alpha.params["log_alpha"])[0])
    assert np.sign(log_alpha - np.log(agent.alpha_init)) == direction
