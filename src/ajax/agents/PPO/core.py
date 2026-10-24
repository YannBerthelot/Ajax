"""PPO's old core module: its names moved to :mod:`ajax.agents.PPO.PPO` and
the shared modules, from which they resolve until 0.5.0."""

from typing import Any

import jax

from ajax.deprecation import moved_names
from ajax.schedule import at


def resolve_clip_coef(agent_config: Any, timestep: int | jax.Array) -> Any:
    """The clip range, a constant or a schedule of the timestep (PPO reads
    ``at(clip_range, timestep)``; kept for downstream callers until 0.5.0)."""
    return at(agent_config.clip_range, timestep)


_new_homes = dict.fromkeys(
    ("AuxiliaryLogs", "PolicyAuxiliaries", "ValueAuxiliaries", "clipped_surrogate"),
    "ajax.agents.PPO.PPO",
)
_new_homes.update(
    dict.fromkeys(
        ("resolve_num_minibatches", "run_epochs", "rollout_actions"),
        "ajax.agents.minibatch",
    )
)
_new_homes.update(
    dict.fromkeys(
        ("approximate_kl", "explained_variance", "mean_policy_std"),
        "ajax.agents.diagnostics",
    )
)
_new_homes.update(
    dict.fromkeys(
        ("policy_entropy", "recompute_log_prob"), "ajax.networks.distributions"
    )
)
__getattr__ = moved_names(__name__, _new_homes)
