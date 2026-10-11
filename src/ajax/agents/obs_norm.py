"""The agent-side observation normaliser moved to
:mod:`ajax.environments.obs_norm`."""

from ajax.deprecation import moved_names
from ajax.environments import obs_norm

__getattr__ = moved_names(
    __name__,
    apply_obs_norm=obs_norm.apply_obs_norm,
    init_agent_obs_norm=obs_norm.init_agent_obs_norm,
    update_obs_norm=obs_norm.update_obs_norm,
)
