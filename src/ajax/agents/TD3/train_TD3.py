"""TD3's training, now in :mod:`ajax.agents.TD3.TD3` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. ``init_TD3``, ``update_agent``,
``update_value_functions``, ``update_policy`` and ``make_train`` became
methods with other parameters: their old names raise TypeError with the
method's call form. ``update_target_networks`` is ``update_targets``."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    (
        "AuxiliaryLogs",
        "PolicyAuxiliaries",
        "ValueAuxiliaries",
        "compute_td3_td_target",
        "gaussian_exploration",
        "make_default_action_pipeline",
        "policy_loss_function",
    ),
    "ajax.agents.TD3.TD3",
)
_new_homes.update(
    {
        "update_target_networks": "ajax.agents.TD3.TD3:update_targets",
        "init_TD3": "ajax.agents.TD3.TD3:TD3.init",
        "update_agent": "ajax.agents.TD3.TD3:TD3.update",
        "update_value_functions": "ajax.agents.TD3.TD3:TD3.update_critic",
        "update_policy": "ajax.agents.TD3.TD3:TD3.update_actor",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
