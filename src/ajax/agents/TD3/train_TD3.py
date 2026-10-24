"""TD3's training, now in :mod:`ajax.agents.TD3.TD3` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. ``init_TD3``, ``update_agent``,
``update_value_functions``, ``update_policy`` and ``make_train`` became
methods, with the agent's signatures."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
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
    | {
        "update_target_networks": "ajax.agents.TD3.TD3:update_targets",
        "init_TD3": "ajax.agents.TD3.TD3:TD3.init",
        "update_agent": "ajax.agents.TD3.TD3:TD3.update",
        "update_value_functions": "ajax.agents.TD3.TD3:TD3.critic_update",
        "update_policy": "ajax.agents.TD3.TD3:TD3.actor_update",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    },
)
