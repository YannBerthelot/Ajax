"""AVG's training, now in :mod:`ajax.agents.AVG.AVG` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. init_AVG, update_AVG_values,
update_agent, update_value_functions, update_policy and make_train became
methods with other parameters: their old names raise TypeError with the
method's call form. value_loss_function is compute_critic_loss and
policy_loss_function compute_policy_loss."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    ("AuxiliaryLogs", "PolicyAuxiliaries", "ValueAuxiliaries", "compute_avg_td_target"),
    "ajax.agents.AVG.AVG",
)
_new_homes.update(
    {
        "value_loss_function": "ajax.agents.AVG.AVG:compute_critic_loss",
        "policy_loss_function": "ajax.agents.AVG.AVG:compute_policy_loss",
        "init_AVG": "ajax.agents.AVG.AVG:AVG.init",
        "update_AVG_values": "ajax.agents.AVG.AVG:AVG.update_step_statistics",
        "update_agent": "ajax.agents.AVG.AVG:AVG.update",
        "update_value_functions": "ajax.agents.AVG.AVG:AVG.update_critic",
        "update_policy": "ajax.agents.AVG.AVG:AVG.update_actor",
        "make_train": "ajax.agents.families:IncrementalAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
