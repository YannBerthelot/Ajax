"""PPO's old train module: its names moved to :mod:`ajax.agents.PPO.PPO` and
the shared modules, from which they resolve until 0.5.0."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    ("policy_loss_function", "value_loss_function"), "ajax.agents.PPO.PPO"
)
_new_homes.update(
    {
        "init_PPO": "ajax.agents.PPO.PPO:PPO.init",
        "update_agent": "ajax.agents.PPO.PPO:PPO.update",
        "make_train": "ajax.agents.families:OnPolicyAgent.make_train",
        "Geometry": "ajax.agents.minibatch:RolloutSplit",
        "minibatch_geometry": "ajax.agents.minibatch:choose_rollout_split",
        "default_bptt_length": "ajax.agents.minibatch",
        "run_epochs": "ajax.agents.minibatch",
        "clip_global_norm": "ajax.agents.update_steps",
        "joint_step": "ajax.agents.update_steps",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
