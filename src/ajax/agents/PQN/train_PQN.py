"""PQN's old train module: its names moved to :mod:`ajax.agents.PQN.PQN` and
:mod:`ajax.agents.families`, from which they resolve until 0.5.0.
``update_agent`` and ``make_train`` became methods with other parameters:
their old names raise TypeError with the method's call form."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    {
        "half_mse_td_loss": "ajax.agents.PQN.PQN:compute_half_mse_td_loss",
        "update_agent": "ajax.agents.PQN.PQN:PQN.update",
        "make_train": "ajax.agents.families:OnPolicyAgent.make_train",
    },
)
