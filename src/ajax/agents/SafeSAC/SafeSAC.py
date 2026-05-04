"""SafeSAC: a SAC variant with a safety V-head sharing the SAC critic's
encoder.

V3 architecture: the safety value lives as an extra ``"v_safety"`` head
on the SAC critic's :class:`MultiHeadMultiCritic`, sharing the encoder
with the Q-heads. This subclass simply default-sets
``extra_critic_head_names=("v_safety",)`` so users instantiating
``SafeSAC(...)`` get a multi-head critic without having to set the
kwarg explicitly. All safety behaviour (pretrain, shield, online
predicate-label update) is wired through the standard Ajax hooks
(``init_transform``, ``action_pipeline``, ``eval_action_transform``,
``auxiliary_update``); see ``safety_experiments.agents.sac_hooks`` for
the SafeSAC-specific factories.

Backward compatibility: passing ``extra_critic_head_names=()``
explicitly disables the safety head and gives plain SAC behaviour.
"""

from ajax.agents.SAC.SAC import SAC


class SafeSAC(SAC):
    name: str = "SafeSAC"

    def __init__(self, *args, **kwargs):
        # Default-add the safety head if the caller didn't already.
        kwargs.setdefault("extra_critic_head_names", ("v_safety",))
        kwargs.setdefault("extra_critic_head_dims", (1,))
        super().__init__(*args, **kwargs)
