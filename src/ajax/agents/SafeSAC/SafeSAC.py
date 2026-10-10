"""SafeSAC: SAC whose critic carries an extra ``"v_safety"`` value head.

The head lives on the SAC critic's :class:`MultiHeadMultiCritic` and shares
its encoder with the Q-heads; this subclass only default-sets
``extra_critic_head_names=("v_safety",)``. SAC's own losses read the Q-heads
alone, so the safety head keeps its initial weights unless an
:class:`~ajax.extensions.base.Extension` trains it (its ``critic_loss``
phase receives the critic params). Passing ``extra_critic_head_names=()``
gives plain SAC.
"""

from ajax.agents.SAC.SAC import SAC


class SafeSAC(SAC):
    name: str = "SafeSAC"

    def __init__(self, *args, **kwargs):
        # Default-add the safety head if the caller didn't already.
        kwargs.setdefault("extra_critic_head_names", ("v_safety",))
        kwargs.setdefault("extra_critic_head_dims", (1,))
        super().__init__(*args, **kwargs)
