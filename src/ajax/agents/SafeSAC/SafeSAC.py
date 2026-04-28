"""SafeSAC: a SAC variant that carries a safety V-head in agent state.

SafeSAC is a thin subclass of :class:`SAC` that does not duplicate any
training machinery. It simply exists as a distinct agent identity for
safety-shielded runs: users wire safety behaviour by passing the standard
SAC hooks (``init_transform`` to pretrain a V-head, ``auxiliary_update``
to keep it fresh, ``action_pipeline`` / ``eval_action_transform`` to
shield actions, ``extra_critic_loss_fn`` / ``extra_actor_loss_fn`` for
auxiliary loss terms). The V-head itself lives in
``SACState.safety_critic_state``.
"""

from ajax.agents.SAC.SAC import SAC


class SafeSAC(SAC):
    name: str = "SafeSAC"
