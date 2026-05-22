"""TD-target modifier research features as composable :class:`Extension`s.

These reshape the SAC Bellman target before it enters the critic loss
(the ``on_target`` phase). Each is behaviour-equivalent to the
corresponding flag-gated path of the pre-refactor ``train_SAC.py``; the
SAC training factory reads the extension stack and assembles the single
``target_modifier`` callable the proven ``update_value_functions``
consumes.

* :class:`IBRL`               — ``ibrl_bootstrap``: add the positive gap
  ``γ(1-d)·max(Q_expert - Q_policy, 0)`` so the value function matches an
  argmax action-selection policy.
* :class:`LCBGatedBootstrap`  — ``lcb_gated_bootstrap``: soft-blend the
  policy and expert TD targets by an LCB-scored sigmoid gate.
* :class:`CriticBlend`        — ``use_critic_blend``: warmup-decaying
  blend of the Bellman target with the frozen-expert value estimate.
* :class:`MCVarianceCorrection` — ``mc_variance_threshold``: replace
  high-ensemble-variance Bellman targets with the MC-pretrained oracle.
* :class:`ValueBox`           — ``use_box``: value-threshold expert
  action override during collection (the ``action`` phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ajax.extensions.base import Extension


@dataclass(frozen=True)
class IBRL(Extension):
    """IBRL bootstrap (``ibrl_bootstrap``).

    Adds ``γ(1-d)·max(min Q_target(s', a_expert') - min Q_target(s', a_π'),
    0)`` to the Bellman target so the value function is consistent with an
    argmax(Q_expert, Q_policy) action-selection rule.
    """

    expert_policy: Callable
    name: str = "ibrl"


@dataclass(frozen=True)
class LCBGatedBootstrap(Extension):
    """LCB-gated bootstrap (``lcb_gated_bootstrap``).

    Scores the policy and expert next-actions by an LCB rule and
    soft-blends the two TD targets with ``p_expert =
    σ((score_e - score_p)/lcb_temperature)``. ``β`` anneals over training.
    """

    expert_policy: Callable
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    name: str = "lcb_gated_bootstrap"


@dataclass(frozen=True)
class CriticBlend(Extension):
    """Warmup-decaying blend of the Bellman target with V_expert.

    Equivalent to ``use_critic_blend``: ``(1-α)·y_bellman + α·V_expert(s')``
    with ``α`` decaying linearly to 0 over ``critic_warmup_frac`` of
    training. Requires a frozen expert critic (MC pre-training).
    """

    expert_policy: Callable
    critic_warmup_frac: float = 0.15
    name: str = "critic_blend"


@dataclass(frozen=True)
class MCVarianceCorrection(Extension):
    """Replace high-variance Bellman targets with the MC oracle.

    Equivalent to ``mc_variance_threshold``: where the inter-critic
    variance exceeds ``threshold`` the Bellman target is swapped for the
    MC-pretrained expert critic's estimate. Requires MC pre-training.
    """

    threshold: float
    name: str = "mc_variance_correction"


@dataclass(frozen=True)
class ValueBox(Extension):
    """Value-threshold expert-action override (``use_box``).

    During collection, override the policy action with the expert's
    whenever the frozen-expert value ``V_expert(s)`` exceeds a curriculum
    threshold that ramps from ``v_min`` to ``v_max`` over training.
    Requires MC pre-training (which supplies ``v_min`` / ``v_max``).
    """

    expert_policy: Callable
    name: str = "value_box"


__all__ = [
    "IBRL",
    "LCBGatedBootstrap",
    "CriticBlend",
    "MCVarianceCorrection",
    "ValueBox",
]
