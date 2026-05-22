"""Exploration research features as composable :class:`Extension`s.

EDGE (Expert Decayed Guided Exploration) substitutes the expert action
for the policy action during collection, gated by a value comparison
that decays over training. The four gating variants of the pre-refactor
``train_SAC.py`` are exposed here; the SAC training factory reads the
extension stack and assembles the ``action_pipeline`` callable that
``collect_experience`` consumes. Every variant is behaviour-equivalent
to its inline counterpart — the gate math lives unchanged in
:mod:`ajax.modules.exploration`.

* :class:`EDGEExploration` — one extension parametrised by ``gate``:
  ``"fixed"`` / ``"argmax"`` / ``"boltzmann"`` (value-gap gates) or
  ``"lcb"`` / ``"argmax_lcb"`` / ``"thompson"`` (quality-aware gates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ajax.extensions.base import Extension

_VALID_GATES = frozenset(
    {"fixed", "argmax", "boltzmann", "lcb", "argmax_lcb", "thompson"}
)


@dataclass(frozen=True)
class EDGEExploration(Extension):
    """Expert Decayed Guided Exploration during collection.

    ``gate`` selects the substitution rule (see module docstring). The
    remaining fields mirror the ``train_SAC.py`` flags one-for-one:

    * ``decay_frac``      — value-gap gates decay to 0 over this fraction.
    * ``tau``             — Boltzmann temperature on the value gap.
    * ``fixed_prob``      — substitution probability for the fixed gate.
    * ``lcb_beta_init`` / ``lcb_beta_decay_k`` — LCB pessimism schedule.
    * ``lcb_temperature`` — softmax temperature of the LCB / Thompson gate.
    * ``lcb_asymmetric``  — LCB on the expert arm, UCB on the policy arm.
    * ``epsilon_floor``   — minimum policy-pick probability (Thompson).
    """

    expert_policy: Callable
    gate: str = "fixed"
    decay_frac: float = 0.30
    tau: float = 1.0
    fixed_prob: float = 0.5
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    lcb_asymmetric: bool = False
    epsilon_floor: float = 0.0
    name: str = "edge_exploration"

    def __post_init__(self):
        if self.gate not in _VALID_GATES:
            raise ValueError(
                f"EDGEExploration.gate must be one of {sorted(_VALID_GATES)}, "
                f"got {self.gate!r}."
            )


__all__ = ["EDGEExploration"]
