"""One-shot pre-training research features as composable :class:`Extension`s.

These run once, before the training loop (the ``pretrain`` phase / the
``init_transform`` slot of ``build_resumable_train`` — skipped on
resume). Each is behaviour-equivalent to the corresponding flag-gated
fresh-init block of the pre-refactor ``train_SAC.py``; the heavy lifting
is delegated to the unchanged pure functions in
:mod:`ajax.modules.pretrain`.

* :class:`MCPretrain`          — ``use_mc_critic_pretrain``: regress a
  frozen expert critic on Monte-Carlo returns, then (optionally) lightly
  nudge the online critic toward it.
* :class:`BellmanPretrain`     — ``use_bellman_critic_pretrain``: the
  legacy Bellman-bootstrapped critic pre-training on the expert buffer.
* :class:`PhiRefresh`          — ``use_phi_refresh``: periodic
  self-consistent refresh of the frozen expert critic during training
  (the ``post_update`` phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ajax.extensions.base import Extension


@dataclass(frozen=True)
class MCPretrain(Extension):
    """Monte-Carlo critic pre-training (``use_mc_critic_pretrain``).

    Regresses a frozen expert critic ``φ*`` on unbiased MC returns from
    expert rollouts, persists ``φ*`` + the value range ``v_min/v_max`` on
    the agent state, and — when ``use_online_light`` is set — applies a
    weak supervised nudge of the online critic toward ``φ*`` to reduce
    seed-to-seed variance at initialisation.
    """

    expert_policy: Callable
    n_mc_steps: int = 10_000
    n_mc_episodes: int = 100
    n_steps: int = 5_000
    use_online_light: bool = True
    online_light_steps: int = 500
    online_light_lr_scale: float = 0.1
    name: str = "mc_pretrain"


@dataclass(frozen=True)
class BellmanPretrain(Extension):
    """Legacy Bellman-bootstrapped critic pre-training.

    Equivalent to ``use_bellman_critic_pretrain`` — mutually exclusive
    with :class:`MCPretrain`. Bootstraps from an untrained critic, so the
    pre-trained values are biased; kept for ablation parity.
    """

    expert_policy: Callable
    n_steps: int = 5_000
    name: str = "bellman_pretrain"


@dataclass(frozen=True)
class PhiRefresh(Extension):
    """Periodic self-consistent refresh of the frozen expert critic.

    Equivalent to ``use_phi_refresh``: every ``interval`` steps run
    ``steps`` self-consistent Bellman updates on expert-flagged buffer
    transitions, keeping ``φ*`` aligned with the live data distribution.
    Requires :class:`MCPretrain` (which creates the refreshable ``φ*``).
    """

    expert_policy: Callable
    interval: int = 500
    steps: int = 20
    name: str = "phi_refresh"


__all__ = ["MCPretrain", "BellmanPretrain", "PhiRefresh"]
