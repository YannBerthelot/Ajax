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
from typing import Any, Callable, Optional

import jax

from ajax.extensions.base import Extension, ExtensionContext
from ajax.modules.pretrain import refresh_phi_star


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

    ``buffer``, ``gamma`` and ``reward_scale`` are populated by the
    SAC training factory's auto-append shim from
    ``agent_config.gamma`` / ``agent_config.reward_scale`` / the agent
    ``buffer``. User-constructed ``PhiRefresh()`` instances leave them
    ``None``; the SAC factory copies the resolved values onto the
    instance with ``dataclasses.replace`` so the ``post_update`` math
    has everything it needs without threading per-step kwargs through
    the phase API.
    """

    expert_policy: Callable
    interval: int = 500
    steps: int = 20
    buffer: Any = None
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    name: str = "phi_refresh"

    def post_update(
        self,
        agent_state: Any,
        ext_state: Any,
        ctx: ExtensionContext,
    ) -> tuple[Any, Any]:
        # ``post_update`` runs once per ``do_update`` (after the collect
        # step, before the inner gradient-step scan). The interval gate
        # below is byte-equivalent to the pre-refactor
        # ``make_runtime_maintenance`` callable: refresh whenever
        # ``timestep % interval == 0``; otherwise pass through unchanged.
        del ctx
        if self.buffer is None or self.gamma is None or self.reward_scale is None:
            # Unconfigured PhiRefresh — no-op (the SAC factory's
            # auto-append shim is what populates these fields).
            return agent_state, ext_state
        interval = self.interval
        steps = self.steps
        gamma = self.gamma
        reward_scale = self.reward_scale
        expert_policy = self.expert_policy
        buffer = self.buffer

        def _do_refresh(s: Any) -> Any:
            new_s, _aux = refresh_phi_star(
                s, buffer, steps, gamma, reward_scale, expert_policy
            )
            return new_s

        new_state = jax.lax.cond(
            agent_state.collector_state.timestep % interval == 0,
            _do_refresh,
            lambda s: s,
            operand=agent_state,
        )
        return new_state, ext_state


__all__ = ["MCPretrain", "BellmanPretrain", "PhiRefresh"]
