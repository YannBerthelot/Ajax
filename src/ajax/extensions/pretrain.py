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
from typing import Any, Callable, Optional, Tuple

import jax

from ajax.extensions.base import Extension, ExtensionContext
from ajax.modules.pretrain import (
    pretrain_critic_mc,
    pretrain_critic_online_light,
    refresh_phi_star,
)


@dataclass(frozen=True)
class MCPretrain(Extension):
    """Monte-Carlo critic pre-training (``use_mc_critic_pretrain``).

    Regresses a frozen expert critic ``φ*`` on unbiased MC returns from
    expert rollouts, persists ``φ*`` + the value range ``v_min/v_max`` on
    the agent state, and — when ``use_online_light`` is set — applies a
    weak supervised nudge of the online critic toward ``φ*`` to reduce
    seed-to-seed variance at initialisation.

    The remaining ``Optional`` fields below carry the closed-over
    context the pre-training math needs (env config, critic
    optimizer/network args, mode, gamma/reward_scale, total timesteps,
    obs-augmentation flag, optionally pre-collected MC tensors, the
    ``use_phi_refresh`` toggle that decides whether to keep the
    optimizer state alive). User-constructed ``MCPretrain()`` instances
    leave them ``None``; the SAC factory's auto-append shim copies the
    resolved values onto the instance with :func:`dataclasses.replace`
    so :meth:`pretrain` has everything it needs without threading
    per-step kwargs through the phase API — same pattern as PhiRefresh.
    """

    expert_policy: Callable
    n_mc_steps: int = 10_000
    n_mc_episodes: int = 100
    n_steps: int = 5_000
    use_online_light: bool = True
    online_light_steps: int = 500
    online_light_lr_scale: float = 0.1
    # Closed-over context populated by the SAC factory shim.
    env_args: Any = None
    network_args: Any = None
    critic_optimizer_args: Any = None
    num_critics: int = 2
    mode: Optional[str] = None
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    total_timesteps: Optional[int] = None
    use_train_frac: bool = False
    augment_obs_with_expert_action: bool = False
    use_phi_refresh: bool = False
    mc_preloaded_data: Optional[Tuple] = None
    name: str = "mc_pretrain"

    def bind_to_agent(self, **agent_context: Any) -> "MCPretrain":
        """Populate env/network/optimizer config + runtime context from the factory.

        Self-contained replacement for the legacy SAC-side
        ``_inject_pretrain_extensions_context`` helper. The agent
        factory calls ``stack.bind_to_agent(env_args=..., network_args=...,
        critic_optimizer_args=..., num_critics=..., mode=..., gamma=...,
        reward_scale=..., total_timesteps=..., use_train_frac=...,
        augment_obs_with_expert_action=..., use_phi_refresh=...,
        mc_preloaded_data=...)`` and this method copies the kwargs it
        cares about onto a new frozen instance.

        ``use_phi_refresh`` is computed by the caller from the stack
        itself (True iff a :class:`PhiRefresh` extension is present);
        kept on the kwarg surface so MCPretrain doesn't need to
        introspect the stack.
        """
        import dataclasses

        env_args = agent_context.get("env_args", None)
        if self.env_args is not None or env_args is None:
            return self
        return dataclasses.replace(
            self,
            env_args=env_args,
            network_args=agent_context.get("network_args"),
            critic_optimizer_args=agent_context.get("critic_optimizer_args"),
            num_critics=agent_context.get("num_critics", self.num_critics),
            mode=agent_context.get("mode"),
            gamma=agent_context.get("gamma"),
            reward_scale=agent_context.get("reward_scale"),
            total_timesteps=agent_context.get("total_timesteps"),
            use_train_frac=agent_context.get("use_train_frac", self.use_train_frac),
            augment_obs_with_expert_action=agent_context.get(
                "augment_obs_with_expert_action",
                self.augment_obs_with_expert_action,
            ),
            use_phi_refresh=agent_context.get("use_phi_refresh", self.use_phi_refresh),
            mc_preloaded_data=agent_context.get(
                "mc_preloaded_data", self.mc_preloaded_data
            ),
        )

    def pretrain(
        self,
        agent_state: Any,
        ext_state: Any,
        ctx: ExtensionContext,
    ) -> tuple[Any, Any]:
        """Build the frozen expert critic ``φ*`` and persist it on ``agent_state``.

        Behaviour-equivalent to the pre-refactor ``use_mc_critic_pretrain``
        block of :func:`make_train.init_fn` in ``ajax.agents.SAC.train_SAC``:
        builds a fresh expert critic, regresses it on unbiased MC returns,
        sets ``expert_critic_params`` / ``expert_v_min`` /
        ``expert_v_max`` on ``agent_state``, and — when
        ``use_online_light`` is set — applies the weak supervised nudge of
        the online critic toward ``φ*``. ``use_phi_refresh`` controls
        whether the φ* optimizer state is kept alive for the
        :class:`PhiRefresh` post-update path.

        The SAC factory's auto-append shim is what populates the closed-
        over context fields below (``env_args``, ``mode``, ``gamma`` …);
        an unconfigured instance is a no-op (returns ``agent_state``
        unchanged) so a stack-only ``MCPretrain(...)`` outside a SAC
        factory call doesn't crash.
        """
        # Bail out early when the shim hasn't filled in the context
        # (e.g. unit-tested standalone). The SAC factory always populates
        # these fields when running through ``make_train``.
        if (
            self.env_args is None
            or self.network_args is None
            or self.critic_optimizer_args is None
            or self.mode is None
            or self.gamma is None
            or self.reward_scale is None
        ):
            return agent_state, ext_state

        # Local import to avoid a circular ``ajax.networks ↔
        # ajax.extensions`` import: networks.py doesn't depend on
        # extensions but the extension's pretrain phase does need the
        # critic builder.
        from ajax.environments.utils import get_action_dim
        from ajax.networks.networks import get_initialized_critic

        rng = ctx.rng
        total_timesteps = self.total_timesteps
        expert_critic_state = get_initialized_critic(
            key=rng,
            env_config=self.env_args,
            critic_optimizer_config=self.critic_optimizer_args,
            network_config=self.network_args,
            num_critics=self.num_critics,
            max_timesteps=total_timesteps if self.use_train_frac else None,
            extra_obs_dim=(
                get_action_dim(self.env_args.env, self.env_args.env_params)
                if self.augment_obs_with_expert_action
                else 0
            ),
        )
        _preloaded = self.mc_preloaded_data
        (
            agent_state,
            frozen_expert_params,
            mc_obs_batched,
            mc_action_batched,
            mc_aux,
            expert_critic_state_trained,
        ) = pretrain_critic_mc(
            agent_state=agent_state,
            expert_critic_state=expert_critic_state,
            expert_policy=self.expert_policy,
            mode=self.mode,
            env_args=self.env_args,
            recurrent=self.network_args.lstm_hidden_size is not None,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            n_mc_steps=self.n_mc_steps,
            n_mc_episodes=self.n_mc_episodes,
            n_steps=self.n_steps,
            max_timesteps=total_timesteps if self.use_train_frac else None,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
            preloaded_obs=_preloaded[0] if _preloaded is not None else None,
            preloaded_action=_preloaded[1] if _preloaded is not None else None,
            preloaded_mc=_preloaded[2] if _preloaded is not None else None,
        )
        agent_state = agent_state.replace(
            expert_critic_params=frozen_expert_params,
            expert_v_min=mc_aux.v_min,
            expert_v_max=mc_aux.v_max,
            # Keep φ* optimizer state alive for periodic refresh (None when disabled)
            expert_critic_state=(
                expert_critic_state_trained if self.use_phi_refresh else None
            ),
        )
        jax.debug.print(
            "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
            "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
            i=mc_aux.initial_loss,
            f=mc_aux.final_loss,
            qm=mc_aux.q_expert_mean,
            qn=mc_aux.q_expert_min,
            qx=mc_aux.q_expert_max,
        )
        if self.use_online_light:
            agent_state = pretrain_critic_online_light(
                agent_state,
                mc_obs_batched,
                mc_action_batched,
                n_steps=self.online_light_steps,
                lr_scale=self.online_light_lr_scale,
            )
            jax.debug.print(
                "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                n=self.online_light_steps,
                s=self.online_light_lr_scale,
            )
        return agent_state, ext_state


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

    def bind_to_agent(self, **agent_context: Any) -> "PhiRefresh":
        """Populate ``buffer`` / ``gamma`` / ``reward_scale`` from the agent factory.

        Self-contained replacement for the legacy SAC-side
        ``_inject_policy_extensions_context`` helper: the agent factory
        calls ``stack.bind_to_agent(buffer=..., gamma=..., reward_scale=...,
        ...)`` and this method copies the kwargs it cares about onto a
        new frozen instance. Unrecognised kwargs are ignored, so the
        same call works uniformly across the whole stack.
        """
        import dataclasses

        buffer = agent_context.get("buffer", None)
        gamma = agent_context.get("gamma", None)
        reward_scale = agent_context.get("reward_scale", None)
        if self.buffer is not None or buffer is None:
            return self
        return dataclasses.replace(
            self, buffer=buffer, gamma=gamma, reward_scale=reward_scale
        )

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
