"""Expert-guidance research features as composable :class:`Extension`s.

Phase 2 of the agent-architecture rework moves SAC's 14 inline research
features out of ``train_SAC.py`` into composable :class:`Extension`
objects. The expert-dependent group lives here:

* :class:`ExpertGuidance`        — frozen expert policy + warmup/buffer/
  pre-training seeding; the shared base the others build on.
* :class:`ExpertObsAugmentation` — append ``a_expert`` to the observation.
* :class:`OnlineBC`              — value-weighted decaying BC loss term.
* :class:`ResidualPolicy`        — execute ``clip(a_expert + scale·a_pi)``.
* :class:`JSRLCurriculum`        — per-episode expert→learner handoff.

Each extension carries only frozen config on ``self`` (a frozen expert
callable, scalar hyper-parameters). The behaviour of every feature is
identical to the corresponding flag-gated path of the pre-refactor
``train_SAC.py`` — the heavy lifting is delegated to the unchanged pure
functions in :mod:`ajax.modules.expert`. The SAC training factory reads
the extension stack and builds the composable hook callables the proven
training functions consume; see :func:`ajax.agents.SAC.sac.make_train`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext
from ajax.modules.expert import (
    compute_online_bc_loss,
    detach_obs_expert_dims,
    residual_action_transform,
)


@dataclass(frozen=True)
class ExpertGuidance(Extension):
    """A frozen expert policy attached to the agent.

    The shared base for every expert-dependent feature. It carries the
    expert callable used by the SAC loop to seed warmup actions, prefill
    the expert replay buffer and (optionally) pre-train the critic. The
    ``expert_fraction`` controls the warmup mix (expert vs uniform);
    ``expert_buffer_n_steps`` how many expert transitions to prefill;
    ``expert_mix_fraction`` how much expert data to blend into each
    training batch. Defaults reproduce ``train_SAC.py``'s.
    """

    expert_policy: Callable
    expert_fraction: float = 0.7
    expert_buffer_n_steps: int = 20_000
    expert_mix_fraction: float = 0.1
    use_expert_guidance: bool = True
    name: str = "expert_guidance"


@dataclass(frozen=True)
class ExpertObsAugmentation(Extension):
    """Append the expert action to the observation seen by the networks.

    Equivalent to ``augment_obs_with_expert_action`` in ``train_SAC.py``:
    layout ``[env_obs | a_expert | train_frac]``. ``detach`` matches the
    old ``detach_obs_aug_action`` — when set, the actor reads the expert
    hint but no gradient flows through those dims.

    The construction-time obs-augmentation (appending ``a_expert`` to the
    observation, which changes the network input dim) is still wired
    through the legacy ``augment_obs_with_expert_action`` flag — the SAC
    factory's auto-append shim echoes it so ``init_SAC`` /
    ``collect_experience`` / the augmented training batch all see the
    correct dim. Only the runtime ``detach`` stop-gradient lives on
    :meth:`on_obs` — that part used to be the
    ``make_policy_obs_preprocessor`` builder, which is gone now.

    ``action_dim`` is required to know where the ``a_expert`` slice
    starts inside the augmented obs layout ``[env_obs | a_expert |
    train_frac]``. User-constructed ``ExpertObsAugmentation()`` instances
    leave it 0; the SAC factory's auto-append shim resolves it from
    ``get_action_dim(env)`` and copies the resolved value onto the
    instance with ``dataclasses.replace`` so :meth:`on_obs` has
    everything it needs without threading per-step kwargs through the
    phase API. The stop-gradient is a no-op when ``detach`` is False or
    ``action_dim`` is 0.
    """

    expert_policy: Callable
    detach: bool = False
    action_dim: int = 0
    name: str = "expert_obs_augmentation"

    def on_obs(
        self,
        obs: jax.Array,
        ext_state: Any,
        ctx: ExtensionContext,
    ) -> jax.Array:
        """Stop-gradient through the ``a_expert`` dims of the augmented obs."""
        del ext_state, ctx
        if not self.detach or self.action_dim <= 0:
            return obs
        return detach_obs_expert_dims(obs, self.action_dim)


@dataclass(frozen=True)
class OnlineBC(Extension):
    """Online, value-weighted, warmup-decaying behavioural-cloning term.

    Equivalent to the ``use_online_bc`` additive actor-loss term: a
    ``bc_coef · w(s) · ||μ_θ(s) - a_expert||²`` penalty that decays to
    zero once ``train_frac`` exceeds ``critic_warmup_frac``. The
    behaviour is identical to the pre-refactor ``make_bc_loss_fn``
    builder: the term is silently skipped when
    ``agent_state.expert_critic_params is None`` (i.e.
    :class:`MCPretrain` hasn't run), which keeps a stack with only this
    extension a no-op rather than an error.

    The actor-loss call site folds every extension via
    ``stack.actor_loss(...)``; this method reads ``pi_loc``,
    ``a_expert``, ``observations`` and ``train_frac`` from ``batch``
    (populated by :func:`policy_loss_function` in
    ``ajax.agents.SAC.sac``).
    """

    expert_policy: Callable
    bc_coef: float = 1.0
    critic_warmup_frac: float = 0.15
    name: str = "online_bc"

    def actor_loss(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        ctx: ExtensionContext,
    ) -> Any:
        del agent_state, ext_state, ctx
        # ``policy_loss_function`` (in ``ajax.agents.SAC.sac``) is what
        # populates ``batch`` here. Every operand the legacy
        # ``compute_online_bc_loss`` consumes is threaded through that
        # dict so this method needs nothing off ``agent_state`` — see
        # the Phase 2b commit notes on actor-loss plumbing.
        expert_critic_params = batch.get("expert_critic_params", None)
        a_expert = batch.get("a_expert", None)
        train_frac = batch.get("train_frac", None)
        # Mirror legacy ``bc_loss_fn is not None and
        # expert_critic_params is not None`` gate: no-op when MC
        # pre-training hasn't supplied φ*, or when ``critic_warmup_frac
        # <= 0`` (the BC term would always be zero anyway).
        if (
            expert_critic_params is None
            or a_expert is None
            or train_frac is None
            or self.critic_warmup_frac <= 0
        ):
            return 0.0
        return compute_online_bc_loss(
            batch["pi_loc"],
            a_expert,
            batch["critic_state"],
            expert_critic_params,
            batch["observations"],
            train_frac,
            self.critic_warmup_frac,
            batch["expert_v_min"],
            batch["expert_v_max"],
            self.bc_coef,
        )


@dataclass(frozen=True)
class ResidualPolicy(Extension):
    """Residual RL (Johannink et al. 2019).

    The executed action is ``clip(a_expert + scale·a_pi, -1, 1)``. The
    actor's policy gradient and the TD-target bootstrap both flow through
    the residual transform. Equivalent to ``use_residual_rl``.

    The actor-loss / TD-target call sites pull this extension's
    :meth:`transform_action` helper directly off the stack; the
    collection-time substitution stays with ``make_action_pipeline``
    via the legacy ``use_residual_rl=True`` flag the resolver echoes.
    """

    expert_policy: Callable
    scale: float = 1.0
    name: str = "residual_policy"

    def transform_action(self, actions: jax.Array, a_expert: jax.Array) -> jax.Array:
        """``clip(a_expert + scale·a_pi, -1, 1)`` — the residual mix."""
        return residual_action_transform(actions, a_expert, scale=self.scale)

    def eval_action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Eval-time residual mix: same clip math as :meth:`transform_action`.

        The SAC eval loop (:func:`ajax.evaluate.step_environment`) folds
        any non-None eval_action through a thin callable; the SAC factory
        builds that callable off this extension's
        :meth:`transform_action` when no explicit
        ``eval_action_transform`` is passed. See
        :func:`_build_residual_policy_eval_transform` in
        ``ajax.agents.SAC.sac``.
        """
        del agent_state, ext_state, rng, ctx
        raw_actions = obs["raw_actions"]
        expert_actions = obs["expert_actions"]
        return self.transform_action(raw_actions, expert_actions)


@dataclass(frozen=True)
class JSRLCurriculum(Extension):
    """True JSRL per-episode handoff (Uchendu et al. 2023).

    While ``step_in_episode < H_t`` the expert acts; otherwise the
    learner does. ``H_t`` decays linearly from ``episode_length`` to 0
    over the first ``decay_frac`` fraction of training. Equivalent to
    ``jsrl_curriculum`` in ``train_SAC.py``.

    The :meth:`action` phase owns the per-episode substitution. It reads
    ``post_warmup_action`` and ``expert_action`` off the SAC action
    pipeline's batch dict (the same dict :class:`EDGEExploration` and
    :class:`~ajax.extensions.target_mods.ValueBox` use) and per-env
    ``step_in_episode`` off ``agent_state.collector_state``; the SAC
    pipeline still owns the ``step_in_episode`` increment/reset (in
    :func:`collect_experience`).
    """

    expert_policy: Callable
    episode_length: int = 1000
    decay_frac: float = 0.5
    name: str = "jsrl_curriculum"

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Substitute the expert action while ``step_in_episode < H_t``."""
        del ext_state, rng
        post_warmup_action = obs["post_warmup_action"]
        expert_action = obs["expert_action"]

        total_timesteps = max(int(ctx.total_steps), 1)
        global_t = agent_state.collector_state.timestep
        train_frac = global_t / jnp.maximum(total_timesteps, 1)
        curriculum_progress = jnp.clip(train_frac / self.decay_frac, 0.0, 1.0)
        H_t = self.episode_length * (1.0 - curriculum_progress)
        step_in_ep = agent_state.collector_state.step_in_episode
        use_expert_jsrl = step_in_ep.astype(jnp.float32) < H_t
        use_expert_jsrl = use_expert_jsrl.reshape(
            (-1,) + (1,) * (post_warmup_action.ndim - 1)
        )
        return jnp.where(use_expert_jsrl, expert_action, post_warmup_action)


def first_of_type(stack, cls) -> Optional[Extension]:
    """Return the first extension that is an instance of ``cls``, else None."""
    for ext in stack:
        if isinstance(ext, cls):
            return ext
    return None


__all__ = [
    "ExpertGuidance",
    "ExpertObsAugmentation",
    "OnlineBC",
    "ResidualPolicy",
    "JSRLCurriculum",
    "first_of_type",
]
