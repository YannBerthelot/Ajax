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
from typing import Callable, Optional

from ajax.extensions.base import Extension


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
    """

    expert_policy: Callable
    detach: bool = False
    name: str = "expert_obs_augmentation"


@dataclass(frozen=True)
class OnlineBC(Extension):
    """Online, value-weighted, warmup-decaying behavioural-cloning term.

    Equivalent to the ``use_online_bc`` additive actor-loss term: a
    ``bc_coef · w(s) · ||μ_θ(s) - a_expert||²`` penalty that decays to
    zero once ``train_frac`` exceeds ``critic_warmup_frac``.
    """

    expert_policy: Callable
    bc_coef: float = 1.0
    critic_warmup_frac: float = 0.15
    name: str = "online_bc"


@dataclass(frozen=True)
class ResidualPolicy(Extension):
    """Residual RL (Johannink et al. 2019).

    The executed action is ``clip(a_expert + scale·a_pi, -1, 1)``. The
    actor's policy gradient and the TD-target bootstrap both flow through
    the residual transform. Equivalent to ``use_residual_rl``.
    """

    expert_policy: Callable
    scale: float = 1.0
    name: str = "residual_policy"


@dataclass(frozen=True)
class JSRLCurriculum(Extension):
    """True JSRL per-episode handoff (Uchendu et al. 2023).

    While ``step_in_episode < H_t`` the expert acts; otherwise the
    learner does. ``H_t`` decays linearly from ``episode_length`` to 0
    over the first ``decay_frac`` fraction of training. Equivalent to
    ``jsrl_curriculum`` in ``train_SAC.py``.
    """

    expert_policy: Callable
    episode_length: int = 1000
    decay_frac: float = 0.5
    name: str = "jsrl_curriculum"


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
