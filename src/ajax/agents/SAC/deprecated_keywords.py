"""SAC's deprecated keywords, and what each now stands for.

Each keyword here warns and builds its replacement, an extension or an
option every agent has (CLAUDE.md rule 10, expand then contract). SAC
declares them as the fields of SACDeprecatedKeywords, so SAC.py shows none
of them. Once AjaxExperiments passes the replacements, B7 deletes this file
and the keywords with it.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ajax.controller import policy_action_dim, transforms_of
from ajax.deprecation import warn_inert
from ajax.extensions.expert import ExpertGains, ExpertWarmup, StorePolicyAction
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)
from ajax.extensions.pretrain import ExpertPrefill
from ajax.schedule import Schedule, linear

#: SAC's warm-up mix while expert_fraction is left unset (None), until B7.
DEFAULT_WARMUP_FRACTION = 0.7
#: SAC's expert prefill while expert_buffer_n_steps is left unset (None),
#: until B7.
DEFAULT_PREFILL_STEPS = 20_000


def augmentation_features(
    expert: Any, *, progress: bool, state: bool, action: bool, stacklevel: int = 3
) -> tuple:
    """The observation features SAC's augmentation keywords stand for, in
    their columns' order: the training progress (use_train_frac), the
    expert's state (augment_obs_with_expert_state, a stateful expert's) and
    its action (augment_obs_with_expert_action, with an expert)."""
    if progress or state or action:
        warnings.warn(
            "SAC's use_train_frac, augment_obs_with_expert_state and"
            " augment_obs_with_expert_action will be removed: pass"
            " TrainingProgressInput(), ExpertStateInput.for_expert(expert) and"
            " ExpertActionInput() (ajax.extensions.observe) in extensions.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    stateful = expert is not None and hasattr(expert, "init_state")
    return (
        *([TrainingProgressInput()] if progress else []),
        *([ExpertStateInput.for_expert(expert)] if state and stateful else []),
        *([ExpertActionInput()] if action and expert is not None else []),
    )


def warmup_extension(
    expert: Any,
    fraction: Optional[float],
    extensions: Sequence[Any],
    learning_starts: int,
    stacklevel: int = 3,
) -> tuple:
    """The ExpertWarmup that SAC's expert_fraction stands for (None: 0.7
    with an expert), beside the run's extensions: none without an expert,
    under an action transform or without warm-up steps, where SAC never
    mixed. A warm-up built from the default warns that the default will go;
    a fraction passed warns that the keyword will go, and raises next to an
    ExpertWarmup."""
    given = any(isinstance(extension, ExpertWarmup) for extension in extensions)
    if fraction is not None:
        if given:
            raise ValueError("Pass SAC's expert_fraction or ExpertWarmup, not both.")
        warnings.warn(
            f"SAC's expert_fraction will be removed: pass ExpertWarmup(fraction="
            f"{fraction}) (ajax.extensions.expert) in extensions, or none for no"
            " expert in the warm-up.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    share = DEFAULT_WARMUP_FRACTION if fraction is None else fraction
    transformed = bool(transforms_of(extensions))
    if expert is None or given or transformed or learning_starts <= 0 or share <= 0:
        return ()
    if fraction is None:
        warnings.warn(
            "SAC with an expert_policy adds ExpertWarmup(fraction=0.7) by"
            " default; this default will be removed: pass it in extensions to"
            " keep it.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    return (ExpertWarmup(fraction=share),)


def prefill_extension(
    expert: Any,
    n_steps: Optional[int],
    extensions: Sequence[Any],
    stacklevel: int = 3,
) -> tuple:
    """The ExpertPrefill that SAC's expert_buffer_n_steps stands for (None:
    20 000 steps with an expert), beside the run's extensions: none without
    an expert or without steps. A prefill built from the default warns that
    the default will go; a number of steps passed warns that the keyword
    will go, and raises next to an ExpertPrefill."""
    given = any(isinstance(extension, ExpertPrefill) for extension in extensions)
    if n_steps is not None:
        if given:
            raise ValueError(
                "Pass SAC's expert_buffer_n_steps or ExpertPrefill, not both."
            )
        warnings.warn(
            f"SAC's expert_buffer_n_steps will be removed: pass ExpertPrefill("
            f"n_steps={n_steps}) (ajax.extensions.pretrain) in extensions, or"
            " none for no expert prefill.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    steps = DEFAULT_PREFILL_STEPS if n_steps is None else n_steps
    if expert is None or given or steps <= 0:
        return ()
    if n_steps is None:
        warnings.warn(
            f"SAC with an expert_policy adds ExpertPrefill(n_steps={steps}) by"
            " default; this default will be removed: pass it in extensions to"
            " keep it.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    return (ExpertPrefill(n_steps=steps),)


def stored_action_extension(
    expert: Any, store_policy_action: bool, extensions: Sequence[Any], stacklevel: int
) -> tuple:
    """The StorePolicyAction that SAC's store_policy_action stands for, where
    the keyword acted: through SAC's expert pipeline, so with an expert and
    outside gain mode (StorePolicyAction acts in every run). Set, the
    keyword warns that it will go, and raises next to a StorePolicyAction."""
    if not store_policy_action:
        return ()
    if any(isinstance(extension, StorePolicyAction) for extension in extensions):
        raise ValueError(
            "Pass SAC's store_policy_action or StorePolicyAction, not both."
        )
    warnings.warn(
        "SAC's store_policy_action will be removed: pass StorePolicyAction()"
        " (ajax.extensions.expert) in extensions; it stores the policy's action"
        " with or without an expert, where the keyword acted only with one"
        " outside gain mode.",
        FutureWarning,
        stacklevel=stacklevel,
    )
    gains = any("actuate" in extension.implemented_phases() for extension in extensions)
    if expert is None or gains:
        return ()
    return (StorePolicyAction(),)


def translate_deprecated_keywords(
    expert: Any,
    *,
    extensions: Sequence[Any],
    learning_starts: int,
    expert_fraction: Optional[float],
    expert_buffer_n_steps: Optional[int],
    normalize_observations: bool,
    normalize_obs_running: bool,
    use_train_frac: bool,
    augment_obs_with_expert_state: bool,
    augment_obs_with_expert_action: bool,
    use_pid_policy: bool,
    jsrl_curriculum: Optional[bool],
    skip_critic_pretrain: bool,
    store_policy_action: bool = False,
    stacklevel: int = 3,
) -> tuple[tuple, bool]:
    """The extensions SAC's deprecated keywords stand for, which SAC appends
    to the caller's extensions, and its normalize_observations. Every
    keyword set warns, pointing stacklevel frames up, at the caller of
    SAC(). The prefill comes last, so every other extension keeps its place
    in the stack and its key."""
    # The critic pretraining skip_critic_pretrain switched off is gone: True
    # changes nothing, False cannot be honoured.
    if not skip_critic_pretrain:
        raise ValueError(
            "SAC no longer pretrains its critic on the cloning dataset; "
            "use the MCPretrain extension."
        )
    if normalize_obs_running:
        warnings.warn(
            "SAC's normalize_obs_running is normalize_observations, which a"
            " replay agent applies itself; pass normalize_observations=True:"
            " normalize_obs_running will be removed.",
            FutureWarning,
            stacklevel=stacklevel,
        )
    built = augmentation_features(
        expert,
        progress=use_train_frac,
        state=augment_obs_with_expert_state,
        action=augment_obs_with_expert_action,
        stacklevel=stacklevel + 1,
    )
    if use_pid_policy:
        warnings.warn(
            "SAC's use_pid_policy will be removed: pass ExpertGains()"
            " (ajax.extensions.expert) in extensions.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        built = (*built, ExpertGains())
    built = (
        *built,
        *warmup_extension(
            expert,
            expert_fraction,
            (*extensions, *built),
            learning_starts,
            stacklevel + 1,
        ),
    )
    built = (
        *built,
        *stored_action_extension(
            expert, store_policy_action, (*extensions, *built), stacklevel + 1
        ),
    )
    built = (
        *built,
        *prefill_extension(
            expert, expert_buffer_n_steps, (*extensions, *built), stacklevel + 1
        ),
    )
    warn_inert(
        "SAC",
        "SAC counts each env's steps in its episode when an extension reads"
        " them (JSRLCurriculum).",
        stacklevel=stacklevel + 1,
        jsrl_curriculum=jsrl_curriculum,
    )
    return built, normalize_observations or normalize_obs_running


@dataclass(kw_only=True, eq=False, repr=False)
class SACDeprecatedKeywords:
    """SAC's deprecated keywords, each a field so the run config logs it,
    translated into what replaces it when SAC is built
    (translate_deprecated_arguments, a deliberate workaround B7 deletes).

    Args:
        expert_buffer_n_steps: ExpertPrefill(n_steps); None, with an
            expert, 20 000 steps.
        expert_fraction: ExpertWarmup(fraction); None, with an expert, 0.7.
        use_pid_policy: ExpertGains().
        use_train_frac: TrainingProgressInput().
        augment_obs_with_expert_action: ExpertActionInput().
        augment_obs_with_expert_state: ExpertStateInput.for_expert(expert).
        normalize_obs_running: normalize_observations.
        jsrl_curriculum: No effect: SAC counts each env's steps in its
            episode for the extensions that read them.
        skip_critic_pretrain: No effect; False raises (MCPretrain replaces
            the critic pretraining it switched off).
        store_policy_action: StorePolicyAction(), with an expert outside
            gain mode.
        target_entropy_initial_per_dim: With target_entropy_ramp_frac, a
            linear ramp of the target entropy per dimension to
            target_entropy_per_dim over that share of the run.
        target_entropy_ramp_frac: The share of the run the ramp lasts.
    """

    expert_buffer_n_steps: Optional[int] = None
    expert_fraction: Optional[float] = None
    use_pid_policy: bool = False
    use_train_frac: bool = False
    augment_obs_with_expert_action: bool = False
    augment_obs_with_expert_state: bool = False
    normalize_obs_running: bool = False
    jsrl_curriculum: Optional[bool] = None
    skip_critic_pretrain: bool = True
    store_policy_action: bool = False
    target_entropy_initial_per_dim: Optional[float] = None
    target_entropy_ramp_frac: float = 0.5

    def translate_deprecated_arguments(self: Any) -> dict[str, Any]:
        """The extensions the keywords stand for, after the caller's, and
        normalize_observations; each keyword set warns at SAC's caller (five
        frames up: the dataclass's __init__ and __post_init__ between)."""
        if self.target_entropy_initial_per_dim is not None:
            warnings.warn(
                "SAC's target_entropy_initial_per_dim and target_entropy_ramp_frac"
                " will be removed: pass target_entropy_per_dim="
                "linear(initial, final, steps) (ajax.schedule).",
                FutureWarning,
                stacklevel=4,
            )
        built, normalize_observations = translate_deprecated_keywords(
            self.expert_policy,
            extensions=self.extensions,
            learning_starts=self.learning_starts,
            expert_fraction=self.expert_fraction,
            expert_buffer_n_steps=self.expert_buffer_n_steps,
            normalize_observations=self.normalize_observations,
            normalize_obs_running=self.normalize_obs_running,
            use_train_frac=self.use_train_frac,
            augment_obs_with_expert_state=self.augment_obs_with_expert_state,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
            use_pid_policy=self.use_pid_policy,
            jsrl_curriculum=self.jsrl_curriculum,
            skip_critic_pretrain=self.skip_critic_pretrain,
            store_policy_action=self.store_policy_action,
            stacklevel=5,
        )
        return {
            "extensions": (*self.extensions, *built),
            "normalize_observations": normalize_observations,
        }

    def target_entropy_over(self: Any, total_timesteps: int) -> Schedule:
        """The target entropy, or the deprecated ramp from
        target_entropy_initial_per_dim over target_entropy_ramp_frac of the
        run, each times the action's width before the ramp is built."""
        if self.target_entropy_initial_per_dim is None:
            return self.target_entropy
        width = policy_action_dim(self.env_args)
        return linear(
            self.target_entropy_initial_per_dim * width,
            self.target_entropy,
            self.target_entropy_ramp_frac * total_timesteps,
        )
