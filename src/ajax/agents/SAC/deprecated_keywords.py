"""SAC's deprecated keywords, and what each now stands for.

Each keyword here warns and builds its replacement, an extension or an
option every agent has (CLAUDE.md rule 10, expand then contract). Once
AjaxExperiments passes the replacements, B7 deletes this file and the
keywords with it.
"""

import warnings
from collections.abc import Sequence
from typing import Any, Optional

from ajax.controller import transforms_of
from ajax.deprecation import warn_inert
from ajax.extensions.expert import ExpertGains, ExpertWarmup
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)

#: SAC's warm-up mix while expert_fraction is left unset (None), until B7.
DEFAULT_WARMUP_FRACTION = 0.7


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


def translate_deprecated_keywords(
    expert: Any,
    *,
    extensions: Sequence[Any],
    learning_starts: int,
    expert_fraction: Optional[float],
    normalize_observations: bool,
    normalize_obs_running: bool,
    use_train_frac: bool,
    augment_obs_with_expert_state: bool,
    augment_obs_with_expert_action: bool,
    use_pid_policy: bool,
    jsrl_curriculum: Optional[bool],
    skip_critic_pretrain: bool,
) -> tuple[tuple, bool]:
    """The extensions SAC's deprecated keywords stand for, which SAC appends
    to the caller's extensions, and its normalize_observations. Every
    keyword set warns, pointing at the caller of SAC()."""
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
            stacklevel=3,
        )
    built = augmentation_features(
        expert,
        progress=use_train_frac,
        state=augment_obs_with_expert_state,
        action=augment_obs_with_expert_action,
        stacklevel=4,
    )
    if use_pid_policy:
        warnings.warn(
            "SAC's use_pid_policy will be removed: pass ExpertGains()"
            " (ajax.extensions.expert) in extensions.",
            FutureWarning,
            stacklevel=3,
        )
        built = (*built, ExpertGains())
    built = (
        *built,
        *warmup_extension(
            expert, expert_fraction, (*extensions, *built), learning_starts, 4
        ),
    )
    warn_inert(
        "SAC",
        "SAC counts each env's steps in its episode when an extension reads"
        " them (JSRLCurriculum).",
        stacklevel=4,
        jsrl_curriculum=jsrl_curriculum,
    )
    return built, normalize_observations or normalize_obs_running
