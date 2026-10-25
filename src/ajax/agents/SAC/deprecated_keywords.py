"""SAC's deprecated keywords, and what each now stands for.

Each keyword here warns and builds its replacement, an extension or an
option every agent has (CLAUDE.md rule 10, expand then contract). Once
AjaxExperiments passes the replacements, B7 deletes this file and the
keywords with it.
"""

import warnings
from typing import Any, Optional

from ajax.deprecation import warn_inert
from ajax.extensions.expert import ExpertGains
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)


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


def translate_deprecated_keywords(
    expert: Any,
    *,
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
    to the caller's, and its normalize_observations. Every keyword set warns,
    pointing at the caller of SAC()."""
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
    extensions = augmentation_features(
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
        extensions = (*extensions, ExpertGains())
    warn_inert(
        "SAC",
        "SAC counts each env's steps in its episode when an extension reads"
        " them (JSRLCurriculum).",
        stacklevel=4,
        jsrl_curriculum=jsrl_curriculum,
    )
    return extensions, normalize_observations or normalize_obs_running
