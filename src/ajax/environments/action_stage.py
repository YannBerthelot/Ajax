"""The action stage: the extensions that substitute the executed action
during collection (an action phase), around any agent's action pipeline.

An extension acting there declares its action_slot, before or after the
warm-up's choice (ACTION_SLOTS): EDGEExploration and JSRLCurriculum
substitute the expert's action for the policy's before it, so the warm-up's
uniform or expert steps keep theirs; ValueBox overrides whichever action
was chosen after it. Each reads and writes a dict of the step's quantities
(its obs argument), proposes the action of its slot, and writes where it
handed the expert the step (expert_mask), which the replay flags
(is_expert). The pre-warm-up gates draw on a key threaded through the dict
(gate_rng), the collector's own stream.

The collector applies the stage to the agent's pipeline, after the expert's
call and before gain mode's actuation (expert_collect_kwargs); without an
extension in a slot the pipeline runs as it is, decided before tracing.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ajax.extensions.base import ExtensionContext

#: The stage's slots for an action extension: before the warm-up's choice
#: (EDGEExploration, JSRLCurriculum) and after it (ValueBox).
ACTION_SLOTS = ("pre_warmup", "post_warmup")


def slotted_extensions(extensions: Any, slot: str) -> tuple:
    """The (index, extension) pairs of the stack's extensions acting in
    slot, in stack order."""
    return tuple(
        (index, extension)
        for index, extension in enumerate(extensions)
        if getattr(extension, "action_slot", None) == slot
    )


def check_action_slots(extensions: Any) -> None:
    """Refuse an extension with an action phase and no slot of ACTION_SLOTS:
    the stage has these two places to run one, and runs no other."""
    unslotted = [
        extension.name
        for extension in extensions
        if "action" in extension.implemented_phases()
        and getattr(extension, "action_slot", None) not in ACTION_SLOTS
    ]
    if unslotted:
        raise ValueError(
            "The action stage runs an action extension only in a slot of"
            f" {ACTION_SLOTS}, which {unslotted} declare none (action_slot)."
        )


def dispatch_action_phase(
    agent_state: Any,
    acting: Any,
    indexed_extensions: tuple,
    step: dict,
    slot: str,
    rng: jax.Array,
    total_steps: int,
) -> tuple[dict, list]:
    """Fold the (index, extension) pairs' action phase over step, which
    they read and write: each proposal that is not None replaces step[slot],
    in stack order, so the last one wins (JSRL after EDGE). Returns step and
    the masks of where each handed the expert the step (expert_mask), in
    the order they were written."""
    context = ExtensionContext(step=acting.timestep, rng=rng, total_steps=total_steps)
    masks = []
    for index, extension in indexed_extensions:
        proposed = extension.action(
            agent_state, agent_state.ext_state[index], step, rng, context
        )
        if proposed is not None:
            step[slot] = proposed
        if "expert_mask" in step:
            masks.append(step.pop("expert_mask"))
    return step, masks


def as_env_column(mask: jax.Array, n_envs: int) -> jax.Array:
    """mask, one entry per env or one for every env (EDGE's fixed gate draws
    a single one), as the (n_envs, 1) float column the replay's is_expert
    holds."""
    mask = mask.astype(jnp.float32)
    if mask.ndim == 0:
        return jnp.broadcast_to(mask, (n_envs, 1))
    return mask.reshape(n_envs, 1)


def apply_action_extensions(
    pipeline: Callable, extensions: Any, n_envs: int, total_timesteps: int
) -> Callable:
    """pipeline with its stack's action extensions applied to its result:
    pipeline itself when none acts in a slot.

    The pre-warm-up slot proposes from the pipeline's executed action,
    which the step executes outside the warm-up; the post-warm-up slot then
    overrides the step's action, warm-up or not. The replay flags the steps
    the expert took: the pipeline's flags, and outside the warm-up any
    slot's mask too. A pipeline storing an action of its own (a discrete
    index's column, DAgger's choice) stores the substituted one, in its
    layout."""
    pre_warmup = slotted_extensions(extensions, "pre_warmup")
    post_warmup = slotted_extensions(extensions, "post_warmup")
    if not pre_warmup and not post_warmup:
        return pipeline

    def staged(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        result = pipeline(
            agent_state, acting, raw_obs, rng, uniform, mix_key, action_key
        )
        env_action = result.env_action
        # The expert's action in the env action's layout (a discrete index's
        # (n_envs,) vector, not the expert's column).
        expert_action = acting.expert_action.reshape(env_action.shape).astype(
            env_action.dtype
        )
        gate_rng = rng if result.rng is None else result.rng
        masks = []
        if pre_warmup:
            step = {
                "policy_action": result.policy_action,
                "expert_action": expert_action,
                "post_warmup_action": env_action,
                # The critics' input, its features included.
                "observations": acting.obs,
                "raw_obs": raw_obs,
                "gate_rng": gate_rng,
            }
            step, masks = dispatch_action_phase(
                agent_state,
                acting,
                pre_warmup,
                step,
                "post_warmup_action",
                gate_rng,
                total_timesteps,
            )
            proposed, gate_rng = step["post_warmup_action"], step["gate_rng"]
            env_action = jax.lax.cond(
                uniform, lambda: result.env_action, lambda: proposed
            )
        box = {}
        if post_warmup:
            step = {
                "expert_action": expert_action,
                "env_action": env_action,
                "observations": acting.obs,
            }
            box, overriding = dispatch_action_phase(
                agent_state,
                acting,
                post_warmup,
                step,
                "env_action",
                gate_rng,
                total_timesteps,
            )
            env_action = box["env_action"]
            masks.extend(overriding)
        # A pipeline without warm-up flags (explore_pipeline without a
        # warm-up chooser) flags no step.
        flags = result.is_expert_flag
        if flags is None:
            flags = jnp.zeros((n_envs, 1), jnp.float32)
        substituted = flags
        for mask in masks:
            substituted = jnp.maximum(substituted, as_env_column(mask, n_envs))
        stored = result.buffer_action
        if stored is not None:
            stored = env_action.reshape(stored.shape).astype(stored.dtype)
        return result._replace(
            env_action=env_action,
            is_expert_flag=jax.lax.cond(uniform, lambda: flags, lambda: substituted),
            rng=gate_rng,
            in_value_box=box.get("in_value_box", result.in_value_box),
            entry_bonus=box.get("entry_bonus", result.entry_bonus),
            box_entry=box.get("box_entry", result.box_entry),
            buffer_action=stored,
        )

    return staged
