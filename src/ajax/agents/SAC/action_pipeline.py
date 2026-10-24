"""SAC's action pipeline (collection-time action selection).

The pipeline owns SAC's collection bookkeeping (``is_expert_flag``,
``buffer_action``) and the warmup expert/uniform mix; the shared collector
calls the expert and tracks its state. Extensions that declare an ``action_slot`` substitute
the action through their :meth:`~ajax.extensions.base.Extension.action`
phase: ``"pre_warmup"`` ones (EDGEExploration, JSRLCurriculum) before the
warmup choice, ``"post_warmup"`` ones (ValueBox) after it. They read and
write a dict of the step's quantities passed as ``obs``.
"""

import jax
import jax.numpy as jnp

from ajax.controller import TransformInputs
from ajax.environments.interaction import (
    ActionPipelineResult,
    get_action_and_log_probs,
)
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.networks.networks import phi_star_critic_state

#: The pipeline's slots for an ``action`` extension: before the warmup
#: choice (EDGEExploration, JSRLCurriculum) and after it (ValueBox).
ACTION_SLOTS = ("pre_warmup", "post_warmup")


def check_action_extensions(extension_stack: ExtensionStack, dispatched: bool) -> None:
    """Reject the ``action`` extensions the pipeline never dispatches: one
    without an ``action_slot`` of :data:`ACTION_SLOTS`, or any, or a residual
    policy (an extension implementing ``transform_action``), when the
    pipeline does not dispatch (``dispatched`` False: no expert policy, or
    gain mode, which acts through the default pipeline). SAC calls it at
    construction."""
    acting = [e for e in extension_stack if "action" in e.implemented_phases()]
    unslotted = [
        e.name for e in acting if getattr(e, "action_slot", None) not in ACTION_SLOTS
    ]
    if unslotted:
        raise ValueError(
            f"SAC dispatches the action phase only to extensions declaring an"
            f" action_slot in {ACTION_SLOTS}; {unslotted} declare none."
        )
    residual = [
        e for e in extension_stack if "transform_action" in e.implemented_phases()
    ]
    if (acting or residual) and not dispatched:
        raise ValueError(
            "SAC dispatches the action phase only with an expert_policy and"
            f" without gain mode; {[e.name for e in acting + residual]}"
            " would be ignored."
        )


def _dispatch_actions(
    agent_state, acting, extensions_indexed, batch, slot, rng, total_steps
):
    """Fold the ``(index, extension)`` pairs' ``action`` phase over
    ``batch``, which they read and write: each non-None proposal replaces
    ``batch[slot]``. Stack order, so the last proposal wins (JSRL after
    EDGE)."""
    ctx = ExtensionContext(step=acting.timestep, rng=rng, total_steps=total_steps)
    for idx, ext in extensions_indexed:
        proposed = ext.action(agent_state, agent_state.ext_state[idx], batch, rng, ctx)
        if proposed is not None:
            batch[slot] = proposed
    return batch


def make_action_pipeline(
    expert_policy,
    recurrent,
    env_args,
    extension_stack: ExtensionStack,
    # The run's residual policy (ResidualPolicy), whose ``transform_action``
    # makes the post-warmup action residual.
    residual=None,
    # Off-policy correctness ablation: when True, the pipeline returns
    # the policy's sampled action as buffer_action (instead of letting
    # collect_experience fall back to env_action = the executed action).
    store_policy_action=False,
    total_timesteps=1,
    expert_fraction=0.7,
):
    """Compose the SAC action pipeline for collect_experience.

    Returns None for vanilla SAC (no expert).

    All boolean flags are resolved at Python level (trace time), so the
    compiled graph only contains the active branches.
    """
    if expert_policy is None:
        return None

    # The extensions of each collection-time slot, classified at trace
    # time by their declared ``action_slot``.
    extensions = extension_stack.extensions

    def _slot(name):
        return tuple(
            (i, e)
            for i, e in enumerate(extensions)
            if getattr(e, "action_slot", None) == name
        )

    _pre_warmup_exts, _post_warmup_exts = map(_slot, ACTION_SLOTS)

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        # The expert's action this step, which the collector computed from
        # its tracked state (expert_collect_kwargs).
        expert_action = acting.expert_action

        # The actor reads the observation with its features (an expert's
        # action among them), as collection stored it.
        action, log_probs, _raw_action, _agent_state = get_action_and_log_probs(
            action_key=action_key,
            agent_state=agent_state,
            recurrent=recurrent,
            acting=acting,
        )

        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=action.shape
        )

        if residual is not None:
            inputs = TransformInputs(raw_obs=raw_obs, expert_action=expert_action)
            post_warmup_action = residual.transform_action(action, inputs)
        else:
            post_warmup_action = action

        # --- Pre-warmup action extensions (EDGEExploration, JSRLCurriculum).
        # A gate's randomness is threaded through ``gate_rng`` (read and
        # overwritten) so the collector's key stream is the gate's own.
        pre = {}
        if _pre_warmup_exts:
            pre = _dispatch_actions(
                agent_state,
                acting,
                _pre_warmup_exts,
                {
                    "policy_action": action,
                    "expert_action": expert_action,
                    "post_warmup_action": post_warmup_action,
                    # The critics' input, its features included, and the
                    # critic the gates score with: phi* once it exists.
                    "observations": acting.obs,
                    "critic_params": (
                        agent_state.expert_critic_params
                        if agent_state.expert_critic_params is not None
                        else agent_state.critic_state.params
                    ),
                    # phi* is applied without the agent's statistics.
                    "critic_state": (
                        phi_star_critic_state(agent_state.critic_state)
                        if agent_state.expert_critic_params is not None
                        else agent_state.critic_state
                    ),
                    "raw_obs": raw_obs,
                    "gate_rng": rng,
                },
                "post_warmup_action",
                rng,
                total_timesteps,
            )
            post_warmup_action, rng = pre["post_warmup_action"], pre["gate_rng"]

        if residual is not None:
            warmup_action = uniform_action
            use_expert_this_step = jnp.zeros((), dtype=jnp.bool_)
        else:
            # Its own key: on mix_key the decision would be the uniform
            # action's first draw (env 0, dim 0), and select on it.
            decision_key = jax.random.fold_in(mix_key, 1)
            use_expert_this_step = jax.random.uniform(decision_key) < expert_fraction
            warmup_action = jnp.where(
                use_expert_this_step, expert_action, uniform_action
            )

        env_action = jax.lax.cond(
            uniform, lambda: warmup_action, lambda: post_warmup_action
        )

        # --- Post-warmup action extensions (ValueBox): they rewrite the
        # executed action whichever branch produced it, and record
        # ``in_value_box`` / ``entry_bonus`` for the transition.
        no_box = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
        post = {}
        if _post_warmup_exts:
            post = _dispatch_actions(
                agent_state,
                acting,
                _post_warmup_exts,
                {
                    "expert_action": expert_action,
                    "env_action": env_action,
                    # The actor's and critics' input, its features included.
                    "observations": acting.obs,
                },
                "env_action",
                rng,
                total_timesteps,
            )
            env_action = post["env_action"]
        in_value_box = post.get("in_value_box", no_box)
        entry_bonus = post.get("entry_bonus", no_box)
        box_entry = post.get("box_entry")

        # --- Expert flag tracking: where an extension handed the expert
        # the step. Each substitutes the expert's action where its mask is
        # set, so in either stack order the expert acted where any is.
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        masks = (
            pre.get("_edge_use_expert"),
            pre.get("_jsrl_use_expert"),
            post.get("in_value_box"),
        )
        for mask in masks:
            if mask is not None:
                _post_expert = jnp.maximum(_post_expert, mask.astype(jnp.float32))
        _warmup_expert = jnp.ones_like(
            action[..., :1], dtype=jnp.float32
        ) * use_expert_this_step.astype(jnp.float32)
        is_expert_flag = jax.lax.cond(
            uniform, lambda: _warmup_expert, lambda: _post_expert
        )

        # Off-policy correctness ablation hook: write the policy's would-
        # be action to the buffer when the flag is set. Default behaviour
        # (None) leaves buffer_action falling back to env_action — i.e.,
        # the action that actually generated (s', r). Storing the policy
        # action breaks this contract on steps where the gate selected
        # the expert; used to justify the design choice.
        _buffer_action_field = action if store_policy_action else None
        return ActionPipelineResult(
            env_action=env_action,
            policy_action=action,
            log_probs=log_probs,
            is_expert_flag=is_expert_flag,
            in_value_box=in_value_box,
            entry_bonus=entry_bonus,
            box_entry=box_entry,
            rng=rng,
            buffer_action=_buffer_action_field,
        )

    return pipeline
