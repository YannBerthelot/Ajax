"""SAC's action pipeline (collection-time action selection) and its
``next_expert_fn``.

The pipeline owns SAC's collection bookkeeping (``is_expert_flag``,
``buffer_action``, expert-state threading), the gain-policy
short-circuit, the expert-action obs augmentation and the warmup
expert/uniform mix. Extensions that declare an ``action_slot`` substitute
the action through their :meth:`~ajax.extensions.base.Extension.action`
phase: ``"pre_warmup"`` ones (EDGEExploration, JSRLCurriculum) before the
warmup choice, ``"post_warmup"`` ones (ValueBox) after it. They read and
write a dict of the step's quantities passed as ``obs``.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from ajax.environments.interaction import (
    ActionPipelineResult,
    get_action_and_log_probs,
)
from ajax.extensions.base import ExtensionContext, ExtensionStack


def _resolve_expert_state(expert_policy, collector_state, n_envs):
    """Return the stateful expert's per-step input state.

    Resets entries whose previous step was a terminal/truncation to the
    zero-initialised state — match the legacy pre-refactor reset
    semantics (autoreset has already advanced obs at this point).
    """
    expert_zero_batched = expert_policy.init_state(n_envs)
    last_done = collector_state.last_terminated.astype(
        jnp.bool_
    ) | collector_state.last_truncated.astype(jnp.bool_)
    current_expert_state = (
        collector_state.expert_state
        if collector_state.expert_state is not None
        else expert_zero_batched
    )
    return jax.tree.map(
        lambda cur, zero: jnp.where(
            last_done.reshape(last_done.shape + (1,) * (cur.ndim - last_done.ndim)),
            zero,
            cur,
        ),
        current_expert_state,
        expert_zero_batched,
    )


def _gain_policy_step(
    *,
    agent_state,
    expert_policy,
    env_args,
    recurrent,
    raw_obs,
    rng,
    uniform,
    mix_key,
    action_key,
    anchor_gains,
    gain_log_scale,
):
    """Gain-policy short-circuit branch of the action pipeline.

    The actor predicts a gain-space action (per-gain log-scale); we
    convert it back to a physical control via the expert's
    ``step_with_gains`` hook. This branch bypasses every collection-time
    Extension override (no EDGE / box / JSRL semantics here); it is the
    only path that returns early from ``pipeline``.
    """
    collector_state = agent_state.collector_state
    expert_state_in = _resolve_expert_state(
        expert_policy, collector_state, env_args.n_envs
    )
    _raw_for_expert = raw_obs if raw_obs is not None else collector_state.last_obs
    action, log_probs, _raw_action, _agent_state = get_action_and_log_probs(
        action_key=action_key,
        agent_state=agent_state,
        recurrent=recurrent,
        uniform=False,
    )
    uniform_action = jax.random.uniform(
        mix_key, minval=-1.0, maxval=1.0, shape=action.shape
    )
    gain_action = jax.lax.cond(uniform, lambda: uniform_action, lambda: action)
    gains = anchor_gains * jnp.exp(gain_log_scale * gain_action)
    env_action, new_expert_state = expert_policy.step_with_gains(
        expert_state_in, _raw_for_expert, gains
    )
    is_expert_flag = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
    return ActionPipelineResult(
        env_action=env_action,
        policy_action=action,
        log_probs=log_probs,
        is_expert_flag=is_expert_flag,
        in_value_box=jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32),
        entry_bonus=jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32),
        rng=rng,
        new_expert_state=new_expert_state,
        buffer_action=gain_action,
    )


def _dispatch_actions(agent_state, extensions_indexed, batch, slot, rng, total_steps):
    """Fold the ``(index, extension)`` pairs' ``action`` phase over
    ``batch``, which they read and write: each non-None proposal replaces
    ``batch[slot]``. Stack order, so the last proposal wins (JSRL after
    EDGE)."""
    ctx = ExtensionContext(
        step=agent_state.collector_state.timestep, rng=rng, total_steps=total_steps
    )
    for idx, ext in extensions_indexed:
        proposed = ext.action(agent_state, agent_state.ext_state[idx], batch, rng, ctx)
        if proposed is not None:
            batch[slot] = proposed
    return batch


def make_action_pipeline(
    expert_policy,
    recurrent,
    env_args,
    extension_stack: Optional[ExtensionStack] = None,
    # Action transforms
    use_residual_rl=False,
    residual_scale=1.0,
    use_pid_policy=False,
    # Obs augmentation
    augment_obs_with_expert_action=False,
    # Off-policy correctness ablation: when True, the pipeline returns
    # the policy's sampled action as buffer_action (instead of letting
    # collect_experience fall back to env_action = the executed action).
    store_policy_action=False,
    # Context
    total_timesteps=1,
    expert_fraction=0.7,
):
    """Compose the SAC action pipeline for collect_experience.

    Returns None for vanilla SAC (no expert). When provided, the pipeline
    handles obs augmentation, action selection, and warmup expert/uniform
    mixing; the collection-time substitution features (EDGE / ValueBox /
    JSRL) own their gate math on their :class:`Extension` ``action``
    phase method — this pipeline dispatches each in the canonical
    legacy ordering (EDGE → JSRL → warmup cond → ValueBox) and only
    retains SAC-side bookkeeping (``is_expert_flag``, ``buffer_action``,
    expert-state threading, ``in_value_box`` / ``entry_bonus``
    recording for buffer-write suppression).

    All boolean flags are resolved at Python level (trace time), so the
    compiled graph only contains the active branches.
    """
    if expert_policy is None:
        return None

    # Stateful experts (FunctionalExpertPolicy) expose init_state; stateless
    # callables (e.g. make_noise_expert_policy) don't. Resolve at Python level
    # so the traced graph only contains the active branch.
    expert_is_stateful = hasattr(expert_policy, "init_state")

    # Gain-policy mode: actor output is interpreted as PID gain modulation.
    # Pre-compute static anchor gains + scaling once; reuse per step.
    gain_policy_mode = use_pid_policy and hasattr(expert_policy, "learnable_fields")
    if use_pid_policy and not gain_policy_mode:
        raise ValueError(
            "use_pid_policy=True but expert_policy lacks learnable_fields."
        )
    if gain_policy_mode:
        _anchor_gains = expert_policy.anchor_gains  # (n_gains,)
        _gain_log_scale = jnp.log(10.0)

    # The extensions of each collection-time slot, classified at trace
    # time by their declared ``action_slot``.
    extensions = extension_stack.extensions if extension_stack is not None else ()

    def _slot(name):
        return tuple(
            (i, e)
            for i, e in enumerate(extensions)
            if getattr(e, "action_slot", None) == name
        )

    _pre_warmup_exts, _post_warmup_exts = _slot("pre_warmup"), _slot("post_warmup")

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        collector_state = agent_state.collector_state

        # --- Gain-policy short-circuit ---
        # Gain-policy mode is a single-feature path that bypasses every
        # other collection-time override; it has no Extension
        # representation yet.
        if gain_policy_mode:
            return _gain_policy_step(
                agent_state=agent_state,
                expert_policy=expert_policy,
                env_args=env_args,
                recurrent=recurrent,
                raw_obs=raw_obs,
                rng=rng,
                uniform=uniform,
                mix_key=mix_key,
                action_key=action_key,
                anchor_gains=_anchor_gains,
                gain_log_scale=_gain_log_scale,
            )

        # --- Stateful expert call (PID integrator / derivative carry) ---
        # Reset the expert's internal state at the start of a new episode
        # (detected via the *previous* step's done flag, since autoreset has
        # already produced the fresh first obs by now).
        _raw_for_expert = raw_obs if raw_obs is not None else collector_state.last_obs
        if expert_is_stateful:
            expert_state_in = _resolve_expert_state(
                expert_policy, collector_state, env_args.n_envs
            )
            expert_action, new_expert_state = expert_policy(
                expert_state_in, _raw_for_expert
            )
            expert_action = jax.lax.stop_gradient(expert_action)
            new_expert_state = jax.lax.stop_gradient(new_expert_state)
        else:
            expert_action = jax.lax.stop_gradient(expert_policy(_raw_for_expert))
            new_expert_state = None

        # --- Obs augmentation: [env_obs | a_expert] ---
        if augment_obs_with_expert_action:
            _last_obs = agent_state.collector_state.last_obs
            _augmented_obs = jnp.concatenate([_last_obs, expert_action], axis=-1)
            agent_state_for_actor = agent_state.replace(
                collector_state=agent_state.collector_state.replace(
                    last_obs=_augmented_obs
                )
            )
        else:
            agent_state_for_actor = agent_state
            _augmented_obs = None

        action, log_probs, _raw_action, _agent_state = get_action_and_log_probs(
            action_key=action_key,
            agent_state=agent_state_for_actor,
            recurrent=recurrent,
            uniform=False,
        )

        # --- Uniform action (for warmup) ---
        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=action.shape
        )

        # --- Post-warmup action (default) ---
        if use_residual_rl:
            post_warmup_action = jnp.clip(
                expert_action + residual_scale * action, -1.0, 1.0
            )
        else:
            post_warmup_action = action

        # --- Pre-warmup action extensions (EDGEExploration, JSRLCurriculum).
        # A gate's randomness is threaded through ``gate_rng`` (read and
        # overwritten) so the collector's key stream is the gate's own.
        pre = {}
        if _pre_warmup_exts:
            pre = _dispatch_actions(
                agent_state,
                _pre_warmup_exts,
                {
                    "policy_action": action,
                    "expert_action": expert_action,
                    "post_warmup_action": post_warmup_action,
                    "obs_for_edge": (
                        _augmented_obs
                        if augment_obs_with_expert_action
                        else agent_state.collector_state.last_obs
                    ),
                    "edge_critic_params": (
                        agent_state.expert_critic_params
                        if agent_state.expert_critic_params is not None
                        else agent_state.critic_state.params
                    ),
                    "critic_state": agent_state.critic_state,
                    "raw_obs": raw_obs,
                    "gate_rng": rng,
                },
                "post_warmup_action",
                rng,
                total_timesteps,
            )
            post_warmup_action, rng = pre["post_warmup_action"], pre["gate_rng"]

        # --- Warmup action ---
        if use_residual_rl:
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

        # --- Final action selection (warmup vs post-warmup) ---
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
                _post_warmup_exts,
                {
                    "expert_action": expert_action,
                    "env_action": env_action,
                    "raw_obs": raw_obs,
                    "total_timesteps": total_timesteps,
                },
                "env_action",
                rng,
                total_timesteps,
            )
            env_action = post["env_action"]
        in_value_box = post.get("in_value_box", no_box)
        entry_bonus = post.get("entry_bonus", no_box)

        # --- Expert flag tracking: where an extension handed the expert
        # the step.
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        for mask in (pre.get("_edge_use_expert"), post.get("in_value_box")):
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
            rng=rng,
            new_expert_state=new_expert_state,
            buffer_action=_buffer_action_field,
            a_expert=expert_action,
        )

    return pipeline


def make_next_expert_fn(expert_policy):
    """Build a callable that returns ``a_expert(s_{t+1}, expert_state_{t+1})``.

    The action pipeline already produced the post-step expert state
    while consuming s_t; we feed it back together with raw s_{t+1} to
    get the expert action that *would have been taken at the next
    step*. Stored in the buffer so the residual-RL TD target evaluates
    the bootstrap Q on the same residual-transformed action
    distribution the critic was trained on. Returns None when the
    expert is not provided (so collect_experience writes zeros).
    """
    if expert_policy is None:
        return None
    expert_is_stateful = hasattr(expert_policy, "init_state")

    def next_expert_fn(post_step_expert_state, raw_next_obs):
        if expert_is_stateful and post_step_expert_state is not None:
            a_next, _ = expert_policy(post_step_expert_state, raw_next_obs)
            return a_next
        return expert_policy(raw_next_obs)

    return next_expert_fn
