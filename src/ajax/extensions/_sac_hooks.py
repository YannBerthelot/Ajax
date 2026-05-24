"""SAC action pipeline + ``next_expert_fn`` helper.

After Phase 2b every SAC research feature whose math could be moved
onto a real :class:`~ajax.extensions.base.Extension` phase method
(``on_target`` / ``actor_loss`` / ``action`` / ``eval_action`` /
``post_update`` / ``pretrain`` / ``on_obs``) has done so. What remains
in this module is the SAC-specific *action pipeline* glue —
collection-time bookkeeping (``is_expert_flag`` / ``buffer_action`` /
expert-state threading), the gain-policy short-circuit, construction-
time obs augmentation, warmup expert/uniform mixing — plus the small
helper that builds ``next_expert_fn``. The pipeline dispatches the
collection-time substitution extensions (EDGEExploration /
JSRLCurriculum / ValueBox) via their :meth:`Extension.action` method.

:mod:`ajax.agents.SAC.sac` reads an :class:`~ajax.extensions.base.\
ExtensionStack` and calls these helpers to assemble the action
pipeline the proven training functions consume. The thin
:class:`~ajax.extensions.base.Extension` config classes live in
``ajax/extensions/{expert,target_mods,exploration,pretrain}.py``; this
module is their shared private machinery.
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_action_and_log_probs
from ajax.extensions.base import ExtensionContext, ExtensionStack

# ---------------------------------------------------------------------------
# Action pipeline — composable exploration for collect_experience
# ---------------------------------------------------------------------------


class ActionPipelineResult(NamedTuple):
    """Result from the SAC action pipeline used by collect_experience.

    The diagnostic LCB telemetry fields (``q_advantage``,
    ``critic_sigma_actor``, ``critic_sigma_expert``, ``p_expert_max``)
    that the pre-refactor pipeline returned have been dropped: they
    were pure observability (not exercised by any parity / equivalence
    test) and will be re-added cleanly via the ``eval_metrics`` phase.
    """

    env_action: jax.Array  # action sent to env
    policy_action: jax.Array  # actor's original action (stored in transition)
    log_probs: jax.Array  # actor's log probs
    is_expert_flag: jax.Array  # expert tracking for buffer
    in_value_box: jax.Array  # box membership (zeros when no box)
    entry_bonus: jax.Array  # box entry bonus (zeros when no box)
    rng: jax.Array  # updated rng (EDGE may consume it)
    new_expert_state: Optional[Any] = None  # updated PID/expert state after this step
    # Action to write into the replay buffer. None => fall back to env_action.
    # In gain-policy mode this is the raw actor output (gains) since Q operates
    # in gain-space while env_action is the PID-derived control.
    buffer_action: Optional[jax.Array] = None
    # Expert action computed at this step with the correct (stateful) expert
    # internal state. Stored in the Transition so the residual-RL actor
    # loss can read it back instead of recomputing the expert with a fresh
    # zero state on a buffer-sampled obs (which silently drops the
    # integrator state for stateful PIDs).
    a_expert: Optional[jax.Array] = None


# Names of the three collection-time action-substitution extensions
# whose math has been migrated onto :meth:`Extension.action`. The
# pipeline below routes by ``name`` rather than ``isinstance`` so the
# resolver stays free of cross-module circular imports.
_PRE_WARMUP_OVERRIDE_NAMES = frozenset({"edge_exploration", "jsrl_curriculum"})
_POST_WARMUP_OVERRIDE_NAMES = frozenset({"value_box"})


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
    # Actor samples gain-space action (shape: (n_envs, n_gains))
    action, log_probs = get_action_and_log_probs(
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


def _apply_pre_warmup_overrides(
    *,
    agent_state,
    extensions_indexed,
    action,
    expert_action,
    post_warmup_action,
    augmented_obs,
    augment_obs_with_expert_action,
    raw_obs,
    rng,
    total_timesteps,
):
    """Dispatch :class:`EDGEExploration` / :class:`JSRLCurriculum`.

    They consume / produce a Python dict threaded through the ``obs``
    argument of :meth:`Extension.action` — duck-typed; mirrors the
    ``batch``-dict pattern used by ``on_target`` / ``actor_loss``. Each
    extension writes its decision back into the dict; iteration order
    matches the legacy pipeline (EDGE first, then JSRL, so JSRL wins
    when both are stacked).
    """
    edge_use_expert = jnp.zeros_like(action[..., :1], dtype=jnp.bool_)
    if not extensions_indexed:
        return post_warmup_action, rng, edge_use_expert

    edge_critic_params = (
        agent_state.expert_critic_params
        if agent_state.expert_critic_params is not None
        else agent_state.critic_state.params
    )
    obs_for_edge = (
        augmented_obs
        if augment_obs_with_expert_action
        else agent_state.collector_state.last_obs
    )
    ext_batch: dict = {
        "policy_action": action,
        "expert_action": expert_action,
        "post_warmup_action": post_warmup_action,
        "obs_for_edge": obs_for_edge,
        "edge_critic_params": edge_critic_params,
        "critic_state": agent_state.critic_state,
        "raw_obs": raw_obs,
        "gate_rng": rng,
    }
    ctx = ExtensionContext(
        step=agent_state.collector_state.timestep,
        rng=rng,
        total_steps=total_timesteps,
    )
    ext_state = agent_state.ext_state
    for idx, ext in extensions_indexed:
        proposed = ext.action(agent_state, ext_state[idx], ext_batch, rng, ctx)
        if proposed is not None:
            post_warmup_action = proposed
            ext_batch["post_warmup_action"] = post_warmup_action
    # Thread the gate's updated rng back to the pipeline + pull the
    # EDGE substitution mask for is_expert_flag bookkeeping.
    rng = ext_batch.get("gate_rng", rng)
    if "_edge_use_expert" in ext_batch:
        edge_use_expert = ext_batch["_edge_use_expert"]
    return post_warmup_action, rng, edge_use_expert


def _apply_post_warmup_overrides(
    *,
    agent_state,
    extensions_indexed,
    env_action,
    expert_action,
    raw_obs,
    rng,
    box_v_min,
    box_v_max,
    total_timesteps,
    n_envs,
):
    """Dispatch :class:`~ajax.extensions.target_mods.ValueBox`.

    Runs after the warmup vs post-warmup ``jax.lax.cond`` so it can
    rewrite the executed action regardless of which branch produced it.
    Writes back ``in_value_box`` / ``entry_bonus`` so the SAC pipeline
    can record them on the transition (buffer-write suppression, reward
    shaping).
    """
    in_value_box = jnp.zeros((n_envs, 1), dtype=jnp.float32)
    entry_bonus = jnp.zeros((n_envs, 1), dtype=jnp.float32)
    if not extensions_indexed:
        return env_action, in_value_box, entry_bonus

    ctx = ExtensionContext(
        step=agent_state.collector_state.timestep,
        rng=rng,
        total_steps=total_timesteps,
    )
    box_batch: dict = {
        "expert_action": expert_action,
        "env_action": env_action,
        "raw_obs": raw_obs,
        "box_v_min": box_v_min,
        "box_v_max": box_v_max,
        "total_timesteps": total_timesteps,
    }
    ext_state = agent_state.ext_state
    for idx, ext in extensions_indexed:
        proposed = ext.action(agent_state, ext_state[idx], box_batch, rng, ctx)
        if proposed is not None:
            env_action = proposed
            box_batch["env_action"] = env_action
    if "in_value_box" in box_batch:
        in_value_box = box_batch["in_value_box"]
    if "entry_bonus" in box_batch:
        entry_bonus = box_batch["entry_bonus"]
    return env_action, in_value_box, entry_bonus


def make_action_pipeline(
    expert_policy,
    recurrent,
    env_args,
    extension_stack: Optional[ExtensionStack] = None,
    # Box bounds: only consumed when a :class:`ValueBox` is present in
    # the extension stack. The SAC factory passes the MC-pretrain
    # ``expert_v_min`` / ``expert_v_max`` (or 0.0 on resume / no MC).
    box_v_min=0.0,
    box_v_max=0.0,
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

    # Pre-classify the extensions in the stack into the two
    # collection-time slots (pre-warmup vs post-warmup). Both happen
    # at trace time so the compiled graph only contains the active
    # branches; the lists are Python-level and never carried into
    # ``pipeline``.
    extensions = (
        tuple(extension_stack.extensions) if extension_stack is not None else ()
    )
    _pre_warmup_exts = tuple(
        (i, e) for i, e in enumerate(extensions) if e.name in _PRE_WARMUP_OVERRIDE_NAMES
    )
    _post_warmup_exts = tuple(
        (i, e)
        for i, e in enumerate(extensions)
        if e.name in _POST_WARMUP_OVERRIDE_NAMES
    )
    _has_edge = any(e.name == "edge_exploration" for _, e in _pre_warmup_exts)
    _has_value_box = bool(_post_warmup_exts)

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

        # --- Policy action ---
        action, log_probs = get_action_and_log_probs(
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
        # ``trunc_condition`` is an env-defined safe-region indicator,
        # unrelated to the value-box. The residual / non-residual branch
        # here is independent of the Extension stack.
        in_box = (
            env_args.env.trunc_condition(
                agent_state.collector_state.env_state, env_args.env_params
            )
            if "trunc_condition" in dir(env_args.env)
            else jnp.zeros_like(action[..., :1])
        )
        if use_residual_rl:
            post_warmup_action = jnp.clip(
                expert_action + residual_scale * action, -1.0, 1.0
            )
        else:
            post_warmup_action = (1 - in_box) * action + in_box * expert_action

        # --- Collection-time override extensions
        # (:class:`EDGEExploration`, :class:`JSRLCurriculum`) ---
        post_warmup_action, rng, _edge_use_expert = _apply_pre_warmup_overrides(
            agent_state=agent_state,
            extensions_indexed=_pre_warmup_exts,
            action=action,
            expert_action=expert_action,
            post_warmup_action=post_warmup_action,
            augmented_obs=_augmented_obs,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            raw_obs=raw_obs,
            rng=rng,
            total_timesteps=total_timesteps,
        )

        # --- Warmup action ---
        if use_residual_rl:
            warmup_action = uniform_action
            use_expert_this_step = jnp.zeros((), dtype=jnp.bool_)
        else:
            use_expert_this_step = jax.random.uniform(mix_key) < expert_fraction
            warmup_action = jnp.where(
                use_expert_this_step, expert_action, uniform_action
            )

        # --- Final action selection (warmup vs post-warmup) ---
        env_action = jax.lax.cond(
            uniform, lambda: warmup_action, lambda: post_warmup_action
        )

        # --- ValueBox post-warmup override ---
        # ValueBox runs AFTER the warmup vs post-warmup ``jax.lax.cond``
        # so it can rewrite the executed action regardless of which
        # branch produced it. It writes back ``in_value_box`` /
        # ``entry_bonus`` so the SAC pipeline can record them on the
        # transition (buffer-write suppression, reward shaping).
        env_action, in_value_box, entry_bonus = _apply_post_warmup_overrides(
            agent_state=agent_state,
            extensions_indexed=_post_warmup_exts,
            env_action=env_action,
            expert_action=expert_action,
            raw_obs=raw_obs,
            rng=rng,
            box_v_min=box_v_min,
            box_v_max=box_v_max,
            total_timesteps=total_timesteps,
            n_envs=env_args.n_envs,
        )

        # --- Expert flag tracking ---
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        if _has_edge:
            _post_expert = jnp.maximum(
                _post_expert, _edge_use_expert.astype(jnp.float32)
            )
        if _has_value_box:
            _post_expert = jnp.maximum(_post_expert, in_value_box.astype(jnp.float32))
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
