"""SAC research-feature hook builders (extracted from train_SAC.py).

These ``make_*`` functions are the *implementations* of SAC's 14 research
features — EDGE exploration, the value box, residual RL, the critic
target modifiers (IBRL / LCB-gated bootstrap / critic-blend /
MC-correction), online BC, obs augmentation, φ* refresh, the eval-time
action transforms. They are moved here verbatim from the pre-refactor
``train_SAC.py`` so the feature code lives outside the agent file while
remaining byte-identical (behaviour-preserving).

:mod:`ajax.agents.SAC.sac` reads an :class:`~ajax.extensions.base.\
ExtensionStack` and calls these builders to assemble the composable hook
callables the proven training functions consume. The thin
:class:`~ajax.extensions.base.Extension` config classes live in
``ajax/extensions/{expert,target_mods,exploration,pretrain}.py``; this
module is their shared private machinery.
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_action_and_log_probs
from ajax.modules.expert import (
    detach_obs_expert_dims,
)
from ajax.modules.exploration import (
    box_action_override,
    box_compute_state,
    box_compute_threshold,
    edge_argmax_gate,
    edge_boltzmann_gate,
    edge_compute_asym_scores,
    edge_compute_decay,
    edge_compute_lcb_scores,
    edge_compute_thompson_stats,
    edge_compute_value_gap,
    edge_fixed_gate,
    edge_lcb_argmax_gate,
    edge_lcb_gate,
    edge_thompson_gate,
)

# ---------------------------------------------------------------------------
# Action pipeline — composable exploration for collect_experience
# ---------------------------------------------------------------------------


class ActionPipelineResult(NamedTuple):
    """Result from the SAC action pipeline used by collect_experience."""

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
    # Live diagnostic stats from the LCB gate (per-step batch means). Used
    # for tensorboard telemetry; nan when the pipeline doesn't compute them
    # (no expert / non-LCB gate / warmup).
    q_advantage: Optional[jax.Array] = None  # mean(mu_actor - mu_expert)
    critic_sigma_actor: Optional[jax.Array] = None  # mean(sigma_actor)
    critic_sigma_expert: Optional[jax.Array] = None  # mean(sigma_expert)
    p_expert_max: Optional[jax.Array] = None  # max(p_t) of LCB softmax gate
    # Expert action computed at this step with the correct (stateful) expert
    # internal state. Stored in the Transition so the residual-RL actor
    # loss can read it back instead of recomputing the expert with a fresh
    # zero state on a buffer-sampled obs (which silently drops the
    # integrator state for stateful PIDs).
    a_expert: Optional[jax.Array] = None


def _apply_lcb_gate(
    score_e,
    score_p,
    rng,
    lcb_temperature,
    argmax,
):
    """Pick LCB gate: argmax (deterministic) or softmax (default)."""
    if argmax:
        return edge_lcb_argmax_gate(score_e, score_p, rng)
    return edge_lcb_gate(score_e, score_p, rng, lcb_temperature)


def make_action_pipeline(
    expert_policy,
    recurrent,
    env_args,
    # Box
    use_box=False,
    box_v_min=0.0,
    box_v_max=0.0,
    # EDGE
    use_expert_guided_exploration=False,
    exploration_decay_frac=0.30,
    exploration_tau=1.0,
    exploration_boltzmann=False,
    exploration_argmax=False,
    fixed_exploration_prob=0.5,
    # Quality-aware (LCB) gate — alternative to argmax/boltzmann/fixed
    exploration_lcb=False,
    # Variant of the LCB path that swaps the softmax gate for a
    # deterministic argmax (score_e > score_p) while keeping LCB
    # scoring. Populates the (argmax, LCB) corner of the 2x2 ablation.
    exploration_argmax_lcb=False,
    exploration_thompson=False,
    lcb_beta_init=1.0,
    lcb_beta_decay_k=2.0,
    lcb_temperature=1.0,
    epsilon_floor=0.0,
    # When True, the LCB gate uses asymmetric pessimism: LCB on the
    # expert arm (conservative about following an unreliable expert),
    # UCB on the policy arm (give the policy credit for its uncertainty
    # so the gate hands control whenever the critic is unsure about
    # a policy action). Fixes the symmetric LCB's tendency to over-
    # penalize the policy in its own exploration regions. Set by the
    # ``r_edge_bow`` method.
    lcb_asymmetric=False,
    # Action transforms
    use_residual_rl=False,
    residual_scale=1.0,
    use_pid_policy=False,
    # True JSRL: per-episode handoff at step_in_episode < H_t, with H_t
    # decreasing over training (Uchendu et al. 2023). episode_length and
    # decay_frac control the curriculum schedule.
    jsrl_curriculum=False,
    jsrl_episode_length=1000,
    jsrl_decay_frac=0.5,
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
    handles obs augmentation, action selection (EDGE, box, residual, PID),
    and warmup expert/uniform mixing.

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

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        collector_state = agent_state.collector_state

        # Live LCB / Thompson telemetry. Populated by the gating branches
        # that compute critic ensemble stats; stays NaN otherwise (so
        # vanilla SAC, edge-eps, ibrl, jsrl, residual all log NaN here,
        # which downstream filtering drops cleanly).
        _diag_q_advantage = jnp.nan
        _diag_sigma_actor = jnp.nan
        _diag_sigma_expert = jnp.nan
        _diag_p_expert_max = jnp.nan

        # --- Gain-policy short-circuit ---
        if gain_policy_mode:
            expert_zero_batched = expert_policy.init_state(env_args.n_envs)
            last_done = collector_state.last_terminated.astype(
                jnp.bool_
            ) | collector_state.last_truncated.astype(jnp.bool_)
            current_expert_state = (
                collector_state.expert_state
                if collector_state.expert_state is not None
                else expert_zero_batched
            )
            expert_state_in = jax.tree.map(
                lambda cur, zero: jnp.where(
                    last_done.reshape(
                        last_done.shape + (1,) * (cur.ndim - last_done.ndim)
                    ),
                    zero,
                    cur,
                ),
                current_expert_state,
                expert_zero_batched,
            )
            _raw_for_expert = (
                raw_obs if raw_obs is not None else collector_state.last_obs
            )
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
            gains = _anchor_gains * jnp.exp(_gain_log_scale * gain_action)
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

        # --- Stateful expert call (PID integrator / derivative carry) ---
        # Reset the expert's internal state at the start of a new episode
        # (detected via the *previous* step's done flag, since autoreset has
        # already produced the fresh first obs by now).
        if expert_is_stateful:
            expert_zero_batched = expert_policy.init_state(env_args.n_envs)
            last_done = collector_state.last_terminated.astype(
                jnp.bool_
            ) | collector_state.last_truncated.astype(jnp.bool_)
            current_expert_state = (
                collector_state.expert_state
                if collector_state.expert_state is not None
                else expert_zero_batched
            )
            expert_state_in = jax.tree.map(
                lambda cur, zero: jnp.where(
                    last_done.reshape(
                        last_done.shape + (1,) * (cur.ndim - last_done.ndim)
                    ),
                    zero,
                    cur,
                ),
                current_expert_state,
                expert_zero_batched,
            )
            _raw_for_expert = (
                raw_obs if raw_obs is not None else collector_state.last_obs
            )
            expert_action, new_expert_state = expert_policy(
                expert_state_in, _raw_for_expert
            )
            expert_action = jax.lax.stop_gradient(expert_action)
            new_expert_state = jax.lax.stop_gradient(new_expert_state)
        else:
            _raw_for_expert = (
                raw_obs if raw_obs is not None else collector_state.last_obs
            )
            expert_action = jax.lax.stop_gradient(expert_policy(_raw_for_expert))
            new_expert_state = None

        # --- Box state computation ---
        if use_box:
            train_frac = agent_state.collector_state.timestep / total_timesteps
            threshold = box_compute_threshold(box_v_min, box_v_max, train_frac)
            obs_for_box = agent_state.collector_state.last_obs
            raw_for_box = raw_obs if raw_obs is not None else obs_for_box[..., :-1]
            in_value_box, entry_bonus, _ = box_compute_state(
                obs_for_box,
                raw_for_box,
                expert_policy,
                agent_state.critic_state,
                agent_state.expert_critic_params,
                threshold,
                agent_state.collector_state.last_in_box,
            )
        else:
            in_value_box = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
            entry_bonus = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)

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

        # --- Post-warmup action ---
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

        # --- EDGE (Expert Decayed Guided Exploration) ---
        if use_expert_guided_exploration:
            # When augment_obs_with_expert_action is on, the critic was
            # initialised with augmented input (env_obs + expert_action),
            # so the EDGE gating must feed it the same shape. Use the
            # already-built _augmented_obs from the policy block above.
            obs_for_edge = (
                _augmented_obs
                if augment_obs_with_expert_action
                else agent_state.collector_state.last_obs
            )
            edge_critic_params = (
                agent_state.expert_critic_params
                if agent_state.expert_critic_params is not None
                else agent_state.critic_state.params
            )
            decay = edge_compute_decay(
                agent_state.collector_state.timestep,
                total_timesteps,
                exploration_decay_frac,
            )
            if exploration_thompson:
                mu_e, sigma_e, mu_p, sigma_p, q_policy = edge_compute_thompson_stats(
                    obs_for_edge,
                    action,
                    expert_action,
                    agent_state.critic_state,
                    edge_critic_params,
                )
                use_expert_edge, rng = edge_thompson_gate(
                    mu_e,
                    sigma_e,
                    mu_p,
                    sigma_p,
                    rng,
                    lcb_temperature,
                    epsilon_floor=epsilon_floor,
                )
                gap = mu_e - mu_p
                _diag_q_advantage = jnp.mean(mu_p - mu_e)
                _diag_sigma_actor = jnp.mean(sigma_p)
                _diag_sigma_expert = jnp.mean(sigma_e)
            elif exploration_lcb:
                # Quality-aware: LCB scores + Boltzmann gate. Anneals beta
                # toward 0 over training so the rule reduces to argmax-Q
                # (IBRL-like) once the critic is well-calibrated.
                progress = jnp.clip(
                    agent_state.collector_state.timestep
                    / jnp.maximum(total_timesteps, 1),
                    0.0,
                    1.0,
                )
                beta_eff = lcb_beta_init * jnp.power(1.0 - progress, lcb_beta_decay_k)
                # Asymmetric variant (r_edge_bow): LCB on expert arm,
                # UCB on policy arm. Inverts the over-conservative
                # penalty of symmetric LCB at policy-exploration states.
                _scores_fn = (
                    edge_compute_asym_scores
                    if lcb_asymmetric
                    else edge_compute_lcb_scores
                )
                (
                    score_e,
                    score_p,
                    q_policy,
                    _mu_p,
                    _mu_e,
                    _sigma_p,
                    _sigma_e,
                ) = _scores_fn(
                    obs_for_edge,
                    action,
                    expert_action,
                    agent_state.critic_state,
                    edge_critic_params,
                    beta_eff,
                )
                use_expert_edge, rng = _apply_lcb_gate(
                    score_e,
                    score_p,
                    rng,
                    lcb_temperature,
                    exploration_argmax_lcb,
                )
                # gap kept for diagnostic logging compat
                gap = score_e - score_p
                # Live LCB telemetry (mean over batch).
                _diag_q_advantage = jnp.mean(_mu_p - _mu_e)
                _diag_sigma_actor = jnp.mean(_sigma_p)
                _diag_sigma_expert = jnp.mean(_sigma_e)
                # Coverage Lemma diagnostic: max(p_expert) over the
                # collection batch. With softmax gate, this is
                # max sigmoid((score_e - score_p) / tau); with the
                # argmax_lcb variant the empirical max is in {0, 1}.
                _diag_p_expert_max = jnp.max(
                    jax.nn.sigmoid(
                        (score_e - score_p) / jnp.maximum(lcb_temperature, 1e-6)
                    )
                )
            else:
                gap, q_policy = edge_compute_value_gap(
                    obs_for_edge,
                    action,
                    expert_action,
                    agent_state.critic_state,
                    edge_critic_params,
                )
                if exploration_argmax:
                    use_expert_edge, rng = edge_argmax_gate(gap, decay, rng)
                elif exploration_boltzmann:
                    use_expert_edge, rng = edge_boltzmann_gate(
                        gap,
                        decay,
                        rng,
                        q_policy,
                        exploration_tau,
                    )
                else:
                    use_expert_edge, rng = edge_fixed_gate(
                        gap,
                        decay,
                        rng,
                        fixed_exploration_prob,
                    )
            post_warmup_action = jnp.where(
                use_expert_edge, expert_action, post_warmup_action
            )

        # --- True JSRL curriculum: per-episode handoff at H_t ---
        # H_t decreases linearly from jsrl_episode_length to 0 over the
        # first jsrl_decay_frac fraction of training. While
        # step_in_episode < H_t, the expert acts; otherwise the learner
        # acts. Both arms' transitions go to the buffer (unified).
        if jsrl_curriculum:
            _global_t = agent_state.collector_state.timestep
            _train_frac = _global_t / jnp.maximum(total_timesteps, 1)
            _curriculum_progress = jnp.clip(_train_frac / jsrl_decay_frac, 0.0, 1.0)
            _H_t = jsrl_episode_length * (1.0 - _curriculum_progress)
            _step_in_ep = agent_state.collector_state.step_in_episode
            # Shape (n_envs,); broadcast into the action mask.
            _use_expert_jsrl = _step_in_ep.astype(jnp.float32) < _H_t
            _use_expert_jsrl = _use_expert_jsrl.reshape(
                (-1,) + (1,) * (post_warmup_action.ndim - 1)
            )
            post_warmup_action = jnp.where(
                _use_expert_jsrl, expert_action, post_warmup_action
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

        # --- Expert flag tracking ---
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        if use_expert_guided_exploration:
            _post_expert = jnp.maximum(
                _post_expert, use_expert_edge.astype(jnp.float32)
            )
        if use_box:
            _post_expert = jnp.maximum(_post_expert, in_value_box.astype(jnp.float32))
        _warmup_expert = jnp.ones_like(
            action[..., :1], dtype=jnp.float32
        ) * use_expert_this_step.astype(jnp.float32)
        is_expert_flag = jax.lax.cond(
            uniform, lambda: _warmup_expert, lambda: _post_expert
        )

        # --- Final action selection ---
        env_action = jax.lax.cond(
            uniform, lambda: warmup_action, lambda: post_warmup_action
        )

        # --- Box action override ---
        if use_box:
            env_action = box_action_override(env_action, expert_action, in_value_box)

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
            q_advantage=_diag_q_advantage,
            critic_sigma_actor=_diag_sigma_actor,
            critic_sigma_expert=_diag_sigma_expert,
            p_expert_max=_diag_p_expert_max,
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


# ---------------------------------------------------------------------------
# Policy modifiers — composable obs preprocessing, action transform, BC loss
# ---------------------------------------------------------------------------


def make_policy_obs_preprocessor(
    augment_obs_with_expert_action, detach_obs_aug_action, action_dim
):
    """Compose obs preprocessing for policy: stop-gradient expert-action dims.

    Returns None when not needed (no augmentation or no detach).
    """
    if not (augment_obs_with_expert_action and detach_obs_aug_action):
        return None

    def preprocess(observations):
        return detach_obs_expert_dims(observations, action_dim)

    return preprocess


# ---------------------------------------------------------------------------
# Eval action transform — composable residual RL / PID for evaluate.py
# ---------------------------------------------------------------------------


def make_eval_action_transform(
    use_residual_rl=False, use_pid_policy=False, residual_scale: float = 1.0
):
    """Compose eval-time action transform for residual RL.

    Returns None for vanilla SAC (default box-based handover in evaluate.py).
    Gain-mode pid_policy is handled directly in evaluate.step_environment via
    the pid_gain_policy flag (needs expert_state access), not through here.
    """
    if use_pid_policy:
        return None
    if use_residual_rl:

        def transform(raw_actions, expert_actions, obs, agent_state):
            return jnp.clip(expert_actions + residual_scale * raw_actions, -1.0, 1.0)

        return transform
    return None
