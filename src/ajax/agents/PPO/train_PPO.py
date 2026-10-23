"""PPO (Schulman et al., 2017), with brax's minibatch geometry as an option.

Each iteration collects an ``(n_steps, n_envs)`` rollout, then runs
``n_epochs`` passes of a critic and an actor step on every minibatch of it
(the clipped surrogate: :mod:`ajax.agents.PPO.core`). The rollout's
:class:`Geometry` decides how it is minibatched and where GAE is computed:

* ``flat`` (feedforward default): GAE once on the whole rollout, then a flat
  shuffle of its ``n_steps * n_envs`` samples (the paper's Algorithm 1,
  OpenAI Baselines' ppo2);
* ``time`` (with ``unroll_length``): fragments of ``unroll_length`` steps
  shuffled into minibatches, GAE recomputed in every minibatch with the
  current critic (brax's ``compute_ppo_loss``);
* ``recurrent``: GAE once on the whole rollout, then a shuffle of whole
  sequences of ``length`` steps, each replayed from the carry it started
  from.
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from ajax.agents.loop import TrainLoop, extension_key
from ajax.agents.PPO.core import (
    AuxiliaryLogs,
    PolicyAuxiliaries,
    ValueAuxiliaries,
    approximate_kl,
    clipped_surrogate,
    explained_variance,
    mean_policy_std,
    policy_entropy,
    recompute_log_prob,
    resolve_clip_coef,
    resolve_num_minibatches,
    rollout_actions,
    run_epochs,
)
from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import (
    _compute_gae,
    get_minibatches_from_batch,
    get_minibatches_preserving_time,
    split_fragments,
)
from ajax.environments.interaction import (
    get_pi,
    get_pi_sequence,
    init_collector_state,
    reset,
)
from ajax.environments.obs_norm import normalized_obs
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    env_keys,
    raw_obs_columns,
)
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import (
    get_initialized_actor_critic,
    predict_value,
    predict_value_sequence,
    state_value,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


def init_PPO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    expert_policy: Optional[Callable] = None,
) -> PPOState:
    """The initial actor, critic and collector (holding ``expert_policy``'s
    initial state when it is stateful)."""
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)

    continuous = check_if_environment_has_continuous_actions(
        env_args.env, env_params=env_args.env_params
    )
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=continuous,
        action_value=False,
        squash=network_args.squash,
        num_critics=1,
    )
    mode = env_args.mode
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode=mode, expert_policy=expert_policy
    )

    return PPOState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    extra_loss_fn: Optional[Callable] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[ValueAuxiliaries, dict]]:
    """``vf_coef * 0.5 * MSE(V(obs), targets)`` plus ``extra_loss_fn``'s
    term, with the metrics and ``extra_loss_fn``'s (``{}`` without one). A
    recurrent critic replays the sequence from ``initial_hidden``,
    ``resets`` flagging the observations that start an episode."""
    if recurrent:
        assert resets is not None and initial_hidden is not None
        v_preds, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        v_preds = predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
        )
    v_preds = v_preds.squeeze(0)  # the single critic keeps the ensemble axis

    loss = vf_coef * 0.5 * jnp.mean((v_preds - value_targets) ** 2)
    metrics: dict = {}
    if extra_loss_fn is not None:
        extra, metrics = extra_loss_fn(
            critic_params, critic_states, observations, value_targets
        )
        loss = loss + extra

    aux = ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
        explained_variance=explained_variance(v_preds, value_targets),
    )
    return loss, (aux, metrics)


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, dict]]:
    """The clipped surrogate minus ``ent_coef`` times the entropy bonus,
    plus ``extra_loss_fn``'s term; with ``advantage_normalization`` the
    advantages are normalised per minibatch. A recurrent actor replays the
    sequence from ``initial_hidden`` (the live carry on ``actor_state`` has
    advanced past this rollout), ``resets`` flagging episode starts."""
    if recurrent:
        assert resets is not None
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        pi, _ = get_pi(actor_state, actor_params, observations)

    new_log_probs = recompute_log_prob(pi, actions, raw_actions)
    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    surrogate, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    loss_actor = surrogate.mean()
    entropy = policy_entropy(pi, entropy_rng).mean()

    total_loss = loss_actor - ent_coef * entropy
    metrics: dict = {}
    if extra_loss_fn is not None:
        # A discrete policy has no mean action.
        pi_mean = None if isinstance(pi, distrax.Categorical) else pi.mean()
        extra, metrics = extra_loss_fn(actor_params, actor_state, observations, pi_mean)
        total_loss = total_loss + extra

    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
        approx_kl=approximate_kl(new_log_probs - log_probs),
        policy_std=mean_policy_std(pi),
    )
    return total_loss, (aux, metrics)


def _stack_critic_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
) -> Optional[Callable]:
    """The stack's additive critic-loss term on ``key`` as a
    ``value_loss_function`` ``extra_loss_fn``: ``(critic_params,
    critic_states, observations, value_targets) -> (term, metrics)``;
    ``None`` for an empty stack. The extensions read their state from
    ``agent_state``.
    """
    if not extension_stack:
        return None

    def critic_loss(critic_params, critic_states, observations, value_targets):
        _batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": critic_params,
            "critic_state": critic_states,
        }
        return extension_stack.fold_critic_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return critic_loss


def _stack_actor_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
    raw_observations: Optional[jax.Array] = None,
) -> Optional[Callable]:
    """The stack's additive actor-loss term on ``key`` as a
    ``policy_loss_function`` ``extra_loss_fn``: ``(actor_params,
    actor_state, observations, pi_mean) -> (term, metrics)``, the minibatch's
    ``raw_observations`` beside them; ``None`` for an empty stack."""
    if not extension_stack:
        return None

    def actor_loss(actor_params, actor_state, observations, pi_mean):
        _batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        return extension_stack.fold_actor_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return actor_loss


@dataclasses.dataclass(frozen=True)
class Geometry:
    """How an update minibatches its rollout (the module docstring)."""

    kind: str  # "time", "flat" or "recurrent"
    num_minibatches: int
    length: int  # steps per fragment ("time") or sequence ("recurrent")


def default_bptt_length(n_steps: int, n_envs: int, num_minibatches: int) -> int:
    """The longest sequences that cut every env's rollout evenly and split
    exactly into ``num_minibatches``; an explicit ``bptt_length`` splits
    only if it divides this one.

    Libraries disagree on a recurrent PPO's BPTT length (RLlib 20 steps,
    Sample Factory the rollout, Pleines et al., 2022, tuned per task), so the
    minibatches set it. When ``n_envs`` is a multiple of ``num_minibatches``
    it is the whole rollout, each minibatch a subset of the envs, as in
    Baselines' ppo2 (ea25b9e, ppo2.py:167-180), CleanRL's ppo_atari_lstm
    (fe8d8a0, :296-313) and PureJaxRL's ppo_rnn (31756b1, :258-291).
    Otherwise each rollout is cut into the fewest equal pieces that split:
    ``num_minibatches // gcd(n_envs, num_minibatches)`` of them, which must
    divide ``n_steps`` (ValueError if not; no length would split then).
    docs/guide/memory.md compares the libraries and says how to set it.
    """
    pieces = num_minibatches // math.gcd(n_envs, num_minibatches)
    if n_steps % pieces:
        raise ValueError(
            f"Recurrent PPO cannot split {n_envs} envs' rollouts of"
            f" n_steps={n_steps} into {num_minibatches} minibatches of whole"
            f" sequences: each rollout would need {pieces} equal pieces. Choose"
            " n_steps, n_envs and num_minibatches (batch_size when it is 0) so"
            " that they split (docs/guide/memory.md)."
        )
    return n_steps // pieces


def minibatch_geometry(
    agent_config: PPOConfig, n_steps: int, n_envs: int, recurrent: bool
) -> Geometry:
    """The rollout's :class:`Geometry`.

    ``time`` only when ``unroll_length`` asks for it (``n_steps`` gives
    whole rollouts per env): with as many minibatches as envs, each
    minibatch is then one env's trajectory, its advantages normalised
    within it. It needs fragments that split evenly into more than one
    minibatch: recomputing GAE on a single minibatch, the whole rollout,
    with a critic that just stepped only makes the value targets oscillate
    (the flake of the one-env probing fixture). A recurrent rollout is a
    pool of sequences of ``bptt_length`` steps (by default
    :func:`default_bptt_length`) that must split exactly into the
    minibatches, as Baselines' and CleanRL's recurrent PPO assert: a
    ``bptt_length`` that does not divide the default raises ValueError.
    """
    num_minibatches = resolve_num_minibatches(agent_config)
    if recurrent:
        longest = default_bptt_length(n_steps, n_envs, num_minibatches)
        length = agent_config.bptt_length or longest
        if longest % length:
            raise ValueError(
                f"Recurrent PPO minibatches whole sequences: {n_envs} envs'"
                f" rollouts of n_steps={n_steps} cut into sequences of"
                f" bptt_length={length} steps do not split into"
                f" {num_minibatches} minibatches. bptt_length must divide"
                f" {longest}, the longest length that does; None picks it"
                " (docs/guide/memory.md)."
            )
        return Geometry("recurrent", num_minibatches, length)
    unroll = agent_config.unroll_length
    if (
        unroll is not None
        and num_minibatches > 1
        and n_steps % unroll == 0
        and n_steps // unroll * n_envs % num_minibatches == 0
    ):
        return Geometry("time", num_minibatches, unroll)
    return Geometry("flat", num_minibatches, n_steps)


def _recurrent_values(
    agent_state: PPOState, rollout: Transition, start: PPOState, length: int
) -> tuple[PPOState, jax.Array, jax.Array, jax.Array, tuple]:
    """A recurrent rollout's values, next values and reset flags, and the
    actor's and critic's carries at the start of each sequence of
    ``length`` steps.

    One critic pass gives the values and advances the critic carry, which
    collection never steps: it ends as the next rollout's start carry. Run
    as a scan over the sequences, it also yields their start carries. The
    actor's are recomputed sequence by sequence with the current params (one
    more pass, skipped for whole-rollout sequences, whose start carry is
    exact): a sequence never starts from a zero carry mid-episode, the
    variant that makes truncated BPTT worse than none. The pass runs step by
    step (the memories' acting path, equal to the sequence one), and after
    each step reads ``V(s_{t+1})`` from the carry that consumed ``obs[t]``,
    on ``next_obs[t]``: the next observation, or the episode's final one
    where it ended. A time limit thus bootstraps on its own episode, as
    sb3-contrib's RecurrentPPO does (its pre-reset LSTM state on the
    terminal observation), not on the reset observation with a fresh
    carry; GAE masks terminal steps.

    Returns the state (critic carry advanced), the values, the next values,
    the reset flags (``resets[t]``: ``obs[t]`` starts an episode) and the
    ``(actor, critic)`` start carries.
    """
    n_steps, n_envs = rollout.obs.shape[:2]
    n_sequences = n_steps // length
    initial_done = jnp.logical_or(
        start.collector_state.last_terminated, start.collector_state.last_truncated
    ).astype(bool)
    dones_seq = jnp.logical_or(rollout.terminated, rollout.truncated).squeeze(-1)
    resets = jnp.concatenate(
        [initial_done[None], dones_seq[:-1].astype(bool)], axis=0
    )  # (T, B)
    obs_seqs = rollout.obs.reshape(n_sequences, length, *rollout.obs.shape[1:])
    next_obs_seqs = rollout.next_obs.reshape(obs_seqs.shape)
    resets_seqs = resets.reshape(n_sequences, length, n_envs)
    critic, actor = agent_state.critic_state, agent_state.actor_state
    going_on = jnp.zeros((1, n_envs), bool)

    def critic_step(carry: Any, step: tuple) -> tuple[Any, tuple]:
        obs, next_obs, step_resets = step
        value, carry = predict_value_sequence(
            critic, critic.params, obs[None], step_resets[None], carry
        )
        next_value, _ = predict_value_sequence(
            critic, critic.params, next_obs[None], going_on, carry
        )
        # (num = 1, T = 1, B, 1) -> (B, 1)
        return carry, (value.squeeze((0, 1)), next_value.squeeze((0, 1)))

    def critic_chunk(carry: Any, seq: tuple) -> tuple[Any, tuple]:
        carry_next, values = jax.lax.scan(critic_step, carry, seq)
        return carry_next, (values, carry)

    assert start.critic_state.hidden_state is not None
    critic_end, ((seq_values, seq_next_values), critic_starts) = jax.lax.scan(
        critic_chunk,
        jax.lax.stop_gradient(start.critic_state.hidden_state),
        (obs_seqs, next_obs_seqs, resets_seqs),
    )
    # (S, L, B, 1) -> (T, B, 1)
    values = seq_values.reshape(n_steps, *seq_values.shape[2:])
    next_values = seq_next_values.reshape(values.shape)

    initial_actor_hidden = jax.lax.stop_gradient(start.actor_state.hidden_state)
    if n_sequences > 1:

        def actor_chunk(carry: Any, seq: tuple) -> tuple[Any, Any]:
            obs, seq_resets = seq
            _, carry_next = get_pi_sequence(actor, actor.params, obs, seq_resets, carry)
            return carry_next, carry

        _, actor_starts = jax.lax.scan(
            actor_chunk, initial_actor_hidden, (obs_seqs, resets_seqs)
        )
    else:
        actor_starts = jax.tree.map(lambda x: x[None], initial_actor_hidden)

    agent_state = agent_state.replace(
        critic_state=critic.replace(hidden_state=jax.lax.stop_gradient(critic_end))
    )
    carries = jax.lax.stop_gradient((actor_starts, critic_starts))
    return agent_state, values, next_values, resets, carries


def _split_carries(x: jax.Array, perm: jax.Array, num_minibatches: int) -> jax.Array:
    """``(S, B, ...)`` start carries to ``(k, S * B // k, ...)``, sequence
    ``s * B + b`` in the order of ``perm`` (as :func:`split_fragments`)."""
    x = x.reshape(-1, *x.shape[2:])
    x = jnp.take(x, perm, axis=0)
    return x.reshape(num_minibatches, -1, *x.shape[1:])


def _sequence_minibatches(
    batch: dict, carries: tuple, rng: jax.Array, num_minibatches: int, length: int
) -> dict:
    """A recurrent rollout's minibatches of whole sequences, each with the
    actor's and the critic's carry at its start (the critic's ensemble axis
    second)."""
    n_steps, n_envs = batch["obs"].shape[:2]
    perm = jax.random.permutation(rng, n_steps // length * n_envs)

    def split(x: jax.Array) -> jax.Array:
        return _split_carries(x, perm, num_minibatches)

    actor_starts, critic_starts = carries
    return split_fragments(batch, perm, num_minibatches, length) | {
        "actor_hidden": jax.tree.map(split, actor_starts),
        "critic_hidden": jax.tree.map(
            jax.vmap(split, in_axes=1, out_axes=1), critic_starts
        ),
    }


def clip_global_norm(grads: Any, max_norm: Optional[float], eps: float = 0.0) -> Any:
    """``grads`` (any tree) times ``min(1, max_norm / (norm + eps))``, their
    global ``norm`` taken over every leaf at once; ``None`` leaves them.

    With ``eps`` 0 it is ``tf.clip_by_global_norm``, ppo2's (baselines
    ea25b9e ppo2/model.py:105-107); torch's ``clip_grad_norm_``, which APO's
    reference code calls, adds 1e-6 to the norm (torch 1.5.1,
    ``torch/nn/utils/clip_grad.py:33-37``)."""
    if max_norm is None:
        return grads
    coef = jnp.minimum(1.0, max_norm / (optax.global_norm(grads) + eps))
    return jax.tree.map(lambda g: g * coef, grads)


def joint_step(
    actor: Any,
    critic: Any,
    actor_grads: Any,
    critic_grads: Any,
    max_grad_norm: Optional[float],
    norm_eps: float = 0.0,
) -> tuple[Any, Any]:
    """Both networks' Adam steps on their gradients clipped together
    (:func:`clip_global_norm`), as ppo2's one optimiser, and APO's
    reference code's, clip every parameter's gradient at once. Adam is per
    coordinate, so the two Adams are one over the whole model."""
    actor_grads, critic_grads = clip_global_norm(
        (actor_grads, critic_grads), max_grad_norm, norm_eps
    )
    return (
        actor.apply_gradients(grads=actor_grads),
        critic.apply_gradients(grads=critic_grads),
    )


def _normalisation_info(env_state: Any, mode: str) -> Any:
    """The running statistics an env normaliser keeps in ``env_state``
    (``None`` without one)."""
    if mode == "brax":
        return (env_state.info or {}).get("normalization_info")
    return getattr(env_state, "normalization_info", None)


def _force_reset(
    agent_state: PPOState, env_args: EnvironmentConfig, mode: str
) -> PPOState:
    """Reset every env on a fresh key (brax's ``num_resets_per_eval``).

    The reset observations are normalised with the agent's running
    statistics, which they do not count in. A reward normaliser in the env
    stack keeps its running statistics: its reset restarts them, so the
    saved ones are put back. The cut episodes are abandoned, not finished:
    the next return starts from zero, and they are flagged truncated, so a
    recurrent actor and critic restart their memory on the reset
    observation, as at any episode start.
    """
    agent_state, reset_key = agent_state.next_keys()
    saved = _normalisation_info(agent_state.collector_state.env_state, mode)
    reset_keys = env_keys(reset_key, mode, env_args.n_envs)
    new_obs, new_env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
    if saved is not None:
        if mode == "brax":
            info = {**new_env_state.info, "normalization_info": saved}
            new_env_state = new_env_state.replace(info=info)
        else:
            new_env_state = new_env_state.replace(normalization_info=saved)
    if env_args.store_normalized_obs:
        new_obs = normalized_obs(
            new_obs,
            agent_state.collector_state.obs_norm_info,
            raw_obs_columns(env_args.env),
        )
    returns = agent_state.collector_state.episodic_return_state
    returns = returns.replace(
        cumulative_reward=jnp.zeros_like(returns.cumulative_reward)
    )
    collector = agent_state.collector_state
    new_collector = collector.replace(
        _env_state=new_env_state,
        last_obs=new_obs,
        episodic_return_state=returns,
        last_truncated=jnp.ones_like(collector.last_truncated),
    )
    return agent_state.replace(collector_state=new_collector)


def update_agent(
    agent_state: PPOState,
    rollout: Transition,
    start: PPOState,
    agent_config: PPOConfig,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    total_n_updates: int,
) -> tuple[PPOState, AuxiliaryLogs]:
    """One PPO update on the ``(n_steps, n_envs)`` ``rollout`` collected
    from ``start``: ``n_epochs`` passes over its minibatches, then brax's
    periodic forced reset."""
    n_steps, n_envs = rollout.obs.shape[:2]
    geometry = minibatch_geometry(agent_config, n_steps, n_envs, recurrent)
    actions, log_probs = rollout_actions(rollout)
    # The update's key: it shuffles the rollout, and the rollout's folds draw
    # on keys derived from it.
    agent_state, shuffle_key = agent_state.next_keys()
    reward_key, target_key = jax.random.split(extension_key(shuffle_key))
    rewards, reward_metrics = rollout.reward, dict[str, Any]()
    if "reward" in extension_stack.implemented_phases():
        # The extensions' rewards replace the env's (GAIL's discriminator).
        agent_state, rewards, reward_metrics = extension_stack.fold_reward(
            agent_state,
            {
                "observations": rollout.obs,
                "raw_observations": rollout.raw_obs,
                "actions": actions,
            },
            rewards,
            agent_state.collector_state.timestep,
            reward_key,
            total_timesteps,
        )
    batch = {
        "obs": rollout.obs,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": rollout.raw_action,
    }
    if extension_stack:
        # The environment's own observations, which an expert reads.
        batch["raw_obs"] = rollout.raw_obs
    rollout_metrics: dict = {}  # the extensions' on the rollout's targets

    if geometry.kind == "time":
        # GAE in each minibatch, with the critic of that step.
        batch |= {
            "next_obs": rollout.next_obs,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "rewards": rewards,
        }
    else:
        if recurrent:
            agent_state, values, next_values, resets, carries = _recurrent_values(
                agent_state, rollout, start, geometry.length
            )
            batch["resets"] = resets
        else:
            values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.obs,
            ).squeeze(0)
            next_values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.next_obs,
            ).squeeze(0)
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=rewards,
            terminateds=rollout.terminated,
            truncateds=rollout.truncated,
            gamma=agent_config.gamma,
            gae_lambda=agent_config.gae_lambda,
        )
        if extension_stack:
            target_batch = {
                "observations": rollout.obs,
                "next_observations": rollout.next_obs,
                "rewards": rewards,
                "terminated": rollout.terminated,
                "truncated": rollout.truncated,
                "gae": gae,
                "values": values,
                "next_values": next_values,
                "gamma": agent_config.gamma,
            }
            value_targets, rollout_metrics = extension_stack.fold_on_target(
                agent_state,
                target_batch,
                value_targets,
                agent_state.collector_state.timestep,
                target_key,
                total_timesteps,
            )
        batch |= {"targets": value_targets, "gae": gae}

    k, length = geometry.num_minibatches, geometry.length

    def minibatches(key: jax.Array) -> dict:
        """One epoch's partition of the rollout."""
        if geometry.kind == "time":
            return get_minibatches_preserving_time(batch, key, k, length)
        if geometry.kind == "recurrent":
            return _sequence_minibatches(batch, carries, key, k, length)
        return get_minibatches_from_batch(batch, key, k)

    def minibatch_step(agent_state: PPOState, mb: dict) -> tuple[PPOState, Any]:
        """A critic and an actor step, both from the state before either."""
        agent_state, ent_rng = agent_state.next_keys()
        target_metrics: dict = {}
        critic, actor = agent_state.critic_state, agent_state.actor_state
        # The extensions' target and loss terms, each on a key of its own.
        target_key, critic_key, actor_key = jax.random.split(extension_key(ent_rng), 3)
        critic_extra = _stack_critic_loss(
            extension_stack, agent_state, critic_key, total_timesteps
        )
        actor_extra = _stack_actor_loss(
            extension_stack, agent_state, actor_key, total_timesteps, mb.get("raw_obs")
        )

        if geometry.kind == "time":
            # brax's compute_ppo_loss: GAE from the current critic, held
            # constant (no gradient reaches the critic through it).
            values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic, critic_params=critic.params, x=mb["obs"]
                ).squeeze(0)
            )
            next_values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic,
                    critic_params=critic.params,
                    x=mb["next_obs"],
                ).squeeze(0)
            )
            gae, value_targets = _compute_gae(
                values=values,
                next_values=next_values,
                rewards=mb["rewards"],
                terminateds=mb["terminated"],
                truncateds=mb["truncated"],
                gamma=agent_config.gamma,
                gae_lambda=agent_config.gae_lambda,
            )
            gae = jax.lax.stop_gradient(gae)
            value_targets = jax.lax.stop_gradient(value_targets)
            if extension_stack:
                target_batch = {
                    "observations": mb["obs"],
                    "next_observations": mb["next_obs"],
                    "rewards": mb["rewards"],
                    "terminated": mb["terminated"],
                    "truncated": mb["truncated"],
                    "gae": gae,
                    "values": values,
                    "next_values": next_values,
                    "gamma": agent_config.gamma,
                }
                value_targets, target_metrics = extension_stack.fold_on_target(
                    agent_state,
                    target_batch,
                    value_targets,
                    agent_state.collector_state.timestep,
                    target_key,
                    total_timesteps,
                )
        else:
            gae, value_targets = mb["gae"], mb["targets"]

        resets = mb.get("resets")
        (_, (v_aux, v_metrics)), v_grads = jax.value_and_grad(
            value_loss_function, has_aux=True
        )(
            critic.params,
            critic,
            mb["obs"],
            value_targets,
            recurrent,
            vf_coef=agent_config.vf_coef,
            extra_loss_fn=critic_extra,
            resets=resets,
            initial_hidden=mb.get("critic_hidden"),
        )
        (_, (p_aux, p_metrics)), p_grads = jax.value_and_grad(
            policy_loss_function, has_aux=True
        )(
            actor.params,
            actor,
            mb["obs"],
            mb["actions"],
            mb["log_probs"],
            gae,
            recurrent,
            resolve_clip_coef(agent_config, agent_state.collector_state.timestep),
            agent_config.ent_coef,
            agent_config.normalize_advantage,
            extra_loss_fn=actor_extra,
            raw_actions=mb["raw_actions"],
            entropy_rng=ent_rng,
            resets=resets,
            initial_hidden=mb.get("actor_hidden"),
        )
        actor, critic = joint_step(
            actor, critic, p_grads, v_grads, agent_config.max_grad_norm
        )
        agent_state = agent_state.replace(critic_state=critic, actor_state=actor)
        # Each network's norm before the joint clip.
        p_aux = p_aux.replace(grad_norm=optax.global_norm(p_grads))
        v_aux = v_aux.replace(grad_norm=optax.global_norm(v_grads))
        metrics = merge_metrics(target_metrics, v_metrics, p_metrics)
        return agent_state, AuxiliaryLogs(policy=p_aux, value=v_aux, extensions=metrics)

    agent_state, aux = run_epochs(
        agent_state, shuffle_key, minibatches, minibatch_step, agent_config.n_epochs
    )
    # The flat and recurrent updates fold on_target once, over the rollout.
    aux = aux.replace(
        extensions=merge_metrics(reward_metrics, rollout_metrics, aux.extensions)
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    # brax's num_resets_per_eval: every env reset every reset_every updates.
    # Playground envs draw a fresh initial state at every episode end by
    # default (build_env_from_id(fresh_reset=True)); with fresh_reset=False
    # their auto-reset replays the first reset state, and without these
    # resets a run sees only n_envs initial conditions.
    if agent_config.num_resets_per_eval > 0:
        num_evals_after_init = max(int(agent_config.num_evals) - 1, 1)
        reset_every = jnp.maximum(
            1,
            jnp.ceil(
                total_n_updates
                / (num_evals_after_init * agent_config.num_resets_per_eval)
            ).astype(jnp.int32),
        )
        should_reset = (agent_state.n_updates % reset_every == 0) & (
            agent_state.n_updates > 0
        )
        agent_state = jax.lax.cond(
            should_reset,
            lambda s: _force_reset(s, env_args, mode),
            lambda s: s,
            agent_state,
        )
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: PPOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    extensions: Sequence = (),
):
    """PPO's train function: an ``n_steps`` rollout per env, then one
    update, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent and extensions:
        raise NotImplementedError("Recurrent PPO does not support extensions yet.")
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    )

    def init(key: jax.Array) -> PPOState:
        return init_PPO(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
        )

    def update(
        agent_state: PPOState, rollout: Transition, start: PPOState
    ) -> tuple[PPOState, AuxiliaryLogs]:
        return update_agent(
            agent_state,
            rollout,
            start,
            agent_config,
            env_args,
            loop.mode,
            recurrent,
            loop.stack,
            total_timesteps,
            # brax's reset cadence divides the whole training: the updates
            # of one call over the run, from 0 to its horizon.
            loop.n_rollouts(agent_config.n_steps, total_timesteps),
        )

    return loop.on_policy(
        init,
        update,
        agent_config.n_steps,
        recurrent=recurrent,
        expose_rollout=agent_config.expose_recent_rollout,
        eval_kwargs={"value_estimate": ValueEstimate(state_value, agent_config.gamma)},
    )
