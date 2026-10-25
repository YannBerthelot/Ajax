"""The steps an agent's update is written in: a loss's gradients
(compute_gradients), gradient steps with the extensions' loss terms added
(critic_step, actor_step, apply_gradients_clipped_together for two networks
clipped together), the extensions' target and post-update phases, the gates
(when) and keys (extension_key) around them, which ajax.agents.loop
re-exports, and the target critics' Polyak update SAC, ASAC, REDQ and TQC
share (update_target_networks).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax

from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.log import unevaluated

#: The keys of the extensions' batches the shared steps below build, which
#: an agent on them declares (ActorCritic.phase_batch_keys), with any it
#: adds: apply_on_target's, apply_critic_loss's, and those
#: actor_step adds, and with them the rows' entries a replay agent
#: passes (REPLAY_ACTOR_KEYS: SAC's, TD3's, CrossQ's and SAC's core's).
TARGET_KEYS = frozenset(
    {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "dones",
        "gamma",
        "reward_scale",
        "raw_next_observations",
        "next_a_expert",
        "next_action",
    }
)


CRITIC_LOSS_KEYS = frozenset(
    {
        "observations",
        "actions",
        "next_observations",
        "targets",
        "critic_params",
        "critic_state",
        "actor_state",
        "return_to_go",
    }
)


ACTOR_STEP_KEYS = frozenset({"pi_mean", "actor_params", "actor_state", "agent_loss"})


REPLAY_ACTOR_KEYS = ACTOR_STEP_KEYS.union(
    {"observations", "raw_observations", "actions", "a_expert", "critic_state"}
)


#: A replay actor-critic's on the three steps (REDQ, ASAC, CrossQ, TQC).
REPLAY_BATCH_KEYS: dict[str, frozenset[str]] = {
    "on_target": TARGET_KEYS,
    "critic_loss": CRITIC_LOSS_KEYS,
    "actor_loss": REPLAY_ACTOR_KEYS,
}


def apply_post_update(
    extensions: ExtensionStack, agent_state: Any, step: Any, total_steps: int
) -> Any:
    """Apply the extensions' post_update at step on a fresh key split
    from agent_state.rng (no key is drawn without extensions)."""
    if not extensions:
        return agent_state
    key, rng = jax.random.split(agent_state.rng)
    return extensions.apply_post_update(
        agent_state.replace(rng=rng), step, key, total_steps
    )


def when(pred: Any, fn: Callable[[Any], Any], state: Any, fill: Any = None) -> Any:
    """fn(state) where pred holds, else state unchanged: a policy
    delay, a target refresh, a warm-up.

    fn returns the new state, or the new state and its metrics. Skipped,
    the metrics have fn's structure, read from its trace (with the names
    the extensions log), filled with fill: by default NaN, -1 in integer
    leaves (ajax.log.unevaluated), which the logger drops. Under the
    seed vmap a pred read from per-seed state runs fn every time
    and selects; one from unbatched counters runs it only where it holds.
    """
    # The branches close over state rather than take it as an operand:
    # with state an operand, SAC's warm-ups (an fn returning a state
    # computed outside) compiled to other float rounding than the
    # lax.cond they replaced.
    out = jax.eval_shape(fn, state)
    # PyTreeDef compares by structure; its stub does not declare __eq__.
    if jax.tree.structure(out) == jax.tree.structure(state):  # type: ignore[operator]
        return jax.lax.cond(pred, lambda: fn(state), lambda: state)
    metrics = (
        unevaluated(out[1])
        if fill is None
        else jax.tree.map(lambda s: jnp.full(s.shape, fill, s.dtype), out[1])
    )
    return jax.lax.cond(pred, lambda: fn(state), lambda: (state, metrics))


#: What a step's key is folded with into its extensions' key. fold_in(k, i)
#: is split(k, n)[i] under JAX's partitionable threefry, so the tag lies far
#: past any split's length: no key a step splits off its own is theirs.
EXTENSIONS = 0x45585453  # "EXTS"


def extension_key(key: jax.Array) -> jax.Array:
    """The extensions' key of a step on key, which the step's own draws
    (on key and the keys it splits off it) never get."""
    return jax.random.fold_in(key, EXTENSIONS)


def compute_gradients(
    loss: Callable[[Any], tuple[jax.Array, Any]], params: Any
) -> tuple[Any, Any]:
    """The gradients of loss at params, and what loss returns beside its value.

    loss takes the parameters alone and returns (value, aux): the number to
    minimise and what to keep from computing it (its metrics). An agent fixes
    the loss's other inputs first, in a small local function. params is one
    network's parameters, or a tuple of several networks' (APO's summed loss),
    and the gradients have the same structure. A caller that logs the loss's
    value puts it in aux.
    """
    return jax.grad(loss, has_aux=True)(params)


def gradient_step(train_state: Any, loss_fn: Callable) -> tuple[Any, Any, jax.Array]:
    """One optimiser step down loss_fn(params) -> (loss, aux).

    Returns the stepped train state, the loss's aux and the global norm
    of the gradients, before any clipping the optimiser does (a caller that
    does not log it leaves it to dead-code elimination).
    """
    grads, aux = compute_gradients(loss_fn, train_state.params)
    return train_state.apply_gradients(grads=grads), aux, optax.global_norm(grads)


def apply_on_target(
    extensions: ExtensionStack,
    agent_state: Any,
    batch: Any,
    target_q: jax.Array,
    key: jax.Array,
    total_timesteps: int,
    *,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: Optional[float],
    reward_scale: float,
    target_extras: Optional[dict] = None,
) -> tuple[jax.Array, dict]:
    """target_q with the extensions' on_target applied, held
    constant, and their metrics: critic_step's first half, which an
    agent whose target comes out of its own critic pass (CrossQ) calls
    inside its loss.

    batch is a Transition. rewards, dones (where the
    bootstrap is cut), gamma (None for an average-reward agent) and
    reward_scale are what the target used; target_extras join the
    extensions' batch (TD3's smoothed next actions). The phase draws on a key
    derived from key, the step's (extension_key).
    """
    target_batch = {
        "observations": batch.obs,
        "actions": batch.action,
        "next_observations": batch.next_obs,
        "rewards": rewards,
        "dones": dones,
        "gamma": gamma,
        "reward_scale": reward_scale,
        # The env's own s' and the agent's expert's action there, from its
        # state; None where the replay keeps neither (a run without an expert).
        "raw_next_observations": batch.raw_next_obs,
        "next_a_expert": batch.next_a_expert,
        # The dataset's action at s' (NaN where unknown); None unstored.
        "next_action": batch.next_action,
    }
    target_batch.update(target_extras or {})
    target_key, _ = jax.random.split(extension_key(key))
    step = agent_state.collector_state.timestep
    target_q, metrics = extensions.apply_on_target(
        agent_state, target_batch, target_q, step, target_key, total_timesteps
    )
    return jax.lax.stop_gradient(target_q), metrics


def apply_critic_loss(
    extensions: ExtensionStack,
    agent_state: Any,
    batch: Any,
    params: Any,
    target_q: jax.Array,
    key: jax.Array,
    total_timesteps: int,
) -> tuple[jax.Array | float, dict]:
    """The extensions' critic_loss terms at the critic's params (the
    differentiated ones) on batch against the held target_q, and
    their metrics: critic_step's second half, on a key derived from
    key, the step's."""
    loss_batch = {
        "observations": batch.obs,
        "actions": batch.action,
        "next_observations": batch.next_obs,
        "targets": target_q,
        "critic_params": params,
        "critic_state": agent_state.critic_state,
        "actor_state": agent_state.actor_state,
        "return_to_go": batch.return_to_go,
    }
    _, loss_key = jax.random.split(extension_key(key))
    step = agent_state.collector_state.timestep
    return extensions.apply_critic_loss(
        agent_state, loss_batch, step, loss_key, total_timesteps
    )


def critic_step(
    agent_state: Any,
    batch: Any,
    target_q: jax.Array,
    value_loss: Callable[[Any, jax.Array], tuple[jax.Array, Any]],
    extensions: ExtensionStack,
    key: jax.Array,
    total_timesteps: int,
    **target_args: Any,
) -> tuple[Any, Any, jax.Array, dict]:
    """A replay agent's critic step on batch (a Transition).

    Applies the extensions' on_target to target_q, held constant
    (apply_on_target, which target_args reach: what the target
    used), then steps the critic down value_loss(params, target_q) ->
    (loss, aux) plus their critic_loss terms
    (apply_critic_loss). The phases draw on keys derived from key,
    the step's, on which the agent may draw too. Returns the stepped critic
    state, aux, the gradients' norm (gradient_step) and the two
    phases' metrics.
    """
    target_q, target_metrics = apply_on_target(
        extensions,
        agent_state,
        batch,
        target_q,
        key,
        total_timesteps,
        **target_args,
    )

    def loss_fn(params: Any) -> tuple[jax.Array, Any]:
        loss, aux = value_loss(params, target_q)
        extra, loss_metrics = apply_critic_loss(
            extensions, agent_state, batch, params, target_q, key, total_timesteps
        )
        return loss + extra, (aux, loss_metrics)

    critic_state, (aux, loss_metrics), grad_norm = gradient_step(
        agent_state.critic_state, loss_fn
    )
    return critic_state, aux, grad_norm, merge_metrics(target_metrics, loss_metrics)


def actor_step(
    agent_state: Any,
    policy_loss: Callable[[Any], tuple[jax.Array, Any, Optional[jax.Array]]],
    extensions: ExtensionStack,
    key: jax.Array,
    total_timesteps: int,
    **batch: Any,
) -> tuple[Any, Any, jax.Array, dict]:
    """A replay agent's actor step.

    Steps the actor down policy_loss(params) -> (loss, aux, pi_mean)
    (pi_mean: the policy's mean action at the rows, None for a discrete
    policy) plus the extensions' actor_loss terms, on a key
    derived from key, the step's (extension_key). Their batch is
    batch, the rows' entries the agent passes (observations,
    raw_observations, actions, a_expert, critic_state), with
    the differentiated actor_params, the actor_state, pi_mean and
    agent_loss.
    Returns the stepped actor state, aux, the gradients' norm
    (gradient_step) and the phase's metrics.
    """
    actor_state = agent_state.actor_state

    def loss_fn(params: Any) -> tuple[jax.Array, Any]:
        loss, aux, pi_mean = policy_loss(params)
        actor_batch = dict(batch)
        actor_batch.update(
            {
                "pi_mean": pi_mean,
                "actor_params": params,
                "actor_state": actor_state,
                "agent_loss": loss,
            }
        )
        step = agent_state.collector_state.timestep
        extra, metrics = extensions.apply_actor_loss(
            agent_state, actor_batch, step, extension_key(key), total_timesteps
        )
        return loss + extra, (aux, metrics)

    actor_state, (aux, metrics), grad_norm = gradient_step(actor_state, loss_fn)
    return actor_state, aux, grad_norm, metrics


def clip_global_norm(grads: Any, max_norm: Optional[float], eps: float = 0.0) -> Any:
    """grads (any tree) times min(1, max_norm / (norm + eps)), their
    global norm taken over every leaf at once; None leaves them.

    With eps 0 it is tf.clip_by_global_norm, ppo2's (baselines
    ea25b9e ppo2/model.py:105-107); torch's clip_grad_norm_, which APO's
    reference code calls, adds 1e-6 to the norm (torch 1.5.1,
    torch/nn/utils/clip_grad.py:33-37)."""
    if max_norm is None:
        return grads
    coef = jnp.minimum(1.0, max_norm / (optax.global_norm(grads) + eps))
    return jax.tree.map(lambda g: g * coef, grads)


def apply_gradients_clipped_together(
    actor: Any,
    critic: Any,
    actor_grads: Any,
    critic_grads: Any,
    max_grad_norm: Optional[float],
    norm_eps: float = 0.0,
) -> tuple[Any, Any, jax.Array, jax.Array]:
    """One Adam step of each network, their gradients first clipped together
    to one global norm (clip_global_norm), as ppo2's single optimiser and
    APO's reference code clip every parameter's gradient at once. Adam works
    per coordinate, so the two Adams act as one over the whole model.

    Returns the stepped actor and critic, then each one's gradient norm
    before the clip."""
    clipped_actor_grads, clipped_critic_grads = clip_global_norm(
        (actor_grads, critic_grads), max_grad_norm, norm_eps
    )
    actor = actor.apply_gradients(grads=clipped_actor_grads)
    critic = critic.apply_gradients(grads=clipped_critic_grads)
    actor_grad_norm = optax.global_norm(actor_grads)
    critic_grad_norm = optax.global_norm(critic_grads)
    return actor, critic, actor_grad_norm, critic_grad_norm


def joint_step(
    actor: Any,
    critic: Any,
    actor_grads: Any,
    critic_grads: Any,
    max_grad_norm: Optional[float],
    norm_eps: float = 0.0,
) -> tuple[Any, Any]:
    """The former name of apply_gradients_clipped_together, which returns the
    stepped actor and critic alone, as it did in 0.4.0 (then
    ajax.agents.PPO.train_PPO.joint_step); 0.5.0 removes it."""
    actor, critic, _, _ = apply_gradients_clipped_together(
        actor, critic, actor_grads, critic_grads, max_grad_norm, norm_eps
    )
    return actor, critic


# The former names, which the agents not yet declared on a family call;
# 0.5.0 removes them.
fold_post_update = apply_post_update
fold_on_target = apply_on_target
fold_critic_loss = apply_critic_loss


def update_target_networks(agent_state: Any, tau: float) -> Any:
    """Polyak-average the target critics towards the critics."""
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )
