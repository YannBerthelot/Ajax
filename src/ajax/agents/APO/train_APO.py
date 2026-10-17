"""APO (Ma et al., 2021): Average-reward Policy Optimization.

PPO for the average-reward criterion: each rollout updates an EMA of the
reward rate ``rho`` (``alpha`` its rate) and of the mean value ``b``; the
advantages are GAE on the differential TD error ``r - rho + V(s') - V(s)``
(no discount, cut at episode ends), and the critic fits the differential value with the
value-bias penalty ``nu b``. Each minibatch takes one step of both networks
down their summed loss, its gradient clipped over both at once.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from ajax.agents.APO.state import APOConfig, APOState
from ajax.agents.loop import TrainLoop
from ajax.agents.PPO.core import (
    AuxiliaryLogs,
    PolicyAuxiliaries,
    ValueAuxiliaries,
    clipped_surrogate,
    policy_entropy,
    recompute_log_prob,
    resolve_clip_coef,
    resolve_num_minibatches,
    rollout_actions,
    run_epochs,
)
from ajax.agents.PPO.train_PPO import init_PPO
from ajax.agents.PPO.utils import _compute_gae, get_minibatches_from_batch
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import predict_value
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    clip_coef: float,
    ent_coef: float,
    raw_actions: Optional[jax.Array] = None,
    entropy_key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, Optional[jax.Array]]]:
    """PPO's clipped surrogate with PPO's entropy bonus (APO's Gaussian's,
    exact; a squashed policy's executed action's, sampled with
    ``entropy_key``); also returns the policy mean for extensions (``None``
    for a discrete policy, which has no mean action)."""
    pi, _ = get_pi(actor_state, actor_params, observations)
    new_log_probs = recompute_log_prob(pi, actions, raw_actions)
    entropy = policy_entropy(pi, entropy_key)
    ratio = jnp.exp(new_log_probs - log_probs)
    assert ratio.shape[0] == gae.shape[0], (
        f"Mismatch between ratio shape ({ratio.shape}) and gae shape ({gae.shape})"
    )
    loss_actor, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    total_loss = (loss_actor - ent_coef * entropy.mean()).mean()
    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy.mean(),
    )
    pi_mean = None if isinstance(pi, distrax.Categorical) else pi.mean()
    return total_loss, (aux, pi_mean)


def init_APO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
) -> APOState:
    """PPO's initial state (APO's networks: :class:`ajax.agents.APO.APO`),
    the reward rate and the value bias at 0."""
    state = init_PPO(
        key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
    )
    return APOState(**vars(state), average_reward=0.0, b=0.0)


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    nu: float,
    b: float | jax.Array,
    vf_coef: float = 1.0,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """The differential value loss ``vf_coef 0.5 (V(s) - (target - nu b))^2``.

    APO's official code (``xtma/apo``, ``apo/algos/apg/``) subtracts
    ``nu b`` from the critic's target, so with ``b`` the mean value each fit
    pulls the values back towards zero mean (Ma et al., 2021, the value
    bias penalty); adding it would push them away and let ``|b|`` grow.
    ``vf_coef`` is its ``value_loss_coeff`` (``appo.py:152-153``).
    """
    # The single critic still has the ensemble's leading axis.
    v_preds = predict_value(critic_states, critic_params, observations).squeeze(0)
    loss = vf_coef * 0.5 * jnp.mean((v_preds - (value_targets - nu * b)) ** 2)
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


def clip_grad_norm(grads: Any, max_norm: Optional[float]) -> Any:
    """``grads`` scaled down to global norm ``max_norm`` when above it, as
    torch's ``clip_grad_norm_`` scales the gradients of all the parameters
    it is given, its ``1e-6`` included (torch 1.5.1, rlpyt's pin,
    ``torch/nn/utils/clip_grad.py:33-37``); ``None`` leaves them."""
    if max_norm is None:
        return grads
    coef = jnp.minimum(1.0, max_norm / (optax.global_norm(grads) + 1e-6))
    return jax.tree.map(lambda g: g * coef, grads)


def update_networks(
    agent_state: APOState,
    minibatch: dict,
    agent_config: APOConfig,
    b: jax.Array,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[APOState, AuxiliaryLogs]:
    """One step of both networks on ``minibatch`` (``update_agent``'s), as
    APO's official code takes it (``xtma/apo``, ``apo/algos/apg/``): one
    Adam over the whole model (``base.py:28``) down the clipped surrogate
    plus ``vf_coef`` times the value loss minus ``ent_coef`` times the
    entropy (``appo.py:145-158``), its gradient clipped to global norm
    ``max_grad_norm`` over every parameter (``appo.py:106-112``).

    Adam is per coordinate, so the two networks' own Adams, stepped together
    on that one gradient, are that Adam (each its learning rate, as torch's
    parameter groups would). The extensions' critic and actor terms join the
    sum, on two keys drawn from the minibatch's extension key.
    """
    actor_state, critic_state = agent_state.actor_state, agent_state.critic_state
    critic_key, actor_key = jax.random.split(minibatch["extension_key"])
    timestep = agent_state.collector_state.timestep

    def loss_fn(params: tuple) -> Tuple[jax.Array, AuxiliaryLogs]:
        actor_params, critic_params = params
        policy_loss, (policy_aux, pi_mean) = policy_loss_function(
            actor_params,
            actor_state,
            observations=minibatch["obs"],
            actions=minibatch["actions"],
            log_probs=minibatch["log_probs"],
            gae=minibatch["gae"],
            clip_coef=resolve_clip_coef(agent_config, timestep),
            ent_coef=agent_config.ent_coef,
            raw_actions=minibatch["raw_actions"],
            entropy_key=minibatch["entropy_key"],
        )
        value_loss, value_aux = value_loss_function(
            critic_params,
            critic_state,
            minibatch["obs"],
            minibatch["targets"],
            agent_config.nu,
            b,
            agent_config.vf_coef,
        )
        critic_batch = {
            "observations": minibatch["obs"],
            "targets": minibatch["targets"],
            "critic_params": critic_params,
            "critic_state": critic_state,
        }
        actor_batch = {
            "observations": minibatch["obs"],
            "raw_observations": minibatch["raw_obs"],
            "pi_mean": pi_mean,
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        extra = extension_stack.fold_critic_loss(
            agent_state, critic_batch, timestep, critic_key, total_timesteps
        ) + extension_stack.fold_actor_loss(
            agent_state, actor_batch, timestep, actor_key, total_timesteps
        )
        loss = policy_loss + value_loss + extra
        return loss, AuxiliaryLogs(policy=policy_aux, value=value_aux)

    grads, aux = jax.grad(loss_fn, has_aux=True)(
        (actor_state.params, critic_state.params)
    )
    actor_grads, critic_grads = clip_grad_norm(grads, agent_config.max_grad_norm)
    return (
        agent_state.replace(
            actor_state=actor_state.apply_gradients(grads=actor_grads),
            critic_state=critic_state.apply_gradients(grads=critic_grads),
        ),
        aux,
    )


def update_agent(
    agent_state: APOState,
    transition: Transition,
    agent_config: APOConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[APOState, AuxiliaryLogs]:
    """One APO update on an ``(n_steps, n_envs)`` rollout: the reward-rate
    and value-bias EMAs over every step, differential GAE, then
    ``n_epochs`` epochs of minibatch steps.

    The GAE is PPO's with ``gamma = 1`` on ``r - rho``, as in APO's official
    code (``xtma/apo``, ``apo/algos/utils.py``,
    ``generalized_advantage_estimation`` with ``discount = 1``, and
    ``apo/algos/apg/base.py``, ``process_returns``): ``V(s')`` is dropped
    on a termination and the lambda-carry is cut at every episode end. The
    reference's ``done`` includes time limits and it drops time-limited
    samples from the loss (``bootstrap_timelimit``), having only the reset
    observation; Ajax keeps the final observation, so a truncated step
    bootstraps on it and is trained on (PPO's convention here).
    """
    critic_state = agent_state.critic_state
    values = predict_value(critic_state, critic_state.params, transition.obs).squeeze(0)
    next_values = predict_value(
        critic_state, critic_state.params, transition.next_obs
    ).squeeze(0)

    average_reward = (
        1 - agent_config.alpha
    ) * agent_state.average_reward + agent_config.alpha * jnp.mean(transition.reward)

    b = (1 - agent_config.alpha) * agent_state.b + agent_config.alpha * jnp.mean(values)
    agent_state = agent_state.replace(average_reward=average_reward, b=b)

    gae, value_targets = _compute_gae(
        rewards=transition.reward - average_reward,
        values=values,
        next_values=next_values,
        terminateds=transition.terminated,
        truncateds=transition.truncated,
        gamma=1.0,
        gae_lambda=agent_config.gae_lambda,
    )

    # A key is drawn only with extensions.
    if extension_stack:
        target_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        target_batch = {
            "observations": transition.obs,
            "next_observations": transition.next_obs,
            "rewards": transition.reward,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "gae": gae,
            "values": values,
            "average_reward": average_reward,
            "gamma": None,  # APO is average-reward (differential V)
        }
        value_targets = extension_stack.fold_on_target(
            agent_state,
            target_batch,
            value_targets,
            agent_state.collector_state.timestep,
            target_key,
            total_timesteps,
        )

    # Normalise advantages ONCE over the full rollout (PPO here normalises
    # per minibatch, as brax's loss does; unverified): per-minibatch
    # normalisation uses noisy mean/std estimates on tiny minibatches.
    if agent_config.normalize_advantage:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    actions, log_probs = rollout_actions(transition)
    batch = {
        "obs": transition.obs,
        "actions": actions,
        "targets": value_targets,
        "gae": gae,
        "log_probs": log_probs,
        "raw_obs": transition.raw_obs,
        "raw_actions": transition.raw_action,
    }
    shuffle_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    num_minibatches = resolve_num_minibatches(agent_config)

    def minibatches(key: jax.Array) -> dict:
        """One epoch's partition of the rollout, each minibatch with a key
        for its entropy sample and one for the extensions' loss terms,
        derived from the epoch's key so the agent's own key stream is drawn
        as before."""
        parts = get_minibatches_from_batch(batch, key, num_minibatches)
        keys = jax.random.split(jax.random.fold_in(key, 1), num_minibatches)
        ext_keys = jax.random.split(jax.random.fold_in(key, 2), num_minibatches)
        return parts | {"entropy_key": keys, "extension_key": ext_keys}

    def step(agent_state: APOState, mb: dict) -> Tuple[APOState, AuxiliaryLogs]:
        return update_networks(
            agent_state, mb, agent_config, b, extension_stack, total_timesteps
        )

    agent_state, aux = run_epochs(
        agent_state, shuffle_key, minibatches, step, agent_config.n_epochs
    )
    return agent_state.replace(n_updates=agent_state.n_updates + 1), aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: APOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    expert_policy: Optional[Callable] = None,
    extensions: Sequence = (),
):
    """APO's train function: an ``n_steps`` rollout per env, then one
    update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
    )

    def init(key: jax.Array) -> APOState:
        return init_APO(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
        )

    def update(agent_state: APOState, rollout: Transition, _start: APOState) -> Any:
        return update_agent(
            agent_state, rollout, agent_config, loop.stack, total_timesteps
        )

    return loop.on_policy(
        init,
        update,
        agent_config.n_steps,
        expose_rollout=agent_config.expose_recent_rollout,
        eval_kwargs={"avg_reward_mode": True, "expert_policy": expert_policy},
    )
