"""APO (Ma et al., 2021): Average-reward Policy Optimization.

PPO for the average-reward criterion: each rollout updates an EMA of the
reward rate ``rho`` (``alpha`` its rate) and of the mean value ``b``; the
advantages are GAE on the differential TD error ``r - rho + V(s') - V(s)``
(no discount), and the critic fits the differential value with the
value-bias penalty ``nu b``.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.APO.state import APOConfig, APOState
from ajax.agents.APO.utils import _compute_gae
from ajax.agents.cloning import CloningConfig, pretrain_on_expert
from ajax.agents.loop import TrainLoop, gradient_step
from ajax.agents.PPO.utils import get_minibatches_from_batch
from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.interaction import get_pi, init_collector_state
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.networks import get_initialized_actor_critic, predict_value
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: float
    log_probs: float
    old_log_probs: float
    clip_fraction: float
    entropy: float


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: float
    predictions: float
    targets: float


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def compute_entropy_and_log_probs(pi, actions, raw_actions=None):
    """Return new log_probs and entropy with shape normalization.

    For SquashedNormal policies, pass the pre-tanh ``raw_actions``
    (stored at collection time) to get a numerically stable log_prob
    via ``pi.log_prob_from_raw(raw_actions)`` -- avoids the unstable
    ``arctanh(post_tanh_action)`` path inside distrax. See m4 audit
    (May 2026) and ``SquashedNormal.log_prob_from_raw`` docstring.
    """
    if isinstance(pi, distrax.Categorical):
        new_log_probs = jnp.expand_dims(pi.log_prob(actions.squeeze(-1)), -1)
        entropy = jnp.expand_dims(pi.entropy(), -1)
    elif isinstance(pi, SquashedNormal) and raw_actions is not None:
        # SAFE recompute via the helper -- never inverts tanh.
        new_log_probs = pi.log_prob_from_raw(raw_actions)
        entropy = pi.unsquashed_entropy()
    else:
        new_log_probs = pi.log_prob(actions).sum(-1, keepdims=True)
        entropy = (
            pi.unsquashed_entropy() if isinstance(pi, SquashedNormal) else pi.entropy()
        )
    return new_log_probs, entropy


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    clip_coef: float,
    ent_coef: float,
    # Pre-tanh raw action for SquashedNormal log_prob recompute (m4).
    # See SquashedNormal.log_prob_from_raw and PPO's policy_loss_function
    # for the rationale. None ⇒ fall back to the standard
    # pi.log_prob(actions) path (Categorical or unsquashed policies).
    raw_actions: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, Optional[jax.Array]]]:
    """Clipped surrogate loss; also returns the policy mean for extensions
    (``None`` for a discrete policy, which has no mean action)."""
    pi, _ = get_pi(actor_state, actor_params, observations)
    new_log_probs, entropy = compute_entropy_and_log_probs(
        pi, actions, raw_actions=raw_actions
    )
    ratio = jnp.exp(new_log_probs - log_probs)
    assert (
        ratio.shape[0] == gae.shape[0]
    ), f"Mismatch between ratio shape ({ratio.shape}) and gae shape ({gae.shape})"
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()
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


def update_policy(
    agent_state: APOState,
    observations: jax.Array,
    actions: jax.Array,
    gae: jax.Array,
    log_probs: jax.Array,
    clip_coef: float,
    ent_coef: float,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    raw_observations: Optional[jax.Array] = None,
    raw_actions: Optional[jax.Array] = None,
) -> Tuple[APOState, PolicyAuxiliaries]:
    """The clipped-surrogate actor step on one minibatch."""
    actor_state = agent_state.actor_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, PolicyAuxiliaries]:
        loss, (aux, pi_mean) = policy_loss_function(
            params,
            actor_state,
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            gae=gae,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            raw_actions=raw_actions,  # m4: pre-tanh for SquashedNormal recompute
        )
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
            "actor_params": params,
            "actor_state": actor_state,
        }
        extra = extension_stack.fold_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            agent_state.rng,
            total_timesteps,
        )
        return loss + extra, aux

    actor_state, aux = gradient_step(actor_state, loss_fn)
    return agent_state.replace(actor_state=actor_state), aux


def init_APO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    window_size: int = 10,
    pid_actor_config: Optional[PIDActorConfig] = None,
) -> APOState:
    rng, init_key, collector_key = jax.random.split(key, num=3)
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=check_if_environment_has_continuous_actions(
            env_args.env, env_params=env_args.env_params
        ),
        action_value=False,
        # Average-reward PPO (Ma et al. 2021). Continuous-action APO on
        # bounded control envs (brax / mujoco_playground) suffers the
        # same Gaussian-saturates-at-clip pathology as PPO when
        # squash=False (unbounded Normal + external ClipActionBrax).
        # Squashing natively keeps the policy in [-1, 1] with a correct
        # log-prob Jacobian. Discrete-action APO ignores this flag.
        squash=True,
        # brax-style policy noise: a single learnable scalar log_std per
        # action dim (state-independent), init at log(std=1.0)=0.0 —
        # ~3.5x more exploration than Ajax's legacy state-dependent
        # log_std with bias=-1 (std≈0.37). See PPO for the rationale.
        log_std_state_independent=True,
        log_std_init=0.0,
        # brax-style mean-head init: see PPO for the rationale. Legacy
        # orthogonal(0.01) gives an essentially-zero deterministic eval
        # action; lecun_uniform produces moderate initial actions that
        # train_reward >> eval_reward feedback was traced to.
        mean_kernel_init="lecun_uniform",
        # brax MLP has no output LayerNorm; Ajax's Encoder adds one by
        # default. Disable to match brax (see PPO for full rationale).
        disable_encoder_output_norm=True,
        num_critics=1,
        pid_actor_config=pid_actor_config,
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode="gymnax" if check_env_is_gymnax(env_args.env) else "brax",
        window_size=window_size,
    )
    return APOState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
        average_reward=0.0,
        b=0.0,
    )


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    nu: float,
    b: float,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """The differential value loss ``0.5 (V(s) - nu b - target)^2``."""
    # The single critic still has the ensemble's leading axis.
    v_preds = predict_value(critic_states, critic_params, observations).squeeze(0)
    loss = 0.5 * jnp.mean(((v_preds - nu * b) - value_targets) ** 2)  # classic MSE
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


def update_value_functions(
    agent_state: APOState,
    observations: jax.Array,
    value_targets: jax.Array,
    nu: float,
    b: float,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[APOState, ValueAuxiliaries]:
    """The critic step on one minibatch."""
    critic_state = agent_state.critic_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, ValueAuxiliaries]:
        loss, aux = value_loss_function(
            params, critic_state, observations, value_targets, nu, b
        )
        loss_batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": params,
            "critic_state": critic_state,
        }
        extra = extension_stack.fold_critic_loss(
            agent_state,
            loss_batch,
            agent_state.collector_state.timestep,
            agent_state.rng,
            total_timesteps,
        )
        return loss + extra, aux

    critic_state, aux = gradient_step(critic_state, loss_fn)
    return agent_state.replace(critic_state=critic_state), aux


def update_epoch(
    agent_state: APOState,
    minibatches: tuple,
    agent_config: APOConfig,
    b: float,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[APOState, AuxiliaryLogs]:
    """One epoch: a critic and an actor step on every minibatch (minibatch
    axis leading); the metrics stacked per minibatch."""

    def minibatch_step(agent_state: APOState, mb: tuple) -> Tuple[APOState, Any]:
        (
            observations,
            actions,
            _terminated,
            _truncated,
            value_targets,
            gae,
            log_probs,
            raw_observations,
            raw_action,
        ) = mb
        agent_state, aux_value = update_value_functions(
            agent_state,
            observations,
            value_targets,
            agent_config.nu,
            b,
            extension_stack,
            total_timesteps,
        )
        clip_coef = (
            agent_config.clip_range(agent_state.collector_state.timestep)
            if callable(agent_config.clip_range)
            else agent_config.clip_range
        )
        agent_state, aux_policy = update_policy(
            agent_state,
            observations,
            actions,
            gae,
            log_probs,
            clip_coef,
            agent_config.ent_coef,
            extension_stack,
            total_timesteps,
            raw_observations=raw_observations,
            raw_actions=raw_action,  # m4: SquashedNormal log_prob recompute
        )
        return agent_state, AuxiliaryLogs(policy=aux_policy, value=aux_value)

    return jax.lax.scan(minibatch_step, agent_state, minibatches)


def update_agent(
    agent_state: APOState,
    transition: Transition,
    agent_config: APOConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[APOState, AuxiliaryLogs]:
    """One APO update on an ``(n_steps, n_envs)`` rollout: the reward-rate
    and value-bias EMAs, differential GAE, then ``n_epochs`` epochs of
    minibatch steps."""
    critic_state = agent_state.critic_state
    values = predict_value(critic_state, critic_state.params, transition.obs).squeeze(0)
    last_value = (
        predict_value(critic_state, critic_state.params, transition.next_obs[-1:])
        .squeeze(0)
        .squeeze(0)  # don't need the first dimension for a single transition
    )
    dones = transition.terminated

    average_reward = (
        1 - agent_config.alpha
    ) * agent_state.average_reward + agent_config.alpha * jnp.mean(transition.reward)

    b = (1 - agent_config.alpha) * agent_state.b + agent_config.alpha * jnp.mean(values)
    agent_state = agent_state.replace(average_reward=average_reward, b=b)

    gae, value_targets = _compute_gae(
        values=values,
        last_value=last_value,
        rewards=transition.reward,
        dones=dones,
        gae_lambda=agent_config.gae_lambda,
        average_reward=average_reward,
    )

    # Extension on_target: reshape the value targets after GAE. APO is
    # average-reward: gamma None says so. A key is drawn only with
    # extensions.
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

    # Normalise advantages ONCE over the full rollout (brax PPO's
    # convention): per-minibatch normalisation uses noisy mean/std
    # estimates on tiny minibatches.
    if agent_config.normalize_advantage:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    assert transition.log_prob is not None  # an on-policy rollout carries it
    batch = (
        transition.obs,
        (
            jnp.expand_dims(transition.action, axis=-1)
            if jnp.ndim(transition.action)
            < 3  # discrete case without trailing dimension
            else transition.action
        ),
        transition.terminated,
        transition.truncated,
        value_targets,
        gae,
        (
            jnp.expand_dims(transition.log_prob, axis=-1)
            if jnp.ndim(transition.log_prob)
            < 3  # discrete case without trailing dimension
            else transition.log_prob.sum(-1, keepdims=True)
        ),
        transition.raw_obs,
        # Pre-tanh raw_action for SquashedNormal log_prob recompute
        # (m4 fix). For non-squashed / discrete policies this equals
        # ``transition.action``, so same shape.
        transition.raw_action,
    )

    shuffle_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    # num_minibatches: brax-style when set explicitly (positive),
    # independent of batch_size and n_steps; otherwise legacy Ajax,
    # derived from the batch_size / n_steps ratio.
    if agent_config.num_minibatches > 0:
        num_minibatches = agent_config.num_minibatches
    else:
        assert (
            max(agent_config.batch_size, agent_config.n_steps)
            % min(agent_config.batch_size, agent_config.n_steps)
            == 0
        ), (
            "can't evenly break n_steps into batch size chunks,"
            f" n_steps={agent_config.n_steps} batch_size={agent_config.batch_size}"
        )
        num_minibatches = max(agent_config.batch_size, agent_config.n_steps) // min(
            agent_config.batch_size, agent_config.n_steps
        )
    minibatches = get_minibatches_from_batch(
        batch, rng=shuffle_key, num_minibatches=num_minibatches
    )

    def epoch(agent_state: APOState, _: Any) -> Tuple[APOState, AuxiliaryLogs]:
        return update_epoch(
            agent_state, minibatches, agent_config, b, extension_stack, total_timesteps
        )

    agent_state, aux = jax.lax.scan(
        epoch, agent_state, xs=None, length=agent_config.n_epochs
    )
    # The metrics, averaged over every epoch and minibatch.
    aux = jax.tree.map(jnp.mean, aux)
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
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    extensions: Sequence = (),
):
    """APO's train function: an ``n_steps`` rollout per env, then one
    update, per iteration."""
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key: jax.Array, pretrain_key: jax.Array) -> APOState:
        agent_state = init_APO(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            pid_actor_config=pid_actor_config,
        )
        return pretrain_on_expert(
            agent_state,
            pretrain_key,
            cloning_args,
            expert_policy,
            env_args,
            agent_config,
            actor_optimizer_args,
            critic_optimizer_args,
        )

    def update(agent_state: APOState, rollout: Transition) -> Any:
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
