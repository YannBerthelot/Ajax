from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state
from jax.tree_util import Partial as partial

from ajax.environments.interaction import (
    collect_experience_from_expert_policy,
)
from ajax.utils import get_one


@struct.dataclass
class CloningConfig:
    actor_epochs: int = 10
    critic_epochs: int = 10
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    actor_batch_size: int = 64
    critic_batch_size: int = 64
    pre_train_n_steps: int = int(1e5)
    imitation_coef: float = 1e-3
    distance_to_stable: Optional[Callable] = None
    imitation_coef_offset: float = 1e-3
    action_scale: float = 0.1


def get_imitation_coef(
    cloning_args: CloningConfig, total_timesteps: int
) -> Union[float, Callable]:
    imitation_coef = cloning_args.imitation_coef

    def imitation_coef_schedule(init_val):
        def imitation_coef(t, total_timesteps):
            return (1 - (t / total_timesteps)) * init_val

        return imitation_coef

    # if "auto" not in str(imitation_coef):
    if "lin" in str(imitation_coef):
        imitation_coef = (
            imitation_coef_schedule(float(imitation_coef.split("_")[1]))
            if isinstance(imitation_coef, str)
            else imitation_coef
        )
        imitation_coef = (
            partial(imitation_coef, total_timesteps=total_timesteps)
            if callable(imitation_coef)
            else imitation_coef
        )
    if "auto" in str(imitation_coef):
        imitation_coef = float("".join(imitation_coef.split("_")[1:]))
    return imitation_coef


def batchify(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reshape x into (num_batches, batch_size, ...) padding last batch if needed."""
    n = x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    pad = n_batches * batch_size - n
    if pad > 0:
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
    return x.reshape(n_batches, batch_size, *x.shape[1:])


@partial(
    jax.jit,
    static_argnames=[
        "actor_lr",
        "actor_epochs",
        "actor_batch_size",
        "critic_lr",
        "critic_epochs",
        "critic_batch_size",
    ],
)
def pre_train(
    rng: jax.Array,
    actor_state: train_state.TrainState,
    critic_state: train_state.TrainState,
    dataset: Sequence,  # Sequence[Transition]
    # Actor hyperparameters
    actor_lr: float = 1e-3,
    actor_epochs: int = 10,
    actor_batch_size: int = 64,
    # Critic hyperparameters
    critic_lr: float = 1e-3,
    critic_epochs: int = 10,
    critic_batch_size: int = 64,
) -> Tuple[train_state.TrainState, train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Pre-train actor (behavioral cloning) and critic (TD(0)) from a dataset of transitions.
    Returns trained states and metrics dict with per-epoch actor/critic losses.
    """
    obs = dataset.obs
    actions = dataset.action
    # rewards = dataset.reward
    # terminated = dataset.terminated
    # next_obs = dataset.next_obs
    metrics = {
        "actor_loss": jnp.zeros((actor_epochs,)),
        "critic_loss": jnp.zeros((critic_epochs,)),
    }

    # --------------------------
    # Actor pre-training
    # --------------------------
    bc_actor_state = train_state.TrainState.create(
        apply_fn=actor_state.apply_fn,
        params=actor_state.params,
        tx=optax.adam(actor_lr),
    )

    def actor_loss_fn(params, batch_obs, batch_actions):
        pi = bc_actor_state.apply_fn(params, batch_obs)
        # if isinstance(pi, distrax.Categorical):
        #     pred_actions = pi.mode()
        # else:
        #     pred_actions = pi.mean()  # deterministic prediction
        return -pi.log_prob(batch_actions).sum(-1, keepdims=True).mean()

    def actor_train_step(state, batch_obs, batch_actions):
        loss, grads = jax.value_and_grad(actor_loss_fn)(
            state.params, batch_obs, batch_actions
        )
        return state.apply_gradients(grads=grads), loss

    def actor_epoch_step(carry, rng_epoch):
        state = carry
        perm = jax.random.permutation(rng_epoch, obs.shape[0])
        obs_shuffled = obs[perm]
        actions_shuffled = actions[perm]

        obs_batches = batchify(obs_shuffled, actor_batch_size)
        act_batches = batchify(actions_shuffled, actor_batch_size)

        def batch_step(carry, batch):
            state = carry
            b_obs, b_act = batch
            new_state, loss = actor_train_step(state, b_obs, b_act)
            return new_state, loss

        state, batch_losses = jax.lax.scan(
            batch_step, state, (obs_batches, act_batches)
        )
        return state, jnp.mean(batch_losses)

    rng, rng_actor = jax.random.split(rng)
    rng_epochs = jax.random.split(rng_actor, actor_epochs)
    bc_actor_state, actor_losses = jax.lax.scan(
        actor_epoch_step, bc_actor_state, rng_epochs
    )
    metrics["actor_loss"] = actor_losses

    # --------------------------
    # Critic pre-training
    # --------------------------
    # bc_critic_state = train_state.TrainState.create(
    #     apply_fn=critic_state.apply_fn,
    #     params=critic_state.params,
    #     tx=optax.adam(critic_lr),
    # )

    # def critic_loss_fn(
    #     params, batch_obs, batch_rewards, batch_next_obs, batch_terminated
    # ):
    #     v_pred = bc_critic_state.apply_fn(params, batch_obs).squeeze(-1)
    #     v_next = bc_critic_state.apply_fn(params, batch_next_obs).squeeze(-1)
    #     if "gamma" in agent_config.__dict__:
    #         td_target = batch_rewards.squeeze(-1) + agent_config.gamma * v_next * (
    #             1.0 - batch_terminated.squeeze(-1)
    #         )
    #         return jnp.mean((v_pred - td_target) ** 2)
    #     else:  # average-reward
    #         td_target = batch_rewards - jnp.mean(batch_rewards) + v_next
    #         b = jnp.mean(v_pred)
    #         jax.debug.breakpoint()
    #         return jnp.mean(((v_pred - agent_config.nu * b) - td_target) ** 2)

    # def critic_train_step(
    #     state, batch_obs, batch_rewards, batch_next_obs, batch_terminated
    # ):
    #     loss, grads = jax.value_and_grad(critic_loss_fn)(
    #         state.params, batch_obs, batch_rewards, batch_next_obs, batch_terminated
    #     )
    #     return state.apply_gradients(grads=grads), loss

    # def critic_epoch_step(carry, rng_epoch):
    #     state = carry
    #     perm = jax.random.permutation(rng_epoch, obs.shape[0])
    #     obs_shuffled = obs[perm]
    #     rewards_shuffled = rewards[perm]
    #     next_obs_shuffled = next_obs[perm]
    #     terminated_shuffled = terminated[perm]

    #     obs_batches = batchify(obs_shuffled, critic_batch_size)
    #     rew_batches = batchify(rewards_shuffled, critic_batch_size)
    #     next_obs_batches = batchify(next_obs_shuffled, critic_batch_size)
    #     term_batches = batchify(terminated_shuffled, critic_batch_size)

    #     def batch_step(carry, batch):
    #         state = carry
    #         b_obs, b_rew, b_next_obs, b_term = batch
    #         new_state, loss = critic_train_step(state, b_obs, b_rew, b_next_obs, b_term)
    #         return new_state, loss

    #     state, batch_losses = jax.lax.scan(
    #         batch_step,
    #         state,
    #         (obs_batches, rew_batches, next_obs_batches, term_batches),
    #     )
    #     return state, jnp.mean(batch_losses)

    # rng, rng_critic = jax.random.split(rng)
    # rng_epochs = jax.random.split(rng_critic, critic_epochs)
    # bc_critic_state, critic_losses = jax.lax.scan(
    #     critic_epoch_step, bc_critic_state, rng_epochs
    # )
    # metrics["critic_loss"] = critic_losses

    # Update original states with trained parameters
    actor_state = actor_state.replace(params=bc_actor_state.params)
    # critic_state = critic_state.replace(params=bc_critic_state.params)
    return actor_state, critic_state, metrics


def get_pre_trained_agent(
    agent_state,
    expert_policy,
    expert_key,
    env_args,
    cloning_args,
    mode,
    agent_config,
    actor_optimizer_args,
    critic_optimizer_args,
):
    # dataset is examples of observations and actions taken
    dataset = collect_experience_from_expert_policy(
        expert_policy,
        rng=expert_key,
        env_args=env_args,
        mode=mode,
        n_timesteps=cloning_args.pre_train_n_steps,
    )
    jax.clear_caches()
    actor_state, critic_state, metrics = pre_train(
        rng=expert_key,
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        dataset=dataset,
        actor_lr=actor_optimizer_args.learning_rate,
        critic_lr=critic_optimizer_args.learning_rate,
        actor_epochs=cloning_args.actor_epochs,
        critic_epochs=cloning_args.critic_epochs,
        actor_batch_size=cloning_args.actor_batch_size,
        critic_batch_size=cloning_args.critic_batch_size,
    )
    return agent_state.replace(actor_state=actor_state, critic_state=critic_state)


def get_cloning_args(
    cloning_args: Optional[CloningConfig], total_timesteps: int
) -> Tuple:
    imitation_coef = 0.0
    distance_to_stable = get_one
    imitation_coef_offset = 0.0
    pre_train_n_steps = 0
    action_scale = 1.0

    if cloning_args is not None:
        imitation_coef = get_imitation_coef(  # type: ignore[assignment]
            cloning_args=cloning_args, total_timesteps=total_timesteps
        )
        distance_to_stable = cloning_args.distance_to_stable or get_one
        imitation_coef_offset = cloning_args.imitation_coef_offset
        pre_train_n_steps = cloning_args.pre_train_n_steps
        action_scale = cloning_args.action_scale

    return {
        "imitation_coef": imitation_coef,
        "imitation_coef_offset": imitation_coef_offset,
        "distance_to_stable": distance_to_stable,
        "action_scale": action_scale,
    }, pre_train_n_steps


@partial(
    jax.jit,
    static_argnames=[
        "expert_policy",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
)
def compute_imitation_score(
    pi: distrax.Distribution,
    expert_policy: Callable[[jax.Array], float],
    raw_observations: jax.Array,
    distance_to_stable: Callable[[jax.Array], float],
    imitation_coef_offset: float,
) -> jax.Array:
    # imitation_loss = (
    #     -pi.log_prob(expert_policy(raw_observations)).sum(-1, keepdims=True)
    #     if expert_policy is not None
    #     else jnp.zeros(1)
    # )
    imitation_loss = jnp.mean(
        jnp.square(pi.mode() if isinstance(pi, distrax.Categorical) else pi.mean()),
        axis=-1,
        keepdims=True,
    )
    EPS = 1e-9

    distance = (
        (1 / (distance_to_stable(raw_observations) + EPS)) + imitation_coef_offset
    )  # small offset to prevent it going too low while avoiding max (which is conditional on the actual value) for performance
    distance = jnp.expand_dims(distance, -1)
    return distance * imitation_loss
