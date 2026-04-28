from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
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
    # Per-component pretrain switches. Default keeps backwards-compat:
    # actor BC enabled, critic MC disabled (matches the legacy behaviour).
    skip_actor_pretrain: bool = False
    skip_critic_pretrain: bool = True
    # When True, reset only the log_std head after BC training (preserves
    # the BC'd mean, restores entropy). See pre_train docstring.
    reset_log_std_after_bc: bool = False
    # When True, reset BOTH mean and log_std heads after BC, keeping only
    # the encoder BC-warmed. See pre_train docstring.
    reset_actor_head_after_bc: bool = False
    # BC actor loss: "mse" (legacy: MSE on unsquashed mean targeting
    # atanh(clipped expert)) or "nll" (NLL on SquashedNormal with action
    # clipped to (-1+eps, 1-eps) and log_std lower-clipped). NLL is the
    # principled choice; MSE was a numerical-stability fallback that
    # collapses on heavily-saturated PID experts (mu drifts toward
    # majority-sign atanh(±0.999) ≈ ±3.8 and minority samples never
    # recover, see diag_bc_quality_v2 results 2026-04-26).
    bc_loss_type: str = "nll"
    bc_min_log_std: float = -1.0
    bc_action_clip_eps: float = 1e-3


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
        "skip_actor",
        "skip_critic",
        "gamma",
        "reset_log_std_after_bc",
        "reset_actor_head_after_bc",
        "augment_obs_with_expert_action",
        "bc_loss_type",
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
    # Pretrain mode controls
    skip_actor: bool = False,
    skip_critic: bool = True,  # Default-off for backward compat with SAC users
    gamma: float = 0.99,
    # If True, after BC training the SAC actor's mean (and encoder), reset
    # the ``log_std`` head's params to its init values (zeros kernel,
    # constant -1.0 bias → std ≈ 0.37). Preserves the BC'd mean.
    reset_log_std_after_bc: bool = False,
    # If True, after BC reset BOTH ``mean`` and ``log_std`` head subtrees
    # to fresh init. Keeps only the encoder (trunk) BC-warmed: trunk
    # features encode expert-relevant state representations, but the
    # action map is random. Online RL re-learns the head fast on top of
    # informative features without any sharp policy break (since the
    # actor outputs are random-init scale, no entropy compression).
    reset_actor_head_after_bc: bool = False,
    augment_obs_with_expert_action: bool = False,
    bc_loss_type: str = "nll",
    bc_min_log_std: float = -1.0,
    bc_action_clip_eps: float = 1e-3,
) -> Tuple[train_state.TrainState, train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Pre-train actor (behavioral cloning) and critic (TD(0)) from a dataset of transitions.
    Returns trained states and metrics dict with per-epoch actor/critic losses.
    """
    obs = dataset.obs
    actions = dataset.action
    if augment_obs_with_expert_action:
        # Match the training-time obs format: [env_obs, expert_action].
        # dataset.actions IS the expert action for each obs (the dataset
        # was collected by running the expert).
        obs = jnp.concatenate([obs, actions], axis=-1)
    # Standardize BC obs (only) for training: raw obs has wildly
    # varying scale (env state + flattened PID integrators can span 6
    # orders of magnitude), which leaves the encoder's first layer
    # with mostly-saturated / dead ReLUs and BC mode-collapses near the
    # marginal action mean. The actor is then trained to expect
    # standardised inputs; the caller seeds the agent's runtime
    # obs_norm_info with these same stats so get_pi / predict_value
    # apply matching normalisation online.
    obs_flat = obs.reshape(-1, obs.shape[-1])
    obs_mean = obs_flat.mean(axis=0)
    obs_std = obs_flat.std(axis=0) + 1e-6
    obs = (obs - obs_mean) / obs_std
    rewards = dataset.reward
    terminated = dataset.terminated
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
        if bc_loss_type == "nll":
            # NLL on the SquashedNormal with action clipped to (-1+eps, 1-eps)
            # and log_std lower-clipped at bc_min_log_std. Clip on the action
            # keeps tanh^{-1} finite inside log_prob; the log_std clip
            # prevents entropy collapse on saturated samples (which would
            # otherwise let scale → 0 and dominate the loss).
            from ajax.agents.SAC.utils import SquashedNormal
            eps = bc_action_clip_eps
            target = jnp.clip(batch_actions, -1.0 + eps, 1.0 - eps)
            loc = pi.distribution.loc
            scale = pi.distribution.scale
            scale = jnp.maximum(scale, jnp.exp(bc_min_log_std))
            pi_clipped = SquashedNormal(loc, scale)
            return -pi_clipped.log_prob(target).sum(-1, keepdims=True).mean()
        # Legacy MSE on unsquashed mean targeting atanh(clipped expert).
        # Collapses on heavily-saturated PID experts (see CloningConfig
        # docstring). Kept as a fallback.
        target = jnp.arctanh(jnp.clip(batch_actions, -0.999, 0.999))
        if hasattr(pi, "unsquashed_mean"):
            mu = pi.unsquashed_mean()
        else:
            mu = pi.mean()
        return jnp.square(mu - target).sum(-1, keepdims=True).mean()

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
    if skip_actor:
        # Actor pretraining bypassed (e.g., to keep policy free to deviate
        # from a suboptimal expert). Leave actor_state untouched.
        bc_actor_state = actor_state
    else:
        rng_epochs = jax.random.split(rng_actor, actor_epochs)
        bc_actor_state, actor_losses = jax.lax.scan(
            actor_epoch_step, bc_actor_state, rng_epochs
        )
        metrics["actor_loss"] = actor_losses
        # Optional: reset head subtree(s) of the actor after BC training.
        # ``reset_log_std_after_bc``: reset only log_std (preserves BC'd mean)
        # ``reset_actor_head_after_bc``: reset BOTH mean and log_std (keeps
        #   only the encoder BC-warmed; head is random → natural entropy
        #   → α stays sane; trunk features encode expert-relevant info)
        if reset_log_std_after_bc or reset_actor_head_after_bc:
            from flax.core import freeze, unfreeze
            # Reset values mirror Actor.setup() in networks.py.
            mean_kernel_init_orig = orthogonal(0.01)
            mean_bias_init_orig = constant(0.0)
            new_params = bc_actor_state.params

            def _reset_subtrees(d, rng_key):
                if not hasattr(d, "items") and not isinstance(d, dict):
                    return d, rng_key
                out = {}
                for k, v in d.items():
                    if k == "log_std" and isinstance(v, dict) and (
                        reset_log_std_after_bc or reset_actor_head_after_bc
                    ):
                        new_sub = {}
                        for sub_k, sub_v in v.items():
                            if sub_k == "kernel":
                                new_sub[sub_k] = jnp.zeros_like(sub_v)
                            elif sub_k == "bias":
                                new_sub[sub_k] = jnp.full_like(sub_v, -1.0)
                            else:
                                new_sub[sub_k] = sub_v
                        out[k] = new_sub
                    elif k == "mean" and isinstance(v, dict) and reset_actor_head_after_bc:
                        new_sub = {}
                        for sub_k, sub_v in v.items():
                            if sub_k == "kernel":
                                rng_key, subkey = jax.random.split(rng_key)
                                new_sub[sub_k] = mean_kernel_init_orig(
                                    subkey, sub_v.shape, sub_v.dtype
                                )
                            elif sub_k == "bias":
                                new_sub[sub_k] = jnp.zeros_like(sub_v)
                            else:
                                new_sub[sub_k] = sub_v
                        out[k] = new_sub
                    elif isinstance(v, dict):
                        out[k], rng_key = _reset_subtrees(v, rng_key)
                    else:
                        out[k] = v
                return out, rng_key

            rng, reset_key = jax.random.split(rng)
            new_params_dict = unfreeze(new_params)
            new_params_dict, _ = _reset_subtrees(new_params_dict, reset_key)
            bc_actor_state = bc_actor_state.replace(
                params=freeze(new_params_dict)
            )

    # --------------------------
    # Critic pre-training (Monte Carlo, Q(s, a) -> G_t)
    # --------------------------
    # Each critic in the ensemble fits the same MC return-to-go target
    # G_t = sum_{k>=t} gamma^(k-t) * r_k, with the sum bounded at episode
    # termination (terminated[t] resets the carry). This yields a calibrated
    # initial Q on the expert distribution; both critics share the targets so
    # their *disagreement* is small on expert (s, a) and large elsewhere — the
    # property the quality-aware action pipeline relies on.
    if not skip_critic:
        bc_critic_state = train_state.TrainState.create(
            apply_fn=critic_state.apply_fn,
            params=critic_state.params,
            tx=optax.adam(critic_lr),
        )

        # Compute returns-to-go via reverse scan. dataset shapes are
        # (n_timesteps, n_envs, ...); reduce by leading axis.
        rew_flat = rewards  # (T, n_envs) or (T, n_envs, 1)
        term_flat = terminated.astype(jnp.float32)
        if rew_flat.ndim == term_flat.ndim:
            pass  # ok
        elif rew_flat.ndim == term_flat.ndim + 1 and rew_flat.shape[-1] == 1:
            rew_flat = rew_flat.squeeze(-1)
        elif term_flat.ndim == rew_flat.ndim + 1 and term_flat.shape[-1] == 1:
            term_flat = term_flat.squeeze(-1)

        def _backward_step(carry_G, x):
            r, term = x
            G = r + gamma * carry_G * (1.0 - term)
            return G, G

        _, returns_to_go = jax.lax.scan(
            _backward_step,
            jnp.zeros(rew_flat.shape[1:]),
            (rew_flat[::-1], term_flat[::-1]),
        )
        returns_to_go = returns_to_go[::-1]  # (T, n_envs)

        # Flatten across (T, n_envs) so all transitions are independent samples
        flat_obs = obs.reshape((-1,) + obs.shape[2:])
        flat_act = actions.reshape((-1,) + actions.shape[2:])
        flat_G = returns_to_go.reshape((-1,))

        def critic_loss_fn(params, batch_obs, batch_actions, batch_G):
            x = jnp.concatenate(
                [batch_obs, jax.lax.stop_gradient(batch_actions)], axis=-1
            )
            q_preds = bc_critic_state.apply_fn(params, x)
            # q_preds: (num_critics, batch, 1); target broadcast to same shape
            target = batch_G[None, :, None]
            return jnp.mean((q_preds - target) ** 2)

        def critic_train_step(state, batch_obs, batch_actions, batch_G):
            loss, grads = jax.value_and_grad(critic_loss_fn)(
                state.params, batch_obs, batch_actions, batch_G
            )
            return state.apply_gradients(grads=grads), loss

        def critic_epoch_step(carry, rng_epoch):
            state = carry
            perm = jax.random.permutation(rng_epoch, flat_obs.shape[0])
            obs_shuffled = flat_obs[perm]
            act_shuffled = flat_act[perm]
            G_shuffled = flat_G[perm]

            obs_batches = batchify(obs_shuffled, critic_batch_size)
            act_batches = batchify(act_shuffled, critic_batch_size)
            G_batches = batchify(G_shuffled, critic_batch_size)

            def batch_step(carry, batch):
                state = carry
                b_obs, b_act, b_G = batch
                new_state, loss = critic_train_step(state, b_obs, b_act, b_G)
                return new_state, loss

            state, batch_losses = jax.lax.scan(
                batch_step,
                state,
                (obs_batches, act_batches, G_batches),
            )
            return state, jnp.mean(batch_losses)

        rng, rng_critic = jax.random.split(rng)
        rng_epochs = jax.random.split(rng_critic, critic_epochs)
        bc_critic_state, critic_losses = jax.lax.scan(
            critic_epoch_step, bc_critic_state, rng_epochs
        )
        metrics["critic_loss"] = critic_losses
    else:
        bc_critic_state = critic_state

    # Update original states with trained parameters. Skip the actor write-
    # back when actor pretraining was bypassed; same for critic.
    if not skip_actor:
        actor_state = actor_state.replace(params=bc_actor_state.params)
    if not skip_critic:
        # Sync target_params too if the critic state has them (TD3, SAC).
        new_critic_params = bc_critic_state.params
        if hasattr(critic_state, "target_params") and critic_state.target_params is not None:
            critic_state = critic_state.replace(
                params=new_critic_params, target_params=new_critic_params
            )
        else:
            critic_state = critic_state.replace(params=new_critic_params)
    return actor_state, critic_state, metrics, obs_mean, obs_std


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
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
):
    # dataset is examples of observations and actions taken
    dataset = collect_experience_from_expert_policy(
        expert_policy,
        rng=expert_key,
        env_args=env_args,
        mode=mode,
        n_timesteps=cloning_args.pre_train_n_steps,
        augment_obs_with_expert_state=augment_obs_with_expert_state,
    )
    jax.clear_caches()
    actor_state, critic_state, metrics, obs_mean, obs_std = pre_train(
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
        skip_actor=cloning_args.skip_actor_pretrain,
        skip_critic=cloning_args.skip_critic_pretrain,
        gamma=getattr(agent_config, "gamma", 0.99),
        reset_log_std_after_bc=cloning_args.reset_log_std_after_bc,
        reset_actor_head_after_bc=cloning_args.reset_actor_head_after_bc,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        bc_loss_type=cloning_args.bc_loss_type,
        bc_min_log_std=cloning_args.bc_min_log_std,
        bc_action_clip_eps=cloning_args.bc_action_clip_eps,
    )
    # Seed the agent's running obs_norm_info with the BC dataset stats.
    # The actor / critic params have just been trained on standardised
    # inputs; the runtime get_pi / predict_value will apply the same
    # standardisation via obs_norm_info, so the actor sees a consistent
    # input distribution across BC and online. Online collection then
    # continues to update these stats from this seeded baseline.
    new_state = agent_state.replace(actor_state=actor_state, critic_state=critic_state)
    if agent_state.collector_state.obs_norm_info is not None:
        from ajax.wrappers import NormalizationInfo
        n_envs = agent_state.collector_state.obs_norm_info.mean.shape[0]
        n_samples = float(dataset.obs.reshape(-1, dataset.obs.shape[-1]).shape[0])
        var = obs_std**2
        seeded = NormalizationInfo(
            count=jnp.full((n_envs, 1), n_samples),
            mean=jnp.broadcast_to(obs_mean, (n_envs, obs_mean.shape[0])),
            mean_2=jnp.broadcast_to(var * n_samples, (n_envs, var.shape[0])),
            var=jnp.broadcast_to(var, (n_envs, var.shape[0])),
            returns=None,
        )
        new_state = new_state.replace(
            collector_state=new_state.collector_state.replace(obs_norm_info=seeded),
            actor_state=new_state.actor_state.replace(obs_norm_info=seeded),
            critic_state=new_state.critic_state.replace(obs_norm_info=seeded),
        )
    return new_state


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
    expert_policy: Optional[Callable],
    raw_observations: jax.Array,
    distance_to_stable: Callable,
    imitation_coef_offset: float,
    q_preds: Optional[jax.Array] = None,
    q_expert: Optional[jax.Array] = None,
) -> jax.Array:
    if isinstance(pi, distrax.Categorical) or expert_policy is None:
        return jnp.zeros(1)

    expert_action = jax.lax.stop_gradient(expert_policy(raw_observations))
    mse = jnp.mean(
        jnp.square(pi.mean() - expert_action) / 4.0, axis=-1, keepdims=True
    )  # pure distance, no Q-weighting
    return mse
