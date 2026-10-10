from typing import Dict, Sequence, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.linen.initializers import orthogonal
from flax.training import train_state
from jax.tree_util import Partial as partial

from ajax.environments.interaction import (
    collect_experience_from_expert_policy,
)
from ajax.environments.utils import check_env_is_gymnax


@struct.dataclass
class CloningConfig:
    actor_epochs: int = 10
    actor_lr: float = 1e-3
    actor_batch_size: int = 64
    pre_train_n_steps: int = int(1e5)
    skip_actor_pretrain: bool = False
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


def batchify(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reshape x into (num_batches, batch_size, ...) padding last batch if needed."""
    n = x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    pad = n_batches * batch_size - n
    if pad > 0:
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
    return x.reshape(n_batches, batch_size, *x.shape[1:])


def _reset_actor_heads(
    bc_actor_state: train_state.TrainState,
    rng: jax.Array,
    reset_log_std_after_bc: bool,
    reset_actor_head_after_bc: bool,
) -> train_state.TrainState:
    """Reset the log_std and/or mean head subtrees of an actor TrainState.

    Reset values mirror Actor.setup() in networks.py.
    """
    from flax.core import freeze, unfreeze

    mean_kernel_init_orig = orthogonal(0.01)

    def _reset_subtrees(d, rng_key):
        if not hasattr(d, "items") and not isinstance(d, dict):
            return d, rng_key
        out = {}
        for k, v in d.items():
            if (
                k == "log_std"
                and isinstance(v, dict)
                and (reset_log_std_after_bc or reset_actor_head_after_bc)
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

    new_params_dict = unfreeze(bc_actor_state.params)
    new_params_dict, _ = _reset_subtrees(new_params_dict, rng)
    return bc_actor_state.replace(params=freeze(new_params_dict))


@partial(
    jax.jit,
    static_argnames=[
        "actor_lr",
        "actor_epochs",
        "actor_batch_size",
        "skip_actor",
        "reset_log_std_after_bc",
        "reset_actor_head_after_bc",
        "augment_obs_with_expert_action",
        "bc_loss_type",
    ],
)
def pre_train(
    rng: jax.Array,
    actor_state: train_state.TrainState,
    dataset: Sequence,  # Sequence[Transition]
    actor_lr: float = 1e-3,
    actor_epochs: int = 10,
    actor_batch_size: int = 64,
    skip_actor: bool = False,
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
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Behaviour-clone the actor on a dataset of expert transitions.

    Returns the trained actor state, the per-epoch actor losses and the
    observation mean and standard deviation the actor was trained on.
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
    metrics = {"actor_loss": jnp.zeros((actor_epochs,))}

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
            # NLL with log_std lower-clipped at bc_min_log_std (prevents
            # entropy collapse on saturated samples, where scale → 0 would
            # otherwise dominate the loss).
            from ajax.agents.SAC.utils import SquashedNormal

            eps = bc_action_clip_eps
            if isinstance(pi, SquashedNormal):
                # Clip the action into the squash domain so tanh^{-1}
                # inside log_prob stays finite.
                target = jnp.clip(batch_actions, -1.0 + eps, 1.0 - eps)
                loc = pi.distribution.loc
                scale = jnp.maximum(pi.distribution.scale, jnp.exp(bc_min_log_std))
                pi_clipped = SquashedNormal(loc, scale)
                return -pi_clipped.log_prob(target).sum(-1, keepdims=True).mean()
            # Plain Normal (no squashing): no action clipping needed.
            loc = pi.loc
            scale = jnp.maximum(pi.scale, jnp.exp(bc_min_log_std))
            pi_clipped = distrax.Normal(loc, scale)
            return -pi_clipped.log_prob(batch_actions).sum(-1, keepdims=True).mean()
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
            rng, reset_key = jax.random.split(rng)
            bc_actor_state = _reset_actor_heads(
                bc_actor_state,
                reset_key,
                reset_log_std_after_bc,
                reset_actor_head_after_bc,
            )

    # Skip the write-back when actor pretraining was bypassed.
    if not skip_actor:
        actor_state = actor_state.replace(params=bc_actor_state.params)
    return actor_state, metrics, obs_mean, obs_std


def pretrain_on_expert(
    agent_state,
    key,
    cloning_args,
    expert_policy,
    env_args,
    actor_optimizer_args,
):
    """Behaviour-clone ``expert_policy`` when ``cloning_args`` asks for
    pre-training steps (:func:`get_pre_trained_agent`); else the state as is."""
    if cloning_args is None or cloning_args.pre_train_n_steps <= 0:
        return agent_state
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    return get_pre_trained_agent(
        agent_state,
        expert_policy,
        key,
        env_args,
        cloning_args,
        mode,
        actor_optimizer_args,
    )


def get_pre_trained_agent(
    agent_state,
    expert_policy,
    expert_key,
    env_args,
    cloning_args,
    mode,
    actor_optimizer_args,
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
    actor_state, _metrics, obs_mean, obs_std = pre_train(
        rng=expert_key,
        actor_state=agent_state.actor_state,
        dataset=dataset,
        actor_lr=actor_optimizer_args.learning_rate,
        actor_epochs=cloning_args.actor_epochs,
        actor_batch_size=cloning_args.actor_batch_size,
        skip_actor=cloning_args.skip_actor_pretrain,
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
    new_state = agent_state.replace(actor_state=actor_state)
    if agent_state.collector_state.obs_norm_info is not None:
        from ajax.wrappers import NormalizationInfo

        # Agent-side stats live with leading axis size 1 (see
        # init_agent_obs_norm); broadcast obs_mean / var to (1, obs_dim).
        n_samples = float(dataset.obs.reshape(-1, dataset.obs.shape[-1]).shape[0])
        var = obs_std**2
        mean_1 = obs_mean.reshape(1, -1)
        var_1 = var.reshape(1, -1)
        seeded = NormalizationInfo(
            count=jnp.full((1, 1), n_samples),
            mean=mean_1,
            mean_2=var_1 * n_samples,
            var=var_1,
            returns=None,
        )
        new_state = new_state.replace(
            collector_state=new_state.collector_state.replace(obs_norm_info=seeded),
            actor_state=new_state.actor_state.replace(obs_norm_info=seeded),
            critic_state=new_state.critic_state.replace(obs_norm_info=seeded),
        )
    return new_state
