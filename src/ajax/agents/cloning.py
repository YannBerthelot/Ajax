from typing import Any, Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.tree_util import Partial as partial

from ajax.environments.interaction import augment_obs
from ajax.modules.pid_actor import GAIN_NAMES
from ajax.networks.distributions import Squash, latent
from ajax.state import Transition


def batchify(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reshape x into (num_batches, batch_size, ...) padding last batch if needed."""
    n = x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    pad = n_batches * batch_size - n
    if pad > 0:
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
    return x.reshape(n_batches, batch_size, *x.shape[1:])


# The actor's log-std, whatever its form: Actor's head over the features
# (``log_std``) or state-independent parameter (``log_std_param``), or a
# PID actor's parameter (``log_std``).
_LOG_STD_NAMES = frozenset({"log_std", "log_std_param"})


def _is_output_head(names: list) -> bool:
    """Whether a parameter belongs to the actor's output head: Actor's
    ``mean`` layer, or a PID law's gains (fixed ``kp``, ``ki`` and ``kd``,
    or the ``head`` predicting them from the observation)."""
    if len(names) > 1 and names[-2] == "mean":
        return True
    return "pid" in names and ("head" in names or names[-1] in GAIN_NAMES)


def _reset_actor_heads(cloned: Any, before: Any, reset_actor_head: bool) -> Any:
    """The ``cloned`` actor parameters with the log-std and, with
    ``reset_actor_head``, the output head back at their values ``before``
    cloning (the actor's initial values under CloningPretrain)."""

    def reset(path: Any, new: jax.Array, old: jax.Array) -> jax.Array:
        names = [getattr(k, "key", None) for k in path]
        if _LOG_STD_NAMES.intersection(names):
            return old
        return old if reset_actor_head and _is_output_head(names) else new

    return jax.tree_util.tree_map_with_path(reset, cloned, before)


@partial(
    jax.jit,
    static_argnames=[
        "actor_lr",
        "actor_epochs",
        "actor_batch_size",
        "skip_actor",
        "reset_log_std",
        "reset_actor_head",
        "augment_obs_with_expert_action",
        "raw_columns",
    ],
)
def pre_train(
    rng: jax.Array,
    actor_state: train_state.TrainState,
    dataset: Transition,
    actor_lr: float = 1e-3,
    actor_epochs: int = 10,
    actor_batch_size: int = 64,
    skip_actor: bool = False,
    # If True, after BC training the actor's mean (and encoder), put its
    # log-std back at its values before cloning (head or parameter).
    # Preserves the BC'd mean.
    reset_log_std: bool = False,
    # If True, after BC also put the output head (the ``mean`` layer, or a
    # PID actor's gains) back at its values before cloning. Keeps only the
    # encoder (trunk) BC-warmed: trunk
    # features encode expert-relevant state representations, but the
    # action map is random. Online RL re-learns the head fast on top of
    # informative features without any sharp policy break (since the
    # actor outputs are random-init scale, no entropy compression).
    reset_actor_head: bool = False,
    augment_obs_with_expert_action: bool = False,
    bc_min_log_std: float = -1.0,
    bc_action_clip_eps: float = 1e-3,
    raw_columns: int = 0,
    train_frac_column: Optional[int] = None,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Behaviour-clone the actor on a dataset of expert transitions.

    Returns the trained actor state, the per-epoch actor losses and the
    observation mean and standard deviation the actor was trained on (0
    and 1 on the env's ``raw_columns``, which stay raw, and on the
    training-fraction column ``train_frac_column``: constant 0 before
    training, its standard deviation would be 1e-6).
    """
    obs = dataset.obs
    actions = dataset.action
    # Standardize BC obs (only) for training: raw obs has wildly
    # varying scale (env state + flattened PID integrators can span 6
    # orders of magnitude), which leaves the encoder's first layer
    # with mostly-saturated / dead ReLUs and BC mode-collapses near the
    # marginal action mean. The actor is then trained to expect
    # standardised inputs; the caller seeds the agent's runtime
    # obs_norm_info with these same stats so get_pi / predict_value
    # apply matching normalisation online.
    # The statistics cover the observation, not the expert action appended
    # after it, which stays raw (apply_obs_norm, as at run time).
    obs_mean, obs_std = bc_statistics(obs, raw_columns, train_frac_column)
    obs = (obs - obs_mean) / obs_std
    if augment_obs_with_expert_action:
        # The dataset's actions are the expert's (it ran the expert).
        obs = augment_obs(obs, expert_action=actions)
    metrics = {"actor_loss": jnp.zeros((actor_epochs,))}

    bc_actor_state = train_state.TrainState.create(
        apply_fn=actor_state.apply_fn,
        params=actor_state.params,
        tx=optax.adam(actor_lr),
    )

    def actor_loss_fn(params, batch_obs, batch_actions):
        pi = bc_actor_state.apply_fn(params, batch_obs)
        from ajax.agents.TD3.networks import Deterministic

        if isinstance(pi, Deterministic):
            # A deterministic actor (TD3's) has no distribution to fit, nor
            # a pre-tanh mean: either loss regresses its action on the
            # expert's (its log_prob is minus the squared error).
            return -pi.log_prob(batch_actions).sum(-1, keepdims=True).mean()
        # NLL with log_std lower-clipped at bc_min_log_std (prevents entropy
        # collapse on saturated samples, where scale → 0 would otherwise
        # dominate the loss).
        eps = bc_action_clip_eps
        base = latent(pi)
        floored = distrax.Normal(
            base.loc, jnp.maximum(base.scale, jnp.exp(bc_min_log_std))
        )
        if isinstance(pi, Squash):
            # Clip the action into the squash domain so tanh^{-1} inside
            # log_prob stays finite.
            target = jnp.clip(batch_actions, -1.0 + eps, 1.0 - eps)
            return -Squash(floored).log_prob(target).sum(-1, keepdims=True).mean()
        # Plain Normal (no squashing): no action clipping needed.
        return -floored.log_prob(batch_actions).sum(-1, keepdims=True).mean()

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

    _, rng_actor = jax.random.split(rng)
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
        if reset_log_std or reset_actor_head:
            bc_actor_state = bc_actor_state.replace(
                params=_reset_actor_heads(
                    bc_actor_state.params, actor_state.params, reset_actor_head
                )
            )

    if not skip_actor:
        actor_state = actor_state.replace(params=bc_actor_state.params)
    return actor_state, metrics, obs_mean, obs_std


def bc_statistics(
    obs: jnp.ndarray, raw_columns: int = 0, train_frac_column: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The mean and standard deviation cloning standardises its observations
    with: 0 and 1 on the env's ``raw_columns`` and on the training-fraction
    column (constant 0 before training: its std would be 1e-6)."""
    obs_flat = obs.reshape(-1, obs.shape[-1])
    mean = obs_flat.mean(axis=0).at[:raw_columns].set(0.0)
    std = (obs_flat.std(axis=0) + 1e-6).at[:raw_columns].set(1.0)
    if train_frac_column is not None:
        mean = mean.at[train_frac_column].set(0.0)
        std = std.at[train_frac_column].set(1.0)
    return mean, std
