"""Behaviour cloning of an actor on an expert's transitions before training
(pre_train, which CloningPretrain runs), on observations standardised as
the run will normalise them (bc_statistics, cloning_columns), and the
actor's output head set back or zeroed (zero_output_head)."""

from typing import Any, Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.tree_util import Partial as partial

from ajax.agents.update_steps import compute_gradients
from ajax.controller import feature_width, observation_layout
from ajax.environments.obs_norm import normaliser_std
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.observe import TrainingProgressInput
from ajax.modules.pid_actor import GAIN_NAMES
from ajax.networks.distributions import Deterministic, Squash, latent
from ajax.state import Transition, network_variables

#: The floor of the cloned Gaussian's log-std: a saturated sample's scale
#: cannot collapse to 0, where it would dominate the loss.
CLONING_MIN_LOG_STD = -1.0
#: How far inside (-1, 1) a squashed actor's cloned actions are clipped, so
#: that tanh^-1 in its log-probability stays finite.
CLONING_ACTION_CLIP = 1e-3


def batchify(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reshape x into (num_batches, batch_size, ...), every row in a batch,
    the last batch filled from the first rows: a zero row would teach the
    clone a false example."""
    n = x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    rows = jnp.arange(n_batches * batch_size) % n
    return x[rows].reshape(n_batches, batch_size, *x.shape[1:])


# The actor's log-std, whatever its form: Actor's head over the features
# (log_std) or state-independent parameter (log_std_param), or a
# PID actor's parameter (log_std).
_LOG_STD_NAMES = frozenset({"log_std", "log_std_param"})


def _is_output_head(names: list) -> bool:
    """Whether a parameter belongs to the actor's output head: Actor's
    mean layer, DeterministicActor's head, or a PID law's gains (fixed
    kp, ki and kd, or the head predicting them from the
    observation)."""
    if len(names) > 1 and names[-2] in ("mean", "head"):
        return True
    return "pid" in names and ("head" in names or names[-1] in GAIN_NAMES)


def _reset_actor_heads(cloned: Any, before: Any, reset_actor_head: bool) -> Any:
    """The cloned actor parameters with the log-std and, with
    reset_actor_head, the output head back at their values before
    cloning (the actor's initial values under CloningPretrain)."""

    def reset(path: Any, new: jax.Array, old: jax.Array) -> jax.Array:
        names = [getattr(k, "key", None) for k in path]
        if _LOG_STD_NAMES.intersection(names):
            return old
        return old if reset_actor_head and _is_output_head(names) else new

    return jax.tree_util.tree_map_with_path(reset, cloned, before)


def zero_output_head(params: Any) -> Any:
    """The actor's params with its output head at 0: it starts by outputting
    0 at every observation (the log-std stays, so it still explores)."""

    def zero(path: Any, leaf: jax.Array) -> jax.Array:
        names = [getattr(k, "key", None) for k in path]
        return jnp.zeros_like(leaf) if _is_output_head(names) else leaf

    return jax.tree_util.tree_map_with_path(zero, params)


@partial(
    jax.jit,
    static_argnames=[
        "actor_lr",
        "actor_epochs",
        "actor_batch_size",
        "skip_actor",
        "reset_log_std",
        "reset_actor_head",
        "raw_columns",
        "raw_features",
        "pinned",
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
    reset_log_std: bool = False,
    reset_actor_head: bool = False,
    raw_columns: int = 0,
    raw_features: int = 0,
    pinned: tuple[int, ...] = (),
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Behaviour-clone the actor on a dataset of expert transitions, its
    observations with the agent's features (raw_features trailing columns
    of them left raw, as the normaliser leaves them).

    Args:
        rng: The cloning's key.
        actor_state: The actor to clone.
        dataset: The expert's transitions: their observations and actions.
        actor_lr: Cloning's Adam learning rate.
        actor_epochs: The passes over the dataset.
        actor_batch_size: The rows of each step.
        skip_actor: Leave the actor as it is and return the statistics
            alone, e.g. to keep the policy free to deviate from a
            suboptimal expert.
        reset_log_std: Put the actor's log-std (head or parameter) back at
            its values before cloning, keeping the cloned mean.
        reset_actor_head: Also put the output head (the mean layer, or a PID
            actor's gains) back: only the encoder keeps what cloning taught
            it, features online RL can map to actions again quickly, with no
            sharp break in the policy (its outputs at their initial scale,
            its entropy not compressed).
        raw_columns: The env's observation columns left raw.
        raw_features: The agent's features left raw (cloning_columns).
        pinned: The columns whose statistics are 0 and 1 (cloning_columns).

    Returns the actor state, the per-epoch actor losses (zeros with
    skip_actor) and the observation mean and standard deviation the actor
    was trained on, over all but the raw features (0 and 1 on raw_columns,
    which stay raw, and on the pinned columns).
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
    # The statistics cover the observation, not its raw features (an
    # expert's action), which stay raw (apply_obs_norm, as at run time).
    span = obs.shape[-1] - raw_features
    obs_mean, obs_std = bc_statistics(obs[..., :span], raw_columns, pinned)
    obs = jnp.concatenate([(obs[..., :span] - obs_mean) / obs_std, obs[..., span:]], -1)
    if skip_actor:
        return (
            actor_state,
            {"actor_loss": jnp.zeros((actor_epochs,))},
            obs_mean,
            obs_std,
        )

    bc_actor_state = train_state.TrainState.create(
        apply_fn=actor_state.apply_fn,
        params=actor_state.params,
        tx=optax.adam(actor_lr),
    )

    def actor_loss_fn(params, batch_obs, batch_actions):
        pi = bc_actor_state.apply_fn(network_variables(actor_state, params), batch_obs)
        if isinstance(pi, Deterministic):
            # A deterministic actor (TD3's) has no distribution to fit, nor
            # a pre-tanh mean: either loss regresses its action on the
            # expert's (its log_prob is minus the squared error).
            return -pi.log_prob(batch_actions).sum(-1, keepdims=True).mean()
        # The negative log-likelihood, the log-std floored.
        base = latent(pi)
        floored = distrax.Normal(
            base.loc, jnp.maximum(base.scale, jnp.exp(CLONING_MIN_LOG_STD))
        )
        if isinstance(pi, Squash):
            # Clip the action into the squash domain so tanh^{-1} inside
            # log_prob stays finite.
            target = jnp.clip(
                batch_actions, -1.0 + CLONING_ACTION_CLIP, 1.0 - CLONING_ACTION_CLIP
            )
            return -Squash(floored).log_prob(target).sum(-1, keepdims=True).mean()
        # Plain Normal (no squashing): no action clipping needed.
        return -floored.log_prob(batch_actions).sum(-1, keepdims=True).mean()

    def actor_train_step(state, batch):
        batch_obs, batch_actions = batch

        def loss(params):
            value = actor_loss_fn(params, batch_obs, batch_actions)
            return value, value

        grads, value = compute_gradients(loss, state.params)
        return state.apply_gradients(grads=grads), value

    def actor_epoch_step(state, epoch_key):
        perm = jax.random.permutation(epoch_key, obs.shape[0])
        obs_batches = batchify(obs[perm], actor_batch_size)
        act_batches = batchify(actions[perm], actor_batch_size)
        state, batch_losses = jax.lax.scan(
            actor_train_step, state, (obs_batches, act_batches)
        )
        return state, jnp.mean(batch_losses)

    # The second of two keys, the stream cloning has always drawn.
    _, actor_key = jax.random.split(rng)
    epoch_keys = jax.random.split(actor_key, actor_epochs)
    bc_actor_state, actor_losses = jax.lax.scan(
        actor_epoch_step, bc_actor_state, epoch_keys
    )
    params = bc_actor_state.params
    if reset_log_std or reset_actor_head:
        params = _reset_actor_heads(params, actor_state.params, reset_actor_head)
    return (
        actor_state.replace(params=params),
        {"actor_loss": actor_losses},
        obs_mean,
        obs_std,
    )


def bc_statistics(
    obs: jnp.ndarray, raw_columns: int = 0, pinned: tuple[int, ...] = ()
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The mean and standard deviation cloning standardises its observations
    with, the run's (normaliser_std): 0 and 1 on the env's raw_columns
    and on the pinned ones."""
    obs_flat = obs.reshape(-1, obs.shape[-1])
    mean = obs_flat.mean(axis=0).at[:raw_columns].set(0.0)
    std = normaliser_std(obs_flat.var(axis=0)).at[:raw_columns].set(1.0)
    for column in pinned:
        mean = mean.at[column].set(0.0)
        std = std.at[column].set(1.0)
    return mean, std


def cloning_columns(env_args: Any) -> dict:
    """pre_train's view of the agent's observation features: how many
    raw ones end it, and the training progress's columns, whose statistics
    are pinned (constant 0 before training, its std would be the floor's)."""
    env_width = get_state_action_shapes(env_args.env)[0][-1]
    layout = observation_layout(env_width, env_args.observers, env_args)
    pinned = tuple(
        column
        for observer in env_args.observers
        if isinstance(observer, TrainingProgressInput)
        for column in range(*layout.slice(observer.name).indices(layout.width))
    )
    return {"raw_features": feature_width(env_args, normalised=False), "pinned": pinned}
