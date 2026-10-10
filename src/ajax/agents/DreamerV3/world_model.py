"""The DreamerV3 world-model loss on a replay batch with its context step.

Algorithm E of ``docs/world_models/dreamerv3_spec.md`` restricted to the
world-model terms, fed as in Algorithm J (replay context ``K = 1``,
dreamerv3_spec 5.8): the carry is the posterior latent stored at index 0,
observations and flags are indices ``1..T`` and the previous actions are
``action[0:T]``, the upstream fix ``29eb964`` of the paper-era code
``2411f7d`` (``29eb964:dreamerv3/agent.py:172-183``, ``:225-249``,
``:392-394``; ``docs/world_models/deviations.md`` section 1).

:func:`world_model_loss` returns every term per step ``[B, T]``, unscaled and
reduced over feature dimensions only, as the reference's ``losses`` dict
before scaling; :func:`ajax.agents.DreamerV3.learner.compute_loss` adds the
actor-critic terms and applies ``sum_k scale_k mean(loss_k)`` over all of
them. The world-model part of that sum is returned too (``weighted``), so the
world model can be tested on its own. Parity with the reference: ``tests/agents/DreamerV3/
test_dreamerv3_parity.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ajax.agents.DreamerV3.distributions import (
    OneHot,
    bernoulli_loss,
    draw_onehot_noise,
    symlog_mse,
)
from ajax.agents.DreamerV3.networks import (
    RSSM,
    RSSMFeatures,
    RSSMState,
    WorldModel,
    features,
    observe,
)
from ajax.agents.DreamerV3.state import DreamerV3Config

#: The world-model terms, in the reference's order (``agent.py:237-247``):
#: reconstruction (one term for the flat vector observation, deviation D5),
#: reward, continue, dynamics and representation.
LOSS_TERMS = ("rec", "rew", "con", "dyn", "rep")

sg = jax.lax.stop_gradient


class ReplayContextBatch(NamedTuple):
    """A replay batch of ``T + 1`` rows whose first is context, and the
    posterior latent stored with that context row.

    Rows follow the replay convention (dreamerv3_spec 2.15): ``obs_t``, the
    ``reward_t`` received on entering it (0 at ``is_first``), the flags and
    the action chosen after seeing ``obs_t`` (zeroed at ``is_last``). The
    loss is computed for rows ``1..T``; row 0 contributes its action (the
    first previous action) and, through ``context_deter`` / ``context_stoch``,
    the carry. The reference pops the stored latents off the batch and keeps
    the context's alone (``29eb964:dreamerv3/agent.py:172-180``,
    ``data.pop(k)[:, :K]`` with ``K = 1``), so the replay gathers them at
    the window start only.

    Attributes:
        obs: ``[B, T + 1, O]`` float vector observations.
        action: ``[B, T + 1, A]`` encoded actions
            (:func:`ajax.agents.DreamerV3.networks.encode_action`).
        reward: ``[B, T + 1]``.
        is_first, is_last, is_terminal: ``[B, T + 1]`` bool.
        context_deter: ``[B, D]`` the ``h`` stored with row 0.
        context_stoch: ``[B, S]`` the class indices of the ``z`` stored with
            row 0.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    context_deter: jax.Array
    context_stoch: jax.Array


class PosteriorEntries(NamedTuple):
    """Fresh posterior latents of the trained steps, for the replay write-back.

    ``deter [B, T, D]`` float32 and ``stoch [B, T, S]`` int32 class indices of
    the sampled ``z`` (dreamerv3_spec 5.7, 5.9; ``agent.py:197-202``).
    """

    deter: jax.Array
    stoch: jax.Array


class WorldModelOutput(NamedTuple):
    """Everything the world-model forward pass produces.

    Attributes:
        weighted: ``sum_k scale_k * mean(loss_k)`` over :data:`LOSS_TERMS`
            (``agent.py:393-394`` restricted to the world model).
        losses: per-step unscaled terms ``[B, T]`` keyed by
            :data:`LOSS_TERMS`; ``dyn`` and ``rep`` include the free bits.
        kl: ``KL(posterior || prior)`` ``[B, T]`` before free bits, summed
            over latents (the forward value of both KL terms; no gradient).
        entries: posterior latents for the replay write-back.
        posterior: per-step posterior features ``[B, T, ...]`` with gradients
            (straight-through ``stoch``), for the actor-critic: imagination
            starts and the replay critic, which trains the world model
            through them (``agent.py:331-333``, ``replay_critic_grad:
            True``; dreamerv3_spec 2.16).
        prior_logits: raw prior logits ``[B, T, S, C]`` from the same ``h_t``.
        tokens: encoder outputs ``[B, T, units]``.

    There is no final carry: the reference threads one between train calls
    only for ``consec_train > 1``; with ``consec_train = 1`` the replay
    context always overrides it (dreamerv3_spec 5.10), and imagination
    starts from every posterior step (``imag_start: all``).
    """

    weighted: jax.Array
    losses: dict[str, jax.Array]
    kl: jax.Array
    entries: PosteriorEntries
    posterior: RSSMFeatures
    prior_logits: jax.Array
    tokens: jax.Array


def kl_losses(
    post_logits: jax.Array, prior_logits: jax.Array, unimix: float, free_nats: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dynamics and representation losses with free bits (dreamerv3_spec 2.14).

    From raw logits ``[..., S, C]``: ``dyn = max(KL(sg(post) || prior),
    free_nats)`` trains the prior (and whatever produced its input), ``rep =
    max(KL(post || sg(prior)), free_nats)`` the posterior. Both are KL(post
    || prior), the uniform mixture is applied after the stop-gradient, and the
    clamp acts on the KL summed over latents, per element of ``[...]``
    (``29eb964:dreamerv3/nets.py:97-105``). As there, ``free_nats = 0`` skips
    the clamp (``if free:``), so a KL that float32 rounds below 0 keeps its
    value and gradient. ``free_nats`` is a static Python float (a
    :class:`DreamerV3Config` field), so this is a trace-time branch. Returns
    ``(dyn, rep, kl)`` with ``kl`` the unclamped KL, without gradient.
    """
    dyn_kl = OneHot.from_logits(sg(post_logits), unimix).kl(
        OneHot.from_logits(prior_logits, unimix)
    )
    rep_kl = OneHot.from_logits(post_logits, unimix).kl(
        OneHot.from_logits(sg(prior_logits), unimix)
    )
    dyn, rep = dyn_kl, rep_kl
    if free_nats:
        dyn, rep = jnp.maximum(dyn_kl, free_nats), jnp.maximum(rep_kl, free_nats)
    return dyn, rep, sg(dyn_kl)


def draw_posterior_noise(
    key: jax.Array, config: DreamerV3Config, batch_shape: tuple[int, ...]
) -> jax.Array:
    """Gumbel noise ``[*batch_shape, S, C]`` for the posterior samples."""
    return draw_onehot_noise(key, (*batch_shape, config.stoch, config.classes))


def world_model_loss(
    model: WorldModel, params: dict, batch: ReplayContextBatch, noise: jax.Array
) -> WorldModelOutput:
    """World-model losses of one replay batch (Algorithms E and J).

    Args:
        model: the :class:`WorldModel` (its ``config`` holds the scales).
        params: its parameters.
        batch: ``[B, T + 1]`` replay rows, index 0 being context, and the
            context's stored latent.
        noise: ``[B, T, S, C]`` Gumbel noise of the posterior samples
            (:func:`draw_posterior_noise`).

    The terms (dreamerv3_spec 2.11-2.14), all ``[B, T]``:

    * ``rec``: symlog squared error of the decoder, summed over the
      observation dims, with the 2411f7d tolerance;
    * ``rew``: two-hot cross-entropy of the raw reward
      (:meth:`ajax.distributional.TwoHot.dreamerv3`);
    * ``con``: logistic loss of the soft label ``(1 - 1 / return_horizon) *
      (1 - is_terminal)`` (time-limit truncations keep 0.997);
    * ``dyn = max(KL(sg(post) || prior), free_nats)`` and ``rep =
      max(KL(post || sg(prior)), free_nats)`` (:func:`kl_losses`); the prior
      is computed from the same ``h_t`` as the posterior, which is not
      stop-gradiented, so ``dyn`` also trains the core and everything
      upstream (dreamerv3_spec 2.14).

    Gradient routing (dreamerv3_spec 2.16): the decoder, reward and continue
    heads read ``concat(h_t, z_t)`` without a stop-gradient (``reward_grad``),
    ``z_t`` carries straight-through gradients, BPTT runs through the whole
    scan, and the context carry is data (no gradient). Losses at
    ``is_first`` steps are not masked.
    """
    config = model.config
    apply = model.apply
    variables = {"params": params}
    rssm = RSSM(config)

    # Replay context (29eb964 agent.py:172-183): carry from the latent stored
    # with row 0; observations and flags 1..T; previous actions 0..T-1.
    carry = jax.lax.stop_gradient(
        RSSMState(
            deter=jnp.asarray(batch.context_deter, jnp.float32),
            stoch=jax.nn.one_hot(
                batch.context_stoch, config.classes, dtype=jnp.float32
            ),
        )
    )
    obs = batch.obs[:, 1:]
    prevact = batch.action[:, :-1]
    is_first = batch.is_first[:, 1:]

    tokens = apply(variables, obs, method=WorldModel.encode)
    _, post = observe(rssm, params["rssm"], carry, tokens, prevact, is_first, noise)
    prior_logits = rssm.apply(
        {"params": params["rssm"]}, post.deter, method=RSSM.prior_logits
    )

    dyn, rep, kl = kl_losses(post.logits, prior_logits, config.unimix, config.free_nats)

    feat = features(post.deter, post.stoch)
    recon = apply(variables, feat, method=WorldModel.decode)
    reward_logits = apply(variables, feat, method=WorldModel.reward_logits)
    cont_logit = apply(variables, feat, method=WorldModel.cont_logit)
    cont_target = (1 - batch.is_terminal[:, 1:].astype(jnp.float32)) * config.gamma
    losses = {
        "rec": symlog_mse(recon, obs),
        "rew": config.two_hot.loss(reward_logits, batch.reward[:, 1:]),
        "con": bernoulli_loss(cont_logit, cont_target),
        "dyn": dyn,
        "rep": rep,
    }
    scales = config.loss_scales
    # agent.py:393-394: scale each term, mean over (B, T), sum over terms.
    weighted = jnp.stack([jnp.mean(losses[k] * scales[k]) for k in LOSS_TERMS]).sum()
    entries = PosteriorEntries(
        deter=jax.lax.stop_gradient(post.deter),
        stoch=jnp.argmax(post.stoch, -1).astype(jnp.int32),
    )
    return WorldModelOutput(
        weighted=weighted,
        losses=losses,
        kl=kl,
        entries=entries,
        posterior=post,
        prior_logits=prior_logits,
        tokens=tokens,
    )
