"""One DreamerV3 training step: the joint loss, the optimizers, the slow critic.

The paper-era ``Agent.train`` / ``Agent.loss`` (``danijar/dreamerv3@2411f7d``
with the upstream replay-context fix ``29eb964``, whose line numbers are
cited: ``29eb964:dreamerv3/agent.py:166-223`` and ``:225-399`` are 2411f7d's
``:182-239`` and ``:241-420``), as pure, jittable float32 functions;
dreamerv3_spec Algorithm J and section 3.20, ``docs/world_models/DESIGN.md``
section 6.2. One step on a replay batch with its context row:

1. **one forward pass** (:func:`compute_loss`): the world-model loss on the
   replayed steps (:func:`~ajax.agents.DreamerV3.world_model.
   world_model_loss`, Algorithm E), imagination from every posterior state
   (:func:`~ajax.agents.DreamerV3.actor_critic.imagine`), the actor and
   critic losses with the return normaliser updated then read (Algorithm F)
   and the replay critic (Algorithm G);
2. **one gradient** of ``sum_k mean(scale_k loss_k)`` over the eight terms
   with respect to the world model, actor and critic parameters together,
   at their pre-update values (:func:`loss_and_grads`);
3. **three optimizer updates**, one LaProp instance per train state
   (:mod:`~ajax.agents.DreamerV3.optim`), all fed that joint gradient: the
   reference's single optimizer over all modules (``agent.py:83-91``,
   ``:191-192``), since every step of the optimizer acts per tensor;
4. the **slow critic** update after the step: a hard copy of the critic
   after the first update, then an EMA at ``slow_rate`` (2411f7d
   ``SlowUpdater``, ``jaxutils.py:737-762``; ``agent.py:194``).

:func:`train_step` returns the new state, the fresh posterior latents of the
replayed steps (computed with the pre-update parameters) for the replay
write-back, and the reference's metrics. All randomness comes in through
:class:`TrainNoise` (:func:`draw_train_noise`), so the parity tests can
force the reference's own draws
(``tests/agents/DreamerV3/test_dreamerv3_train_parity.py``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ajax.agents.DreamerV3.actor_critic import (
    ImaginationLoss,
    imagination_loss,
    imagine,
    replay_critic_loss,
)
from ajax.agents.DreamerV3.distributions import (
    BoundedNormal,
    OneHot,
    OneHotPolicy,
    Policy,
    draw_action_noise,
    draw_onehot_noise,
)
from ajax.agents.DreamerV3.networks import (
    RSSM,
    Actor,
    RSSMState,
    WorldModel,
    features,
    init_actor,
    init_critic,
    init_world_model,
    make_critic,
)
from ajax.agents.DreamerV3.optim import make_optimizer
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.world_model import (
    LOSS_TERMS,
    PosteriorEntries,
    ReplayContextBatch,
    draw_posterior_noise,
    world_model_loss,
)
from ajax.normalizers import ReturnNormalizer
from ajax.state import LoadedTrainState

sg = jax.lax.stop_gradient

#: The actor-critic terms, after the world model's :data:`LOSS_TERMS`: the
#: reference's ``actor``, ``critic`` and ``replay_critic``.
ACTOR_CRITIC_TERMS = ("actor", "critic", "repval")
#: All loss terms in the reference's summation order (``agent.py:241-249``,
#: ``:321``, ``:327``, ``:349``, ``:393-394``).
ALL_TERMS = LOSS_TERMS + ACTOR_CRITIC_TERMS


@struct.dataclass
class LearnerState:
    """Everything a training step updates.

    The train states keep their module's parameters (applied as
    ``apply_fn({'params': params}, ...)``) and their own LaProp instance
    (:func:`~ajax.agents.DreamerV3.optim.make_optimizer`).

    Attributes:
        world_model_state: :class:`~ajax.agents.DreamerV3.networks.
            WorldModel` (``enc``, ``rssm``, ``dec``, ``rew``, ``con``; the
            reference's ``dyn`` is ``rssm``).
        actor_state: :class:`~ajax.agents.DreamerV3.networks.Actor`.
        critic_state: the critic (:func:`~ajax.agents.DreamerV3.networks.
            make_critic`); ``target_params`` is the slow critic.
        retnorm: the return normaliser.

    The step counters of the three train states advance together, one per
    :func:`train_step`; the critic's also counts the slow critic's updates.
    """

    world_model_state: LoadedTrainState
    actor_state: LoadedTrainState
    critic_state: LoadedTrainState
    retnorm: ReturnNormalizer


class Params(NamedTuple):
    """The differentiated parameters: one joint gradient over all three."""

    world_model: dict
    actor: dict
    critic: dict


class TrainNoise(NamedTuple):
    """Every random draw of one training step (``B`` rows, ``T`` steps).

    Attributes:
        posterior: ``[B, T, S, C]`` Gumbel noise of the posterior samples.
        prior: ``[B T, H, S, C]`` Gumbel noise of the imagined prior samples,
            start-major (start ``b T + t`` is replayed step ``(b, t)``).
        action: ``[B T, H + 1, A]`` noise of the actions sampled at the start
            state and at each imagined state: Gumbel for discrete actions,
            standard normal for continuous ones.
    """

    posterior: jax.Array
    prior: jax.Array
    action: jax.Array


def draw_train_noise(
    key: jax.Array,
    config: DreamerV3Config,
    batch_size: int,
    length: int,
    action_dim: int,
    discrete: bool,
) -> TrainNoise:
    """The noise of one :func:`train_step` on ``batch_size`` rows of
    ``length`` trained steps (the context row excluded); ``action_dim`` is
    the action's dimension or the number of discrete actions."""
    post_key, prior_key, action_key = jax.random.split(key, 3)
    starts = batch_size * length
    h, s, c = config.imag_horizon, config.stoch, config.classes
    return TrainNoise(
        posterior=draw_posterior_noise(post_key, config, (batch_size, length)),
        prior=draw_onehot_noise(prior_key, (starts, h, s, c)),
        action=draw_action_noise(action_key, (starts, h + 1, action_dim), discrete),
    )


def init_learner(
    key: jax.Array,
    config: DreamerV3Config,
    obs_dim: int,
    action_dim: int,
    discrete: bool,
) -> LearnerState:
    """Initial parameters, optimizers and normaliser.

    ``action_dim`` is the action's dimension, or the number of actions of a
    discrete space. Each train state gets its own LaProp instance with the
    optimizer fields of ``config``. The slow critic starts as a copy of the
    critic. 2411f7d initialises a separate module instead, but both critics
    have a zero output layer, so the slow critic predicts exactly what a copy
    predicts until the first update overwrites it with the critic
    (deviations.md section 1, "Slow critic"; dreamerv3_spec 3.18).

    No two leaves of the returned state share a buffer (the slow critic is
    a copy, not an alias, of the critic), so the state can be donated to a
    jitted :func:`train_step`.
    """
    wm_key, actor_key, critic_key = jax.random.split(key, 3)
    tx = make_optimizer(config)
    critic_params = init_critic(critic_key, config)
    return LearnerState(
        world_model_state=LoadedTrainState.create(
            apply_fn=WorldModel(config, obs_dim).apply,
            params=init_world_model(wm_key, config, obs_dim, action_dim),
            tx=tx,
        ),
        actor_state=LoadedTrainState.create(
            apply_fn=Actor(config, action_dim, discrete).apply,
            params=init_actor(actor_key, config, action_dim, discrete),
            tx=tx,
        ),
        critic_state=LoadedTrainState.create(
            apply_fn=make_critic(config).apply,
            params=critic_params,
            tx=tx,
            target_params=jax.tree.map(jnp.copy, critic_params),
        ),
        retnorm=ReturnNormalizer.create(config.retnorm_rate, config.retnorm_limit),
    )


class LossAux(NamedTuple):
    """What :func:`compute_loss` returns next to the total loss.

    Attributes:
        losses: the unscaled per-element terms keyed by :data:`ALL_TERMS`:
            the world model's ``[B, T]``, ``actor`` and ``critic`` ``[B T,
            H]``, ``repval`` ``[B, T - 1]``.
        entries: the posterior latents of the replayed steps, for the replay
            write-back.
        imagination: Algorithm F's outputs (returns, values, weights, the
            return normaliser after this step's update, ...).
        reward: ``[B T, H + 1]`` the rewards of the imagined trajectories
            (index 0 from the replayed data).
        action: ``[B T, H + 1, A]`` their actions.
        prior_stoch: ``[B T, H, S, C]`` the one-hot samples of the imagined
            prior latents.
        replay_ret: ``[B, T - 1]`` the replay critic's returns.
        metrics: the reference's loss-side metrics (scalars).
    """

    losses: dict[str, jax.Array]
    entries: PosteriorEntries
    imagination: ImaginationLoss
    reward: jax.Array
    action: jax.Array
    prior_stoch: jax.Array
    replay_ret: jax.Array
    metrics: dict[str, jax.Array]


def compute_loss(
    params: Params,
    state: LearnerState,
    batch: ReplayContextBatch,
    noise: TrainNoise,
    *,
    config: DreamerV3Config,
) -> tuple[jax.Array, LossAux]:
    """The total loss ``sum_k mean(scale_k loss_k)`` and its parts (``Agent.loss``).

    ``params`` are the differentiated parameters; ``state`` gives the
    modules (``apply_fn``), the slow critic and the return normaliser. The
    batch has ``B`` rows of ``T + 1`` steps, the first being context
    (:class:`~ajax.agents.DreamerV3.world_model.ReplayContextBatch`).

    Steps (dreamerv3_spec 3.20; ``29eb964:dreamerv3/agent.py:225-399``):

    * the world-model terms on the replayed steps ``1..T`` (Algorithm E);
    * imagination from all ``N = B T`` posterior states, flattened
      batch-major (``agent.py:260-266``, ``imag_start: all``): the start
      action is sampled from the actor at the start state, then ``H``
      imagined steps (:func:`~ajax.agents.DreamerV3.actor_critic.imagine`);
    * the start state's reward and continuation are the replayed ``reward``
      and ``1 - is_terminal`` (2411f7d, ``agent.py:258-259``); the imagined
      states' are the reward head's prediction and the continue head's
      probability (``agent.py:285-286``);
    * every actor and critic input is stop-gradiented (``ac_grads: none``,
      ``agent.py:287-299``), and so are the actions inside the
      log-probabilities (Algorithm F). The reference evaluates the reward
      and continue heads on the imagined features *before* the
      stop-gradient, but their outputs only reach stop-gradiented
      quantities (weights, returns, advantages, targets), so they get no
      gradient there either; they are evaluated on the stop-gradiented
      features here, with the same values;
    * the replay critic reads the replayed posterior features without a
      stop-gradient, so it trains the world model too (Algorithm G,
      ``replay_critic_grad: True``).
    """
    model = WorldModel(config, batch.obs.shape[-1])
    twohot = config.two_hot

    def heads(feat: jax.Array) -> tuple[jax.Array, jax.Array]:
        variables = {"params": params.world_model}
        reward = model.apply(variables, feat, method=WorldModel.reward_logits)
        cont = model.apply(variables, feat, method=WorldModel.cont_logit)
        return twohot.decode(reward), jax.nn.sigmoid(cont)

    def actor(feat: jax.Array) -> Policy:
        return state.actor_state.apply_fn({"params": params.actor}, feat)

    def critic(critic_params: dict, feat: jax.Array) -> jax.Array:
        return state.critic_state.apply_fn({"params": critic_params}, feat)

    slow_params = state.critic_state.target_params
    if slow_params is None:
        raise ValueError("the critic state has no slow critic (target_params)")

    # Replay rollout (agent.py:229-249).
    wm = world_model_loss(model, params.world_model, batch, noise.posterior)
    post = wm.posterior
    b, t = batch.reward.shape[0], batch.reward.shape[1] - 1
    reward = batch.reward[:, 1:]
    is_terminal, is_last = batch.is_terminal[:, 1:], batch.is_last[:, 1:]

    # Imagination rollout (agent.py:251-282).
    def flat(x: jax.Array) -> jax.Array:
        return x.reshape(b * t, *x.shape[2:])

    start = RSSMState(flat(post.deter), flat(post.stoch))
    start_feat = features(start.deter, start.stoch)
    start_action = actor(start_feat).sample(noise.action[:, 0])
    traj = imagine(
        RSSM(config),
        params.world_model["rssm"],
        actor,
        start,
        start_action,
        noise.prior,
        noise.action[:, 1:],
    )
    feat = jnp.concatenate([start_feat[:, None], features(traj.deter, traj.stoch)], 1)
    action = sg(jnp.concatenate([start_action[:, None], traj.action], 1))

    # Annotate (agent.py:284-299).
    inp = sg(feat)
    imag_reward, imag_cont = heads(inp)
    imag_reward = jnp.concatenate([flat(reward)[:, None], imag_reward[:, 1:]], 1)
    start_cont = flat(1 - is_terminal.astype(jnp.float32))
    imag_cont = jnp.concatenate([start_cont[:, None], imag_cont[:, 1:]], 1)
    policy = actor(inp)
    imag = imagination_loss(
        config,
        policy,
        action,
        critic(params.critic, inp),
        twohot.decode(critic(slow_params, inp)),
        imag_reward,
        imag_cont,
        state.retnorm,
    )

    # Replay critic (agent.py:331-352).
    rep_feat = features(post.deter, post.stoch)
    repval, replay_ret = replay_critic_loss(
        config,
        critic(params.critic, rep_feat),
        twohot.decode(critic(slow_params, rep_feat)),
        imag.ret[:, 0].reshape(b, t),
        reward,
        is_terminal,
        is_last,
    )

    # Combine (agent.py:392-394): scale each term, mean, sum over terms.
    losses = {**wm.losses, "actor": imag.actor, "critic": imag.critic, "repval": repval}
    scales = config.loss_scales
    total = jnp.stack([jnp.mean(losses[k] * scales[k]) for k in ALL_TERMS]).sum()

    aux = LossAux(
        losses=losses,
        entries=wm.entries,
        imagination=imag,
        reward=imag_reward,
        action=action,
        prior_stoch=traj.stoch,
        replay_ret=replay_ret,
        metrics={},
    )
    metrics = _loss_metrics(
        config,
        aux,
        policy,
        reward,
        wm.posterior.logits,
        wm.prior_logits,
        wm.tokens,
    )
    return total, aux._replace(metrics=metrics)


def _stats(x: jax.Array, prefix: str) -> dict[str, jax.Array]:
    """``jaxutils.tensorstats`` without the random subsample ``dist``
    (``jaxutils.py:52-66``)."""
    x = jnp.asarray(x, jnp.float32)
    return {
        f"{prefix}/mean": x.mean(),
        f"{prefix}/std": x.std(),
        f"{prefix}/mag": jnp.abs(x).mean(),
        f"{prefix}/min": x.min(),
        f"{prefix}/max": x.max(),
    }


def _loss_metrics(
    config: DreamerV3Config,
    aux: LossAux,
    policy: Policy,
    reward: jax.Array,
    post_logits: jax.Array,
    prior_logits: jax.Array,
    tokens: jax.Array,
) -> dict[str, jax.Array]:
    """The scalar metrics of ``Agent.loss`` (``agent.py:354-390``) and of
    ``RSSM.loss`` (``nets.py:106-109``).

    ``{term}_loss`` and ``{term}_loss_std`` are of the unscaled terms (the
    reference logs them before scaling, ``agent.py:355-356``), under Ajax's
    names (``rec``, ``rew``, ``con``, ``repval`` for the reference's
    ``vector``, ``reward``, ``cont``, ``replay_critic``). Not reproduced:
    the random ``*/dist`` subsamples and the ``rewstats`` / ``constats``
    diagnostics.
    """
    imag = aux.imagination
    metrics = {f"{k}_loss": jnp.mean(v) for k, v in aux.losses.items()}
    metrics.update({f"{k}_loss_std": jnp.std(v) for k, v in aux.losses.items()})
    ret, value = imag.ret, imag.value
    metrics.update(_stats(imag.adv, "adv"))
    metrics.update(_stats(aux.reward, "rew"))
    metrics.update(_stats(imag.weight, "weight"))
    metrics.update(_stats(value, "val"))
    metrics.update(_stats(ret, "ret"))
    metrics.update(_stats((ret - imag.retnorm.lo) / imag.scale, "ret_normed"))
    metrics.update(_stats(aux.replay_ret, "replay_ret"))
    metrics["td_error"] = jnp.abs(ret - value[:, :-1]).mean()
    metrics["ret_rate"] = (jnp.abs(ret) > 1.0).mean()
    action_dim = aux.action.shape[-1]
    if isinstance(policy, OneHotPolicy):
        lo, hi = OneHotPolicy.entropy_range(action_dim)
        metrics.update(_stats(jnp.argmax(aux.action, -1), "act/action"))
    else:
        assert isinstance(policy, BoundedNormal)
        lo, hi = BoundedNormal.entropy_range(action_dim, config.minstd, config.maxstd)
        metrics.update(_stats(aux.action, "act/action"))
    metrics.update(_stats((imag.entropy - lo) / (hi - lo), "rand/action"))
    metrics.update(_stats(imag.entropy, "ent/action"))
    for name, logits in (("prior_ent", prior_logits), ("post_ent", post_logits)):
        entropy = OneHot.from_logits(logits, config.unimix).entropy()
        metrics.update(_stats(entropy, name))
    metrics["data_rew/max"] = jnp.abs(reward).max()
    metrics["pred_rew/max"] = jnp.abs(aux.reward).max()
    metrics["data_rew/mean"] = reward.mean()
    metrics["pred_rew/mean"] = aux.reward.mean()
    metrics["data_rew/std"] = reward.std()
    metrics["pred_rew/std"] = aux.reward.std()
    metrics["activation/embed"] = jnp.abs(tokens).mean()
    return metrics


def learner_params(state: LearnerState) -> Params:
    """The differentiated parameters of ``state``."""
    return Params(
        world_model=state.world_model_state.params,
        actor=state.actor_state.params,
        critic=state.critic_state.params,
    )


def loss_and_grads(
    state: LearnerState,
    batch: ReplayContextBatch,
    noise: TrainNoise,
    *,
    config: DreamerV3Config,
) -> tuple[tuple[jax.Array, LossAux], Params]:
    """``((total, aux), grads)``: one ``jax.value_and_grad`` of
    :func:`compute_loss` over the world model, actor and critic parameters
    together, at their current values (the reference's ``nj.grad`` over
    ``self.modules``, ``agent.py:89-91``; ``jaxutils.py:478-479``)."""
    return jax.value_and_grad(compute_loss, has_aux=True)(
        learner_params(state), state, batch, noise, config=config
    )


def apply_optimizer(
    train_state: LoadedTrainState, grads: dict
) -> tuple[LoadedTrainState, dict]:
    """One optimizer update; returns the new state and the applied updates.

    ``params + updates`` (``optax.apply_updates``, ``jaxutils.py:526``) and
    the step incremented; flax's ``TrainState.apply_gradients``, but the
    updates are returned too (for the metrics).
    """
    updates, opt_state = train_state.tx.update(
        grads, train_state.opt_state, train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)
    new_state = train_state.replace(
        step=train_state.step + 1, params=params, opt_state=opt_state
    )
    return new_state, updates


def update_slow_critic(
    critic_state: LoadedTrainState, count: jax.Array, rate: float
) -> LoadedTrainState:
    """The slow critic after an optimizer step (2411f7d ``SlowUpdater``).

    ``slow <- mix critic + (1 - mix) slow``
    (:meth:`~ajax.state.LoadedTrainState.soft_update`) with ``mix = 1`` when
    ``count``, the number of earlier slow-critic updates, is 0 -- a hard
    copy of the critic after the first optimizer step -- and ``rate`` after
    it, every step (``29eb964:dreamerv3/jaxutils.py:737-762``: ``mix =
    clip(1 [updates == 0] + fraction [updates % 1 == 0], 0, 1)``;
    dreamerv3_spec 3.18; deviations.md section 1, "Slow critic").
    """
    return critic_state.soft_update(
        jnp.where(count == 0, 1.0, rate).astype(jnp.float32)
    )


def train_step(
    state: LearnerState,
    batch: ReplayContextBatch,
    noise: TrainNoise,
    *,
    config: DreamerV3Config,
) -> tuple[LearnerState, PosteriorEntries, dict[str, jax.Array]]:
    """One DreamerV3 update (``Agent.train``, ``29eb964:dreamerv3/agent.py:166-223``).

    :func:`loss_and_grads` at the current parameters, then the three
    optimizer updates with that one gradient (:func:`apply_optimizer`),
    then the slow critic (:func:`update_slow_critic`) and the return
    normaliser's new state. ``config`` must be the configuration the state
    was built with (:func:`init_learner`); jit it as a static argument.

    Returns:
        The new state; the posterior latents of the replayed steps
        ``1..T`` (pre-update parameters, ``agent.py:197-202``) for the
        replay write-back; the reference's metrics: the loss-side ones of
        :func:`compute_loss` and the optimizer's ``opt_loss``,
        ``opt_grad_norm``, ``opt_update_norm``, ``opt_param_norm`` (after
        the update) and ``opt_grad_steps`` (``jaxutils.py:527-549``).
    """
    (total, aux), grads = loss_and_grads(state, batch, noise, config=config)
    wm_state, wm_updates = apply_optimizer(state.world_model_state, grads.world_model)
    actor_state, actor_updates = apply_optimizer(state.actor_state, grads.actor)
    critic_state, critic_updates = apply_optimizer(state.critic_state, grads.critic)
    critic_state = update_slow_critic(
        critic_state, state.critic_state.step, config.slow_rate
    )
    new_state = state.replace(
        world_model_state=wm_state,
        actor_state=actor_state,
        critic_state=critic_state,
        retnorm=aux.imagination.retnorm,
    )
    metrics = dict(aux.metrics)
    metrics["opt_loss"] = total
    metrics["opt_grad_norm"] = optax.global_norm(grads)
    metrics["opt_update_norm"] = optax.global_norm(
        (wm_updates, actor_updates, critic_updates)
    )
    metrics["opt_param_norm"] = optax.global_norm(learner_params(new_state))
    metrics["opt_grad_steps"] = wm_state.step
    return new_state, aux.entries, metrics
