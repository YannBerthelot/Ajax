"""Expert-guidance research features as composable :class:`Extension`s.

* :class:`OnlineBC`              — value-weighted decaying BC loss term.
* :class:`ImitationLoss`         — plain BC loss term towards the expert.
* :class:`ResidualPolicy`        — execute ``clip(a_expert + scale·a_pi)``.
* :class:`JSRLCurriculum`        — per-episode expert→learner handoff.
* :class:`ExpertValueMetrics`    — the critic at the expert's actions, logged.

Each extension carries only frozen config on ``self``; the maths are the
functions below. The expert is SAC's ``expert_policy``: OnlineBC,
ResidualPolicy and JSRLCurriculum read the expert action SAC computes.
ImitationLoss holds its own expert, and reads the actions the agent stored
when that expert is the agent's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import phi_star_critic_state, predict_value


def residual_action_transform(
    actions: jax.Array,
    a_expert: jax.Array,
    scale: float = 1.0,
) -> jax.Array:
    """Residual RL: executed action is clip(a_expert + scale * a_pi, -1, 1).

    The critic was trained on (s, executed_action) tuples, so the policy
    gradient must flow through Q(s, executed_action), not Q(s, a_pi).
    ``scale`` follows Johannink et al. 2019; values < 1 keep the initial
    behaviour close to the expert when the policy is randomly initialised.
    """
    return jnp.clip(a_expert + scale * actions, -1.0, 1.0)


def online_bc_weight(train_frac: jax.Array, critic_warmup_frac: float) -> jax.Array:
    """The online BC term's schedule: 1 at the start, falling linearly to 0
    at ``critic_warmup_frac`` of training."""
    return jnp.maximum(1.0 - train_frac / critic_warmup_frac, 0.0)


def compute_online_bc_loss(
    pi_mean: jax.Array,
    a_expert: jax.Array,
    critic_state,
    expert_critic_params,
    observations: jax.Array,
    train_frac: jax.Array,
    critic_warmup_frac: float,
    expert_v_min: jax.Array,
    expert_v_max: jax.Array,
    bc_coef: float,
) -> jax.Array:
    """Online decaying BC term — value-weighted, warmup-decaying.

    Decays to zero after train_frac exceeds critic_warmup_frac.
    Value-weighting ensures BC is strongest in high-value states.
    """
    bc_weight = online_bc_weight(train_frac, critic_warmup_frac)
    v_star = jax.lax.stop_gradient(
        jnp.min(
            predict_value(
                critic_state=phi_star_critic_state(critic_state),
                critic_params=expert_critic_params,
                x=jnp.concatenate([observations, a_expert], axis=-1),
            ),
            axis=0,
        )
    )
    v_weight = (
        jnp.clip(
            (v_star - expert_v_min) / (expert_v_max - expert_v_min + 1e-6),
            0.0,
            1.0,
        )
        ** 2
    )
    return (
        bc_coef
        * bc_weight
        * (
            v_weight
            * jnp.sum(
                (pi_mean - jax.lax.stop_gradient(a_expert)) ** 2,
                axis=-1,
                keepdims=True,
            )
        ).mean()
    )


def reads_stored_expert_actions(ext: Any, agent_context: dict) -> bool:
    """Whether an extension that carries an expert of its own (IBRL,
    LCBGatedBootstrap, CriticBlend, PhiRefresh) reads that expert's actions
    as collection stored them (``next_a_expert``): when it is the agent's
    ``expert_policy``, whose state collection tracks along each trajectory.
    Another expert is called on the raw observations the extension is
    given, without a state (a stateful one is refused at construction,
    ``check_extension_experts``)."""
    return ext.expert_policy is agent_context.get("expert_policy")


def expert_next_actions(ext: Any, raw_next: Any, stored_next: Any) -> jax.Array:
    """The extension's expert's actions at the replayed rows' s': the
    stored ones (``reads_stored_expert_actions``), else its call on the
    raw next observations."""
    if ext.reads_stored_expert:
        return jax.lax.stop_gradient(stored_next)
    return jax.lax.stop_gradient(ext.expert_policy(raw_next))


@dataclass(frozen=True)
class OnlineBC(Extension):
    """Online, value-weighted, warmup-decaying behavioural-cloning term.

    A ``bc_coef · w(s) · ||μ_θ(s) - a_expert||²`` penalty
    (:func:`compute_online_bc_loss`) that decays to zero once the training
    fraction exceeds ``critic_warmup_frac``. It reads the actor's mean
    action (``pi_mean``), the expert's action (``a_expert``, which SAC puts
    in its actor-loss batch) and the observations from the batch, and the
    frozen expert critic phi* with its value range
    (``expert_critic_params``, ``expert_v_min``, ``expert_v_max``) and the
    critic phi* is applied with from the agent's state, which
    :class:`MCPretrain` fits (refused without it). The expert's action is
    the agent's ``expert_policy``'s, so an agent without one refuses it
    (``requires`` ``agent_expert``); it raises on a batch without ``a_expert`` (an
    agent other than SAC). Until phi* exists the term is 0.
    """

    bc_coef: float = 1.0
    critic_warmup_frac: float = 0.15
    name: str = "online_bc"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "continuous_actions", "phi_star", "q_critic"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "actor_loss": frozenset({"a_expert", "observations", "pi_mean"})
    }

    def actor_loss(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        ctx: ExtensionContext,
    ) -> Any:
        del ext_state
        if "a_expert" not in batch:
            raise ValueError(
                "OnlineBC needs the expert's action ('a_expert') in the"
                " actor-loss batch; only SAC provides it."
            )
        a_expert = batch["a_expert"]
        if agent_state.expert_critic_params is None or self.critic_warmup_frac <= 0:
            return 0.0
        train_frac = ctx.step / ctx.total_steps
        term = compute_online_bc_loss(
            batch["pi_mean"],
            a_expert,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            batch["observations"],
            train_frac,
            self.critic_warmup_frac,
            agent_state.expert_v_min,
            agent_state.expert_v_max,
            self.bc_coef,
        )
        weight = online_bc_weight(train_frac, self.critic_warmup_frac)
        return term, {"online_bc/term": term, "online_bc/weight": weight}


def anneal_weight(step: jax.Array, total_steps: int, anneal_frac: float) -> jax.Array:
    """1 at the start of training, falling linearly to 0 at ``anneal_frac``
    of ``total_steps``, 0 after."""
    progress = step / (anneal_frac * max(total_steps, 1))
    return jnp.clip(1.0 - progress, 0.0, 1.0)


def q_filter_weight(
    critic_state: Any,
    observations: jax.Array,
    a_expert: jax.Array,
    a_policy: jax.Array,
) -> jax.Array:
    """The Q-filter of Nair et al. 2018 (eq. 8), ``(batch, 1)``: 1 where the
    critic's ensemble minimum rates the expert's action above the policy's,
    ``min_k Q_k(s, a_E) > min_k Q_k(s, π(s))``, else 0; held constant."""

    def q_min(a: jax.Array) -> jax.Array:
        x = jnp.concatenate([observations, a], axis=-1)
        return jnp.min(predict_value(critic_state, critic_state.params, x), axis=0)

    keep = q_min(a_expert) > q_min(a_policy)
    return jax.lax.stop_gradient(keep.astype(jnp.float32))


def td3bc_weight(
    critic_state: Any, observations: jax.Array, pi_mean: jax.Array, alpha: float
) -> jax.Array:
    """TD3+BC's ``1 / lambda = mean|Q_1(s, pi(s))| / alpha``, held constant,
    with Q_1 the first critic, the one TD3's actor follows
    (sfujim/TD3_BC@8791ad7, TD3_BC.py:141-143: ``lmbda = alpha /
    Q.abs().mean().detach()``, actor loss ``-lmbda * Q.mean() +
    mse_loss(pi, action)``)."""
    x = jnp.concatenate([observations, jax.lax.stop_gradient(pi_mean)], axis=-1)
    q1 = predict_value(critic_state, critic_state.params, x)[0]
    return jax.lax.stop_gradient(jnp.mean(jnp.abs(q1)) / alpha)


@dataclass(frozen=True)
class ImitationLoss(Extension):
    """Behavioural-cloning regulariser: ``coef · mean((μ(s) - a_E(s))²) / 4``.

    Same form as the actor-side BC term of TD3+BC (Fujimoto & Gu, 2021),
    with a frozen expert's action as the target: ``μ(s)`` is the actor's mean action
    (``pi.mean()``, which is TD3's deterministic action), ``a_E(s)`` the
    expert's action, held constant: its call on the raw observation, or,
    when it is the agent's ``expert_policy``, the action collection stored
    from its tracked state (which a stateful expert, a PID with its
    integral, needs). The mean runs over
    batch and action dims; ``/ 4`` maps the squared distance of two
    actions in ``[-1, 1]`` to ``[0, 1]``. Unlike :class:`OnlineBC` the term
    is unweighted and needs no pre-trained expert critic.

    ``q_filter=True`` counts only the rows where the critic rates the
    expert's action above the policy's (:func:`q_filter_weight`, at
    ``μ(s)``) and still divides by every row, as the paper's fixed ``λ2 =
    1/N_D`` does: the Q-filtered BC of Nair, McGrew, Andrychowicz, Zaremba
    and Abbeel (ICRA 2018, Sec. IV-C, eq. 8). Deviations: replay rows
    labelled with the expert's action, not a demonstration buffer (the
    replay's expert flags would restore it); ``coef`` on the BC term, not
    the paper's ``λ1 = 1e-3`` on the RL term and ``λ2``; the policy's mean
    action and the minimum over the agent's critics (TD3's actor reads its
    first, REDQ's the mean of its ensemble), not DDPG's one critic. The
    filter needs a Q critic, which SAC, TD3, REDQ, ASAC and AVG put in
    their actor-loss batch (``critic_state``); a state-value critic (PPO,
    APO) cannot rate an action. It refuses :class:`ResidualPolicy`, whose
    critic reads the executed action. ``imitation/q_filter_frac`` logs, at
    each evaluation, the share of a replay batch the filter keeps (every
    agent but AVG, which keeps no replay).

    With ``anneal_frac`` the term is multiplied by a weight falling
    linearly from 1 to 0 at that fraction of training (:func:`anneal_weight`),
    as in kickstarting (Schmitt et al., 2018, *Kickstarting Deep
    Reinforcement Learning*, arXiv:1803.03835, eqs. 2-3), whose distillation
    weight on the teacher decays so the student ends on its own RL loss.
    Deviations from kickstarting: a squared error on the mean action
    instead of the cross-entropy between the teacher's and the student's
    policies (the expert is deterministic: it gives an action, not a
    distribution); a linear schedule set by ``anneal_frac`` instead of the
    weights population-based training tunes. ``None`` keeps the weight at 1.

    ``target="replay"`` clones the replayed rows' own actions (``actions``,
    which SAC, TD3, REDQ and ASAC put in their actor-loss batch) instead of
    an expert's: logged data needs no expert, and ``expert_policy`` may be
    None. ``alpha`` gives the term TD3+BC's weight (Fujimoto and Gu, 2021;
    :func:`td3bc_weight`): ``coef · mean|Q_1(s, μ(s))| / alpha · mean((μ(s)
    - a)²)``, without the ``/ 4``. That is the paper's actor loss, ``-λ
    mean Q + mean((π(s) - a)²)`` with ``λ = alpha / mean|Q|``, divided by
    ``λ`` (the agent's own term stays ``-mean Q``): the same direction, at
    a scale Adam's step nearly cancels. With both, on TD3, a transition
    dataset trained offline (``train_offline``) and
    ``normalize_observations=True`` (the dataset's statistics, sqrt(var +
    1e-8) in place of the reference's std + 1e-3), it is TD3+BC.

    TD3, REDQ, ASAC, APO, SAC and PPO put ``pi_mean`` and
    ``raw_observations`` in their actor-loss batch, and all but PPO (which
    takes no expert) the stored ``a_expert``; an agent whose batch has no
    ``pi_mean`` (a discrete-action policy) raises at trace time.
    """

    expert_policy: Optional[Callable]
    coef: float = 1.0
    q_filter: bool = False
    anneal_frac: Optional[float] = None
    target: str = "expert"
    alpha: Optional[float] = None
    # Agent context, bound by bind_to_agent.
    reads_stored_expert: bool = False
    buffer: Any = None  # the Q-filter's logged share samples it
    augment_obs_with_expert_action: bool = False
    name: str = "imitation_loss"
    requires: ClassVar[frozenset[str]] = frozenset({"continuous_actions"})
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "actor_loss": frozenset({"observations", "pi_mean"})
    }
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.anneal_frac is not None and not 0.0 < self.anneal_frac <= 1.0:
            raise ValueError(
                "ImitationLoss's anneal_frac is the fraction of training at"
                f" which its weight reaches 0, in (0, 1]; got {self.anneal_frac}."
            )
        if self.target not in ("expert", "replay"):
            raise ValueError(
                f"ImitationLoss's target is 'expert' or 'replay', got {self.target!r}."
            )
        if self.target == "expert" and self.expert_policy is None:
            raise ValueError(
                "ImitationLoss clones its expert_policy: give it one, or"
                " target='replay' to clone the replayed actions."
            )
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError(f"ImitationLoss's alpha is positive, got {self.alpha}.")

    def experts(self) -> tuple:
        return () if self.expert_policy is None else (self.expert_policy,)

    def needs(self) -> frozenset[str]:
        """The filter and ``alpha`` also read the Q critic; the replayed
        target needs a replay."""
        q = {"q_critic"} if self.q_filter or self.alpha is not None else set()
        return super().needs() | q | ({"replay"} if self.target == "replay" else set())

    def reads(self) -> Mapping[str, frozenset[str]]:
        """The filter and ``alpha`` also read the critic from the actor-loss
        batch, the replayed target its rows' actions."""
        keys = self.batch_keys["actor_loss"]
        if self.q_filter or self.alpha is not None:
            keys = keys | {"critic_state"}
        if self.target == "replay":
            keys = keys | {"actions"}
        return {"actor_loss": keys}

    def bind_to_agent(self, **agent_context: Any) -> "ImitationLoss":
        """Read the expert's stored actions when it is the agent's; with the
        filter, bind the agent's replay buffer (its logged share samples it)
        and SAC's observation augmentation."""
        expert = self.target == "expert"
        stored = expert and reads_stored_expert_actions(self, agent_context)
        bound = dataclasses.replace(self, reads_stored_expert=stored)
        if not self.q_filter:
            return bound
        exts = agent_context.get("extensions", ())
        if any("transform_action" in e.implemented_phases() for e in exts):
            raise ValueError(
                "ImitationLoss(q_filter=True) reads the critic at the policy's"
                " mean action; ResidualPolicy's critic reads the executed one."
            )
        augment = agent_context.get("augment_obs_with_expert_action", False)
        return dataclasses.replace(
            bound,
            buffer=agent_context.get("buffer", self.buffer),
            augment_obs_with_expert_action=augment,
        )

    def _expert_actions(self, stored: Any, raw_obs: jax.Array) -> jax.Array:
        """The expert's actions: the stored ones where the agent keeps them
        (the agent's own expert), else its call on the raw observations."""
        if self.reads_stored_expert and stored is not None:
            return jax.lax.stop_gradient(stored)
        assert self.expert_policy is not None  # target "expert" (__post_init__)
        return jax.lax.stop_gradient(self.expert_policy(raw_obs))

    def actor_loss(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        ctx: ExtensionContext,
    ) -> Any:
        del agent_state, ext_state
        pi_mean = batch.get("pi_mean")
        if pi_mean is None:
            raise ValueError(
                "ImitationLoss pulls the actor's mean action ('pi_mean') towards"
                " the expert's, and this actor-loss batch has none: a discrete"
                " policy has no mean action (the term has no cross-entropy form"
                " yet)."
            )
        obs = batch.get("raw_observations")
        if obs is None:
            obs = batch["observations"]
        if self.target == "replay":
            a_expert = jax.lax.stop_gradient(batch["actions"])
        else:
            a_expert = self._expert_actions(batch.get("a_expert"), obs)
        sq = jnp.mean(jnp.square(pi_mean - a_expert), axis=-1, keepdims=True)
        mse = jnp.mean(sq)  # before the filter and the schedule
        if self.q_filter:
            if "critic_state" not in batch:
                raise ValueError(
                    "ImitationLoss(q_filter=True) needs the agent's Q-critic"
                    " ('critic_state') in the actor-loss batch, and this one"
                    " has none."
                )
            sq = sq * q_filter_weight(
                batch["critic_state"],
                batch["observations"],
                a_expert,
                jax.lax.stop_gradient(pi_mean),
            )
        if self.anneal_frac is not None:
            sq = sq * anneal_weight(ctx.step, ctx.total_steps, self.anneal_frac)
        if self.alpha is None:
            term = self.coef * jnp.mean(sq) / 4.0
        else:
            weight = td3bc_weight(
                batch["critic_state"], batch["observations"], pi_mean, self.alpha
            )
            term = self.coef * weight * jnp.mean(sq)
        return term, {"imitation/term": term, "imitation/mse": mse}

    def eval_metrics(
        self, agent_state: Any, ext_state: Any, rng: jax.Array, ctx: ExtensionContext
    ) -> dict:
        """``imitation/q_filter_frac``, the share of a replay batch the
        Q-filter keeps (an agent with a replay, which it binds)."""
        from ajax.buffers.utils import get_batch_from_buffer
        from ajax.environments.interaction import augment_obs, get_pi

        del ext_state, ctx
        if self.buffer is None:
            return {}
        batch = get_batch_from_buffer(
            self.buffer, agent_state.collector_state.buffer_state, rng
        )
        obs = batch.obs
        if self.augment_obs_with_expert_action:
            obs = augment_obs(obs, expert_action=batch.a_expert)
        actor = agent_state.actor_state  # get_pi applies its observation norm
        pi_mean = get_pi(actor, actor.params, obs)[0].mean()
        a_expert = (
            batch.action
            if self.target == "replay"
            else self._expert_actions(batch.a_expert, batch.raw_obs)
        )
        keep = q_filter_weight(agent_state.critic_state, obs, a_expert, pi_mean)
        return {"imitation/q_filter_frac": keep.mean()}


@dataclass(frozen=True)
class ResidualPolicy(Extension):
    """Residual RL (Johannink et al. 2019).

    The executed action is ``clip(a_expert + scale·a_pi, -1, 1)``: SAC's
    action pipeline applies :meth:`transform_action` at collection (its
    warmup samples uniformly), and the actor's policy gradient, the
    TD-target bootstrap and evaluation flow through the same transform.
    Its presence alone turns residual RL on, with SAC's expert policy;
    ``transform_action`` is a phase only SAC runs, so the other agents
    refuse it.
    """

    scale: float = 1.0
    name: str = "residual_policy"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})

    def transform_action(self, actions: jax.Array, a_expert: jax.Array) -> jax.Array:
        """``clip(a_expert + scale·a_pi, -1, 1)`` — the residual mix."""
        return residual_action_transform(actions, a_expert, scale=self.scale)

    def build_policy_transform(
        self, expert_policy: Optional[Callable]
    ) -> Optional[Callable]:
        """The actor-loss / TD-target residual transform, ``(actions,
        raw_obs, a_expert_precomputed) -> actions``; ``None`` (no residual
        transform) without an ``expert_policy``."""
        if expert_policy is None:
            return None

        scale = self.scale

        def transform(actions, raw_obs, a_expert_precomputed):
            a_exp = (
                a_expert_precomputed
                if a_expert_precomputed is not None
                else jax.lax.stop_gradient(expert_policy(raw_obs))
            )
            return residual_action_transform(actions, a_exp, scale=scale)

        return transform

    def build_eval_transform(self) -> Callable:
        """The eval-time residual transform, ``(raw, expert, obs,
        agent_state) -> actions``, as :func:`ajax.evaluate.step_environment`
        calls it."""
        scale = self.scale

        def transform(raw_actions, expert_actions, obs, agent_state):
            del obs, agent_state
            return residual_action_transform(raw_actions, expert_actions, scale=scale)

        return transform


@dataclass(frozen=True)
class JSRLCurriculum(Extension):
    """True JSRL per-episode handoff (Uchendu et al. 2023).

    While ``step_in_episode < H_t`` the expert acts; otherwise the
    learner does. ``H_t`` decays linearly from ``episode_length`` to 0
    over the first ``decay_frac`` fraction of training.

    The :meth:`action` phase owns the per-episode substitution. It reads
    ``post_warmup_action`` and ``expert_action`` off the SAC action
    pipeline's batch dict (the same dict :class:`EDGEExploration` and
    :class:`~ajax.extensions.target_mods.ValueBox` use) and per-env
    ``step_in_episode`` off ``agent_state.collector_state``; the SAC
    pipeline still owns the ``step_in_episode`` increment/reset (in
    :func:`collect_experience`). It writes ``_jsrl_use_expert`` back, which
    the pipeline reads to flag the steps the expert took.

    Evaluations log the guide horizon ``H_t`` (``jsrl/horizon``) and the
    share of envs the expert drove on the last collection step
    (``jsrl/expert_frac``).
    """

    episode_length: int = 1000
    decay_frac: float = 0.5
    name: str = "jsrl_curriculum"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})
    action_slot: ClassVar[str] = "pre_warmup"
    reads_step_in_episode: ClassVar[bool] = True

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Substitute the expert action while ``step_in_episode < H_t``."""
        del ext_state, rng
        post_warmup_action = obs["post_warmup_action"]
        expert_action = obs["expert_action"]

        H_t = self.horizon(agent_state.collector_state.timestep, ctx.total_steps)
        step_in_ep = agent_state.collector_state.step_in_episode
        use_expert_jsrl = step_in_ep.astype(jnp.float32) < H_t
        use_expert_jsrl = use_expert_jsrl.reshape(
            (-1,) + (1,) * (post_warmup_action.ndim - 1)
        )
        # The pipeline flags the steps the expert took (is_expert_flag).
        obs["_jsrl_use_expert"] = use_expert_jsrl
        return jnp.where(use_expert_jsrl, expert_action, post_warmup_action)

    def horizon(self, timestep: Any, total_steps: int) -> jax.Array:
        """``H_t``: ``episode_length`` at the start, falling linearly to 0 at
        ``decay_frac`` of the run."""
        total_timesteps = max(int(total_steps), 1)
        train_frac = timestep / jnp.maximum(total_timesteps, 1)
        curriculum_progress = jnp.clip(train_frac / self.decay_frac, 0.0, 1.0)
        return self.episode_length * (1.0 - curriculum_progress)

    def eval_metrics(
        self, agent_state: Any, ext_state: Any, rng: jax.Array, ctx: ExtensionContext
    ) -> dict:
        del ext_state, rng
        collector = agent_state.collector_state
        return {
            "jsrl/horizon": self.horizon(collector.timestep, ctx.total_steps),
            "jsrl/expert_frac": collector.last_expert_frac,
        }


__all__ = [
    "ImitationLoss",
    "JSRLCurriculum",
    "OnlineBC",
    "ResidualPolicy",
]


@dataclass(frozen=True)
class ExpertValueMetrics(Extension):
    """The critic at the expert's actions, logged at each evaluation, on a
    replay batch: ``value/q_expert_mean``, the mean of ``min_k Q_k(s,
    a_E)``; ``value/q_gap``, that minus the same at the batch's actions
    (above 0: the critic rates the expert above what was done); and, once
    :class:`~ajax.extensions.pretrain.MCPretrain` has fitted ``φ*``,
    ``value/phi_star_q_gap_ood``, the mean ``|min_k Q_φ*(s, a_E) - min_k
    Q(s, a_E)|``.

    One replay batch per evaluation pays for them, not a critic pass (two
    with ``φ*``) per update for three logged numbers. ``a_E`` is the
    expert's action stored at collection. A replay agent with an
    ``expert_policy`` (SAC, TD3, REDQ, ASAC) binds its replay buffer, its
    expert and its observation augmentation.
    """

    buffer: Any = None
    expert_policy: Optional[Callable] = None
    augment_obs_with_expert_action: bool = False
    name: str = "expert_value_metrics"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "q_critic", "replay"}
    )
    context: ClassVar[frozenset[str]] = frozenset({"buffer", "expert_policy"})
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})

    def bind_to_agent(self, **agent_context: Any) -> "ExpertValueMetrics":
        if self.buffer is not None:
            return self
        buffer, expert = agent_context.get("buffer"), agent_context.get("expert_policy")
        if buffer is None or expert is None:
            raise ValueError(
                "ExpertValueMetrics reads, on a replay batch, the critic at the"
                " actions the agent stored for its expert_policy: it needs a"
                " replay agent (SAC, TD3, REDQ, ASAC) with an expert_policy."
            )
        augment = agent_context.get("augment_obs_with_expert_action", False)
        return dataclasses.replace(
            self,
            buffer=buffer,
            expert_policy=expert,
            augment_obs_with_expert_action=augment,
        )

    def eval_metrics(
        self, agent_state: Any, ext_state: Any, rng: jax.Array, ctx: ExtensionContext
    ) -> dict:
        from ajax.buffers.utils import get_batch_from_buffer
        from ajax.environments.interaction import augment_obs

        del ext_state, ctx
        batch = get_batch_from_buffer(
            self.buffer, agent_state.collector_state.buffer_state, rng
        )
        obs, actions = batch.obs, batch.action
        # The expert's actions as collection stored them (from its state).
        a_expert = batch.a_expert
        if self.augment_obs_with_expert_action:
            obs = augment_obs(obs, expert_action=a_expert)
        critic = agent_state.critic_state

        def q_min(params: Any, a: jax.Array, state: Any = critic) -> jax.Array:
            x = jnp.concatenate([obs, a], axis=-1)
            return jnp.min(predict_value(state, params, x), axis=0)

        q_expert = q_min(critic.params, a_expert)
        out = {
            "value/q_expert_mean": q_expert.mean(),
            "value/q_gap": (q_expert - q_min(critic.params, actions)).mean(),
        }
        phi_star = getattr(agent_state, "expert_critic_params", None)
        if phi_star is not None:
            phi = q_min(phi_star, a_expert, phi_star_critic_state(critic))
            gap = jnp.abs(phi - q_expert).mean()
            out["value/phi_star_q_gap_ood"] = gap
        return out
