"""Expert-guidance research features as composable :class:`Extension`s.

* :class:`OnlineBC`              — value-weighted decaying BC loss term.
* :class:`ImitationLoss`         — plain BC loss term towards the expert.
* :class:`ResidualPolicy`        — execute ``clip(a_expert + scale·a_pi)``.
* :class:`ExpertGains`           — the action sets the expert's gains (gain mode).
* ExpertWarmup                  — the expert acts on a share of the warm-up steps.
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

from ajax.environments.utils import critic_action_units
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.observe import ExpertActionInput
from ajax.networks.networks import phi_star_critic_state, predict_value
from ajax.schedule import progress


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


def extension_expert_actions(ext: Any, raw_observations: Any, stored: Any) -> jax.Array:
    """The extension's expert's actions at raw_observations (a replayed
    batch's s', or the step collection acts on): the stored ones
    (reads_stored_expert_actions), else its call on the raw observations."""
    if ext.reads_stored_expert:
        return jax.lax.stop_gradient(stored)
    return jax.lax.stop_gradient(ext.expert_policy(raw_observations))


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

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def actor_loss(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        context: ExtensionContext,
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
        train_frac = progress(context.step, context.total_steps)
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
    return jnp.clip(1.0 - progress(step, anneal_frac * total_steps), 0.0, 1.0)


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


def td3bc_lambda(
    critic_state: Any, observations: jax.Array, pi_mean: jax.Array, alpha: float
) -> jax.Array:
    """TD3+BC's ``lambda = alpha / mean|Q_1(s, pi(s))|``, held constant,
    with Q_1 the first critic, the one TD3's actor follows
    (sfujim/TD3_BC@8791ad7, TD3_BC.py:141-143: ``lmbda = alpha /
    Q.abs().mean().detach()``, actor loss ``-lmbda * Q.mean() +
    mse_loss(pi, action)``)."""
    x = jnp.concatenate([observations, jax.lax.stop_gradient(pi_mean)], axis=-1)
    q1 = predict_value(critic_state, critic_state.params, x)[0]
    return jax.lax.stop_gradient(alpha / jnp.mean(jnp.abs(q1)))


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
    None. ``alpha`` makes the actor loss TD3+BC's (Fujimoto and Gu, 2021),
    ``λ · (the agent's own) + coef · mean((μ(s) - a)²)`` with ``λ = alpha /
    mean|Q_1(s, μ(s))|`` (:func:`td3bc_lambda`), the actions compared in
    the env's units and without the ``/ 4``: the term adds ``(λ - 1)`` times
    the agent's own loss (``agent_loss`` in its actor batch), already
    counted. On TD3, whose own loss is ``-mean Q_1``, that is the paper's
    ``-λ mean Q + mean((π(s) - a)²)``; ``anneal_frac`` and the filter act
    on the cloning part only. With both, on TD3, a transition
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
    action_scale: Optional[tuple] = None  # with alpha: the env's half-ranges
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

    def uses_expert_actions_as_policy_actions(self) -> bool:
        # The replayed target clones the stored actions, the policy's own.
        return self.target == "expert"

    def needs(self) -> frozenset[str]:
        """The filter and ``alpha`` also read the Q critic; the replayed
        target needs a replay."""
        q = {"q_critic"} if self.q_filter or self.alpha is not None else set()
        replay = {"replay"} if self.target == "replay" else set()
        return super().needs().union(q, replay)

    def reads(self) -> Mapping[str, frozenset[str]]:
        """The filter and ``alpha`` also read the critic from the actor-loss
        batch, the replayed target its rows' actions."""
        keys = self.batch_keys["actor_loss"]
        if self.q_filter or self.alpha is not None:
            keys = keys.union({"critic_state"})
        if self.alpha is not None:
            keys = keys.union({"agent_loss"})
        if self.target == "replay":
            keys = keys.union({"actions"})
        return {"actor_loss": keys}

    def bind_to_agent(self, **agent_context: Any) -> "ImitationLoss":
        """Read the expert's stored actions when it is the agent's; with
        ``alpha``, the env's action half-ranges (TD3+BC compares env actions);
        with the filter, bind the agent's replay buffer (its logged share
        samples it) and SAC's observation augmentation."""
        expert = self.target == "expert"
        stored = expert and reads_stored_expert_actions(self, agent_context)
        bound = dataclasses.replace(self, reads_stored_expert=stored)
        env_args = agent_context.get("env_args")
        if self.alpha is not None and env_args is not None:
            units = critic_action_units(env_args.env, env_args.env_params)
            scale = None if units is None else units[1]
            bound = dataclasses.replace(bound, action_scale=scale)
        if not self.q_filter:
            return bound
        exts = agent_context.get("extensions", ())
        if any("transform_action" in e.implemented_phases() for e in exts):
            raise ValueError(
                "ImitationLoss(q_filter=True) reads the critic at the policy's"
                " mean action; ResidualPolicy's critic reads the executed one."
            )
        return dataclasses.replace(
            bound, buffer=agent_context.get("buffer", self.buffer)
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
        context: ExtensionContext,
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
        diff = pi_mean - a_expert
        if self.alpha is not None and self.action_scale is not None:
            diff = diff * jnp.asarray(self.action_scale, diff.dtype)  # env units
        sq = jnp.mean(jnp.square(diff), axis=-1, keepdims=True)
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
            sq = sq * anneal_weight(context.step, context.total_steps, self.anneal_frac)
        if self.alpha is None:
            term = self.coef * jnp.mean(sq) / 4.0
        else:
            lam = td3bc_lambda(
                batch["critic_state"], batch["observations"], pi_mean, self.alpha
            )
            term = (lam - 1.0) * batch["agent_loss"] + self.coef * jnp.mean(sq)
        return term, {"imitation/term": term, "imitation/mse": mse}

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        """``imitation/q_filter_frac``, the share of a replay batch the
        Q-filter keeps (an agent with a replay, which it binds)."""
        from ajax.buffers.utils import get_batch_from_buffer
        from ajax.environments.interaction import get_pi

        del ext_state, context
        if self.buffer is None:
            return {}
        batch = get_batch_from_buffer(
            self.buffer, agent_state.collector_state.buffer_state, rng
        )
        obs = batch.obs  # with the agent's observation features
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

    The executed action is ``clip(a_expert + scale·a_pi, -1, 1)``, from the
    agent's tracked expert. Collection applies it before the action is
    chosen (the warm-up draws executed actions), the replay stores the
    executed action, the critics score it (the TD-target bootstrap at
    ``next_a_expert`` and the actor's policy gradient flow through the same
    transform, :func:`ajax.controller.executed_actions`), and evaluation
    acts through it. Its presence alone turns residual RL on, with the
    agent's expert policy.
    """

    scale: float = 1.0
    name: str = "residual_policy"
    # Its base is the agent's expert, which an agent learning from that
    # expert's labels alone (DAgger) would learn to cancel, and which an
    # imagined latent state cannot be handed (obs_networks).
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "obs_networks", "reward_learning"}
    )

    def transform_action(self, actions: jax.Array, inputs: Any) -> jax.Array:
        """``clip(a_expert + scale·a_pi, -1, 1)`` — the residual mix."""
        return residual_action_transform(
            actions, inputs.expert_action, scale=self.scale
        )

    def policy_action_for_expert(self, policy_actions: jax.Array) -> jax.Array:
        # A residual of 0 executes clip(a_expert), the expert's own action.
        return jnp.zeros_like(policy_actions)


@dataclass(frozen=True)
class ExpertGains(Extension):
    """Gain mode: the policy's action sets the agent's expert's learnable
    gains, ``anchor_gains · 10**a`` per gain, and the expert acts with them
    (its ``step_with_gains``, which advances its state). The replay and the
    critics hold the policy's action, the gains: the agent acts in a space
    as wide as the expert's learnable gains, and the warm-up draws uniform
    gains. The expert needs ``learnable_fields`` and ``anchor_gains``
    (``register_learnable_gains``).
    """

    name: str = "expert_gains"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "obs_networks", "reward_learning"}
    )

    def policy_action_dim(self, env_args: Any, expert: Any) -> int:
        if not (hasattr(expert, "learnable_fields") and hasattr(expert, "init_state")):
            raise ValueError(
                "ExpertGains needs a stateful expert with learnable_fields"
                " (registered via register_learnable_gains)."
            )
        if any(isinstance(e, ExpertActionInput) for e in env_args.observers):
            raise ValueError(
                "Gain mode cannot take the expert's action as an observation"
                " feature (ExpertActionInput): the actor's output sets the"
                " gains the expert acts with, so the expert's action does not"
                " exist before the actor reads its observation."
            )
        return len(expert.learnable_fields)

    def actuate(self, actions: jax.Array, inputs: Any) -> tuple[jax.Array, Any]:
        expert = inputs.expert
        gains = expert.anchor_gains * jnp.exp(jnp.log(10.0) * actions)
        return expert.step_with_gains(inputs.expert_state, inputs.raw_obs, gains)

    def policy_action_for_expert(self, policy_actions: jax.Array) -> jax.Array:
        # An action of 0 sets the anchor gains: the expert's own behaviour.
        return jnp.zeros_like(policy_actions)


@dataclass(frozen=True)
class ExpertWarmup(Extension):
    """During the warm-up, the agent's expert acts on a share fraction of the
    steps and uniform random actions on the others (SAC's former
    expert_fraction). One draw decides each step for every env, as SAC drew
    it. It acts in the controller only: no state and no key in the learner.
    The replay flags the steps the expert took, where it keeps is_expert."""

    fraction: float
    name: str = "expert_warmup"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})

    def choose_warmup_action(
        self, uniform_actions: jax.Array, inputs: Any
    ) -> tuple[jax.Array, jax.Array]:
        # Its own key: on the uniform actions' key the decision would reuse
        # their first draw (env 0, dimension 0).
        decision_key = jax.random.fold_in(inputs.key, 1)
        expert_took = jax.random.uniform(decision_key) < self.fraction
        actions = jnp.where(expert_took, inputs.expert_action, uniform_actions)
        return actions, expert_took


@dataclass(frozen=True)
class StorePolicyAction(Extension):
    """An off-policy correctness ablation (SAC's former store_policy_action):
    the replay stores the action the policy chose, not the one executed, so
    the rows an expert, a warm-up or a residual acted in no longer hold the
    action that produced their reward and next observation. It acts in the
    controller only: no state and no key in the learner."""

    name: str = "store_policy_action"

    def choose_replay_action(
        self, executed_actions: jax.Array, policy_actions: jax.Array
    ) -> jax.Array:
        del executed_actions
        return policy_actions


@dataclass(frozen=True)
class JSRLCurriculum(Extension):
    """True JSRL per-episode handoff (Uchendu et al. 2023).

    While step_in_episode < H_t the expert acts; otherwise the learner
    does. H_t decays linearly from episode_length to 0 over the first
    decay_frac fraction of training.

    The action phase owns the per-episode substitution, in the action
    stage's pre-warm-up slot (ajax.environments.action_stage). It reads
    post_warmup_action and expert_action off the stage's dict and each
    env's step_in_episode off agent_state.collector_state, which
    collect_experience counts for an extension that reads it. It writes
    expert_mask back, which flags the steps the expert took.

    Evaluations log the guide horizon H_t (jsrl/horizon) and the share of
    envs the expert drove on the last collection step (jsrl/expert_frac).
    """

    episode_length: int = 1000
    decay_frac: float = 0.5
    name: str = "jsrl_curriculum"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})
    action_slot: ClassVar[str] = "pre_warmup"
    reads_step_in_episode: ClassVar[bool] = True

    def uses_expert_actions_as_policy_actions(self) -> bool:
        # It executes the expert's action where the policy's goes.
        return True

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array | None:
        """Substitute the expert action while step_in_episode < H_t."""
        del ext_state, rng
        post_warmup_action = obs["post_warmup_action"]
        expert_action = obs["expert_action"]

        H_t = self.horizon(agent_state.collector_state.timestep, context.total_steps)
        step_in_ep = agent_state.collector_state.step_in_episode
        use_expert_jsrl = step_in_ep.astype(jnp.float32) < H_t
        use_expert_jsrl = use_expert_jsrl.reshape(
            (-1,) + (1,) * (post_warmup_action.ndim - 1)
        )
        # The stage flags the steps the expert took (is_expert_flag).
        obs["expert_mask"] = use_expert_jsrl
        return jnp.where(use_expert_jsrl, expert_action, post_warmup_action)

    def horizon(self, timestep: Any, total_steps: int) -> jax.Array:
        """``H_t``: ``episode_length`` at the start, falling linearly to 0 at
        ``decay_frac`` of the run."""
        train_frac = progress(timestep, total_steps)
        curriculum_progress = jnp.clip(train_frac / self.decay_frac, 0.0, 1.0)
        return self.episode_length * (1.0 - curriculum_progress)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del ext_state, rng
        collector = agent_state.collector_state
        return {
            "jsrl/horizon": self.horizon(collector.timestep, context.total_steps),
            "jsrl/expert_frac": collector.last_expert_frac,
        }


__all__ = [
    "ExpertGains",
    "ExpertWarmup",
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
    ``expert_policy`` (SAC, TD3, REDQ, ASAC) binds its replay buffer and its
    expert.
    """

    buffer: Any = None
    expert_policy: Optional[Callable] = None
    name: str = "expert_value_metrics"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "q_critic", "replay"}
    )
    context: ClassVar[frozenset[str]] = frozenset({"buffer", "expert_policy"})
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

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
        return dataclasses.replace(self, buffer=buffer, expert_policy=expert)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        from ajax.buffers.utils import get_batch_from_buffer

        del ext_state, context
        batch = get_batch_from_buffer(
            self.buffer, agent_state.collector_state.buffer_state, rng
        )
        obs, actions = batch.obs, batch.action
        # The expert's actions as collection stored them (from its state).
        a_expert = batch.a_expert
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
