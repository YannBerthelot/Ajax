"""Soft Actor-Critic (Haarnoja et al., 2018a, 2018b), for continuous actions:
an actor maximising the critics' value plus its entropy, its temperature
learned towards a target entropy.

One update on a replay batch, each line one method below or one of the
equations under it::

    num_critic_updates critic steps on the batch:                 update_critic
        a' ~ pi(. | s'), log pi(a' | s')                    sample_next_actions
        y = scale r + gamma (1 - d) (min_i Q_target_i(s', a') - alpha log pi(a' | s'))
                                     compute_td_target, compute_soft_bellman_target
        Q_i <- argmin mean_i E[(Q_i(s, a) - y)^2]          compute_mean_critic_loss
    actor <- argmin E[alpha log pi(a | s) - min_i Q_i(s, a)]          update_actor
                                                        compute_soft_policy_loss
    alpha <- argmin log alpha sg(-log pi(a | s) - H_target)   update_temperature
                                                         compute_temperature_loss
    Q_target <- tau Q + (1 - tau) Q_target                update_target_networks

The actor's and the temperature's steps are kept from their starts
(policy_update_start, alpha_update_start); the temperature reads the
log-probabilities of a fresh sample of the stepped actor. A residual policy
makes the critics score the executed actions, from the expert's stored ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents import update_steps
from ajax.agents.action_pipeline import check_action_extensions, make_action_pipeline
from ajax.agents.critics import compute_mean_critic_loss
from ajax.agents.families import (
    BELLMAN_CONTEXT,
    EntropyTemperature,
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    PolicyUpdateStart,
    RecurrentReplayAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.recurrent import RecurrentCarries, actor_dist, q_values, sample_replay
from ajax.agents.temperature import (
    SoftActorCriticState,
    TemperatureAuxiliaries,
    create_alpha_train_state,
    get_alpha,
    update_temperature,
)
from ajax.controller import leave_unchanged
from ajax.extensions.base import merge_metrics
from ajax.networks.distributions import latent
from ajax.networks.networks import predict_value
from ajax.perf_utils import final_aux_scan
from ajax.state import LoadedTrainState, Transition
from ajax.types import Architecture

#: SAC's reference initialisation, Glorot-uniform kernels and zero biases in
#: both networks (see SAC.actor_critic_args).
GLOROT = "xavier_uniform"


@dataclass(kw_only=True, eq=False, repr=False)
class SAC(
    EntropyTemperature,
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """Soft Actor-Critic with a learned temperature (the module's update).

    Defaults are the learned-temperature version's (rail-berkeley/
    softlearning): 256-256 ReLU networks, two critics, Adam at 3e-4 with
    TensorFlow's eps 1e-8, tau 0.005, gamma 0.99, a target entropy of -1
    per action dimension, a 1e6 replay, batch 256 and 1e4 warm-up steps.
    The actor's and the temperature's steps wait for 2000 steps
    (policy_update_start, alpha_update_start), inert at that warm-up.

    Args:
        tau: The target critics' Polyak rate.
        fixed_alpha: Keep the temperature at alpha_init (SAC v1's fixed
            temperature); its loss is still computed for the logs.
        num_critics: The critics, whose minimum the targets and the actor
            read.
        num_critic_updates: The critic steps of each update, all on the
            same replay batch (REDQ's steps each sample a fresh one).
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    fixed_alpha: bool = False
    num_critics: int = 2
    num_critic_updates: int = 1
    policy_update_start: int = 2_000
    alpha_update_start: int = 2_000
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(1e4)

    def actor_critic_args(self) -> dict[str, Any]:
        # The initialisation of SAC's reference code, Glorot-uniform
        # kernels and zero biases in both: haarnoja/sac @8258e33
        # (sac/misc/mlp.py:24-25, :41-49: tf.contrib's
        # xavier_initializer(), uniform by default at tensorflow 1.4, its
        # environment.yml's) and rail-berkeley/softlearning @13cf187,
        # the learned-temperature code (softlearning/models/
        # feedforward.py:28-34: Keras Dense, glorot_uniform and zeros by
        # default at tensorflow 2.4.1, its requirements'). Both draw the
        # mean and the log-std from one output layer; here each head
        # takes Glorot's bound for its own width.
        # Recurrent SAC (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its reference (pomdp-baselines), is audited: at
        # the feedforward reference's, its memory cells' actors read 25-28/32.
        args: dict[str, Any] = {"squash": True}
        if self.memory is not None:
            return args
        args.update(
            {
                "encoder_kernel_init": GLOROT,
                "value_kernel_init": GLOROT,
                "mean_kernel_init": GLOROT,
                "log_std_kernel_init": GLOROT,
                "log_std_init": 0.0,
            }
        )
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        # Adam's eps is TensorFlow's default, 1e-8, which softlearning's
        # tf.train.AdamOptimizer keeps (a1253bc sac.py:236-244, :318-324);
        # recurrent SAC's reference (Ni et al., pomdp-baselines e7c19c3
        # policy_rnn.py:75, :93) keeps torch's, also 1e-8.
        return {"eps": 1e-8}

    def setup(self) -> None:
        # Gain mode (ExpertGains) acts through the default pipeline, which
        # dispatches no action extension yet (it runs the warm-up's chooser).
        check_action_extensions(
            self.extension_stack,
            dispatched=self.expert is not None and not self.gain_mode,
        )

    @property
    def gain_mode(self) -> bool:
        """Whether the policy chooses an expert's gains (ExpertGains)."""
        return self.env_args.policy_action_dim is not None

    def choose_behaviour(self, context: TrainContext) -> Optional[Callable]:
        # The action phase's pipeline (EDGE, JSRL, ValueBox) with an expert;
        # gain mode acts through the shared collector's.
        return make_action_pipeline(
            expert_policy=None if self.gain_mode else self.expert,
            recurrent=context.recurrent,
            env_args=context.env_args,
            extension_stack=context.extensions,
            total_timesteps=context.total_timesteps,
        )

    def init(self, key: jax.Array, context: TrainContext) -> SACState:
        """A squashed Gaussian actor, num_critics Q critics with their
        targets, and the temperature."""
        # Deliberate workaround: the fourth key fed SAC's built-in expert
        # prefill, now the ExpertPrefill extension, which draws its own. It
        # is still split and left unused so that every other key, and so
        # every run without a prefill, stays as it was.
        training_key, init_key, collector_key, _, evaluation_key = jax.random.split(
            key, 5
        )
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=self.num_critics,
        )
        collector_state = context.make_collector(collector_key)
        return SACState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
        )

    def update(
        self, agent_state: SACState, context: TrainContext
    ) -> tuple[SACState, AuxiliaryLogs]:
        """num_critic_updates critic steps on one replay batch, then the
        actor's and the temperature's steps on it, each kept from its start,
        then the target critics' Polyak update."""
        sample_key, agent_state = agent_state.get_keys_and_new_state()
        # Sequence replay burns the carries in (R2D2-style); a run with
        # memory refuses an expert (make_context), so the expert's stored
        # actions below are None there.
        batch, carries = sample_replay(
            agent_state,
            context.buffer,
            sample_key,
            context.recurrent,
            self.burn_in,
            self.stored_state,
        )
        # The share of the rows an expert acted in (no stored flags: none).
        expert_share = (
            jnp.zeros(()) if batch.is_expert is None else batch.is_expert.mean()
        )

        def update_critic_once(agent_state: SACState, _: Any) -> tuple:
            agent_state, value_aux, metrics = self.update_critic(
                agent_state, batch, carries, context
            )
            return agent_state, (value_aux, metrics)

        # A carry-only scan that keeps the last step's metrics alone
        # (perf_utils.final_aux_scan).
        agent_state, (value_aux, value_metrics) = final_aux_scan(
            update_critic_once, agent_state, length=self.num_critic_updates
        )
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state, batch, carries, context
        )
        # The actor step runs from the first update, since its log-probs feed
        # the temperature's, and is kept from policy_update_start.
        timestep = agent_state.collector_state.timestep
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            self.policy_update_due(timestep),
            lambda _: (stepped, (policy_aux, policy_metrics)),
            agent_state,
        )
        # The temperature's loss is computed for its logs; its step is kept
        # from alpha_update_start, never with fixed_alpha (alpha stays at
        # alpha_init).
        tempered, temperature_aux = update_temperature(
            agent_state,
            log_probs,
            self.target_entropy_at(timestep),
            compute_temperature_loss,
        )
        temperature_due = jnp.logical_and(
            not self.fixed_alpha, self.temperature_update_due(timestep)
        )
        agent_state = update_steps.run_if(
            temperature_due, lambda _: tempered, agent_state
        )
        agent_state = update_steps.update_target_networks(agent_state, tau=self.tau)
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux.replace(expert_frac_in_buffer=expert_share),
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: SACState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[SACState, ValueAuxiliaries, dict]:
        """The critics' step on SAC's soft target. Under a residual policy
        the bootstrap reads the executed next action, from the expert's
        stored action at s', so the critics are queried where they were
        trained."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        # A truncated row bootstraps on its final observation (bootstrap_obs).
        dones = batch.terminated
        scored = context.make_scored_actions(batch)
        target_q = compute_td_target(
            actor_state=agent_state.actor_state,
            critic_state=agent_state.critic_state,
            next_observations=batch.next_obs,
            dones=dones,
            rewards=batch.reward,
            gamma=self.gamma,
            alpha=get_alpha(agent_state),
            rng=critic_key,
            recurrent=context.recurrent,
            reward_scale=self.reward_scale,
            map_next_actions=scored.map_next_actions,
            carries=carries,
        )
        # The target extensions (MCVarianceCorrection's variance) read the
        # critics' predictions, from a pass of their own: the target comes
        # before the gradient-bearing pass.
        q_preds = None
        if "on_target" in context.extensions.implemented_phases():
            q_preds = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=jnp.concatenate(
                    (batch.obs, jax.lax.stop_gradient(batch.action)), axis=-1
                ),
            )

        def critic_loss(params: FrozenDict, target_q: jax.Array) -> tuple:
            return compute_mean_critic_loss(
                params,
                agent_state.critic_state,
                batch.obs,
                batch.action,
                target_q,
                carries=carries,
            )

        critic_state, aux, grad_norm, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target_q,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=dones,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            # The target's key (IBRL and LCBGatedBootstrap redraw its
            # next-action sample), the critics' predictions and the memory flag.
            target_extras={
                "rng_key": critic_key,
                "q_preds": q_preds,
                "recurrent": context.recurrent,
            },
        )
        value_aux = ValueAuxiliaries(
            critic_loss=aux.critic_loss,
            q_pred_min=aux.q_pred_min,
            var_preds=aux.var_preds,
            expert_frac_in_buffer=jnp.zeros(1),
            grad_norm=grad_norm,
        )
        return agent_state.replace(critic_state=critic_state), value_aux, metrics

    def update_actor(
        self,
        agent_state: SACState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[SACState, PolicyAuxiliaries, jax.Array, dict]:
        """The actor's step on SAC's soft policy loss, the critics scoring a
        residual policy's executed actions from the expert's stored ones,
        which the actor-loss extensions (OnlineBC, ImitationLoss) read as
        their clone target. Also returns the log-probs of a fresh
        sample of the stepped actor, which the temperature's step reads
        (CleanRL's SAC order; cited from memory, unverified)."""
        actor_key, sample_key, agent_state = agent_state.get_keys_and_new_state(2)
        alpha = get_alpha(agent_state)
        scored = context.make_scored_actions(batch)

        def policy_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
            loss, (log_probs, q_values, policy) = compute_soft_policy_loss(
                params,
                agent_state.actor_state,
                agent_state.critic_state,
                batch.obs,
                alpha,
                actor_key,
                carries,
                map_actions=scored.map_actions,
            )
            aux = PolicyAuxiliaries(
                policy_loss=loss,
                log_pi=log_probs.mean(),
                policy_std=latent(policy).stddev().mean(),
                q_min=q_values.mean(),
            )
            return loss, aux, policy.mean()

        actor_state, aux, grad_norm, metrics = update_steps.actor_step(
            agent_state,
            policy_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            observations=batch.obs,
            # The replay keeps them only for a run with an expert: None
            # without one, never the observations passed off as raw.
            raw_observations=batch.raw_obs,
            a_expert=batch.a_expert,
            actions=batch.action,
            critic_state=agent_state.critic_state,
        )
        policy = actor_dist(actor_state, actor_state.params, batch.obs, carries)
        _, log_probs = policy.sample_and_log_prob(seed=sample_key)
        return (
            agent_state.replace(actor_state=actor_state),
            aux.replace(grad_norm=grad_norm),
            jax.lax.stop_gradient(log_probs),
            metrics,
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "SAC"
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
            "warmup",
        }
    )
    # The action phase reaches only the action pipeline's slots, and
    # transform_action the pipeline, with an expert policy
    # (check_action_extensions); choose_warmup_action reaches SAC's pipeline,
    # or the default one under gain mode.
    supported_extension_phases = RecurrentReplayAgent.supported_extension_phases.union(
        {"action"}
    )
    # Its critic step's target_extras join the shared step's on_target keys.
    phase_batch_keys = {  # noqa: RUF012
        **update_steps.REPLAY_BATCH_KEYS,
        "on_target": update_steps.TARGET_KEYS.union(
            {"rng_key", "q_preds", "recurrent"}
        ),
    }
    # MCPretrain's and PhiRefresh's (the expert critic phi*, SACState's
    # fields), and BellmanPretrain's.
    extension_context = RecurrentReplayAgent.extension_context.union(
        BELLMAN_CONTEXT
    ).union({"critic_optimizer_args", "extensions", "network_args", "num_critics"})
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- SAC's equations, in the order update calls them --------------------------
def sample_next_actions(
    actor_state: Any,
    next_observations: jax.Array,
    key: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    map_next_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
) -> tuple[jax.Array, jax.Array]:
    """The bootstrap actions a' ~ pi(.|s') and their log-probabilities,
    summed over the action dimensions; the actions are those the critics
    score, map_next_actions(a') (a residual policy's executed ones)."""
    next_policy = actor_dist(
        actor_state, actor_state.params, next_observations, carries, bootstrap=True
    )
    next_actions, log_probs = next_policy.sample_and_log_prob(seed=key)
    next_actions = map_next_actions(next_actions)
    return next_actions, log_probs.sum(-1, keepdims=True)


def compute_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    rng: jax.Array,
    recurrent: bool,
    reward_scale: float = 1.0,
    map_next_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
    carries: Optional[RecurrentCarries] = None,
) -> jax.Array:
    """SAC's Bellman target r + gamma (1 - d)(min_i Q_target_i(s', a') -
    alpha log pi(a'|s')), a' ~ pi(.|s'), no gradient flowing back through
    it; the target extensions (IBRL, CriticBlend, MCVarianceCorrection) may
    change it before the critic loss reads it.

    With a residual policy, map_next_actions maps the sampled next actions
    to the executed ones, from the expert's stored next actions
    (ajax.controller.make_scored_actions): the bootstrap is queried where
    the critic was trained, clip(a_expert(s') + scale * pi(s'), -1, 1).
    Without it the target Q is out of distribution and training diverges.
    The carries (sequence replay) are set exactly when recurrent.
    """
    del recurrent
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, sample_key, carries, map_next_actions
    )
    q_targets = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    min_q_target = jnp.min(q_targets, axis=0, keepdims=False)
    target = compute_soft_bellman_target(
        min_q_target, log_probs, rewards, dones, gamma, alpha, reward_scale
    )
    return jax.lax.stop_gradient(target)


def compute_soft_bellman_target(
    next_q_values: jax.Array,
    next_log_probs: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    alpha: jax.Array,
    reward_scale: float = 1.0,
) -> jax.Array:
    """SAC's Bellman target r + gamma (1 - d)(Q' - alpha log pi(a'|s')) from
    the bootstrap value next_q_values (the critics' reduced value at (s', a')) and
    a''s log-probabilities, however the agent computed them: a target
    critic's pass (compute_td_target) or CrossQ's joint pass."""
    rewards = rewards * reward_scale
    return rewards + gamma * (1.0 - dones) * (next_q_values - alpha * next_log_probs)


def compute_soft_policy_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    map_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, Any]]:
    """mean(alpha log pi(a|s) - Q(s, a)), a ~ pi(.|s), Q the critics'
    values reduced by q_reduce over the ensemble axis, at map_actions(a),
    the actions the critics score (residual RL's executed ones).

    Returns the loss and (log_probs, q_values, policy). In sequence mode the fresh
    actions are queried at the critic's head, its memory reading the actions
    taken (ajax.agents.recurrent.q_values).
    """
    policy = actor_dist(actor_state, actor_params, observations, carries)
    return compute_soft_policy_objective(
        policy,
        critic_state,
        observations,
        alpha,
        rng,
        carries,
        q_reduce,
        map_actions,
    )


def compute_soft_policy_objective(
    policy: Any,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    map_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, Any]]:
    """compute_soft_policy_loss on the policy at observations, however
    the agent computed it (CrossQ's actor in training)."""
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = policy.sample_and_log_prob(seed=sample_key)
    actions = map_actions(actions)
    reduced_q_values = q_reduce(
        q_values(critic_state, critic_state.params, observations, actions, carries),
        axis=0,
    )
    log_probs = log_probs.sum(-1, keepdims=True)
    assert log_probs.shape == reduced_q_values.shape, (
        f"{log_probs.shape} != {reduced_q_values.shape}"
    )
    return (alpha * log_probs - reduced_q_values).mean(), (
        log_probs,
        reduced_q_values,
        policy,
    )


def compute_temperature_loss(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> tuple[jax.Array, TemperatureAuxiliaries]:
    """SAC's temperature loss, log alpha sg(-log pi - H_target)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (log_alpha * jax.lax.stop_gradient(-log_probs - target_entropy)).mean()
    return loss, TemperatureAuxiliaries(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


def compute_alpha_temperature_loss(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> tuple[jax.Array, TemperatureAuxiliaries]:
    """alpha (H - H_target), H = -mean log pi: the temperature's loss in
    CrossQ's authors' SBX fork (adityab/CrossQ@a318a26,
    sbx/sac/sac.py:414-416), alpha in place of log alpha, so its gradient is
    alpha times compute_temperature_loss's (SBX later took the log)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = alpha * jax.lax.stop_gradient(-log_probs.mean() - target_entropy)
    return loss, TemperatureAuxiliaries(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class SACState(SoftActorCriticState):
    """SAC's state, with the frozen expert critic phi* its extensions fill."""

    expert_critic_params: Optional[Any] = None
    expert_v_min: Optional[Any] = None
    expert_v_max: Optional[Any] = None
    # Mutable phi* state for periodic self-consistent refresh (None when
    # refresh disabled).
    expert_critic_state: Optional[LoadedTrainState] = None


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array  # alpha log pi - Q, before the extensions' terms
    log_pi: jax.Array  # entropy proxy; tracks target_entropy
    policy_std: jax.Array  # mean unsquashed std; lower = more deterministic
    q_min: jax.Array  # Q(s, pi(s)): what the policy optimises
    # Before clipping; the loss builds these metrics before the gradient
    # exists, update_actor fills it in.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    var_preds: jax.Array  # inter-critic variance
    expert_frac_in_buffer: jax.Array  # fraction of sampled batch flagged as expert
    grad_norm: jax.Array  # before clipping


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
