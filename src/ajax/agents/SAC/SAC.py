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
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents import update_steps
from ajax.agents.critics import compute_mean_critic_loss
from ajax.agents.families import (
    BELLMAN_CONTEXT,
    EncoderSharing,
    EntropyTemperature,
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    PolicyUpdateStart,
    RecurrentReplayAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    ni_initialisation,
    q_values,
    sample_replay,
)
from ajax.agents.temperature import (
    SoftActorCriticState,
    TemperatureAuxiliaries,
    create_alpha_train_state,
    get_alpha,
    keep_actor_then_update_temperature,
)
from ajax.buffers.utils import get_buffer
from ajax.controller import leave_unchanged
from ajax.extensions.base import merge_metrics
from ajax.networks.distributions import latent
from ajax.random_keys import split_by_index
from ajax.state import LoadedTrainState, Transition
from ajax.types import Architecture, BufferType

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
    EncoderSharing,
    RecurrentReplayAgent,
):
    """Soft Actor-Critic with a learned temperature (the module's update).

    Defaults are the learned-temperature version's (rail-berkeley/
    softlearning): 256-256 ReLU networks, two critics, Adam at 3e-4 with
    TensorFlow's eps 1e-8, tau 0.005, gamma 0.99, a target entropy of -1
    per action dimension, a 1e6 replay, batch 256 and 1e4 warm-up steps.
    The actor and the temperature step with the critics from the first
    update, as softlearning's single session.run of all its training ops
    does (a1253bc sac.py:348-353); policy_update_start and
    alpha_update_start can delay them.

    n_step above 1 is Ajax's and not SAC's: the critics' target sums n_step
    rewards before it bootstraps, gamma^n in place of gamma, stopping at an
    episode's end. The return is uncorrected, as Rainbow's is (Hessel et al.
    2018): no importance weights for steps the current policy would not have
    taken, and no entropy bonus for the steps between. The first update waits
    for each env's replay to hold n_step rows, whatever learning_starts.

    Args:
        tau: The target critics' Polyak rate.
        fixed_alpha: Keep the temperature at alpha_init (SAC v1's fixed
            temperature); its loss is still computed for the logs.
        num_critics: The critics, whose minimum the targets and the actor
            read.
        num_critic_updates: The critic steps of each update, all on the
            same replay batch (REDQ's steps each sample a fresh one).
        n_step: The steps of the return each target sums (1: SAC's own). A
            run with memory refuses more than 1.
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
    n_step: int = 1
    buffer_size: int = int(1e6)
    # None: 256 rows, or 64 windows with memory (RecurrentReplayAgent).
    batch_size: Optional[int] = None  # type: ignore[assignment]
    learning_starts: int = int(1e4)

    def check(self) -> None:
        if self.n_step < 1:
            raise ValueError(f"SAC's n_step is at least 1, got {self.n_step}.")
        if self.n_step > 1 and self.memory is not None:
            raise ValueError(
                "SAC's n_step folds windows of feedforward rows: a run with"
                " memory replays windows of its own, so it takes n_step=1."
            )

    def run_capabilities(self) -> frozenset[str]:
        capabilities = super().run_capabilities()
        if self.n_step == 1:
            return capabilities
        return capabilities.difference({"one_step_target"})

    def make_replay_buffer(self) -> BufferType:
        if self.n_step == 1:
            return super().make_replay_buffer()
        batch_size = self.batch_size
        assert batch_size is not None  # resolved before the agent is built
        return get_buffer(
            self.buffer_size,
            batch_size,
            self.n_envs,
            n_step=self.n_step,
            gamma=self.gamma,
        )

    def replay_window(self) -> Optional[int]:
        return self.n_step if self.n_step > 1 else super().replay_window()

    def update_start(self) -> int:
        """learning_starts, or later: a window needs n_step rows per env."""
        return max(self.learning_starts, self.n_step * self.n_envs)

    def uniform_steps(self) -> Optional[int]:
        """As the family's, whatever update_start waits for."""
        return self.learning_starts

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
        # Recurrent SAC (memory set) starts as its reference, Ni et al.'s.
        args: dict[str, Any] = {"squash": True}
        if self.memory is not None:
            args.update(ni_initialisation())
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
        collector_state = context.make_collector(
            collector_key, include_seam=self.n_step > 1
        )
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

        def update_critic_once(agent_state: SACState, _: Any) -> tuple:
            agent_state, value_aux, metrics = self.update_critic(
                agent_state, batch, carries, context
            )
            return agent_state, (value_aux, metrics)

        # A carry-only scan that keeps the last step's metrics alone, each
        # step on a key of its index (update_steps.scan_on_indexed_keys).
        agent_state, (value_aux, value_metrics) = update_steps.scan_on_indexed_keys(
            update_critic_once, agent_state, self.num_critic_updates
        )
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state, batch, carries, context
        )
        agent_state, (policy_aux, policy_metrics), temperature_aux = (
            keep_actor_then_update_temperature(
                self,
                agent_state,
                stepped,
                (policy_aux, policy_metrics),
                log_probs,
                context.extensions,
                compute_temperature_loss,
                self.fixed_alpha,
            )
        )
        agent_state = update_steps.update_target_networks(agent_state, tau=self.tau)
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux,
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
        dones = update_steps.bootstrap_cuts(batch.terminated, context.extensions)
        scored = context.make_scored_actions(batch)
        # An n-step row bootstraps as many steps ahead as it sums.
        discount = batch.bootstrap_discount
        gamma = self.gamma if discount is None else discount
        target_q, next_actions, next_values = compute_td_target(
            actor_state=agent_state.actor_state,
            critic_state=agent_state.critic_state,
            next_observations=batch.next_obs,
            dones=dones,
            rewards=batch.reward,
            gamma=gamma,
            alpha=get_alpha(agent_state),
            key=critic_key,
            reward_scale=self.reward_scale,
            map_next_actions=scored.map_next_actions,
            carries=carries,
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

        agent_state, aux, grad_norm, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target_q,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=dones,
            gamma=gamma,
            reward_scale=self.reward_scale,
            next_actions=next_actions,
            next_values=next_values,
        )
        value_aux = ValueAuxiliaries(
            critic_loss=aux.critic_loss,
            q_pred_min=aux.q_pred_min,
            var_preds=aux.var_preds,
            grad_norm=grad_norm,
        )
        return agent_state, value_aux, metrics

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
        (CleanRL's order: vwxyzjn/cleanrl@fe8d8a0,
        cleanrl/sac_continuous_action.py:279-292)."""
        actor_key, sample_key = split_by_index(
            update_steps.derive_actor_key(agent_state), 2
        )
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
            batch,
            agent_state.critic_state,
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
            "entropy_temperature",
            "next_action_target",
            "obs_networks",
            "off_policy",
            "policy_density",
            "one_step_target",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
            "warmup",
        }
    )
    # MCPretrain's and PhiRefresh's (the expert critic phi*, SACState's
    # fields), and BellmanPretrain's.
    extension_context = RecurrentReplayAgent.extension_context.union(
        BELLMAN_CONTEXT
    ).union({"critic_optimizer_args", "extensions", "network_args", "num_critics"})
    stores_expert_actions = True
    supported_buffer_fields = RecurrentReplayAgent.supported_buffer_fields.union(
        {"raw_next_obs", "return_to_go"}
    )


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
    gamma: Union[float, jax.Array],
    alpha: jax.Array,
    key: jax.Array,
    reward_scale: float = 1.0,
    map_next_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
    carries: Optional[RecurrentCarries] = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """SAC's Bellman target r + gamma (1 - d)(min_i Q_target_i(s', a') -
    alpha log pi(a'|s')), a' ~ pi(.|s'), no gradient flowing back through
    it, the next actions a' it scored and the value it bootstrapped,
    min_i Q_target_i(s', a') - alpha log pi(a'|s'); the target extensions
    (IBRL, CriticBlend, MCVarianceCorrection) may change it before the
    critic loss reads it.

    With a residual policy, map_next_actions maps the sampled next actions
    to the executed ones, from the expert's stored next actions
    (ajax.controller.make_scored_actions): the bootstrap is queried where
    the critic was trained, clip(a_expert(s') + scale * pi(s'), -1, 1).
    Without it the target Q is out of distribution and training diverges.
    The carries are set for sequence replay alone.
    """
    sample_key, _ = jax.random.split(key)
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
    next_values = jax.lax.stop_gradient(min_q_target - alpha * log_probs)
    return jax.lax.stop_gradient(target), next_actions, next_values


def compute_soft_bellman_target(
    next_q_values: jax.Array,
    next_log_probs: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: Union[float, jax.Array],
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
    key: jax.Array,
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
        key,
        carries,
        q_reduce,
        map_actions,
    )


def compute_soft_policy_objective(
    policy: Any,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    alpha: jax.Array,
    key: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    map_actions: Callable[[jax.Array], jax.Array] = leave_unchanged,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, Any]]:
    """compute_soft_policy_loss on the policy at observations, however
    the agent computed it (CrossQ's actor in training)."""
    sample_key, _ = jax.random.split(key)
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


# -- State and logged metrics -------------------------------------------------
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
    grad_norm: jax.Array  # before clipping


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
