"""Randomized Ensembled Double Q-learning (Chen et al., 2021), for continuous
actions: SAC with an ensemble of critics updated several times per env step,
each target the minimum over a random subset of them.

One update, Algorithm 1 in the order of REDQ's official code
(watchernyu/REDQ@7b5d1bf, redq/algos/redq_sac.py), each line one method
below or one of the equations under it::

    num_critic_updates times, each on a fresh replay batch:
        a' ~ pi(. | s'), log pi(a' | s')              sample_next_actions (SAC's)
        y = scale r + gamma (1 - d) (min over subset_size random target
            critics of Q_i(s', a') - alpha log pi(a' | s'))  compute_redq_td_target
        Q_i <- argmin sum_i E[(Q_i(s, a) - y)^2]   update_critic, compute_critic_loss
        Q_target <- tau Q + (1 - tau) Q_target
    actor <- argmin E[alpha log pi(a | s) - mean_i Q_i(s, a)] on the last batch
                                       update_actor, compute_soft_policy_loss (SAC's)
    alpha <- argmin log alpha sg(-log pi(a | s) - H_target)
                                                 compute_temperature_loss (SAC's)

The actor reads the critics from before the last critic step, and the
critic and actor steps read the temperature REDQ's code reads (0.2 before
its first step, read_temperature). The SVGD variant's kernel repulsion
between the critics is the KernelRepulsion extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents import update_steps
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
from ajax.agents.recurrent import RecurrentCarries, q_values, sample_replay
from ajax.agents.SAC.SAC import (
    compute_soft_policy_loss,
    compute_temperature_loss,
    sample_next_actions,
)
from ajax.agents.temperature import (
    SoftActorCriticState,
    TemperatureAuxiliaries,
    create_alpha_train_state,
    get_alpha,
    update_temperature,
)
from ajax.controller import critic_action_units, make_scored_action, policy_action_dim
from ajax.extensions.base import merge_metrics
from ajax.perf_utils import final_aux_scan
from ajax.schedule import Schedule
from ajax.state import LoadedTrainState, Transition
from ajax.types import Architecture

#: REDQ's reference initialisation (see REDQ.actor_critic_args).
GLOROT = "xavier_uniform"
#: MBPO's per-task target entropies (Janner et al., 2019), each the whole
#: action's: the default of REDQ's official code (watchernyu/REDQ@7b5d1bf,
#: redq/algos/core.py:10-13, read at redq/algos/redq_sac.py:57-58), here
#: under the ids of brax's versions of those gym MuJoCo tasks.
MBPO_TARGET_ENTROPY = {
    "hopper": -1.0,
    "halfcheetah": -3.0,
    "walker2d": -3.0,
    "ant": -4.0,
    "humanoid": -2.0,
}
#: The temperature REDQ's code reads until its first temperature step:
#: __init__ sets self.alpha = exp(log_alpha), then self.alpha = alpha, the
#: constructor's 0.2 (watchernyu/REDQ@7b5d1bf, redq/algos/redq_sac.py:61,
#: :80), likely a slip, kept as the original runs.
FIRST_ALPHA = 0.2


@dataclass(kw_only=True, eq=False, repr=False)
class REDQ(
    EntropyTemperature,
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """REDQ: an ensemble of num_critics SAC critics, each target the minimum
    over subset_size of them drawn at random, num_critic_updates critic
    steps per env step (the module's update).

    Defaults are REDQ's official code's (watchernyu/REDQ@7b5d1bf): ten
    critics, a subset of two, 20 critic steps per env step (utd_ratio),
    Adam at 3e-4 with torch's eps 1e-8 (redq_sac.py:48, :51), xavier-uniform
    layers, a log-probability subtracting log(1 - tanh^2 + 1e-6)
    (core.py:232).

    Args:
        alpha_init: The temperature's initial value: 1.0 as in REDQ's
            official code, which learns log_alpha from 0 (redq_sac.py:59-61).
            Its self.alpha = alpha at :80 then sets the temperature to the
            unused alpha=0.2 default until its first temperature step
            (:214-236), on the last of the 20 critic updates (utd_ratio) of
            the first env step that trains: those updates' targets and the
            first actor loss read 0.2, and so do Ajax's (FIRST_ALPHA).
        target_entropy_per_dim: The target entropy per action dimension, a
            schedule of the timestep, or "mbpo" for MBPO's per-task target
            entropy (MBPO_TARGET_ENTROPY), the default of REDQ's official
            code; an env id outside that table raises.
        num_critic_updates: The critic steps of each update, each on a fresh
            replay batch (SAC's same-named steps share one).
        subset_size: The random target critics whose minimum each target
            reads.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    # Wider than the shared temperature's Schedule: REDQ's also takes "mbpo".
    target_entropy_per_dim: Union[Schedule, Literal["mbpo"]] = -1.0  # type: ignore[assignment]
    num_critic_updates: int = 20
    num_critics: int = 10
    subset_size: int = 2
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(1e4)

    def actor_critic_args(self) -> dict[str, Any]:
        # The initialisation of REDQ's reference code: watchernyu/REDQ
        # @7b5d1bf applies weights_init_ (redq/algos/core.py:18-22,
        # xavier_uniform with gain 1, bias 0) to every Linear layer of
        # its policy and Q networks, the mean and log-std layers
        # included (:96-98, :194-198), whatever their comments say.
        # Recurrent REDQ (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its recurrent reference (pomdp-baselines), is
        # audited: at REDQ's, its memory cells' actors read 24-29/32.
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
        # torch's Adam eps, which the reference's Adams keep (redq_sac.py:48,
        # :51, torch 1.3.1); recurrent REDQ's reference (Ni et al.,
        # pomdp-baselines e7c19c3 policy_rnn.py:75, :93) keeps it too.
        return {"eps": 1e-8}

    def network_settings(self) -> dict[str, Any]:
        # REDQ's log-probability subtracts log(1 - tanh^2 + 1e-6)
        # (redq/algos/core.py:232).
        return {"squash_log_det_eps": 1e-6}

    @property
    def target_entropy(self) -> Schedule:
        """MBPO's target entropy for the env (target_entropy_per_dim
        "mbpo"), else the shared rule's: per dimension times the policy's
        action width."""
        per_dim = self.target_entropy_per_dim
        if isinstance(per_dim, str):
            width = policy_action_dim(self.env_args)
            return resolve_target_entropy(per_dim, width, self.env_id)
        return super().target_entropy

    def init(self, key: jax.Array, context: TrainContext) -> REDQState:
        """A squashed Gaussian actor, num_critics Q critics with their
        targets, and the temperature."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=self.num_critics,
        )
        collector_state = context.make_collector(collector_key)
        # REDQ's official critics read act_limit * tanh, the env's units
        # (watchernyu/REDQ@7b5d1bf, core.py:238): so do Ajax's.
        units = critic_action_units(context.env_args)
        return REDQState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state.replace(action_units=units),
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
        )

    def update(
        self, agent_state: REDQState, context: TrainContext
    ) -> tuple[REDQState, AuxiliaryLogs]:
        """num_critic_updates critic and target steps, each on a fresh replay
        batch, then one actor step and one temperature step on the last
        batch (Chen et al., 2021, Algorithm 1), each kept from its start.

        The actor step reads the critics from before the last critic step,
        as REDQ's official code, which computes the actor loss
        (redq_sac.py:220) before the critics' optimizers step (:242);
        Algorithm 1 reads them after. The critic and actor steps read
        read_temperature's temperature; the temperature step updates the
        learned one."""
        learned = agent_state.alpha

        def update_critic_on_a_fresh_batch(
            agent_state: REDQState, _: Any
        ) -> tuple[REDQState, Any]:
            sample_key, agent_state = agent_state.get_keys_and_new_state()
            critics_before = agent_state.critic_state.params
            batch, carries = sample_replay(
                agent_state,
                context.buffer,
                sample_key,
                context.recurrent,
                self.burn_in,
                self.stored_state,
            )
            agent_state, value_aux, value_metrics = self.update_critic(
                agent_state, batch, carries, context
            )
            agent_state = update_steps.update_target_networks(agent_state, self.tau)
            return agent_state, (
                value_aux,
                value_metrics,
                batch,
                carries,
                critics_before,
            )

        # A carry-only scan: only the last critic step's metrics, batch and
        # critics before it are kept (perf_utils.final_aux_scan).
        agent_state, last = final_aux_scan(
            update_critic_on_a_fresh_batch,
            read_temperature(agent_state),
            length=self.num_critic_updates,
        )
        value_aux, value_metrics, batch, carries, critics_before = last
        critic_state = agent_state.critic_state
        timestep = agent_state.collector_state.timestep
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state.replace(
                critic_state=critic_state.replace(params=critics_before)
            ),
            batch,
            carries,
            context,
        )
        stepped = stepped.replace(critic_state=critic_state, alpha=learned)
        # The actor step runs from the first update, since its log-probs feed
        # the temperature's, and is kept from policy_update_start.
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            self.policy_update_due(timestep),
            lambda _: (stepped, (policy_aux, policy_metrics)),
            agent_state.replace(alpha=learned),
        )
        tempered, temperature_aux = update_temperature(
            agent_state,
            log_probs,
            self.target_entropy_at(timestep, context.total_timesteps),
            compute_temperature_loss,
        )
        agent_state = update_steps.run_if(
            self.temperature_update_due(timestep), lambda _: tempered, agent_state
        )
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: REDQState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[REDQState, ValueAuxiliaries, dict]:
        """One step of the ensemble on the random-subset target."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        alpha = get_alpha(agent_state)
        # A truncated row bootstraps on its final observation (bootstrap_obs).
        dones = batch.terminated
        target_q, next_log_probs = compute_redq_td_target(
            agent_state.actor_state,
            agent_state.critic_state,
            critic_key,
            batch.next_obs,
            dones,
            batch.reward,
            self.gamma,
            alpha,
            self.subset_size,
            self.reward_scale,
            carries,
            context.executed,
            batch.next_a_expert,
        )

        def critic_loss(
            params: FrozenDict, target_q: jax.Array
        ) -> tuple[jax.Array, Any]:
            return compute_critic_loss(
                params,
                agent_state.critic_state,
                batch.obs,
                batch.action,
                target_q,
                next_log_probs,
                carries,
            )

        critic_state, aux, _, metrics = update_steps.critic_step(
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
        )
        return agent_state.replace(critic_state=critic_state), aux, metrics

    def update_actor(
        self,
        agent_state: REDQState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[REDQState, PolicyAuxiliaries, jax.Array, dict]:
        """The actor step on alpha log pi(a|s) - mean_i Q_i(s, a), the
        critics scoring a residual policy's executed actions; also returns
        its sample's log-probs, which the temperature step reads."""
        actor_key, agent_state = agent_state.get_keys_and_new_state()
        alpha = get_alpha(agent_state)
        critic_state = agent_state.critic_state
        transform = make_scored_action(context.executed, batch.a_expert)

        def policy_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
            loss, (log_probs, q_values, policy) = compute_soft_policy_loss(
                params,
                agent_state.actor_state,
                critic_state,
                batch.obs,
                alpha,
                actor_key,
                carries,
                jnp.mean,
                transform,
            )
            return loss, (loss, log_probs, q_values), policy.mean()

        actor_state, (loss, log_probs, q_mean), _, metrics = update_steps.actor_step(
            agent_state,
            policy_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            observations=batch.obs,
            raw_observations=batch.raw_obs,
            a_expert=batch.a_expert,
            actions=batch.action,
            critic_state=critic_state,
        )
        aux = PolicyAuxiliaries(
            policy_loss=loss, log_pi=log_probs.mean(), q_mean=q_mean.mean()
        )
        return agent_state.replace(actor_state=actor_state), aux, log_probs, metrics

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "REDQ"
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
    extension_context = RecurrentReplayAgent.extension_context.union(BELLMAN_CONTEXT)
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- REDQ's equations, in the order update calls them ------------------------
def resolve_target_entropy(
    per_dim: Union[float, Literal["mbpo"]], action_dim: int, env_id: object
) -> float:
    """per_dim per action dimension, or with "mbpo" MBPO's target entropy for
    env_id."""
    if not isinstance(per_dim, str):
        return per_dim * action_dim
    if per_dim != "mbpo":
        raise ValueError(
            f"target_entropy_per_dim takes a number or 'mbpo', not {per_dim!r}."
        )
    if not isinstance(env_id, str) or env_id not in MBPO_TARGET_ENTROPY:
        raise ValueError(
            f"target_entropy_per_dim='mbpo': MBPO's table has no {env_id!r}; it"
            f" has {sorted(MBPO_TARGET_ENTROPY)}."
        )
    return MBPO_TARGET_ENTROPY[env_id]


def read_temperature(agent_state: REDQState) -> REDQState:
    """agent_state with the temperature REDQ's code reads: FIRST_ALPHA before
    the first temperature step, exp(log_alpha) from it on."""
    log_alpha = agent_state.alpha.params["log_alpha"]
    read = jnp.where(agent_state.alpha.step == 0, jnp.log(FIRST_ALPHA), log_alpha)
    alpha = agent_state.alpha.replace(params=FrozenDict({"log_alpha": read}))
    return agent_state.replace(alpha=alpha)


def compute_redq_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    subset_size: int,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """REDQ's Bellman target: the minimum over a random subset of the target
    critics, at the next actions executed (executed, a residual policy's).
    Returns the target, no gradient flowing back through it, and the next
    actions' log-probs."""
    rewards = rewards * reward_scale
    sample_key, subset_key = jax.random.split(rng)
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, sample_key, carries, executed, next_a_expert
    )
    # The ensemble axis stays leading in sequence mode too, so the subset
    # sampling is the same.
    q_targets = q_values(
        critic_states,
        critic_states.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    sampled_indexes = jax.random.choice(
        subset_key, q_targets.shape[0], shape=(subset_size,), replace=False
    )
    min_q_target = jnp.min(q_targets[sampled_indexes], axis=0)

    target = rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)
    return jax.lax.stop_gradient(target), log_probs


def compute_critic_loss(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    next_log_probs: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> tuple[jax.Array, ValueAuxiliaries]:
    """Every critic regresses on the one target: the sum over the critics
    of each one's mean squared error, as REDQ's official
    mse_criterion(q_prediction_cat, y_q) * self.num_Q (redq_sac.py:207),
    whose per-critic Adam optimizers (:49-51) each see their critic's own
    error. A recurrent critic's error is its mean over time and batch."""
    q_preds = q_values(critic_states, critic_params, observations, actions, carries)
    loss = jnp.sum(
        jnp.mean((q_preds - target_q) ** 2, axis=tuple(range(1, q_preds.ndim)))
    )
    q_std, mean_pairwise_q_dist = q_ensemble_divergence(q_preds)
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
        ensemble_q_std=q_std.flatten(),
        mean_pairwise_q_dist=mean_pairwise_q_dist.flatten(),
    )


def q_ensemble_divergence(q_preds: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scale-free diagnostics for how spread out the critic ensemble is,
    computed with no gradient (telemetry only): the std across the critics of
    each (s, a) value, averaged over the batch (ensemble_q_std), and the mean
    L2 distance between the critics' flattened values over all off-diagonal
    pairs (mean_pairwise_q_dist)."""
    ensemble_values = jax.lax.stop_gradient(q_preds)
    ensemble_size = ensemble_values.shape[0]
    ensemble_std = jnp.std(ensemble_values, axis=0).mean()

    flat_values = ensemble_values.reshape(ensemble_size, -1)
    differences = flat_values[:, None, :] - flat_values[None, :, :]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1) + 1e-12)
    off_diagonal_sum = distances.sum() - jnp.trace(distances)
    mean_pairwise = off_diagonal_sum / (ensemble_size * (ensemble_size - 1))
    return ensemble_std, mean_pairwise


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class REDQState(SoftActorCriticState):
    """REDQ's state: a soft actor-critic's, nothing more."""


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_mean: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    log_probs: jax.Array
    # Scale-free ensemble divergence diagnostics (no gradient): whether the
    # ensemble diversified in function space.
    ensemble_q_std: jax.Array
    mean_pairwise_q_dist: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
REDQConfig = REDQ.declare_config(
    "gamma target_entropy tau learning_starts reward_scale num_critic_updates"
    " num_critics subset_size burn_in sequence_length stored_state"
)
