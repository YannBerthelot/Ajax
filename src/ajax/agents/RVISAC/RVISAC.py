"""RVI-SAC (Hisaki & Ono, ICML 2024, https://arxiv.org/abs/2408.01972): SAC
for the average reward, its critics learned by RVI Q-learning.

The critics learn the differential soft Q, bootstrapped without discount and
without a cut at terminations, on r - xi, where the offset xi follows the
batch mean of the target's next values (the reference function f(Q),
ajax.agents.centering). A termination is a step into the reset state, which
the replay stores as its next observation (interaction.bootstrap_obs), and
costs c: a Lagrange multiplier that holds the termination rate, which a reset
critic learns, at most at eps (ajax.agents.reset_cost). One update on a replay
batch, in the order of the authors' code (yhisaki/average-reward-drl@b5b0d31,
average_reward_drl/algorithms/rvi_sac.py:139-254)::

    a' ~ pi(. | s'), V'(s') = min_i Q_target_i(s', a') - alpha log pi(a' | s')
                                                  sample_next_actions (SAC's)
    y = r - c 1(terminated) - xi + V'(s')                compute_rvi_td_target
    Q_i <- argmin sum_i E[(Q_i(s, a) - y)^2]                    update_critic
    Q_reset <- argmin E[(Q_reset(s, a) - (1(terminated) - xi_reset
                         + Q_reset_target(s', a')))^2]    step_reset_critic
    xi <- (1 - kappa) xi + kappa mean V'(s')         RVIBatchMean.next_offset
    xi_reset <- (1 - kappa) xi_reset + kappa mean Q_reset_target(s', a')
    actor <- argmin E[alpha log pi(a | s) - min_i Q_i(s, a)]
                                  update_actor, compute_soft_policy_loss (SAC's)
    alpha <- argmin alpha sg(-log pi(a | s) - H_target)
                                     compute_alpha_temperature_loss (CrossQ's)
    c <- max(0, c after an Adam step down -c (xi_reset - eps))  step_reset_cost
    Q_target, Q_reset_target <- tau Q + (1 - tau) Q_target
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents import update_steps
from ajax.agents.centering import (
    CenteringInputs,
    CenteringRule,
    RVIBatchMean,
    TDErrorCentering,
)
from ajax.agents.critics import compute_mean_critic_loss, critic_value_readers
from ajax.agents.families import (
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
from ajax.agents.reset_cost import (
    ORTHOGONAL_THIRD,
    TORCH_ADAM_EPS,
    ResetCostSettings,
    ResetCostState,
    create_reset_cost_state,
    reset_charge,
    reset_critic_input,
    step_reset_cost,
    step_reset_critic,
    update_reset_target,
)
from ajax.agents.SAC.SAC import (
    compute_alpha_temperature_loss,
    compute_soft_policy_loss,
    sample_next_actions,
)
from ajax.agents.temperature import (
    SoftActorCriticState,
    TemperatureAuxiliaries,
    create_alpha_train_state,
    get_alpha,
    update_temperature,
)
from ajax.controller import feature_width, policy_action_dim
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import merge_metrics
from ajax.random_keys import derive_key
from ajax.state import Transition
from ajax.types import Architecture

#: The authors' twin critics (rvi_sac.py:52-61), whose minimum the target and
#: the actor read.
NUM_CRITICS = 2


@dataclass(kw_only=True, eq=False, repr=False)
class RVISAC(
    EntropyTemperature,
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """RVI-SAC: SAC on the average reward by RVI Q-learning (the module's
    update). It takes no discount.

    Every default is the authors' (their Table 1, experiments/conf/algo/
    rvi_sac.yaml and common.yaml): one learning rate, 3e-4, for every Adam
    (torch's eps 1e-8), no gradient clip, the temperature learned from 1
    towards -dim A.

    Args:
        centering: The reference function f(Q) the offset xi follows
            (ajax.agents.centering): the batch mean of the next values at
            rate 0.005, their kappa (rvi_sac.yaml; the constructor's 0.01 is
            not their runs'), or another rule (ReferencePoint is their
            rvi_sac_with_reference).
        reset_cost: The reset cost's settings (ajax.agents.reset_cost):
            their termination-rate target eps 1e-3, a 64-64 reset critic, its
            rate following at kappa; None charges no cost.
        fixed_alpha: Keep the temperature at alpha_init, as SAC's option.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    tau: float = 0.005
    reward_scale: float = 1.0
    fixed_alpha: bool = False
    centering: CenteringRule = field(default_factory=RVIBatchMean)
    reset_cost: Optional[ResetCostSettings] = field(default_factory=ResetCostSettings)
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(1e4)

    def actor_critic_args(self) -> dict[str, Any]:
        # The authors' initialisation (rvi_sac.py:54-89, modules.py:40-43,
        # :79-83): every layer orthogonal with gain sqrt(1/3) and a zero
        # bias. Their actor draws the mean and the log-std from one output
        # layer, one orthogonal matrix; here each head draws its own.
        # Recurrent RVI-SAC (memory set) keeps Ajax's initialisation, as
        # recurrent SAC and ASAC do until Ni et al.'s is audited.
        args: dict[str, Any] = {"squash": True}
        if self.memory is not None:
            return args
        init = ("encoder_kernel_init", "mean_kernel_init", "log_std_kernel_init")
        args.update(dict.fromkeys((*init, "value_kernel_init"), ORTHOGONAL_THIRD))
        args.update({"log_std_init": 0.0})  # the log-std head's zero bias
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": TORCH_ADAM_EPS}

    def network_settings(self) -> dict[str, Any]:
        # The authors evaluate at the mean of tanh at mu +- sigma
        # (modules.py:115-116), not at tanh(mu); their log-prob is torch's
        # exact TanhTransform, Ajax's default.
        return {"squash_sigma_point_mean": True}

    def init(self, key: jax.Array, context: TrainContext) -> RVISACState:
        """A squashed Gaussian actor, two Q critics with their targets, the
        temperature, the offset xi from 0 and the reset cost's state."""
        training_key, init_key = derive_key(key, "updates"), derive_key(key, "networks")
        collector_key = derive_key(key, "collection")
        evaluation_key = derive_key(key, "evaluation")
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=NUM_CRITICS,
        )
        reset_state = None
        if self.reset_cost is not None:
            # A key derived by name: the other networks draw what they would
            # without a reset cost.
            reset_key = derive_key(init_key, "reset_critic")
            observation_shape, _ = get_state_action_shapes(context.env_args.env)
            width = observation_shape[-1] + feature_width(context.env_args)
            width += policy_action_dim(context.env_args)
            reset_state = create_reset_cost_state(
                reset_key,
                width,
                self.reset_cost.architecture,
                self.reset_cost.learning_rate,
            )
        return RVISACState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=context.make_collector(collector_key),
            offset=jnp.zeros(()),
            reset_cost=reset_state,
        )

    def update(
        self, agent_state: RVISACState, context: TrainContext
    ) -> tuple[RVISACState, AuxiliaryLogs]:
        """One update on a replay batch: the critics', the reset critic's,
        the offsets', the actor's, the temperature's, the reset cost's and
        the targets' steps, in the authors' order."""
        sample_key, agent_state = agent_state.get_keys_and_new_state()
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
        # The temperature reads the actor's own sample, drawn before its step
        # (rvi_sac.py:216-241).
        timestep = agent_state.collector_state.timestep
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state, batch, carries, context
        )
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            self.policy_update_due(timestep),
            lambda _: (stepped, (policy_aux, policy_metrics)),
            agent_state,
        )
        tempered, temperature_aux = update_temperature(
            agent_state,
            log_probs,
            self.target_entropy_at(timestep),
            compute_alpha_temperature_loss,
        )
        if not self.fixed_alpha:
            agent_state = update_steps.run_if(
                self.temperature_update_due(timestep), lambda _: tempered, agent_state
            )
        reset, settings = agent_state.reset_cost, self.reset_cost
        if reset is not None and settings is not None:
            reset = step_reset_cost(reset, settings.target_rate)
            reset = update_reset_target(reset, self.tau)
        agent_state = update_steps.update_target_networks(agent_state, self.tau)
        agent_state = agent_state.replace(reset_cost=reset)
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: RVISACState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[RVISACState, ValueAuxiliaries, dict]:
        """The critics' step on the RVI target, then the reset critic's and
        the offsets' (rvi_sac.py:151-202)."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        critic_state = agent_state.critic_state
        scored = context.make_scored_actions(batch)
        next_actions, next_log_probs = sample_next_actions(
            agent_state.actor_state,
            batch.next_obs,
            derive_key(critic_key, "next_actions"),
            carries,
            scored.map_next_actions,
        )
        next_q = q_values(
            critic_state,
            critic_state.target_params,
            batch.next_obs,
            next_actions,
            carries,
            bootstrap=True,
        )
        next_values = jnp.min(next_q, axis=0) - get_alpha(agent_state) * next_log_probs
        predictions, target_value_at = critic_value_readers(
            critic_state, batch.obs, batch.action
        )
        inputs = CenteringInputs(next_values, target_value_at=target_value_at)
        offset = self.centering.offset_for_target(agent_state.offset, inputs)
        charged = batch.reward
        if agent_state.reset_cost is not None:
            charged = charged - reset_charge(agent_state.reset_cost, batch.terminated)
        target = compute_rvi_td_target(charged, offset, next_values, self.reward_scale)

        def critic_loss(params: FrozenDict, target_q: jax.Array) -> tuple:
            loss, aux = compute_mean_critic_loss(
                params, critic_state, batch.obs, batch.action, target_q, carries
            )
            # Each critic's own mean squared error, summed (rvi_sac.py:176-178).
            summed = NUM_CRITICS * loss
            return summed, dataclasses.replace(aux, critic_loss=summed)

        # RVI-SAC continues through terminations: its target cuts nowhere.
        agent_state, aux, grad_norm, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=charged,
            dones=jnp.zeros_like(batch.terminated),
            gamma=None,
            reward_scale=self.reward_scale,
            next_actions=next_actions,
            next_values=next_values,
            predictions=predictions,
            target_value_at=target_value_at,
        )
        if isinstance(self.centering, TDErrorCentering):
            inputs = dataclasses.replace(inputs, td_errors=target - predictions())
        offset = self.centering.next_offset(offset, inputs)
        reset, settings = agent_state.reset_cost, self.reset_cost
        reset_metrics: dict[str, Any] = {}
        unset = jnp.full((), jnp.nan)
        if reset is not None and settings is not None:
            reset, reset_metrics = step_reset_critic(
                reset,
                reset_critic_input(batch.obs, batch.action),
                reset_critic_input(batch.next_obs, next_actions),
                batch.terminated,
                settings.rate_tau,
                settings.gain,
            )
        value_aux = ValueAuxiliaries(
            critic_loss=aux.critic_loss,
            q_pred_min=aux.q_pred_min,
            var_preds=aux.var_preds,
            grad_norm=grad_norm,
            target_q=jnp.mean(target),
            offset=offset,
            reset_rate=unset if reset is None else reset.rate,
            reset_cost=unset if reset is None else reset.cost.params["cost"],
            reset_critic_loss=reset_metrics.get("reset_critic_loss", unset),
        )
        return agent_state.replace(offset=offset, reset_cost=reset), value_aux, metrics

    def update_actor(
        self,
        agent_state: RVISACState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[RVISACState, PolicyAuxiliaries, jax.Array, dict]:
        """The actor step on alpha log pi(a|s) - min_i Q_i(s, a), the
        critics scoring a residual policy's executed actions; also returns
        its sample's log-probs (the temperature reads them)."""
        actor_key = update_steps.derive_actor_key(agent_state)
        alpha = get_alpha(agent_state)
        critic_state = agent_state.critic_state
        scored = context.make_scored_actions(batch)

        def policy_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
            loss, (log_probs, q_min, policy) = compute_soft_policy_loss(
                params,
                agent_state.actor_state,
                critic_state,
                batch.obs,
                alpha,
                actor_key,
                carries,
                jnp.min,
                scored.map_actions,
            )
            return loss, (loss, log_probs, q_min), policy.mean()

        actor_state, (loss, log_probs, q_min), _, metrics = update_steps.actor_step(
            agent_state,
            policy_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            batch,
            critic_state,
        )
        aux = PolicyAuxiliaries(
            policy_loss=loss, log_pi=log_probs.mean(), q_min=q_min.mean()
        )
        return agent_state.replace(actor_state=actor_state), aux, log_probs, metrics

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "RVISAC"
    capabilities = frozenset(
        {
            "actor",
            "continuing_target",
            "critic_ensemble",
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
    stores_expert_actions = True
    supported_buffer_fields = RecurrentReplayAgent.supported_buffer_fields.union(
        {"raw_next_obs"}
    )


# -- RVI-SAC's equation -------------------------------------------------------
def compute_rvi_td_target(
    rewards: jax.Array,
    offset: Union[float, jax.Array],
    next_values: jax.Array,
    reward_scale: float = 1.0,
) -> jax.Array:
    """r - xi + V'(s') (Eq. 17), r already charged its reset cost, no cut at
    a termination and no gradient flowing back through it."""
    return jax.lax.stop_gradient(rewards * reward_scale - offset + next_values)


# -- State and logged metrics -------------------------------------------------
@struct.dataclass(kw_only=True)
class RVISACState(SoftActorCriticState):
    """RVI-SAC's state: a soft actor-critic's plus the offset xi and the reset
    cost's (None without one)."""

    offset: jax.Array
    reset_cost: Optional[ResetCostState]


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array  # summed over the critics, as the authors'
    q_pred_min: jax.Array
    var_preds: jax.Array  # inter-critic variance
    grad_norm: jax.Array
    target_q: jax.Array
    offset: jax.Array  # xi, the reward rate the critics are centred on
    reset_rate: jax.Array  # xi_reset, the termination rate the reset critic reads
    reset_cost: jax.Array  # c
    reset_critic_loss: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
