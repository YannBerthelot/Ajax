"""Average-Reward Soft Actor-Critic (Adamczyk et al., 2025,
https://arxiv.org/abs/2501.09080v2), for continuous actions: SAC for the
average-reward criterion.

The critics learn the differential soft Q, bootstrapped without discount on
r - theta, where theta tracks the entropy-regularised reward rate. Every
target is shifted by the target critics' value at the origin Q(0, 0), their
reference point, and a termination drops the next value and is charged a
penalty learned from the rewards (p_0 scales it), so that episodic tasks fit
the continuing criterion. One update on a replay batch, in the order of the
authors' code (BobMak/asac@4b68862, darer/ASAC.py), each line one method
below or one of the equations under it::

    P <- (1 - tau_theta) P + tau_theta p_0 max r[terminated]
                                            compute_episode_termination_penalty
    r_P = r - P terminated            get_episode_termination_penalized_rewards
    a' ~ pi(. | s'), log pi(a' | s')                  sample_next_actions (SAC's)
    y = r_P - theta + (1 - d) (min_i Q_target_i(s', a') - alpha (log pi(a' | s')
        - log(1 / nA))) - Q_target(0, 0)                    compute_asac_td_target
    Q_i <- argmin sum_i E[(Q_i(s, a) - y)^2] / 2    update_critic, compute_critic_loss
    actor <- argmin E[alpha log pi(a | s) - min_i Q_i(s, a)]
                                       update_actor, compute_soft_policy_loss (SAC's)
    alpha <- argmin log alpha sg(-log pi(a | s) - H_target), unless fixed_alpha
    theta <- (1 - tau_theta) theta + tau_theta mean(r - alpha (log pi - log(1 / nA)))
                                                                    update_theta
    Q_target <- tau Q + (1 - tau) Q_target
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents import update_steps
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
from ajax.controller import make_scored_action
from ajax.extensions.base import merge_metrics
from ajax.networks.memory import zeros_carry_like
from ajax.networks.networks import (
    action_value_input,
    predict_value,
    predict_value_sequence,
)
from ajax.networks.utils import TORCH_LINEAR
from ajax.state import LoadedTrainState, Transition
from ajax.types import Architecture

#: The authors' twin critics (darer/Models.py), whose minimum the targets
#: and the actor read.
NUM_CRITICS = 2


@dataclass(kw_only=True, eq=False, repr=False)
class ASAC(
    EntropyTemperature,
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """ASAC: SAC on the average reward, its targets differential (the
    module's update). It takes no discount.

    The learning rates, the gradient clip (each critic and the actor on its
    own; the temperature unclipped) and tau_theta are the authors'
    (BobMak/asac @4b68862: every hparams/*/asac.yaml sets learning_rate 5e-4
    for the critics, actor_learning_rate 1e-4 and max_grad_norm 10;
    darer/ASAC.py:87, :92, :104 for the temperature's 1e-4, :228-229 and
    :249-250 for the clips), with torch's Adam eps 1e-8.

    Args:
        tau_theta: Rate of the theta and penalty averages (darer/ASAC.py:188,
            :213). The authors tune it per environment, 0.0005 to 0.1; 0.005
            is their default (darer/BaseAgent.py:80) and six of their 11
            configs'.
        alpha_init: The temperature, or its initial value when fixed_alpha
            is False.
        fixed_alpha: Keep the temperature at alpha_init. The authors'
            published runs fix it at 1/beta: 0.2, and 0.05 for Humanoid and
            Swimmer (hparams/*/asac.yaml, darer/ASAC.py:56-57,
            experiments/finetuned_runs.py:67-76). False learns it towards
            the target entropy, as SAC does.
        p_0: The termination penalty's scale (darer/ASAC.py:187).
    """

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 5e-4
    alpha_learning_rate: Optional[float] = 1e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    max_grad_norm: Optional[float] = 10.0
    tau: float = 0.005
    tau_theta: float = 0.005
    reward_scale: float = 1.0
    alpha_init: float = 0.2
    fixed_alpha: bool = True
    p_0: float = 20
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(1e4)

    def actor_critic_args(self) -> dict[str, Any]:
        # The initialisation of ASAC's reference code: BobMak/asac
        # @4b68862 builds its actor from stable-baselines3 2.4.0's SAC
        # Actor (darer/ASAC.py:15, :78-82; sac/policies.py:85, :102-103
        # at v2.4.0) and its critics from Qsa (darer/Models.py:409-411),
        # all torch nn.Linear layers at torch's default initialisation:
        # "torch_linear", whose layers draw their biases too.
        # Recurrent ASAC (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its recurrent reference (pomdp-baselines), is
        # audited, as SAC and REDQ do.
        args: dict[str, Any] = {"squash": True}
        if self.memory is not None:
            return args
        init = ("encoder_kernel_init", "mean_kernel_init", "log_std_kernel_init")
        args.update(dict.fromkeys((*init, "value_kernel_init"), TORCH_LINEAR))
        args.update({"log_std_init": None})
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        # torch's Adam eps, which the reference's Adams keep (darer/ASAC.py:
        # 87-92, torch 2.8.0); recurrent ASAC's reference (Ni et al.,
        # pomdp-baselines e7c19c3 policy_rnn.py:75, :93) keeps it too.
        return {"eps": 1e-8}

    def network_settings(self) -> dict[str, Any]:
        # The reference's actor is stable-baselines3's squashed Gaussian
        # (2.4.0, distributions.py), whose log-prob subtracts log(1 - tanh^2
        # + 1e-6).
        return {"squash_log_det_eps": 1e-6}

    def init(self, key: jax.Array, context: TrainContext) -> ASACState:
        """A squashed Gaussian actor, two Q critics with their targets, the
        temperature, the reward rate theta and the termination penalty, both
        from 0."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=NUM_CRITICS,
        )
        collector_state = context.make_collector(collector_key)
        return ASACState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
            episode_termination_penalty=jnp.zeros(()),
            theta=0.0,
        )

    def update(
        self, agent_state: ASACState, context: TrainContext
    ) -> tuple[ASACState, AuxiliaryLogs]:
        """One update on a replay batch: the termination penalty, then the
        critic, actor, temperature, theta and target steps, the actor's and
        the temperature's kept from their starts."""
        sample_key, agent_state = agent_state.get_keys_and_new_state()
        batch, carries = sample_replay(
            agent_state,
            context.buffer,
            sample_key,
            context.recurrent,
            self.burn_in,
            self.stored_state,
        )
        episode_termination_penalty = compute_episode_termination_penalty(
            agent_state.episode_termination_penalty,
            batch.reward,
            batch.terminated,
            self.p_0,
            self.tau_theta,
        )
        rewards = get_episode_termination_penalized_rewards(
            episode_termination_penalty, batch.reward, batch.terminated
        )
        agent_state = agent_state.replace(
            episode_termination_penalty=episode_termination_penalty
        )
        agent_state, value_aux, value_metrics = self.update_critic(
            agent_state, batch, rewards, carries, context
        )
        # The temperature and theta read the actor's own sample, drawn before
        # its step, as the authors' code does (darer/ASAC.py:146-213).
        timestep = agent_state.collector_state.timestep
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state, batch, carries, context
        )
        agent_state = update_steps.run_if(
            self.policy_update_due(timestep), lambda _: stepped, agent_state
        )
        # fixed_alpha drops the temperature's step but keeps its logs, as
        # SAC's does.
        tempered, temperature_aux = update_temperature(
            agent_state,
            log_probs,
            self.target_entropy_at(timestep, context.total_timesteps),
            compute_temperature_loss,
        )
        if not self.fixed_alpha:
            agent_state = update_steps.run_if(
                self.temperature_update_due(timestep), lambda _: tempered, agent_state
            )
        # In the target's units: the reward rate theta is subtracted from the
        # scaled rewards there. It averages the rewards without the penalty,
        # as the authors' code (darer/ASAC.py:191).
        agent_state = update_theta(
            agent_state,
            self.tau_theta,
            batch.reward * self.reward_scale,
            log_probs,
            batch.action.shape[-1],
        )
        agent_state = update_steps.update_target_networks(agent_state, self.tau)
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux,
            theta=ThetaAuxiliaries(
                theta=agent_state.theta,
                episode_termination_penalty=episode_termination_penalty,
            ),
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: ASACState,
        batch: Transition,
        rewards: jax.Array,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[ASACState, ValueAuxiliaries, dict]:
        """The critics' step on the differential target of the penalised
        rewards."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        alpha = get_alpha(agent_state)
        dones = batch.terminated
        target_q, next_log_probs = compute_asac_td_target(
            agent_state.actor_state,
            agent_state.critic_state,
            critic_key,
            batch.next_obs,
            dones,
            rewards,
            agent_state.theta,
            alpha,
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

        # ASAC is average-reward: no gamma, None tells the extensions so.
        critic_state, aux, _, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target_q,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=rewards,
            dones=dones,
            gamma=None,
            reward_scale=self.reward_scale,
        )
        return agent_state.replace(critic_state=critic_state), aux, metrics

    def update_actor(
        self,
        agent_state: ASACState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[ASACState, PolicyAuxiliaries, jax.Array, dict]:
        """The actor step on alpha log pi(a|s) - min_i Q_i(s, a), the
        critics scoring a residual policy's executed actions; also returns
        its sample's log-probs (the temperature and theta read them)."""
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
                jnp.min,
                transform,
            )
            return loss, (loss, log_probs, q_values), policy.mean()

        actor_state, (loss, log_probs, q_min), _, metrics = update_steps.actor_step(
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
            policy_loss=loss, log_pi=log_probs.mean(), q_min=q_min.mean()
        )
        return agent_state.replace(actor_state=actor_state), aux, log_probs, metrics

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "ASAC"
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
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
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs"})


# -- ASAC's equations, in the order update calls them ------------------------
def compute_episode_termination_penalty(
    episode_termination_penalty: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    p_0: float,
    tau: float,
) -> jax.Array:
    """Move the penalty by tau towards p_0 times the largest reward
    gather(rewards, 0, terminated) reads, as the authors' code
    (darer/ASAC.py:187, with 20 for p_0): a row reads the reward of the
    flattened batch's row 0 when it did not terminate, of row 1 when it did.
    Their comment means the mean non-terminal reward."""
    rows = terminated.reshape(-1).astype(jnp.int32)
    p_bar = p_0 * jnp.max(rewards.reshape(-1)[rows])
    episode_termination_penalty = episode_termination_penalty * (1 - tau) + tau * p_bar
    return episode_termination_penalty


def get_episode_termination_penalized_rewards(
    episode_termination_penalty: jax.Array, rewards: jax.Array, terminated: jax.Array
) -> jax.Array:
    """The rewards, a terminated row's charged the penalty."""
    return rewards - episode_termination_penalty * terminated


def uniform_log_prob(action_dim: int) -> float:
    """log(1 / nA), the authors' reference point for the entropy term
    (darer/ASAC.py:54, :182, :191): 0 for one action dimension."""
    return -math.log(action_dim)


def compute_asac_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    theta: float | jax.Array,
    alpha: jax.Array,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """r - theta + (1 - done) (min_i Q_target_i(s', a') - alpha (log pi(a'|s')
    - log(1 / nA))) - Q_target(0, 0), r with a termination's penalty.

    As the authors' code (darer/ASAC.py:179-205): a termination drops the
    next value and is charged the penalty; the origin shift applies to every
    row. a' is executed (executed, a residual policy's). Returns the target,
    no gradient flowing back through it, and the next actions' log-probs.
    """
    rewards = rewards * reward_scale
    sample_key, _ = jax.random.split(rng)
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, sample_key, carries, executed, next_a_expert
    )
    q_targets = q_values(
        critic_states,
        critic_states.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    # The origin shift Q(0, 0): one zero input; in sequence mode at an
    # episode start (a batch of 1, a fresh carry, no previous action).
    zero_obs = jnp.zeros((1, next_observations.shape[-1]))
    zero_action = jnp.zeros((1, next_actions.shape[-1]))
    assert critic_states.target_params is not None
    if carries is None:
        zero_x = action_value_input(zero_obs, zero_action)
        shift_value = predict_value(critic_states, critic_states.target_params, zero_x)
    else:
        shift_value, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_states.target_params,
            x=action_value_input(zero_obs, zero_action, zero_action)[None],
            resets=jnp.zeros((1, 1), dtype=bool),
            initial_hidden=zeros_carry_like(
                carries.target_critic_hidden, 1, batch_axis=1
            ),
        )
    q1_target, q2_target = jnp.split(q_targets, 2, axis=0)
    min_q_target = jnp.minimum(q1_target, q2_target).squeeze(0)
    logpi0 = uniform_log_prob(next_actions.shape[-1])
    next_value = (1.0 - dones) * (min_q_target - alpha * (log_probs - logpi0))
    target = rewards - theta + next_value - jnp.mean(shift_value)
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
    """Both critics regress on the one differential target."""
    q_preds = q_values(critic_states, critic_params, observations, actions, carries)
    assert target_q.shape == q_preds.shape[1:], f"{target_q.shape} != {q_preds.shape}"
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    loss_q1 = 0.5 * jnp.mean((q1_pred.squeeze(0) - target_q) ** 2)
    loss_q2 = 0.5 * jnp.mean((q2_pred.squeeze(0) - target_q) ** 2)
    total_loss = loss_q1 + loss_q2
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q1_pred=q1_pred.mean().flatten(),
        q2_pred=q2_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
    )


def update_theta(
    agent_state: ASACState,
    tau: float,
    rewards: jax.Array,
    log_probs: jax.Array,
    action_dim: int,
) -> ASACState:
    """Track the entropy-regularised reward rate: an exponential average of
    mean(r - alpha (log pi(a|s) - log(1 / nA))) over the batch, a ~ pi(.|s)."""
    alpha = get_alpha(agent_state)
    log_pi = log_probs.sum(-1, keepdims=True) - uniform_log_prob(action_dim)
    new_theta = jnp.mean(rewards - alpha * log_pi)
    theta = agent_state.theta * (1 - tau) + tau * new_theta
    return agent_state.replace(theta=theta)


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class ASACState(SoftActorCriticState):
    """ASAC's state: a soft actor-critic's plus the reward rate and the
    termination penalty."""

    episode_termination_penalty: jax.Array
    theta: Union[float, jax.Array]


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q1_pred: jax.Array
    q2_pred: jax.Array
    target_q: jax.Array
    log_probs: jax.Array


@struct.dataclass
class ThetaAuxiliaries:
    theta: Union[float, jax.Array]
    episode_termination_penalty: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    theta: ThetaAuxiliaries
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
ASACConfig = ASAC.declare_config(
    "target_entropy tau tau_theta learning_starts reward_scale p_0 burn_in"
    " sequence_length stored_state"
)
