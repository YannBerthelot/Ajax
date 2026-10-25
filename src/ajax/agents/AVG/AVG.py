"""Action Value Gradient (Vasan et al., 2024, https://arxiv.org/abs/2411.15370),
Algorithm 1, for continuous actions: incremental deep RL, one env step and
then one update on that single step, with no replay and no target network.

Each step, in the order of the authors' code (gauthamvasan/avg, avg.py),
each line one method below or one of the equations under it::

    the TD-error scale's statistics: the step's entropy-regularised reward
    and discount and, where an episode ended, its return
                                update_step_statistics, compute_td_error_scaling
    a' ~ pi(. | s'), log pi(a' | s')                  sample_next_actions (SAC's)
    y = scale r + gamma (1 - d) (min_i Q_i(s', a') - alpha log pi(a' | s'))
                                                         compute_avg_td_target
    Q <- argmin ((Q(s, a) - y) / scale of the TD error)^2
                                                  update_critic, compute_critic_loss
    actor <- argmin alpha log pi(a | s) - Q(s, a) at the action taken
                                                   update_actor, compute_policy_loss

The critic's and the actor's steps both read the networks from before
either; alpha stays at its initial value. AVG has no memory option: its
single-step updates give length-1 BPTT, so memory weights could not learn
temporal structure without eligibility traces or RTRL.
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
from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    IncrementalAgent,
    PolicyUpdateStart,
    TrainContext,
)
from ajax.agents.recurrent import q_values
from ajax.agents.SAC.SAC import sample_next_actions
from ajax.agents.temperature import (
    SoftActorCriticState,
    TemperatureAuxiliaries,
    create_alpha_train_state,
    get_alpha,
)
from ajax.environments.interaction import get_pi
from ajax.extensions.base import merge_metrics
from ajax.state import AlphaConfig, LoadedTrainState, Transition
from ajax.types import Architecture, Replaceable

#: The reference's initialisation (gauthamvasan/avg @6b6822c, avg.py:16-20,
#: :59-62, :97-98): every Linear layer, the mean and log-std heads
#: included, starts orthogonal with gain 1 and a zero bias.
ORTHOGONAL = "orthogonal(1.0)"


@dataclass(kw_only=True, eq=False, repr=False)
class AVG(PolicyUpdateStart, GradientClipping, ExpertGuided, IncrementalAgent):
    """AVG: an incremental soft actor-critic whose critic's TD error is
    divided by a running estimate of its scale (the module's update). It
    always normalises its observations, never its rewards, and squashes its
    actions; with learning_starts above 0 it acts uniformly until then.

    Defaults are the authors' (gauthamvasan/avg): 256-256 leaky-ReLU
    networks with penultimate normalisation, Adam at 6.3e-3 (actor) and
    8.7e-3 (critic) with beta_1 0, beta_2 0.999 and torch's eps 1e-8
    (avg.py:116-117, torch >= 2.7.0), a fixed alpha of 0.07.

    Args:
        alpha_init: The entropy coefficient, which never steps.
        beta_1: Adam's first-moment decay (AVG uses 0).
        beta_2: Adam's second-moment decay.
        num_critics: The critics, whose minimum the target and the actor
            read.
    """

    actor_learning_rate: float = 6.3e-3
    critic_learning_rate: float = 8.7e-3
    actor_architecture: Architecture = ("256", "leaky_relu", "256", "leaky_relu")
    critic_architecture: Architecture = ("256", "leaky_relu", "256", "leaky_relu")
    gamma: float = 0.99
    reward_scale: float = 1.0
    alpha_init: float = 0.07
    beta_1: float = 0
    beta_2: float = 0.999
    num_critics: int = 1
    learning_starts: int = 0

    def actor_critic_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "normalize_observations": True,
            "squash": True,
            "encoder_kernel_init": ORTHOGONAL,
            "value_kernel_init": ORTHOGONAL,
            "mean_kernel_init": ORTHOGONAL,
            "log_std_kernel_init": ORTHOGONAL,
            "log_std_init": 0.0,
        }
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        return {"beta_1": self.beta_1, "beta_2": self.beta_2, "eps": 1e-8}

    def network_settings(self) -> dict[str, Any]:
        # The reference's penultimate normalisation, and its MultivariateNormal
        # takes exp(log_std) as the covariance's diagonal (avg.py:70-72): the
        # head is read as a log-variance.
        return {"penultimate_normalization": True, "log_variance_head": True}

    @property
    def alpha_args(self) -> AlphaConfig:
        """AVG keeps its entropy coefficient at alpha_init, as its reference
        code does (avg.py): the temperature's optimizer never steps."""
        return AlphaConfig(learning_rate=0.0, alpha_init=self.alpha_init)

    def init(self, key: jax.Array, context: TrainContext) -> AVGState:
        """A squashed Gaussian actor, num_critics Q critics without targets,
        the fixed temperature and the TD-error scale's statistics."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=self.num_critics,
        )
        collector_state = context.make_collector(collector_key)
        # One running statistic per scalar (reward, discount, return), shared
        # by every env.
        no_moments = RunningMoments(
            value=jnp.zeros((context.env_args.n_envs, 1)),
            count=jnp.zeros((1,)),
            mean=jnp.zeros((1, 1)),
            mean_2=jnp.zeros((1, 1)),
        )
        return AVGState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
            reward=no_moments,
            gamma=no_moments,
            G_return=no_moments,
            scaling_coef=jnp.ones((1, 1)),
        )

    def update_step_statistics(
        self, agent_state: AVGState, transition: Transition, context: TrainContext
    ) -> AVGState:
        """Add the step just taken to the statistics behind the TD-error
        scale: its entropy-regularised reward and discount and, where an
        episode ended, its return."""
        del context
        alpha = get_alpha(agent_state)
        assert transition.log_prob is not None
        soft_reward = transition.reward - alpha * jax.lax.stop_gradient(
            transition.log_prob
        ).sum(-1, keepdims=True)
        reward = agent_state.reward.replace(value=soft_reward)
        gamma = agent_state.gamma.replace(
            value=self.gamma * (1 - transition.terminated)
        )
        new_return = agent_state.G_return.value + soft_reward
        ended = transition.terminated.astype(bool)
        # The statistics read the returns of the episodes that ended (NaN
        # marks the others); those restart at 0, the others accumulate.
        scaling_coef, reward, gamma, G_return = compute_td_error_scaling(
            reward,
            gamma,
            G_return=agent_state.G_return.replace(
                value=jnp.where(ended, new_return, jnp.nan)
            ),
        )
        G_return = G_return.replace(value=jnp.where(ended, 0.0, new_return))
        return agent_state.replace(
            reward=reward, gamma=gamma, G_return=G_return, scaling_coef=scaling_coef
        )

    def update(
        self, agent_state: AVGState, transition: Transition, context: TrainContext
    ) -> tuple[AVGState, AuxiliaryLogs]:
        """One update on the step just taken: a critic and an actor step,
        both from the current parameters (the actor's loss sees the critic
        before its step, the critic's TD target the actor before its), the
        actor's kept from policy_update_start; alpha is not updated.

        The official code's order (gauthamvasan/avg, AVG.update in
        incremental_rl/avg_ablation.py and in avg.py): both losses are built
        before popt.step(), then qopt.step(). Vasan et al. (2024), Algorithm
        1, writes the critic's update line before the actor's; the paper's
        runs come from that code.
        """
        critic_updated, value_aux, value_metrics = self.update_critic(
            agent_state, transition, context
        )
        # The actor step reads the networks from before the critic's, its key
        # drawn after the critic's; the expert's stored actions reach the
        # actor-loss extensions in a run with an expert.
        a_expert = transition.a_expert if self.expert is not None else None
        policy_updated, policy_aux, policy_metrics = self.update_actor(
            agent_state.replace(rng=critic_updated.rng), transition, a_expert, context
        )
        stepped = critic_updated.replace(
            actor_state=policy_updated.actor_state, rng=policy_updated.rng
        )
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            self.policy_update_due(agent_state.collector_state.timestep),
            lambda _: (stepped, (policy_aux, policy_metrics)),
            critic_updated,
        )
        log_alpha = agent_state.alpha.params["log_alpha"]
        return agent_state, AuxiliaryLogs(
            temperature=TemperatureAuxiliaries(
                alpha=jnp.exp(log_alpha),
                log_alpha=log_alpha,
                effective_target_entropy=jnp.nan,  # alpha is not tuned
            ),
            policy=policy_aux,
            value=value_aux,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self, agent_state: AVGState, transition: Transition, context: TrainContext
    ) -> tuple[AVGState, ValueAuxiliaries, dict]:
        """The critic step on the scaled TD error of the step, at the action
        executed (the policy's stored one; a residual policy's from the
        stored expert action)."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        alpha = get_alpha(agent_state)
        critic_state = agent_state.critic_state
        # Only a termination cuts the bootstrap: the official code passes
        # terminated alone as the update's done (avg.py), so a time limit
        # still bootstraps on the final observation, which next_obs holds
        # (get_final_obs).
        dones = transition.terminated
        target_q, next_log_probs = compute_avg_td_target(
            agent_state.actor_state,
            critic_state,
            critic_state.params,
            critic_key,
            transition.next_obs,
            dones,
            transition.reward,
            self.gamma,
            alpha,
            self.reward_scale,
            context.executed,
            transition.next_a_expert,
        )
        action = transition.action
        if context.executed is not None:
            action = context.executed(action, transition.a_expert)

        def critic_loss(
            params: FrozenDict, target_q: jax.Array
        ) -> tuple[jax.Array, Any]:
            return compute_critic_loss(
                params,
                critic_state,
                transition.obs,
                action,
                target_q,
                next_log_probs,
                agent_state.scaling_coef,
            )

        critic_state, aux, _, metrics = update_steps.critic_step(
            agent_state,
            transition,
            target_q,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=transition.reward,
            dones=dones,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
        )
        return agent_state.replace(critic_state=critic_state), aux, metrics

    def update_actor(
        self,
        agent_state: AVGState,
        transition: Transition,
        a_expert: Optional[jax.Array],
        context: TrainContext,
    ) -> tuple[AVGState, PolicyAuxiliaries, dict]:
        """The actor step through the action taken; the extensions also read
        the env's own observations, the policy's mean action and a_expert,
        the agent's expert's action from its state (None without one)."""
        actor_key, agent_state = agent_state.get_keys_and_new_state()
        alpha = get_alpha(agent_state)
        actor_state = agent_state.actor_state
        raw_actions = transition.raw_action  # the pre-squash sample, as collected
        assert raw_actions is not None

        def policy_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
            loss, aux = compute_policy_loss(
                params,
                actor_state,
                agent_state.critic_state,
                transition.obs,
                raw_actions,
                alpha,
                context.executed,
                a_expert,
            )
            # The mean action is the extensions' (unused without them,
            # compiled out).
            pi_mean = get_pi(actor_state, params, transition.obs)[0].mean()
            return loss, aux, pi_mean

        actor_state, aux, _, metrics = update_steps.actor_step(
            agent_state,
            policy_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            observations=transition.obs,
            raw_observations=transition.raw_obs,
            a_expert=a_expert,
            critic_state=agent_state.critic_state,
        )
        return agent_state.replace(actor_state=actor_state), aux, metrics

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "AVG"
    # No replay: AVG learns from each step as it comes. Its warm-up, before a
    # learning_starts above 0, acts uniformly or by the warm-up's chooser.
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
            "reward_learning",
            "warmup",
        }
    )
    phase_batch_keys = {  # noqa: RUF012
        "on_target": update_steps.TARGET_KEYS,
        "critic_loss": update_steps.CRITIC_LOSS_KEYS,
        "actor_loss": update_steps.ACTOR_STEP_KEYS.union(
            {"observations", "raw_observations", "a_expert", "critic_state"}
        ),
    }
    # The step's transition carries what a replay batch holds: the expert's
    # stored actions and, for a reader, the raw next observation.
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs"})


# -- AVG's equations, in the order update calls them -------------------------
def compute_td_error_scaling(
    reward: RunningMoments,
    gamma: RunningMoments,
    G_return: RunningMoments,
) -> tuple[jax.Array, RunningMoments, RunningMoments, RunningMoments]:
    """AVG's TD-error scale, the authors' TDErrorScaler (gauthamvasan/avg,
    incremental_rl/td_error_scaler.py, after Schaul et al. 2021):
    sqrt(max(var(r) + E[G^2] var(gamma), 1e-4)), or 1 while that variance
    is at most 0.01 and no return has been seen. The reward, discount and
    squared-return statistics are updated first. G_return.value is NaN
    unless an episode just ended: only the returns of the episodes that
    ended count, however many envs run."""
    reward = _add_to_moments(reward, reward.value)
    gamma = _add_to_moments(gamma, gamma.value)
    G_return = _add_to_moments(G_return, jnp.square(G_return.value))
    variance = jnp.maximum(_variance(reward) + G_return.mean * _variance(gamma), 1e-4)
    unscaled = (variance <= 0.01) & (G_return.count == 0)
    scaling = jnp.where(unscaled, jnp.ones_like(variance), jnp.sqrt(variance))
    return scaling, reward, gamma, G_return


def _add_to_moments(moments: RunningMoments, values: jax.Array) -> RunningMoments:
    """Add the entries of values (one per env) to the moments' running count,
    mean and sum of squared deviations (Chan et al.'s parallel update). A
    NaN entry is absent: it neither moves the mean nor counts."""
    valid = ~jnp.isnan(values)
    valid_count = jnp.sum(valid)
    values = jnp.where(valid, values, 0.0)
    batch_mean = jnp.sum(values) / jnp.maximum(valid_count, 1)
    batch_squared_deviations = jnp.sum(
        jnp.where(valid, jnp.square(values - batch_mean), 0.0)
    )
    total = moments.count + valid_count
    share = valid_count / jnp.maximum(total, 1)
    delta = batch_mean - moments.mean
    return moments.replace(
        count=total,
        mean=moments.mean + delta * share,
        mean_2=moments.mean_2
        + batch_squared_deviations
        + jnp.square(delta) * moments.count * share,
    )


def _variance(moments: RunningMoments) -> jax.Array:
    """The population variance, 0 before any value."""
    return moments.mean_2 / jnp.maximum(moments.count, 1)


def compute_avg_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    critic_params: FrozenDict,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    reward_scale: float,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """AVG's Bellman target, from the current critic (no target network),
    at the next action executed (executed, a residual policy's)."""
    rewards = rewards * reward_scale
    sample_key, _ = jax.random.split(rng)
    next_actions, next_log_probs = sample_next_actions(
        actor_state,
        next_observations,
        sample_key,
        executed=executed,
        next_a_expert=next_a_expert,
    )
    q_target = jnp.min(
        q_values(critic_states, critic_params, next_observations, next_actions),
        axis=0,
    )
    target = jax.lax.stop_gradient(
        rewards + (1.0 - dones) * gamma * (q_target - alpha * next_log_probs),
    )
    return target, next_log_probs


def compute_critic_loss(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    next_log_probs: jax.Array,
    scaling_coef: jax.Array,
) -> tuple[jax.Array, ValueAuxiliaries]:
    """The squared TD error divided by its running scale."""
    q_pred = jnp.min(
        q_values(critic_states, critic_params, observations, actions), axis=0
    )
    assert target_q.shape == q_pred.shape, f"{target_q.shape} != {q_pred.shape}"
    scaled_delta = (q_pred - target_q) / scaling_coef
    total_loss = jnp.mean(scaled_delta**2)
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q_pred=q_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
        scaling_coef=scaling_coef.mean().flatten(),
    )


def compute_policy_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    raw_actions: jax.Array,
    alpha: jax.Array,
    executed: Optional[Callable] = None,
    a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, PolicyAuxiliaries]:
    """alpha log pi(a|s) - Q(s, a) at the action taken (executed, with a
    residual policy's executed).

    The action is reparameterised with the noise eps of its stored
    pre-squash sample u: a = tanh(mu(s) + sigma(s) eps), eps = (u - mu(s)) /
    sigma(s). The actor has not changed since it sampled u, so a is the
    action executed and the gradient flows through it, as in AVG's update.
    """
    policy, _ = get_pi(actor_state, actor_params, observations)
    loc, scale = policy.distribution.loc, policy.distribution.scale
    eps = jax.lax.stop_gradient((raw_actions - loc) / scale)
    raw = loc + scale * eps
    log_probs = policy.log_prob_from_raw(raw).sum(-1, keepdims=True)
    action = jnp.tanh(raw)
    if executed is not None:
        action = executed(action, a_expert)
    q_pred = jnp.min(
        q_values(critic_states, critic_states.params, observations, action),
        axis=0,
    )
    assert log_probs.shape == q_pred.shape, f"{log_probs.shape} != {q_pred.shape}"
    loss = (alpha * log_probs - q_pred).mean()
    return loss, PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_min=q_pred.mean()
    )


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass
class RunningMoments(Replaceable):
    """The latest value of one scalar stream (AVG's reward, discount or
    squared return) and its running count, mean and mean_2; not the
    observation normaliser's ajax.types.NormalizationInfo."""

    value: jax.Array
    count: jax.Array
    mean: jax.Array
    mean_2: jax.Array


@struct.dataclass(kw_only=True)
class AVGState(SoftActorCriticState):
    """AVG's state: a soft actor-critic's plus the TD-error scale statistics."""

    reward: RunningMoments
    gamma: RunningMoments
    G_return: RunningMoments
    scaling_coef: jnp.ndarray


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred: jax.Array
    target_q: jax.Array
    log_probs: jax.Array
    scaling_coef: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
