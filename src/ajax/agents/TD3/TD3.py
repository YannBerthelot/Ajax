"""Twin Delayed DDPG (Fujimoto, van Hoof and Meger, ICML 2018), Algorithm 1:
DDPG with three changes against the critics' overestimation, for continuous
actions.

One update on a replay batch, in the order of the reference
(sfujim/TD3@34770cc, TD3.py train), each line one function below::

    a' = clip(mu_target(s') + clip(N(0, sigma), -c, c), -1, 1)
    y = scale r + gamma (1 - d) min_i Q_target_i(s', a')     compute_td3_td_target
    Q_i <- argmin sum_i E[(Q_i(s, a) - y)^2]                   compute_critic_loss
    every policy_delay-th update:
        mu <- argmax E[Q_1(s, mu(s))]                          compute_policy_loss
        Q_target, mu_target <- tau (Q, mu) + (1 - tau) (Q_target, mu_target)
                                                                    update_targets

The actor is deterministic (ajax.networks.deterministic: tanh(mu));
collection explores around it (make_gaussian_exploration), with uniform
actions during the warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents import update_steps
from ajax.agents.critics import compute_mean_critic_loss
from ajax.agents.families import (
    BELLMAN_CONTEXT,
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    PolicyUpdateStart,
    RecurrentReplayAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.recurrent import RecurrentCarries, actor_dist, q_values, sample_replay
from ajax.controller import critic_action_units
from ajax.environments.interaction import Explore, explore_pipeline
from ajax.extensions.base import merge_metrics
from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.utils import TORCH_LINEAR
from ajax.schedule import Schedule, at
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture


@dataclass(kw_only=True, eq=False, repr=False)
class TD3(
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """Twin Delayed DDPG: twin critics, a smoothed target action and a
    delayed actor (the module's equations).

    Defaults are the authors' code as updated in 2019 (sfujim/TD3 a8d53f7,
    ade6260), not the paper's 400-300, lr 1e-3, batch 100 and 1e4 warm-up:
    256-256 ReLU plain MLPs; Adam at 3e-4 with torch's eps 1e-8; tau 0.005,
    gamma 0.99; exploration N(0, 0.1); target noise N(0, 0.2) clipped to
    0.5; policy delay 2; a 1e6 replay, batch 256, 25e3 warm-up steps. Its
    layers keep torch's nn.Linear initialisation (bc98e62 TD3.py:24-26,
    :43-50), kernel and bias uniform in +-1/sqrt(fan_in), the actor's last
    layer included (DDPG's uniform(+-3e-3) is in neither), and its critics
    read the env's units, max_action * tanh. With memory, TD3 keeps
    Ajax's previous initialisation until Ni et al.'s reference
    (pomdp-baselines) is audited, as SAC, REDQ and ASAC do.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    num_critics: int = 2
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: Schedule = 0.1  # or a function of the timestep
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(25e3)
    #: Collection's action selection, in place of make_gaussian_exploration
    #: (AjaxExperiments' TD3 variants).
    action_pipeline: Optional[Callable] = None

    def actor_critic_args(self) -> dict[str, Any]:
        if self.memory is not None:
            return {"squash": True}
        init = ("encoder_kernel_init", "mean_kernel_init", "value_kernel_init")
        args: dict[str, Any] = {"squash": True}
        args.update(dict.fromkeys(init, TORCH_LINEAR))
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # torch.optim.Adam's default eps (sfujim/TD3)

    def setup(self) -> None:
        # The user's pipeline replaces the default one, which alone runs the
        # warm-up's chooser.
        if self.action_pipeline is not None and self.env_args.warmups:
            names = [warmup.name for warmup in self.env_args.warmups]
            raise ValueError(
                f"TD3's action_pipeline never sees the warm-up's chooser {names}:"
                " pass one or the other."
            )

    def choose_behaviour(self, context: TrainContext) -> Optional[Callable]:
        if self.action_pipeline is not None:
            return self.action_pipeline
        transform = next(iter(context.env_args.transforms), None)
        return make_default_action_pipeline(
            context.recurrent,
            self.exploration_noise,
            transform,
            context.env_args.warmups,
        )

    def keeps_actor_metrics(self) -> bool:
        """The policy delay skips the actor on some updates."""
        return True

    def init(self, key: jax.Array, context: TrainContext) -> TD3State:
        """A deterministic actor and num_critics Q critics, with targets."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            build=get_initialized_td3_actor_critic,
            num_critics=self.num_critics,
        )
        # sfujim/TD3's critics read max_action * tanh, the env's units: so do
        # TD3's (static, from the concrete env params).
        units = critic_action_units(context.env_args)
        return TD3State(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state.replace(action_units=units),
            collector_state=context.make_collector(collector_key),
        )

    def update(
        self, agent_state: TD3State, context: TrainContext
    ) -> tuple[TD3State, AuxiliaryLogs]:
        """A critic step on a replay batch and, every policy_delay
        updates, an actor step and the targets' Polyak update."""
        sample_key, agent_state = agent_state.get_keys_and_new_state()
        # Sequence replay burns the TARGET actor's carry in as well: TD3's
        # bootstrap action comes from it.
        batch, carries = sample_replay(
            agent_state,
            context.buffer,
            sample_key,
            context.recurrent,
            self.burn_in,
            self.stored_state,
            burn_target_actor=True,
        )
        agent_state, value_aux, value_metrics = self.update_critic(
            agent_state, batch, carries, context
        )
        # sfujim/TD3 counts the update first (total_it += 1), then steps the
        # actor when the count is a multiple of the delay: its first actor
        # step is the policy_delay-th update, not the first.
        do_policy = update_steps.all_of(
            ((agent_state.n_updates + 1) % self.policy_delay) == 0,
            self.policy_update_due(agent_state.collector_state.timestep),
        )
        update_actor_on_batch = partial(
            self.update_actor, batch=batch, carries=carries, context=context
        )
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            do_policy, update_actor_on_batch, agent_state
        )
        agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
        return agent_state, AuxiliaryLogs(
            policy=policy_aux,
            value=value_aux,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: TD3State,
        batch: Transition,
        carries: Any,
        context: TrainContext,
    ) -> tuple[TD3State, ValueAuxiliaries, dict]:
        """The critics' step on the smoothed, clipped double-Q target."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        # A truncated row bootstraps on its final observation (bootstrap_obs).
        target_q, clip_fraction, next_actions = compute_td3_td_target(
            agent_state.actor_state,
            agent_state.critic_state,
            critic_key,
            batch.next_obs,
            batch.terminated,
            batch.reward,
            self.gamma,
            self.target_policy_noise,
            self.target_noise_clip,
            self.reward_scale,
            carries,
            context.executed,
            batch.next_a_expert,
        )
        compute_critic_loss_on_batch = partial(
            compute_critic_loss,
            agent_state.critic_state,
            batch,
            carries,
            self.num_critics,
            clip_fraction,
        )
        critic_state, aux, grad_norm, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target_q,
            compute_critic_loss_on_batch,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=batch.terminated,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            # The smoothed next actions the target scored, for a target term
            # on them (TargetBC).
            target_extras={"next_actions": next_actions},
        )
        aux = aux.replace(grad_norm=grad_norm)
        return agent_state.replace(critic_state=critic_state), aux, metrics

    def update_actor(
        self,
        agent_state: TD3State,
        batch: Transition,
        carries: Any,
        context: TrainContext,
    ) -> tuple[TD3State, tuple[PolicyAuxiliaries, dict]]:
        """The deterministic policy-gradient step on -Q1(s, mu(s)), then
        both targets' Polyak update."""
        actor_key, agent_state = agent_state.get_keys_and_new_state()

        def policy_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
            loss, (aux, pi_mean) = compute_policy_loss(
                params,
                agent_state.actor_state,
                agent_state.critic_state,
                batch.obs,
                carries,
                context.executed,
                batch.a_expert,
            )
            return loss, aux, pi_mean

        actor_state, aux, grad_norm, metrics = update_steps.actor_step(
            agent_state,
            policy_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            observations=batch.obs,
            raw_observations=batch.raw_obs,
            a_expert=batch.a_expert,
            actions=batch.action,
            critic_state=agent_state.critic_state,
        )
        aux = aux.replace(grad_norm=grad_norm)
        agent_state = update_targets(
            agent_state.replace(actor_state=actor_state), self.tau
        )
        return agent_state, (aux, metrics)

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "TD3"
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
            "warmup",
        }
    )
    # Its smoothed next actions join the shared step's on_target keys.
    phase_batch_keys = {  # noqa: RUF012
        **update_steps.REPLAY_BATCH_KEYS,
        "on_target": update_steps.TARGET_KEYS.union({"next_actions"}),
    }
    extension_context = RecurrentReplayAgent.extension_context.union(BELLMAN_CONTEXT)
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"next_action", "raw_next_obs", "return_to_go"})


# -- TD3's equations, in the order update calls them -------------------------
def compute_td3_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """y = scale r + gamma (1 - d) min_i Q_target_i(s', a') with a' =
    clip(mu_target(s') + clip(N(0, sigma), -c, c), -1, 1), the share of the
    noise samples the clip to [-c, c] changed, and those smoothed next
    actions, executed (executed, a residual policy's, from the expert's
    stored next actions)."""
    rewards = rewards * reward_scale
    next_action = actor_dist(
        actor_state,
        actor_state.target_params,
        next_observations,
        carries,
        bootstrap=True,
        target_actor=True,
    ).mean()
    noise = jax.random.normal(rng, next_action.shape) * target_policy_noise
    clip_fraction = (jnp.abs(noise) > target_noise_clip).mean()
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    if executed is not None:
        next_action = executed(next_action, next_a_expert)
    q_targets = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_action,
        carries,
        bootstrap=True,
    )
    target = rewards + gamma * (1.0 - dones) * jnp.min(q_targets, axis=0)
    return jax.lax.stop_gradient(target), clip_fraction, next_action


def compute_critic_loss(
    critic: LoadedTrainState,
    batch: Transition,
    carries: Optional[RecurrentCarries],
    num_critics: int,
    clip_fraction: jax.Array,
    params: FrozenDict,
    target_q: jax.Array,
) -> tuple[jax.Array, ValueAuxiliaries]:
    """Every critic regresses on the one target; their squared errors are
    summed, as sfujim/TD3 sums its two critics' MSEs (TD3.py train):
    compute_mean_critic_loss averages them, and averaging would halve the gradient,
    which Adam's eps then weighs twice as much."""
    loss, aux = compute_mean_critic_loss(
        params, critic, batch.obs, batch.action, target_q, carries
    )
    loss = loss * num_critics
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        target_q=target_q.mean().flatten(),
        q_pred_min=aux.q_pred_min,
        q_gap=aux.q_gap,
        target_noise_clip_frac=clip_fraction,
    )


def compute_policy_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, tuple[PolicyAuxiliaries, jax.Array]]:
    """TD3 actor loss -Q1(s, mu(s)), at the executed action with a
    residual policy (executed); also returns mu(s) for extensions."""
    actions = actor_dist(actor_state, actor_params, observations, carries).mean()
    scored = actions if executed is None else executed(actions, a_expert)
    q_first = q_values(
        critic_state, critic_state.params, observations, scored, carries
    )[0]  # TD3 uses Q1 only for the actor objective.
    loss = -q_first.mean()
    return loss, (PolicyAuxiliaries(policy_loss=loss, q_mean=q_first.mean()), actions)


def update_targets(agent_state: TD3State, tau: float) -> TD3State:
    """Polyak-average both target networks, the actor's and the critics'
    (update_steps.update_target_networks averages the critics' only)."""
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau),
        actor_state=agent_state.actor_state.soft_update(tau=tau),
    )


# -- TD3's exploration, which collection runs ---------------------------------
def make_gaussian_exploration(exploration_noise: Schedule) -> Explore:
    """TD3's behaviour policy: clip(mu(s) + N(0, exploration_noise), -1, 1),
    the noise's scale a schedule of the timestep."""

    def explore(policy: Any, key: jax.Array, step: jax.Array) -> jax.Array:
        mean_action = policy.mean()
        sigma = at(exploration_noise, step)
        noise = jax.random.normal(key, mean_action.shape) * sigma
        return jnp.clip(mean_action + noise, -1.0, 1.0)

    return explore


def make_default_action_pipeline(
    recurrent: bool,
    exploration_noise: Schedule,
    transform: Any = None,
    warmups: tuple = (),
) -> Callable:
    """TD3's action pipeline: make_gaussian_exploration, through the run's
    residual transform where it has one, uniform actions during the
    warm-up, or its chooser's (warmups: ExpertWarmup)."""
    return explore_pipeline(
        make_gaussian_exploration(exploration_noise),
        recurrent,
        transform=transform,
        warmups=warmups,
    )


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class TD3State(BaseAgentState):
    """TD3 carries only actor + critic + collector (no temperature, no expert state)."""


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    q_mean: jax.Array
    # Before clipping; the loss builds these metrics before the gradient
    # exists, the step fills it in.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_min: jax.Array
    q_gap: jax.Array  # mean max - min over the critics: |Q1 - Q2| for two
    target_noise_clip_frac: jax.Array  # share of the target noise clipped
    grad_norm: jax.Array | float = jnp.nan  # as the policy's


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
