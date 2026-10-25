"""CrossQ (Bhatt et al., ICLR 2024): SAC without target networks, for
continuous actions.

One update on a replay batch, in the order of the reference
(adityab/CrossQ@a318a26, sbx/sac/sac.py:195-400), each line one method
below or one of SAC's equations (core, ajax.agents.SAC.core)::

    a' ~ pi(. | s'), log pi(a' | s')                     core.sample_next_actions
    Q(s, a), Q(s', a') in one training pass over both               critic_update
    y = sg(scale r + gamma (1 - d) (min_i Q_i(s', a') - alpha log pi(a' | s')))
                                                         core.soft_bellman_target
    Q <- argmin E[(Q(s, a) - y)^2], its running statistics kept
    every policy_delay-th update, on the same batch:                 actor_update
      actor <- argmin E[alpha log pi(a | s) - min_i Q_i(s, a)]
                                                       core.soft_policy_objective
      alpha <- argmin alpha sg(-log pi(a | s) - H_target)
                                                   core.alpha_temperature_loss_fn

Batch Renormalization in the actor and the critics, and one training pass of
the critics over the rows (s, a) and their next states (s', a')
together, so that both share the batch statistics: the bootstrap reads that
pass's second half, held constant. a' comes from the actor reading its
running statistics, and so does every critic read outside the critic loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.serialization import to_state_dict

from ajax.agents import steps
from ajax.agents.critics import CoreCriticAux, q_regression_loss
from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    ReplayAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.recurrent import actor_dist_train, sample_replay
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.agents.SAC.state import SoftACState
from ajax.controller import policy_action_dim
from ajax.extensions.base import merge_metrics
from ajax.networks.networks import action_value_input, predict_value_train
from ajax.state import AlphaConfig, Transition

#: The authors' layers are flax's defaults: lecun-normal kernels, zero biases.
LECUN = "lecun_normal"


@dataclass(kw_only=True, eq=False, repr=False)
class CrossQ(
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    ReplayAgent,
):
    """CrossQ: SAC without target networks, its actor and critics
    batch-renormalised, its critics trained on one pass over the rows and
    their next states (the module's equations).

    Defaults are the authors' train.py -algo crossq
    (adityab/CrossQ@a318a26, train.py:126-136, :171-200): Batch
    Renorm MLPs, the actor 256 wide and the critics 2048; a squashed
    Gaussian actor whose log-std head starts at 0; Adam at 1e-3 with beta1
    0.5 and optax's eps 1e-8; the actor and the temperature stepping every
    3rd update; a target entropy of -1 per action dimension.

    Args:
        learning_rate: The actor's and the critics' (and the temperature's,
            unless alpha_learning_rate is given).
        actor_width: The units of the actor's two hidden layers.
        critic_width: The units of the critics' two hidden layers; the
            paper's are 2048 wide (SBX's 1024).
        adam_beta_1: The actor's and the critics' Adam beta1; the
            temperature's Adam keeps 0.9, as the authors'.
    """

    learning_rate: float = 1e-3
    alpha_learning_rate: Optional[float] = None
    actor_width: int = 256
    critic_width: int = 2048
    num_critics: int = 2
    gamma: float = 0.99
    policy_delay: int = 3
    adam_beta_1: float = 0.5
    reward_scale: float = 1.0
    alpha_init: float = 1.0
    target_entropy_per_dim: float = -1.0
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = 5000

    def actor_critic_args(self) -> dict[str, Any]:
        return {
            "actor_learning_rate": self.learning_rate,
            "critic_learning_rate": self.learning_rate,
            "actor_architecture": batch_renorm_mlp(self.actor_width),
            "critic_architecture": batch_renorm_mlp(self.critic_width),
            "squash": True,
            "encoder_kernel_init": LECUN,
            "value_kernel_init": LECUN,
            "mean_kernel_init": LECUN,
            "log_std_kernel_init": LECUN,
            "log_std_init": 0.0,
        }

    def optimizer_settings(self) -> dict[str, Any]:
        # The authors' optax.adam: its default eps, their beta1 (train.py).
        return {"eps": 1e-8, "beta_1": self.adam_beta_1}

    @property
    def alpha_args(self) -> AlphaConfig:
        """The temperature's learning rate and initial value."""
        rate = self.alpha_learning_rate
        return AlphaConfig(
            learning_rate=self.learning_rate if rate is None else rate,
            alpha_init=self.alpha_init,
        )

    @property
    def target_entropy(self) -> float:
        """target_entropy_per_dim times the policy's action width (gain
        mode's gains)."""
        return self.target_entropy_per_dim * policy_action_dim(self.env_args)

    def init(self, key: jax.Array, context: TrainContext) -> CrossQState:
        """A squashed Gaussian actor, num_critics Q critics without
        targets, and the temperature."""
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor_state, critic_state = context.actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=self.num_critics,
        )
        collector_state = context.collector(collector_key)
        return CrossQState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor_state,
            # No target networks: nothing reads a copy of the critics.
            critic_state=critic_state.replace(target_params=None),
            alpha=core.create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
        )

    def update(
        self, state: CrossQState, context: TrainContext
    ) -> tuple[CrossQState, AuxiliaryLogs]:
        """A critic step on a fresh replay batch, then, on the
        policy_delay-th, an actor and a temperature step on the same
        batch (the authors' (n_updates + i + 1) % policy_delay,
        sbx/sac/sac.py:201)."""
        sample_key, state = state.get_keys_and_new_state()
        batch, _ = sample_replay(state, context.buffer, sample_key)
        state, value, value_metrics = self.critic_update(state, batch, context)
        due = (state.n_updates + 1) % self.policy_delay == 0
        actor_update = partial(self.actor_update, batch=batch, context=context)
        state, (policy, temperature, policy_metrics) = steps.when(
            due, actor_update, state
        )
        state = state.replace(n_updates=state.n_updates + 1)
        return state, AuxiliaryLogs(
            temperature=temperature,
            policy=policy,
            value=value,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def critic_update(
        self, state: CrossQState, batch: Transition, context: TrainContext
    ) -> tuple[CrossQState, CoreCriticAux, dict]:
        """One critic step on one training pass over (s, a) and (s',
        a'), its running statistics kept (sbx/sac/sac.py:251-342);
        a' is executed (a residual policy's)."""
        key, next_key, state = state.get_keys_and_new_state(2)
        alpha = jnp.exp(state.alpha.params["log_alpha"])
        next_actions, next_log_probs = core.sample_next_actions(
            state.actor_state,
            batch.next_obs,
            next_key,
            executed=context.executed,
            next_a_expert=batch.next_a_expert,
        )
        critic_state = state.critic_state
        rows = jnp.concatenate(
            [
                action_value_input(batch.obs, batch.action),
                action_value_input(batch.next_obs, next_actions),
            ]
        )

        def loss_fn(params: Any) -> tuple[jax.Array, Any]:
            q, batch_stats = predict_value_train(critic_state, params, rows)
            q_now, q_next = jnp.split(q, 2, axis=1)
            target = core.soft_bellman_target(
                jnp.min(q_next, axis=0),
                next_log_probs,
                batch.reward,
                batch.terminated,
                self.gamma,
                alpha,
                self.reward_scale,
            )
            target, target_metrics = steps.apply_on_target(
                context.extensions,
                state,
                batch,
                target,
                key,
                context.total_timesteps,
                rewards=batch.reward,
                dones=batch.terminated,
                gamma=self.gamma,
                reward_scale=self.reward_scale,
            )
            loss, aux = q_regression_loss(q_now, target)
            extra, loss_metrics = steps.apply_critic_loss(
                context.extensions,
                state,
                batch,
                params,
                target,
                key,
                context.total_timesteps,
            )
            metrics = merge_metrics(target_metrics, loss_metrics)
            return loss + extra, (aux, batch_stats, metrics)

        critic_state, (aux, batch_stats, metrics), _ = steps.gradient_step(
            critic_state, loss_fn
        )
        critic_state = critic_state.replace(batch_stats=batch_stats)
        return state.replace(critic_state=critic_state), aux, metrics

    def actor_update(
        self, state: CrossQState, batch: Transition, context: TrainContext
    ) -> tuple[CrossQState, tuple]:
        """The actor step in training, the critics read with their running
        statistics (at the executed actions, a residual policy's), then the
        temperature's on the same samples (sbx/sac/sac.py:355-400)."""
        key, state = state.get_keys_and_new_state()
        alpha = jnp.exp(state.alpha.params["log_alpha"])
        actor_state, critic_state = state.actor_state, state.critic_state
        transform = None
        if context.executed is not None:
            transform = partial(scored_action, context.executed, batch.a_expert)

        def policy_loss(params: Any) -> tuple[jax.Array, Any, jax.Array]:
            pi, batch_stats = actor_dist_train(actor_state, params, batch.obs)
            loss, (log_probs, q, pi) = core.soft_policy_objective(
                pi, critic_state, batch.obs, alpha, key, action_transform=transform
            )
            aux = PolicyAuxiliaries(
                policy_loss=loss, log_pi=log_probs.mean(), q_min=q.mean()
            )
            return loss, (aux, log_probs, batch_stats), pi.mean()

        actor_state, (aux, log_probs, batch_stats), _, metrics = steps.actor_step(
            state,
            policy_loss,
            context.extensions,
            key,
            context.total_timesteps,
            observations=batch.obs,
            raw_observations=batch.raw_obs,
            a_expert=batch.a_expert,
            actions=batch.action,
            critic_state=critic_state,
        )
        actor_state = actor_state.replace(batch_stats=batch_stats)
        # The authors' temperature loss: alpha, not log alpha, times the gap.
        state, temperature = core.update_temperature(
            state.replace(actor_state=actor_state),
            jax.lax.stop_gradient(log_probs),
            jnp.asarray(self.target_entropy),
            core.alpha_temperature_loss_fn,
        )
        return state, (aux, temperature, metrics)

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "CrossQ"
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
            "reward_learning",
            "warmup",
        }
    )
    trains_batch_norm = True
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- CrossQ's own helpers; its equations are SAC.core's ----------------------
def batch_renorm_mlp(width: int) -> tuple[str, ...]:
    """The authors' MLP: Batch Renorm on the input and after each of two
    ReLU layers of width units (sbx/sac/policies.py:199-231)."""
    layer = (str(width), "relu", "batchrenorm")
    return ("batchrenorm", *layer, *layer)


def scored_action(executed: Callable, a_expert: Any, actions: jax.Array) -> jax.Array:
    """The executed action the critics score (a residual policy's)."""
    return executed(actions, a_expert)


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class CrossQState(SoftACState):
    """CrossQ's state: a soft actor-critic's, its critics without targets."""


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: CoreCriticAux
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
CrossQConfig = CrossQ.declare_config(
    "gamma target_entropy learning_starts reward_scale policy_delay"
)
