"""Implicit Q-Learning (Kostrikov, Nair and Levine, ICLR 2022): offline RL
that never queries the critic at actions outside the data.

One update on a replay batch, in the order of the reference
(ikostrikov/implicit_q_learning@09d7002, learner.py's _update_jit), each
line one function below::

    q = min_i Q_target_i(s, a)                              compute_min_target_q
    V <- argmin E[|expectile - 1(q - V(s) < 0)| (q - V(s))^2]  compute_value_loss
    actor <- argmax E[w log pi(a | s)]          compute_weighted_regression_loss
        w = min(exp(temperature (q - V(s))), 100)      compute_advantage_weights
    Q_i <- argmin sum_i E[(Q_i(s, a) - y)^2]          compute_summed_critic_loss
        y = scale r + gamma (1 - d) V(s')                 compute_bellman_target
    Q_target <- tau Q + (1 - tau) Q_target

The actor's weights w and the critics' target y read V after its step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents import update_steps
from ajax.agents.critics import compute_mean_critic_loss
from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    PolicyUpdateStart,
    ReplayAgent,
    TrainContext,
)
from ajax.agents.recurrent import q_values, sample_replay
from ajax.environments.interaction import get_pi, policy_pipeline
from ajax.extensions.base import merge_metrics
from ajax.networks.networks import predict_value
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture, FloatOrCallable

#: The reference's initialisation (common.py default_init).
ORTHOGONAL_SQRT2 = "orthogonal(np.sqrt(2))"
#: The reference's twin critics, whose losses it sums (critic.py update_q).
NUM_CRITICS = 2


@dataclass(kw_only=True, eq=False, repr=False)
class IQL(
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    ExpertGuided,
    ReplayAgent,
):
    """Implicit Q-Learning: a state value fitted as an expectile of the
    critic over the data's actions, and an actor fitted by
    advantage-weighted regression (the module's equations). Train it on a
    dataset with train_offline (the paper's setting), then fine-tune
    online with train: the same update, the offline rows kept in its
    replay (Sec. 5.3).

    Defaults are the reference code's (configs/mujoco_config.py,
    learner.py, policy.py): expectile 0.7 and temperature 3
    (locomotion; antmaze takes 0.9 and 10); 256-256 ReLU networks,
    orthogonal(sqrt 2) initialisation; Adam at 3e-4 with optax's eps 1e-8
    for the actor, the critics and the value; the actor a Gaussian on a
    tanh mean with one learned log-std per action dimension, clipped to
    [-5, 2]; batch 256, gamma 0.99, tau 0.005. The reference decays the
    actor's offline learning rate to 0 by a cosine over its updates: pass
    actor_learning_rate=optax.cosine_decay_schedule(3e-4, n_updates)
    for that. Its locomotion rewards are scaled by 1000 over the dataset's
    range of returns: reward_scale.
    """

    actor_learning_rate: FloatOrCallable = 3e-4
    critic_learning_rate: FloatOrCallable = 3e-4
    value_learning_rate: FloatOrCallable = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("256", "relu", "256", "relu")
    expectile: float = 0.7
    temperature: float = 3.0
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = 0

    def check(self) -> None:
        if not 0.0 < self.expectile < 1.0:
            raise ValueError(f"IQL's expectile is in (0, 1), got {self.expectile}.")

    def actor_critic_args(self) -> dict[str, Any]:
        return {
            "encoder_kernel_init": ORTHOGONAL_SQRT2,
            "mean_kernel_init": ORTHOGONAL_SQRT2,
            "value_kernel_init": ORTHOGONAL_SQRT2,
            "log_std_state_independent": True,
            "log_std_init": 0.0,
            "log_std_bounds": (-5.0, 2.0),
            "tanh_mean": True,
        }

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # optax.adam's default eps, the reference's Adam

    def setup(self) -> None:
        # V's Adam: the critics', at its own learning rate.
        self.value_optimizer_args = self.critic_optimizer_args.replace(
            learning_rate=self.value_learning_rate
        )

    def choose_behaviour(self, context: TrainContext) -> Optional[Callable]:
        # A residual policy acts on the chosen action, so the warm-up draws
        # residuals (and the replay keeps the one the actor fits, init).
        if context.executed is None:
            return None
        return partial(
            policy_pipeline,
            recurrent=False,
            n_envs=context.env_args.n_envs,
            transform=context.env_args.transforms[0],
            after_choice=True,
            warmups=context.env_args.warmups,
        )

    def init(self, key: jax.Array, context: TrainContext) -> IQLState:
        """A Gaussian actor, two Q critics with their targets, and V(s)."""
        training_key, init_key, value_key, collector_key, evaluation_key = (
            jax.random.split(key, 5)
        )
        actor_state, critic_state = context.make_actor_critic(
            init_key, continuous=True, action_value=True, num_critics=NUM_CRITICS
        )
        _, value_state = context.make_actor_critic(
            value_key,
            critic_optimizer_args=self.value_optimizer_args,
            continuous=True,
            num_critics=1,
        )
        return IQLState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            value_state=value_state,
            collector_state=context.make_collector(
                collector_key, include_policy_action=context.executed is not None
            ),
        )

    def update(
        self, agent_state: IQLState, context: TrainContext
    ) -> tuple[IQLState, AuxiliaryLogs]:
        """One IQL update (the module docstring)."""
        sample_key, critic_key, actor_key, agent_state = (
            agent_state.get_keys_and_new_state(3)
        )
        batch, _ = sample_replay(agent_state, context.buffer, sample_key)
        critic = agent_state.critic_state
        min_q_values = compute_min_target_q(critic, batch)
        compute_value_loss_on_batch = partial(
            compute_value_loss,
            agent_state.value_state,
            batch.obs,
            min_q_values,
            self.expectile,
        )
        value_state, _, _ = update_steps.gradient_step(
            agent_state.value_state, compute_value_loss_on_batch
        )
        value_loss_after_step = compute_value_loss_on_batch(value_state.params)[0]
        state_values = predict_value(value_state, value_state.params, batch.obs)[0]
        advantage_weights = compute_advantage_weights(
            min_q_values, state_values, self.temperature
        )
        # The actor fits the actions it chose: a residual policy's residuals,
        # which the replay keeps apart from the executed actions.
        chosen = batch.action if batch.policy_action is None else batch.policy_action

        def step_actor(state: IQLState) -> tuple[IQLState, tuple]:
            actor_state, actor_loss, _, actor_metrics = update_steps.actor_step(
                state,
                partial(
                    compute_weighted_regression_loss,
                    agent_state.actor_state,
                    batch.obs,
                    chosen,
                    advantage_weights,
                ),
                context.extensions,
                actor_key,
                context.total_timesteps,
                observations=batch.obs,
                raw_observations=batch.raw_obs,
                actions=batch.action,
                critic_state=critic,
            )
            return state.replace(actor_state=actor_state), (actor_loss, actor_metrics)

        stepped, (actor_loss, actor_metrics) = update_steps.run_actor_if(
            self.policy_update_due(agent_state.collector_state.timestep),
            step_actor,
            agent_state,
        )
        target = compute_bellman_target(
            value_state, batch, self.gamma, self.reward_scale
        )
        critic_state, critic_loss, _, critic_metrics = update_steps.critic_step(
            agent_state,
            batch,
            target,
            partial(compute_summed_critic_loss, critic, batch),
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=batch.terminated,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
        )
        # The critic's step read the state from before the actor's.
        agent_state = stepped.replace(
            critic_state=critic_state.soft_update(tau=self.tau),
            value_state=value_state,
            n_updates=agent_state.n_updates + 1,
        )
        return agent_state, AuxiliaryLogs(
            value=Auxiliaries(
                value_loss=value_loss_after_step,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                advantage=jnp.mean(min_q_values - state_values),
                weight=jnp.mean(advantage_weights),
            ),
            extensions=merge_metrics(critic_metrics, actor_metrics),
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "IQL"
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
            "v_critic",
            "reward_learning",
            "warmup",
        }
    )
    phase_batch_keys = {  # noqa: RUF012
        **update_steps.REPLAY_BATCH_KEYS,
        "actor_loss": update_steps.ACTOR_STEP_KEYS.union(
            {"observations", "raw_observations", "actions", "critic_state"}
        ),
    }
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- IQL's equations, in the order update calls them -------------------------
def compute_min_target_q(critic: LoadedTrainState, batch: Transition) -> jax.Array:
    """min_i Q_target_i(s, a) at the rows' actions."""
    per_critic_q_values = q_values(
        critic, critic.target_params, batch.obs, batch.action
    )
    return jnp.min(per_critic_q_values, axis=0)


def compute_expectile_loss(diff: jax.Array, expectile: float) -> jax.Array:
    """|expectile - 1(diff < 0)| diff^2 (critic.py loss)."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff**2


def compute_value_loss(
    value_state: LoadedTrainState,
    obs: jax.Array,
    q_values: jax.Array,
    expectile: float,
    params: Any,
) -> tuple[jax.Array, None]:
    """V's expectile regression on q_values (critic.py update_v)."""
    state_values = predict_value(value_state, params, obs)[0]
    return jnp.mean(compute_expectile_loss(q_values - state_values, expectile)), None


def compute_advantage_weights(
    q_values: jax.Array, state_values: jax.Array, temperature: float
) -> jax.Array:
    """min(exp(temperature (q - V(s))), 100) (actor.py)."""
    return jnp.minimum(jnp.exp(temperature * (q_values - state_values)), 100.0)


def compute_weighted_regression_loss(
    actor_state: LoadedTrainState,
    obs: jax.Array,
    actions: jax.Array,
    weight: jax.Array,
    params: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """-E[weight log pi(a | s)] (actor.py), and pi's mean for the
    extensions."""
    policy, _ = get_pi(actor_state, params, obs)
    log_probs = policy.log_prob(actions).sum(-1, keepdims=True)
    loss = -jnp.mean(weight * log_probs)
    return loss, loss, policy.mean()


def compute_bellman_target(
    value_state: LoadedTrainState, batch: Transition, gamma: float, scale: float
) -> jax.Array:
    """scale r + gamma (1 - d) V(s') (learner.py), on the new V."""
    next_state_values = predict_value(value_state, value_state.params, batch.next_obs)[
        0
    ]
    return scale * batch.reward + gamma * (1.0 - batch.terminated) * next_state_values


def compute_summed_critic_loss(
    critic: LoadedTrainState, batch: Transition, params: Any, target: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The critics' squared errors to the target, summed over the critics as
    the reference sums them: compute_mean_critic_loss averages them, and averaging
    would halve the gradient that Adam's eps and the clip weigh."""
    loss, _ = compute_mean_critic_loss(params, critic, batch.obs, batch.action, target)
    loss = loss * NUM_CRITICS
    return loss, loss


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class IQLState(BaseAgentState):
    """IQL's actor, twin Q critic (with its target), and state-value network."""

    value_state: LoadedTrainState


@struct.dataclass
class Auxiliaries:
    value_loss: jax.Array
    actor_loss: jax.Array
    critic_loss: jax.Array
    advantage: jax.Array  # mean q - V(s) on the batch
    weight: jax.Array  # mean advantage weight


@struct.dataclass
class AuxiliaryLogs:
    value: Auxiliaries
    extensions: dict  # the extensions' training metrics
