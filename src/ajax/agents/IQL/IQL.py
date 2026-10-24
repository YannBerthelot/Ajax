"""Implicit Q-Learning (Kostrikov, Nair and Levine, ICLR 2022): offline RL
that never queries the critic at actions outside the data.

One update on a replay batch, in the order of the reference
(ikostrikov/implicit_q_learning@09d7002, learner.py's _update_jit), each
line one function below::

    q = min_i Q_target_i(s, a)                                     min_target_q
    V <- argmin E[|expectile - 1(q - V(s) < 0)| (q - V(s))^2]         value_loss
    actor <- argmax E[min(exp(temperature (q - V(s))), 100) log pi(a | s)]
                                         awr_weight, awr_loss (on the new V)
    Q_i <- argmin sum_i E[(Q_i(s, a) - scale r - gamma (1 - d) V(s'))^2]
                                                      bellman_target, q_loss
    Q_target <- tau Q + (1 - tau) Q_target
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents import steps
from ajax.agents.critics import critic_loss_fn
from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
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
class IQL(GradientClipping, ObservationNormalization, ExpertGuided, ReplayAgent):
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

    def behaviour(self, context: TrainContext) -> Optional[Callable]:
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
        )

    def init(self, key: jax.Array, context: TrainContext) -> IQLState:
        """A Gaussian actor, two Q critics with their targets, and V(s)."""
        rng, init_key, value_key, collector_key, eval_rng = jax.random.split(key, 5)
        actor_state, critic_state = context.actor_critic(
            init_key, continuous=True, action_value=True, num_critics=NUM_CRITICS
        )
        _, value_state = context.actor_critic(
            value_key,
            critic_optimizer_args=self.value_optimizer_args,
            continuous=True,
            num_critics=1,
        )
        return IQLState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor_state,
            critic_state=critic_state,
            value_state=value_state,
            collector_state=context.collector(
                collector_key, include_policy_action=context.executed is not None
            ),
        )

    def update(
        self, state: IQLState, context: TrainContext
    ) -> tuple[IQLState, AuxiliaryLogs]:
        """One IQL update (the module docstring)."""
        state, sample_key, critic_key, actor_key = state.draw_keys(3)
        batch, _ = sample_replay(state, context.buffer, sample_key)
        critic = state.critic_state
        q = min_target_q(critic, batch)
        fit_v = partial(value_loss, state.value_state, batch.obs, q, self.expectile)
        value_state, _, _ = steps.gradient_step(state.value_state, fit_v)
        v_loss = fit_v(value_state.params)[0]
        v = predict_value(value_state, value_state.params, batch.obs)[0]
        weight = awr_weight(q, v, self.temperature)
        # The actor fits the actions it chose: a residual policy's residuals,
        # which the replay keeps apart from the executed actions.
        chosen = batch.action if batch.policy_action is None else batch.policy_action
        actor_state, a_loss, _, actor_metrics = steps.actor_step(
            state,
            partial(awr_loss, state.actor_state, batch.obs, chosen, weight),
            context.extensions,
            actor_key,
            context.total_timesteps,
            observations=batch.obs,
            raw_observations=batch.raw_obs,
            actions=batch.action,
            critic_state=critic,
        )
        target = bellman_target(value_state, batch, self.gamma, self.reward_scale)
        critic_state, c_loss, _, critic_metrics = steps.critic_step(
            state,
            batch,
            target,
            partial(q_loss, critic, batch),
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=batch.terminated,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
        )
        state = state.replace(
            actor_state=actor_state,
            critic_state=critic_state.soft_update(tau=self.tau),
            value_state=value_state,
            n_updates=state.n_updates + 1,
        )
        return state, AuxiliaryLogs(
            value=Auxiliaries(
                value_loss=v_loss,
                actor_loss=a_loss,
                critic_loss=c_loss,
                advantage=jnp.mean(q - v),
                weight=jnp.mean(weight),
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
        }
    )
    phase_batch_keys = {  # noqa: RUF012
        **steps.REPLAY_BATCH_KEYS,
        "actor_loss": steps.ACTOR_STEP_KEYS.union(
            {"observations", "raw_observations", "actions", "critic_state"}
        ),
    }
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- IQL's equations, in the order update calls them -------------------------
def min_target_q(critic: LoadedTrainState, batch: Transition) -> jax.Array:
    """min_i Q_target_i(s, a) at the rows' actions."""
    q = q_values(critic, critic.target_params, batch.obs, batch.action)
    return jnp.min(q, axis=0)


def expectile_loss(diff: jax.Array, expectile: float) -> jax.Array:
    """|expectile - 1(diff < 0)| diff^2 (critic.py loss)."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff**2


def value_loss(
    value_state: LoadedTrainState,
    obs: jax.Array,
    q: jax.Array,
    expectile: float,
    params: Any,
) -> tuple[jax.Array, None]:
    """V's expectile regression on q (critic.py update_v)."""
    v = predict_value(value_state, params, obs)[0]
    return jnp.mean(expectile_loss(q - v, expectile)), None


def awr_weight(q: jax.Array, v: jax.Array, temperature: float) -> jax.Array:
    """min(exp(temperature (q - V(s))), 100) (actor.py)."""
    return jnp.minimum(jnp.exp(temperature * (q - v)), 100.0)


def awr_loss(
    actor_state: LoadedTrainState,
    obs: jax.Array,
    actions: jax.Array,
    weight: jax.Array,
    params: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """-E[weight log pi(a | s)] (actor.py), and pi's mean for the
    extensions."""
    pi, _ = get_pi(actor_state, params, obs)
    log_probs = pi.log_prob(actions).sum(-1, keepdims=True)
    loss = -jnp.mean(weight * log_probs)
    return loss, loss, pi.mean()


def bellman_target(
    value_state: LoadedTrainState, batch: Transition, gamma: float, scale: float
) -> jax.Array:
    """scale r + gamma (1 - d) V(s') (learner.py), on the new V."""
    next_v = predict_value(value_state, value_state.params, batch.next_obs)[0]
    return scale * batch.reward + gamma * (1.0 - batch.terminated) * next_v


def q_loss(
    critic: LoadedTrainState, batch: Transition, params: Any, target: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The critics' squared errors to the target, summed over the critics as
    the reference sums them: critic_loss_fn averages them, and averaging
    would halve the gradient that Adam's eps and the clip weigh."""
    loss, _ = critic_loss_fn(params, critic, batch.obs, batch.action, target)
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


#: agent.agent_config's class: a view of the fields until 0.5.0.
IQLConfig = IQL.declare_config(
    "expectile temperature gamma tau learning_starts reward_scale"
)
