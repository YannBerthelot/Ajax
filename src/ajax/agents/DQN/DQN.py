"""Deep Q-Network (Mnih et al., 2015), as its reference code,
google-deepmind/dqn @9d9b1d1, runs it, for discrete action spaces.

One iteration on the replay family's loop (ReplayAgent): each env takes an
epsilon-greedy step on the greedy Q-network (DQN.choose_behaviour), uniform
during the warm-up, into the replay; from learning_starts, n_gradient_steps
updates, each line one function below::

    y = scale r + gamma (1 - d) max_a' Q_target(s', a')        compute_dqn_td_target
        Double DQN: Q_target(s', argmax_a' Q(s', a'))    compute_double_dqn_td_target
    one centred-RMSProp step of Q down Huber_1(Q(s, a) - y)
                                         (critics.q_gradient_step, DQN.update_q_network)
    every target_update_interval-th update:
        Q_target <- tau Q + (1 - tau) Q_target

A hard refresh is tau=1 every target_update_interval updates (the
paper's), a Polyak one tau<1 every update. The Q-network lives in
agent_state.actor_state, which the shared evaluation reads (the greedy
policy, argmax_a Q); critic_state mirrors it at init and is never stepped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.critics import (
    ValueAuxiliaries,
    compute_huber_1_td_loss,
    q_gradient_step,
)
from ajax.agents.families import (
    GradientClipping,
    ImageObservations,
    ObservationNormalization,
    ReplayAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.update_steps import derive_extensions_key, run_if
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    epsilon_greedy,
    explore_pipeline,
    reward_statistics,
)
from ajax.environments.utils import get_action_dim
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import merge_metrics
from ajax.networks.networks import reference_cnn
from ajax.networks.q_networks import (
    get_initialized_q_network,
    predict_q,
    q_network_value,
)
from ajax.networks.utils import TORCH_LINEAR
from ajax.perf_utils import final_aux_scan
from ajax.schedule import linear
from ajax.state import BaseAgentState, LoadedTrainState
from ajax.types import Architecture


@dataclass(kw_only=True, eq=False, repr=False)
class DQN(
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ImageObservations,
    ReplayAgent,
):
    """Deep Q-Network (Mnih et al., 2015) for discrete action spaces: a
    Q-network trained by TD against a periodically refreshed target network
    (the module's equations), acting epsilon-greedy.

    Defaults: the reference's learning rate (2.5e-4, run_gpu:25) and
    optimiser (optimizer_settings); a smaller 128-128 ReLU MLP, a 1e5
    replay, batches of 64, 1000 warm-up steps and a refresh every 500
    updates, sized for gymnax's classic-control tasks rather than Atari.
    """

    #: The one Q-network's learning rate and hidden layers.
    learning_rate: float = 2.5e-4
    architecture: Architecture = ("128", "relu", "128", "relu")
    gamma: float = 0.99
    buffer_size: int = int(1e5)
    batch_size: int = 64
    learning_starts: int = 1000
    #: The target's refresh: tau=1 every target_update_interval updates is
    #: the hard one, tau<1 every update a Polyak one.
    tau: float = 1.0
    target_update_interval: int = 500
    reward_scale: float = 1.0
    #: The updates per iteration, once learning_starts is reached.
    n_gradient_steps: int = 1
    #: Linear epsilon-greedy schedule: epsilon goes from epsilon_start to
    #: epsilon_end over the first epsilon_decay_frac of training.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.5
    #: The TD target: None is compute_dqn_td_target; compute_double_dqn_td_target
    #: is Double DQN's.
    td_target_fn: Optional[Callable] = None
    #: The TD loss: None is compute_huber_1_td_loss, the reference's TD-error
    #: clip; compute_mse_td_loss (ajax.agents.critics) the squared error.
    td_loss_fn: Optional[Callable] = None
    #: The Q-network module: None is QNetwork; DuelingQNetwork
    #: (ajax.networks.q_networks) the dueling architecture.
    q_network_cls: Optional[type] = None

    def actor_critic_args(self) -> dict[str, Any]:
        # DQN has one Q-network: both of ActorCritic's slots take it. Its
        # reference builds it from Torch7 nn layers it never re-initialises
        # (dqn/convnet.lua:23-61), and Torch7's nn.Linear draws kernel and
        # bias uniform in +-1/sqrt(fan_in) (torch/nn @8726825,
        # Linear.lua:21-41): "torch_linear", for the MLP encoder and the Q head.
        args: dict[str, Any] = {
            "actor_learning_rate": self.learning_rate,
            "critic_learning_rate": self.learning_rate,
            "actor_architecture": self.architecture,
            "critic_architecture": self.architecture,
            "encoder_kernel_init": TORCH_LINEAR,
            "value_kernel_init": TORCH_LINEAR,
        }
        # The image network of the reference: Mnih et al.'s, or MinAtar's for
        # images too small for it (reference_cnn), at Torch7's initialisation.
        if self.cnn_image_shape is not None and self.cnn_spec is None:
            args["cnn_spec"] = reference_cnn(self.cnn_image_shape)._replace(
                kernel_init=TORCH_LINEAR
            )
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        # The reference's optimiser (NeuralQLearner.lua:265-277): centred
        # RMSProp, both averages decaying at 0.95, eps 0.01 inside the square
        # root of a gradient summed over the batch (:254). Ajax's TD loss
        # averages over the batch, so the same step takes eps 0.01 /
        # batch_size**2.
        eps = 0.01 / self.batch_size**2
        return {"optimizer": "centered_rmsprop", "beta_2": 0.95, "eps": eps}

    def value_estimate(self) -> ValueEstimate:
        # The greedy policy's max_a Q(s, a), not a critic's Q at an actor's
        # action (the family's choice for a q_critic).
        return ValueEstimate(q_network_value, self.gamma, self.reward_scale)

    def make_epsilon_schedule(
        self, context: TrainContext
    ) -> Callable[[Any], jax.Array]:
        """The exploration rate at a timestep: linear over the first
        epsilon_decay_frac of the horizon, then flat."""
        return linear(
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_frac * context.total_timesteps,
        )

    def choose_behaviour(self, context: TrainContext) -> Callable:
        """Epsilon-greedy collection on the greedy Q-network, uniform (or
        the warm-up's chooser) until learning_starts."""
        env_args = context.env_args
        n_actions = get_action_dim(env_args.env, env_args.env_params)
        explore = epsilon_greedy(self.make_epsilon_schedule(context), n_actions)
        return explore_pipeline(explore, False, n_actions, warmups=env_args.warmups)

    def init(self, key: jax.Array, context: TrainContext) -> DQNState:
        """The Q-network and the collector, writing to the replay."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        env_args = context.env_args
        q_state = get_initialized_q_network(
            key=init_key,
            env_config=env_args,
            optimizer_config=context.critic_optimizer_args,
            network_config=context.network_args,
            n_actions=get_action_dim(env_args.env, env_args.env_params),
            q_network_cls=self.q_network_cls,
        )
        return DQNState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=q_state,
            critic_state=q_state,
            collector_state=context.make_collector(collector_key),
        )

    def update(
        self, agent_state: DQNState, context: TrainContext
    ) -> tuple[DQNState, AuxiliaryLogs]:
        """n_gradient_steps steps of the Q-network: the metrics of the last."""

        one_step = lambda agent_state: self.update_q_network(agent_state, context)  # noqa: E731
        if os.environ.get("NESTED_JIT") == "step":
            one_step = jax.jit(one_step)

        def step(agent_state: DQNState, _: Any) -> tuple[DQNState, AuxiliaryLogs]:
            return one_step(agent_state)

        # Carry-only scan: only the final step's metrics are materialised.
        return final_aux_scan(step, agent_state, length=self.n_gradient_steps)

    def update_q_network(
        self, agent_state: DQNState, context: TrainContext
    ) -> tuple[DQNState, AuxiliaryLogs]:
        """One gradient step on a replay batch, then the target's refresh
        every target_update_interval steps; also logs the exploration rate
        at the collector's timestep."""
        sample_key, agent_state = agent_state.get_keys_and_new_state()
        collector = agent_state.collector_state
        batch = get_batch_from_buffer(
            self.buffer,
            collector.buffer_state,
            sample_key,
            reward_statistics(collector.env_state),
        )
        observations, actions, rewards = batch.obs, batch.action, batch.reward
        terminated, next_observations = batch.terminated, batch.next_obs
        # A truncated row bootstraps on its final observation (bootstrap_obs).
        dones = terminated.astype(jnp.float32)
        td_target_fn = self.td_target_fn
        if td_target_fn is None:
            td_target_fn = compute_dqn_td_target
        td_loss_fn = self.td_loss_fn
        if td_loss_fn is None:
            td_loss_fn = compute_huber_1_td_loss
        target_q = td_target_fn(
            q_state=agent_state.actor_state,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
        )
        # The extensions reshape the target, then add their loss terms, each
        # on a key of its own.
        target_key, loss_key = jax.random.split(derive_extensions_key(sample_key))
        target_batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "dones": dones,
            "gamma": self.gamma,
            "reward_scale": self.reward_scale,
        }
        target_q, target_metrics = context.extensions.apply_on_target(
            agent_state,
            target_batch,
            target_q,
            agent_state.collector_state.timestep,
            target_key,
            context.total_timesteps,
        )
        q_state, value_aux, loss_metrics = q_gradient_step(
            agent_state,
            observations,
            actions,
            target_q,
            td_loss_fn,
            context.extensions,
            loss_key,
            context.total_timesteps,
        )
        refresh = (agent_state.n_updates % self.target_update_interval) == 0
        q_state = run_if(
            refresh, lambda q_state: q_state.soft_update(self.tau), q_state
        )
        agent_state = agent_state.replace(
            actor_state=q_state, n_updates=agent_state.n_updates + 1
        )
        epsilon = self.make_epsilon_schedule(context)(collector.timestep)
        return agent_state, AuxiliaryLogs(
            value=value_aux,
            policy=PolicyAuxiliaries(epsilon=epsilon),
            extensions=merge_metrics(target_metrics, loss_metrics),
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "DQN"
    action_space = "discrete"
    capabilities = frozenset(
        {
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
    # Its explore pipeline runs the warm-up's chooser (choose_warmup_action),
    # once DQN takes the expert ExpertWarmup needs.
    supported_extension_phases = LOOP_PHASES.union(
        {"on_target", "critic_loss", "choose_warmup_action"}
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "on_target": frozenset({"actions", "dones", "reward_scale"}),
        "critic_loss": frozenset({"actions", "q_state"}),
    }
    extension_context = frozenset()


# -- DQN's equations, in the order the update calls them ---------------------
def compute_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """DQN's target: y = scale r + gamma (1 - d) max_a' Q_target(s', a')."""
    rewards = rewards * reward_scale
    next_q_values = predict_q(q_state, q_state.target_params, next_observations)
    max_next_q_values = jnp.max(next_q_values, axis=-1, keepdims=True)
    target = rewards + gamma * (1.0 - dones) * max_next_q_values
    return jax.lax.stop_gradient(target)


def compute_double_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """Double DQN's target (van Hasselt et al., 2016): the online network
    selects the next action, the target network scores it, which removes
    the maximisation bias of max_a' Q_target::

        a* = argmax_a' Q(s', a')
        y  = scale r + gamma (1 - d) Q_target(s', a*)

    In place of compute_dqn_td_target: DQN(..., td_target_fn=
    compute_double_dqn_td_target).
    """
    rewards = rewards * reward_scale
    next_online_q_values = predict_q(q_state, q_state.params, next_observations)
    next_actions = jnp.argmax(next_online_q_values, axis=-1, keepdims=True)
    next_target_q_values = predict_q(q_state, q_state.target_params, next_observations)
    next_q_values = jnp.take_along_axis(next_target_q_values, next_actions, axis=-1)
    target = rewards + gamma * (1.0 - dones) * next_q_values
    return jax.lax.stop_gradient(target)


# -- State, logged metrics and the agent_config view --------------------------
@struct.dataclass(kw_only=True)
class DQNState(BaseAgentState):
    """DQN carries a single Q-network, no actor and no temperature: in
    actor_state, which the shared evaluation reads; critic_state mirrors it
    at init and is left inert."""


@struct.dataclass
class PolicyAuxiliaries:
    epsilon: jax.Array  # the exploration rate at the collector's timestep


@struct.dataclass
class AuxiliaryLogs:
    # Nested one level deep so ajax.log.flatten_dict names the metrics
    # value/ then the field's name, as the other agents' auxiliaries are named.
    value: ValueAuxiliaries
    policy: PolicyAuxiliaries
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
DQNConfig = DQN.declare_config(
    (
        "gamma",
        "tau",
        "target_update_interval",
        "learning_starts",
        "reward_scale",
        "n_gradient_steps",
    )
)
