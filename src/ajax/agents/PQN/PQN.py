"""Parallelised Q-Network (Gallici et al., 2024, "Simplifying Deep Temporal
Difference Learning"), as its reference code, purejaxql (mttga/purejaxql
@47af6d7), runs it; its Q-network is PQNNetwork.

One iteration. The on-policy family runs the epsilon-greedy policy for
n_steps in each of n_envs envs
(OnPolicyAgent, PQN.choose_behaviour); each
line after that is a function below::

    V_t = max_a Q(s_{t+1}, a), on the network that collected      PQN.update
    G_t = r_t + gamma ((1 - lambda) V_t + lambda G_{t+1})
          (r_t at an episode's end)                    compute_q_lambda_targets
          no gradient flows back through the targets G_t
    for each of n_epochs, for each minibatch of shuffled single steps:
        one RAdam step of Q down (Q(s_t, a_t) - G_t)^2 / 2     compute_half_mse_td_loss
                                                (critics.q_gradient_step)

PQN is DQN's sibling: it shares the value agents' Q-network wrapper
(ajax.networks.q_networks), their TD step (ajax.agents.critics)
and epsilon-greedy collection, not DQN's code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.critics import ValueAuxiliaries, q_gradient_step
from ajax.agents.families import (
    GradientClipping,
    ImageObservations,
    ObservationNormalization,
    OnPolicyAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.minibatch import get_minibatches_from_batch
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.update_steps import derive_extensions_key
from ajax.environments.interaction import epsilon_greedy, explore_pipeline
from ajax.environments.utils import get_action_dim
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import merge_metrics
from ajax.networks.networks import reference_cnn
from ajax.networks.q_networks import (
    get_initialized_q_network,
    predict_q,
    q_network_value,
)
from ajax.schedule import linear
from ajax.state import BaseAgentState, Transition
from ajax.types import Architecture


@dataclass(kw_only=True, eq=False, repr=False)
class PQN(
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ImageObservations,
    OnPolicyAgent,
):
    """Parallelised Q-Network (Gallici et al., 2024) for discrete action spaces.

    On-policy value-based RL: vectorised rollouts, Q(lambda) regression
    targets, no replay buffer and no target network -- training stability
    instead comes from LayerNorm in the Q-network. The policy is greedy
    over Q, with epsilon-greedy exploration during collection.
    """

    n_envs: int = 8
    #: The one Q-network's learning rate and hidden layers.
    learning_rate: float = 2.5e-4
    architecture: Architecture = ("128", "relu", "128", "relu")
    gamma: float = 0.99
    #: Q(lambda) trace-decay coefficient (0 -> 1-step TD, 1 -> Monte-Carlo).
    q_lambda: float = 0.65
    n_steps: int = 128
    n_epochs: int = 4
    #: Minibatches per epoch; must divide the rollout's n_steps * n_envs rows.
    num_minibatches: int = 4
    reward_scale: float = 1.0
    #: purejaxql's global clip of the gradient, MAX_GRAD_NORM 10 in every
    #: PQN config.
    max_grad_norm: Optional[float] = 10.0
    #: Linear epsilon-greedy schedule: epsilon goes from epsilon_start to
    #: epsilon_end over the first epsilon_decay_frac of training.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.5
    #: The TD loss: None is compute_half_mse_td_loss, purejaxql's;
    #: make_huber_td_loss(delta) (from ajax.agents.critics) the Huber loss.
    td_loss_fn: Optional[Callable] = None

    def actor_critic_args(self) -> dict[str, Any]:
        # PQN has a single Q-network: both of ActorCritic's slots take it.
        args: dict[str, Any] = {
            "actor_learning_rate": self.learning_rate,
            "critic_learning_rate": self.learning_rate,
            "actor_architecture": self.architecture,
            "critic_architecture": self.architecture,
        }
        # The image network of PQN's reference, purejaxql: Mnih et al.'s, or
        # MinAtar's for images too small for it (reference_cnn), each layer
        # he_normal and followed by a LayerNorm.
        if self.cnn_image_shape is not None and self.cnn_spec is None:
            args["cnn_spec"] = reference_cnn(self.cnn_image_shape)._replace(
                kernel_init="he_normal", layer_norm=True
            )
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        # The optimiser of PQN's reference: optax.radam (its eps 1e-8) after
        # a global clip of 10 (mttga/purejaxql @47af6d7 pqn_minatar.py:159-162,
        # pqn_atari.py:194-197, MAX_GRAD_NORM 10 in every PQN config).
        return {"optimizer": "radam", "eps": 1e-8}

    def setup(self) -> None:
        if self.n_steps * self.env_args.n_envs % self.num_minibatches != 0:
            raise ValueError(
                f"the rollout's n_steps * n_envs ({self.n_steps} *"
                f" {self.env_args.n_envs}) rows must be divisible by"
                f" num_minibatches ({self.num_minibatches})."
            )

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
        """Epsilon-greedy collection on the greedy Q-network."""
        env_args = context.env_args
        n_actions = get_action_dim(env_args.env, env_args.env_params)
        epsilon_schedule = self.make_epsilon_schedule(context)
        # purejaxql reads epsilon once per update, at its update count
        # (pqn_gymnax.py:189): here at the first step of the rollout, update
        # count x rollout size.
        rollout_size = self.n_steps * env_args.n_envs

        def epsilon_at_rollout_start(timestep: jax.Array) -> jax.Array:
            return epsilon_schedule(timestep - timestep % rollout_size)

        return explore_pipeline(
            epsilon_greedy(epsilon_at_rollout_start, n_actions),
            False,
            n_actions,
            warmup=False,
        )

    def init(self, key: jax.Array, context: TrainContext) -> PQNState:
        """The Q-network and the collector."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        env_args = context.env_args
        q_state = get_initialized_q_network(
            key=init_key,
            env_config=env_args,
            optimizer_config=context.critic_optimizer_args,
            network_config=context.network_args,
            n_actions=get_action_dim(env_args.env, env_args.env_params),
            q_network_cls=PQNNetwork,
        )
        # The single Q-network lives in actor_state (so the shared eval loop
        # works); critic_state mirrors it at init and is never updated.
        return PQNState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=q_state,
            critic_state=q_state,
            collector_state=context.make_collector(collector_key),
        )

    def update(
        self,
        agent_state: PQNState,
        rollout: Transition,
        agent_state_at_rollout_start: PQNState,
        context: TrainContext,
    ) -> Tuple[PQNState, AuxiliaryLogs]:
        """One PQN update on an (n_steps, n_envs) rollout: Q(lambda)
        targets bootstrapped on the current network (no target network), then
        n_epochs epochs of TD steps, the rollout reshuffled each epoch;
        also logs the epsilon its rollout read (the schedule read once per
        update, as purejaxql reads it)."""
        del agent_state_at_rollout_start  # a feedforward network replays nothing
        extensions, total_timesteps = context.extensions, context.total_timesteps
        next_q = predict_q(
            agent_state.actor_state,
            agent_state.actor_state.params,
            rollout.next_obs,
        )
        next_q_max = jnp.max(next_q, axis=-1, keepdims=True)
        targets = compute_q_lambda_targets(
            rollout.reward * self.reward_scale,
            next_q_max,
            rollout.terminated,
            rollout.truncated,
            self.gamma,
            self.q_lambda,
        )
        # The carried key first, as purejaxql splits it (rng, _rng =
        # jax.random.split(rng), pqn_gymnax.py:318): the reverse of
        # get_keys_and_new_state's order, kept so PQN draws its reference's keys.
        training_key, update_key = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=training_key)
        # The extensions reshape the targets before the gradient is stopped
        # there (stop_gradient below), on a key derived from the update's,
        # then add their loss terms, on keys derived from each epoch's: one
        # per minibatch.
        target_metrics: dict = {}
        if extensions:
            target_batch = {
                "observations": rollout.obs,
                "actions": rollout.action,
                "next_observations": rollout.next_obs,
                "rewards": rollout.reward,
                # Where the target cut the bootstrap, as the replay agents pass.
                "dones": rollout.terminated,
                "terminated": rollout.terminated,
                "truncated": rollout.truncated,
                "gamma": self.gamma,
                "reward_scale": self.reward_scale,
            }
            targets, target_metrics = extensions.apply_on_target(
                agent_state,
                target_batch,
                targets,
                agent_state.collector_state.timestep,
                derive_extensions_key(update_key),
                total_timesteps,
            )
        targets = jax.lax.stop_gradient(targets)
        batch = (rollout.obs, rollout.action, targets)
        td_loss_fn = (
            compute_half_mse_td_loss if self.td_loss_fn is None else self.td_loss_fn
        )

        # PQN always cuts the rollout one way: shuffled single steps, targets fixed above.
        epoch_keys = jax.random.split(update_key, self.n_epochs)

        def run_epoch(agent_state: PQNState, epoch_key: jax.Array) -> Tuple[Any, Any]:
            minibatches = get_minibatches_from_batch(
                batch, epoch_key, self.num_minibatches
            )
            loss_keys = jax.random.split(
                derive_extensions_key(epoch_key), self.num_minibatches
            )

            def step_on_minibatch(
                agent_state: PQNState, minibatch_and_key: tuple
            ) -> Tuple[Any, Any]:
                (observations, actions, targets), loss_key = minibatch_and_key
                q_state, value_aux, loss_metrics = q_gradient_step(
                    agent_state,
                    observations,
                    actions,
                    targets,
                    td_loss_fn,
                    extensions,
                    loss_key,
                    total_timesteps,
                )
                agent_state = agent_state.replace(actor_state=q_state)
                return agent_state, (value_aux, loss_metrics)

            return jax.lax.scan(
                step_on_minibatch, agent_state, (minibatches, loss_keys)
            )

        agent_state, (value_aux, loss_metrics) = jax.lax.scan(
            run_epoch, agent_state, epoch_keys
        )
        # (n_epochs, num_minibatches, ...) -> one (1,) value per metric.
        value_aux, loss_metrics = jax.tree.map(
            lambda metric: metric.mean().reshape((1,)), (value_aux, loss_metrics)
        )
        rollout_size = rollout.reward.shape[0] * rollout.reward.shape[1]
        epsilon_schedule = self.make_epsilon_schedule(context)
        epsilon = epsilon_schedule(agent_state.n_updates * rollout_size).reshape((1,))
        agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
        return agent_state, AuxiliaryLogs(
            value=value_aux,
            policy=PolicyAuxiliaries(epsilon=epsilon),
            extensions=merge_metrics(target_metrics, loss_metrics),
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "PQN"
    action_space = "discrete"
    capabilities = frozenset(
        {"discounted", "obs_networks", "off_policy", "q_critic", "reward_learning"}
    )
    supported_extension_phases = LOOP_PHASES.union({"on_target", "critic_loss"})
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "on_target": frozenset(
            {"actions", "dones", "reward_scale", "terminated", "truncated"}
        ),
        "critic_loss": frozenset({"actions", "q_state"}),
    }


# -- PQN's equations, in the order the update calls them ---------------------
@partial(jax.jit, static_argnames=["gamma", "q_lambda"])
def compute_q_lambda_targets(
    rewards: jax.Array,
    next_q_max: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    q_lambda: float,
) -> jax.Array:
    """Backward Q(lambda) returns over a rollout, as purejaxql's PQN
    computes them (pqn_gymnax.py:216-254 at 47af6d7).

    With the bootstrap V_t := max_a Q(s_{t+1}, a), for each step t:

        G_t = r_t                                         at an end
        G_t = r_t + gamma ((1 - lambda) V_t + lambda G_{t+1})   else

    and the last step's G = r + gamma V. An end is a termination or a
    time limit: gymnax's done flag, which purejaxql reads, does not tell
    them apart. purejaxql seeds its backward pass with the last step's
    bootstrap, V(s_T), as the next value of step T - 2, whose own is
    V(s_{T-1}); that target, and through lambda the ones before it in
    its episode, are not Q(lambda)'s (purejaxql issue #12). Ajax does the
    same.

    All inputs are shaped (n_steps, n_envs, 1); returns the same.
    """
    done = jnp.logical_or(terminateds.astype(bool), truncateds.astype(bool))
    done = done.astype(rewards.dtype)
    bootstraps = (1.0 - done) * next_q_max
    last = rewards[-1] + gamma * bootstraps[-1]
    if rewards.shape[0] == 1:
        return last[None]
    bootstraps = bootstraps.at[-2].set(bootstraps[-1])  # purejaxql's off-by-one

    def step_back(next_return, transition):
        reward, bootstrap, ended = transition
        lambda_return = reward + gamma * (
            (1.0 - q_lambda) * bootstrap + q_lambda * next_return
        )
        lambda_return = (1.0 - ended) * lambda_return + ended * reward
        return lambda_return, lambda_return

    _, returns = jax.lax.scan(
        step_back, last, (rewards[:-1], bootstraps[:-1], done[:-1]), reverse=True
    )
    return jnp.concatenate([returns, last[None]])


def compute_half_mse_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
    """Half the mean-squared TD error, PQN's default: purejaxql's loss
    (mttga/purejaxql @47af6d7 pqn_minatar.py:285), whose gradient its
    global clip of 10 is set against."""
    return 0.5 * jnp.mean((q_taken - target_q) ** 2)


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class PQNState(BaseAgentState):
    """PQN carries a single Q-network -- no replay buffer, no target network.

    The Q-network train state is stored in actor_state so the shared
    evaluation loop (hard-wired to agent_state.actor_state) works
    unchanged; critic_state mirrors it at init and is left inert.
    """


@struct.dataclass
class PolicyAuxiliaries:
    epsilon: jax.Array  # the rate the update's rollout explored at


@struct.dataclass
class AuxiliaryLogs:
    # Nested one level deep so ajax.log.flatten_dict names the metrics
    # value/ then the field's name, as the other agents' auxiliaries are named.
    value: ValueAuxiliaries
    policy: PolicyAuxiliaries
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
PQNConfig = PQN.declare_config(
    (
        "gamma",
        "q_lambda",
        "n_steps",
        "n_epochs",
        "num_minibatches",
        "reward_scale",
        "expose_recent_rollout",
    )
)
