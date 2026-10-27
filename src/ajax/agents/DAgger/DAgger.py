"""DAgger (Ross, Gordon and Bagnell, AISTATS 2011, Alg. 3.1): imitation of
an expert by dataset aggregation, with a supervised loss only::

    D = {}
    for i = 1, 2, ...:
        roll out pi_i = beta_i pi* + (1 - beta_i) pi_hat_i
        D <- D + {(s, pi*(s)) for every state s pi_i visits}
        pi_hat_{i+1} = the policy fitted to D (supervised)

On the replay family's loop (ReplayAgent), at each step every env executes
the expert's action with probability beta_i (compute_beta), else the
actor's deterministic one (DAgger.choose_behaviour; the per-env, per-step
mixing of HumanCompatibleAI/imitation@e5ef188 algorithms/dagger.py,
InteractiveTrajectoryCollector). The replay, which keeps every visited
state with the expert's action there, is D. The fit is online, one gradient
step on a replay batch per tick once the replay holds a batch, rather than
a fit to convergence after each iteration: the paper's guarantee holds for
any no-regret online learner (Sec. 4). The loss is the squared error to the
expert's action for continuous actions and its cross-entropy for discrete
ones (DAgger.update). Neither a reward nor a critic is used.

Deviations: the paper returns the best pi_hat_i on validation, Ajax the last
one, with every evaluation logged; the replay drops its oldest rows past
buffer_size, where D keeps them all.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    ReplayAgent,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.update_steps import ACTOR_STEP_KEYS, actor_step
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import ActionPipelineResult, get_pi
from ajax.environments.utils import flatten_action_column
from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import BaseAgentState
from ajax.types import Architecture


@dataclass(kw_only=True, eq=False, repr=False)
class DAgger(GradientClipping, ObservationNormalization, ExpertGuided, ReplayAgent):
    """DAgger (Ross, Gordon and Bagnell, AISTATS 2011): imitation of an
    expert by dataset aggregation (the module's program).

    At iteration i the expert acts with probability beta_i = beta_decay **
    (i - 1) at each step, the actor otherwise; every visited state joins
    the replay with the expert's action there, and the actor fits those
    actions. It needs no reward and has no critic. The expert follows the
    conventions of the expert guide: it may be stateful; a discrete expert
    returns its action index as an integer (n, 1) column.

    Defaults: the paper's parameter-free beta_i = I(i = 1) (beta_decay=0,
    its Sec. 3: "often performs best in practice"); the rest is the
    reference code's, HumanCompatibleAI/imitation@e5ef188: 500-step
    iterations (rollout_round_min_timesteps) and its behaviour cloning's
    batch of 32, 32-32 tanh network and torch Adam (learning rate 1e-3, eps
    1e-8). Its cloning fits a Gaussian by likelihood with an entropy bonus;
    this actor is deterministic, as the paper's policy, and fits by squared
    error (cross-entropy for discrete actions).
    """

    #: The expert DAgger imitates (ExpertGuided's, required here).
    expert_policy: Callable
    #: beta_i = beta_decay ** (i - 1): 0 is the paper's beta_i = I(i = 1).
    beta_decay: float = 0.0
    #: The timesteps of one iteration i, over all envs.
    iteration_steps: int = 500
    actor_learning_rate: float = 1e-3
    actor_architecture: Architecture = ("32", "tanh", "32", "tanh")
    buffer_size: int = int(1e6)
    batch_size: int = 32
    #: Not a keyword: the first update waits for a batch (setup).
    learning_starts: int = dataclasses.field(init=False)

    def check(self) -> None:
        if not 0.0 <= self.beta_decay <= 1.0:
            raise ValueError(
                f"DAgger's beta_decay is in [0, 1], got {self.beta_decay}."
            )
        if self.expert_policy is None:
            raise ValueError("DAgger imitates an expert: give it an expert_policy.")

    def actor_critic_args(self) -> dict[str, Any]:
        # The critic built beside the actor is never stepped.
        return {
            "critic_learning_rate": self.actor_learning_rate,
            "critic_architecture": self.actor_architecture,
        }

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # torch's Adam eps, the reference's

    def setup(self) -> None:
        # The replay's first batch, as the reference fits once it holds one.
        self.learning_starts = self.batch_size

    def compute_beta(self, timestep: Union[int, jax.Array]) -> jax.Array:
        """beta_i = beta_decay ** (i - 1) at iteration i = timestep //
        iteration_steps + 1; 0 ** 0 = 1."""
        iteration = jnp.asarray(timestep // self.iteration_steps, jnp.float32)
        return jnp.power(jnp.float32(self.beta_decay), iteration)

    def choose_behaviour(self, context: TrainContext) -> Callable:
        """Per env, the expert's action with probability beta_i, else the
        actor's; the expert's action at every state is stored (a_expert):
        the one collection computes before the choice, from the expert's
        state along the trajectory (expert_collect_kwargs)."""
        continuous, n_envs = context.env_args.continuous, context.env_args.n_envs

        def pipeline(
            agent_state, acting, raw_obs, collector_key, uniform, mix_key, action_key
        ):
            del raw_obs, collector_key, uniform, action_key
            actor = agent_state.actor_state
            policy = get_pi(actor, actor.params, acting.obs)[0]
            own = choose_learner_action(policy, continuous)
            # The label in the replay's action layout (a discrete index's column).
            a_expert = acting.expert_action.reshape(own.shape).astype(own.dtype)
            draw = jax.random.uniform(mix_key, (n_envs, 1))
            use_expert = draw < self.compute_beta(acting.timestep)
            action = jnp.where(use_expert, a_expert, own)
            zeros = jnp.zeros((n_envs, 1))
            return ActionPipelineResult(
                env_action=flatten_action_column(action, continuous),
                policy_action=own,
                log_probs=zeros,
                is_expert_flag=use_expert.astype(jnp.float32),
                in_value_box=zeros,
                entry_bonus=zeros,
                buffer_action=action,
                a_expert=a_expert,
            )

        return pipeline

    def init(self, key: jax.Array, context: TrainContext) -> DAggerState:
        """A deterministic actor (TD3's, tanh-squashed) for continuous
        actions, a categorical one for discrete; the critic built beside it
        is unused."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        if context.env_args.continuous:
            actor_state, critic_state = context.make_actor_critic(
                init_key, get_initialized_td3_actor_critic, num_critics=1
            )
        else:
            actor_state, critic_state = context.make_actor_critic(
                init_key, get_initialized_actor_critic, continuous=False
            )
        return DAggerState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=context.make_collector(collector_key),
        )

    def update(
        self, agent_state: DAggerState, context: TrainContext
    ) -> tuple[DAggerState, AuxiliaryLogs]:
        """One supervised step towards the expert's actions on a replay batch."""
        sample_key, actor_key, agent_state = agent_state.get_keys_and_new_state(2)
        collector = agent_state.collector_state
        batch = get_batch_from_buffer(self.buffer, collector.buffer_state, sample_key)
        continuous = context.env_args.continuous

        def compute_imitation_loss(params: Any) -> tuple[jax.Array, jax.Array, Any]:
            policy, _ = get_pi(agent_state.actor_state, params, batch.obs)
            if continuous:
                policy_mean = policy.mean()
                loss = jnp.mean(jnp.square(policy_mean - batch.a_expert))
            else:
                policy_mean = None
                loss = -jnp.mean(policy.log_prob(batch.a_expert[..., 0]))
            return loss, loss, policy_mean

        actor_state, loss, grad_norm, metrics = actor_step(
            agent_state,
            compute_imitation_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            batch,
        )
        agent_state = agent_state.replace(
            actor_state=actor_state, n_updates=agent_state.n_updates + 1
        )
        return agent_state, AuxiliaryLogs(
            policy=PolicyAuxiliaries(imitation_loss=loss, grad_norm=grad_norm),
            dagger=DAggerAuxiliaries(
                beta=self.compute_beta(collector.timestep),
                expert_frac=collector.last_expert_frac,
            ),
            extensions=metrics,
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "DAgger"
    action_space = "any"
    capabilities = frozenset({"actor", "obs_networks", "off_policy", "replay"})
    supported_extension_phases = LOOP_PHASES.union({"actor_loss"})
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "actor_loss": ACTOR_STEP_KEYS.union(
            {"observations", "raw_observations", "a_expert"}
        ),
    }
    # Its replay is not handed to pretraining: it stores the expert's next
    # action, which a transition dataset does not hold (PriorData).
    pretrain_context = frozenset({"env_args"})
    stores_expert_actions = True


def choose_learner_action(policy: Any, continuous: bool) -> jax.Array:
    """The actor's deterministic action: the squashed mean, or the most
    likely action as an int (n, 1) column, as a discrete replay keeps it."""
    if continuous:
        return policy.mean()
    return policy.mode()[..., None].astype(jnp.int32)


# -- State and logged metrics -------------------------------------------------
@struct.dataclass(kw_only=True)
class DAggerState(BaseAgentState):
    """DAgger carries its actor and collector; its critic is unused."""


@struct.dataclass
class PolicyAuxiliaries:
    imitation_loss: jax.Array
    grad_norm: jax.Array | float = jnp.nan  # before clipping


@struct.dataclass
class DAggerAuxiliaries:
    beta: jax.Array  # the iteration's beta_i
    expert_frac: float | jax.Array  # the share of envs the expert drove last step


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    dagger: DAggerAuxiliaries
    extensions: dict  # the extensions' training metrics
