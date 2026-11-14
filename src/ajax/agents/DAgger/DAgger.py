"""DAgger (Ross, Gordon and Bagnell, AISTATS 2011, Alg. 3.1): imitation of
an expert by dataset aggregation, with a supervised loss only::

    D = {}
    for i = 1, 2, ...:
        roll out pi_i = beta_i pi* + (1 - beta_i) pi_hat_i
        D <- D + {(s, pi*(s)) for every state s pi_i visits}
        pi_hat_{i+1} = the policy fitted to D (supervised)

On the replay family's loop (ReplayAgent), at each step every env executes
the expert's action with probability beta_i (beta_schedule), else the
actor's (DAgger.choose_behaviour; the per-env, per-step mixing of
HumanCompatibleAI/imitation@e5ef188 algorithms/dagger.py,
InteractiveTrajectoryCollector). The replay, which keeps every visited
state with the expert's action there, is D. Neither a reward nor a critic
is used.

How the actor fits D is the fit option. IterationFit, the default, is the
reference code's: at the end of each iteration, behaviour cloning over the
whole replay for 4 epochs (DAggerTrainer.extend_and_update), a Gaussian
actor fitted by likelihood with an entropy bonus, which acts by sampling.
OnlineFit is Ajax's earlier fit, an online learner: one gradient step on a
replay batch per step, a deterministic actor regressed onto the expert's
action. The paper's guarantee holds for any no-regret online learner
(Sec. 4).

Deviations: an iteration is iteration_steps steps, every one kept, where
the reference's round also waits for 3 episodes to end
(rollout_round_min_episodes, dagger.py:619, :664-667) and keeps finished
episodes only (its collector saves an episode when it ends,
dagger.py:277-285). Actions are fitted in the agent's [-1, 1] units, as
every Ajax actor acts, where the reference's Gaussian fits the env's own.
The paper returns the best pi_hat_i on validation, Ajax the last one, with
every evaluation logged; the replay drops its oldest rows past buffer_size,
where D keeps them all.
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
from ajax.buffers.utils import (
    as_read,
    as_replay_batch,
    get_batch_from_buffer,
    written_rows,
)
from ajax.environments.interaction import ActionPipelineResult, get_pi
from ajax.environments.utils import flatten_action_column
from ajax.log import unevaluated
from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.networks import get_initialized_actor_critic
from ajax.random_keys import fold_index, split_carry_first
from ajax.state import BaseAgentState
from ajax.types import Architecture

#: The hidden layers' initialiser of the reference's policy, SB3's
#: ActorCriticPolicy with ortho_init (SB3 2.5.0 common/policies.py):
#: orthogonal with gain sqrt(2). Its action head's gain, 0.01, is Actor's
#: default, and its log-std parameter starts at 0.
REFERENCE_HIDDEN_INIT = "orthogonal(np.sqrt(2))"


@dataclass(frozen=True)
class LinearBetaSchedule:
    """beta falls linearly from 1 in the first iteration to 0 after
    rampdown_rounds iterations: the reference code's default,
    LinearBetaSchedule(15) (HumanCompatibleAI/imitation@e5ef188
    algorithms/dagger.py:45-67, :355)."""

    rampdown_rounds: int = 15

    def __post_init__(self) -> None:
        if self.rampdown_rounds < 1:
            raise ValueError(
                f"rampdown_rounds is at least 1, got {self.rampdown_rounds}."
            )

    def __call__(self, iteration: jax.Array) -> jax.Array:
        rounds = jnp.float32(self.rampdown_rounds)
        return jnp.clip((rounds - iteration) / rounds, 0.0, 1.0)


@dataclass(frozen=True)
class ExponentialBetaSchedule:
    """beta_i = decay ** (i - 1), as the reference's ExponentialBetaSchedule
    (dagger.py:70-91); decay 0 is the paper's parameter-free beta_i = I(i = 1)
    (its Sec. 3), which the reference's range (0, 1] leaves out."""

    decay: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"The beta decay is in [0, 1], got {self.decay}.")

    def __call__(self, iteration: jax.Array) -> jax.Array:
        return jnp.power(jnp.float32(self.decay), iteration)  # 0 ** 0 = 1


@dataclass(frozen=True)
class IterationFit:
    """The reference code's fit: behaviour cloning on the whole replay at
    the end of each iteration (DAggerTrainer.extend_and_update,
    HumanCompatibleAI/imitation@e5ef188 algorithms/dagger.py:455-497).
    Each fit makes epochs passes over every row the replay holds
    (DEFAULT_N_EPOCHS, :326), each in a fresh random order, one step per
    minibatch of the agent's batch_size, a short last one dropped (its
    DataLoader's shuffle and drop_last, :445-451). The optimiser's state
    carries over from one fit to the next, as the reference's BC keeps one
    optimiser.

    The actor is the reference's policy, SB3's ActorCriticPolicy (bc.py:347):
    a Gaussian with one learned log-std per action dimension, or a
    categorical for discrete actions. It is fitted by its negative
    log-likelihood of the expert's action minus entropy_weight times its
    entropy (BehaviorCloningLossCalculator, bc.py:130-146; ent_weight 1e-3,
    :286; l2_weight 0, :287, so no L2 term). It acts by sampling, as
    policy.predict does by default (dagger.py:511), and is evaluated at its
    mean or most likely action, as every Ajax actor."""

    epochs: int = 4
    entropy_weight: float = 1e-3

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"IterationFit takes at least 1 epoch, got {self.epochs}.")


@dataclass(frozen=True)
class OnlineFit:
    """Ajax's fit before the reference's became the default: one gradient
    step on a replay batch per step once the replay holds a batch. The actor
    is deterministic (TD3's, tanh-squashed), fitted by the squared error to
    the expert's action, or categorical for discrete actions, fitted by
    cross-entropy; it acts with its mean or most likely action."""


@dataclass(kw_only=True, eq=False, repr=False)
class DAgger(GradientClipping, ObservationNormalization, ExpertGuided, ReplayAgent):
    """DAgger (Ross, Gordon and Bagnell, AISTATS 2011): imitation of an
    expert by dataset aggregation (the module's program).

    At iteration i the expert acts with probability beta_i (beta_schedule)
    at each step, the actor otherwise; every visited state joins
    the replay with the expert's action there, and the actor fits those
    actions (fit). It needs no reward and has no critic. The expert follows
    the conventions of the expert guide: it may be stateful; a discrete
    expert returns its action index as an integer (n, 1) column.

    Defaults are the reference code's, HumanCompatibleAI/imitation@e5ef188:
    beta falling linearly to 0 over 15 iterations (LinearBetaSchedule; the
    paper prefers beta_i = I(i = 1), ExponentialBetaSchedule(0.0), as
    "often performs best in practice", Sec. 3; the code that made the
    reference's numbers ramps); 500-step iterations
    (rollout_round_min_timesteps), each ending with 4 epochs of behaviour
    cloning over the whole replay (IterationFit); its behaviour cloning's
    batch of 32, 32-32 tanh network and torch Adam (learning rate 1e-3, eps
    1e-8).
    """

    #: The expert DAgger imitates (ExpertGuided's, required here).
    expert_policy: Callable
    #: beta_i, the expert's share of iteration i's actions.
    beta_schedule: Union[LinearBetaSchedule, ExponentialBetaSchedule] = (
        dataclasses.field(default_factory=LinearBetaSchedule)
    )
    #: How the actor fits the replay: the reference's IterationFit, or
    #: OnlineFit.
    fit: Union[IterationFit, OnlineFit] = dataclasses.field(
        default_factory=IterationFit
    )
    #: The timesteps of one iteration i, over all envs.
    iteration_steps: int = 500
    actor_learning_rate: float = 1e-3
    actor_architecture: Architecture = ("32", "tanh", "32", "tanh")
    buffer_size: int = int(1e6)
    batch_size: int = 32
    #: Not a keyword: the first update waits for a batch (setup).
    learning_starts: int = dataclasses.field(init=False)
    #: Not a keyword: an IterationFit fits once per iteration (setup).
    train_frequency: int = dataclasses.field(init=False)

    def check(self) -> None:
        if self.expert_policy is None:
            raise ValueError("DAgger imitates an expert: give it an expert_policy.")

    def actor_critic_args(self) -> dict[str, Any]:
        # The critic built beside the actor is never stepped.
        arguments: dict[str, Any] = {
            "critic_learning_rate": self.actor_learning_rate,
            "critic_architecture": self.actor_architecture,
        }
        if isinstance(self.fit, IterationFit):
            # SB3's ActorCriticPolicy, the reference's: its log-std is one
            # parameter per action dimension, starting at 0, never clipped.
            arguments.update(
                {
                    "encoder_kernel_init": REFERENCE_HIDDEN_INIT,
                    "log_std_state_independent": True,
                    "log_std_init": 0.0,
                    "log_std_bounds": None,
                }
            )
        return arguments

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # torch's Adam eps, the reference's

    def setup(self) -> None:
        # The replay's first batch, as the reference fits once it holds one.
        self.learning_starts = self.batch_size
        # An IterationFit's update runs on the step that ends an iteration.
        if isinstance(self.fit, IterationFit):
            self.train_frequency = self.iteration_steps
        else:
            self.train_frequency = 1

    def compute_beta(self, timestep: Union[int, jax.Array]) -> jax.Array:
        """beta_i at iteration i = timestep // iteration_steps + 1, the
        schedules counting iterations from 0 as the reference's rounds."""
        iteration = jnp.asarray(timestep // self.iteration_steps, jnp.float32)
        return self.beta_schedule(iteration)

    def choose_behaviour(self, context: TrainContext) -> Callable:
        """Per env, the expert's action with probability beta_i, else the
        actor's (sampled under an IterationFit, its deterministic one under
        an OnlineFit); the expert's action at every state is stored
        (a_expert): the one collection computes before the choice, from the
        expert's state along the trajectory (expert_collect_kwargs)."""
        continuous, n_envs = context.env_args.continuous, context.env_args.n_envs
        samples = isinstance(self.fit, IterationFit)

        def pipeline(
            agent_state, acting, raw_obs, collector_key, uniform, mix_key, action_key
        ):
            del raw_obs, collector_key, uniform
            actor = agent_state.actor_state
            policy = get_pi(actor, actor.params, acting.obs)[0]
            if samples:
                own = sample_learner_action(policy, continuous, action_key)
            else:
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
        """A Gaussian actor (an IterationFit's) or a deterministic one (TD3's,
        tanh-squashed, an OnlineFit's) for continuous actions, a categorical
        one for discrete; the critic built beside it is unused."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        continuous = context.env_args.continuous
        if continuous and isinstance(self.fit, OnlineFit):
            actor_state, critic_state = context.make_actor_critic(
                init_key, get_initialized_td3_actor_critic, num_critics=1
            )
        else:
            actor_state, critic_state = context.make_actor_critic(
                init_key, get_initialized_actor_critic, continuous=continuous
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
        """The fit: an IterationFit's passes over the whole replay at the end
        of an iteration, or an OnlineFit's one step on a replay batch."""
        collector = agent_state.collector_state
        if isinstance(self.fit, IterationFit):
            agent_state, policy_aux, metrics = self.fit_replay(
                agent_state, context, self.fit
            )
        else:
            agent_state, policy_aux, metrics = self.fit_batch(agent_state, context)
        agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
        return agent_state, AuxiliaryLogs(
            policy=policy_aux,
            dagger=DAggerAuxiliaries(
                beta=self.compute_beta(collector.timestep),
                expert_frac=collector.last_expert_frac,
            ),
            extensions=metrics,
        )

    def fit_batch(
        self, agent_state: DAggerState, context: TrainContext
    ) -> tuple[DAggerState, PolicyAuxiliaries, dict]:
        """An OnlineFit's supervised step towards the expert's actions on a
        replay batch."""
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
        policy_aux = PolicyAuxiliaries(imitation_loss=loss, grad_norm=grad_norm)
        return agent_state.replace(actor_state=actor_state), policy_aux, metrics

    def fit_replay(
        self, agent_state: DAggerState, context: TrainContext, fit: IterationFit
    ) -> tuple[DAggerState, PolicyAuxiliaries, dict]:
        """An IterationFit's behaviour cloning on every row the replay holds:
        fit.epochs passes, each over the rows in a fresh random order, one
        step per minibatch of batch_size, a short last one dropped. Logs the
        mean loss over its steps, and the last step's gradient norm and
        extension metrics (NaN while the replay holds less than a batch).

        Its loops carry the actor alone, never the replay they read: under
        the seed vmap each seed's count of minibatches drives a loop that
        selects every carried value at every step."""
        fit_key, agent_state = agent_state.get_keys_and_new_state()
        buffer_state = agent_state.collector_state.buffer_state
        assert buffer_state is not None  # a replay agent's collector keeps one
        n_envs, ring = jax.tree.leaves(buffer_state.experience)[0].shape[:2]
        slots = n_envs * ring
        # Slot e * ring + r holds env e's row r; each env holds its rows
        # before the write position, or its whole ring once it has wrapped.
        rows = jax.tree.map(
            lambda values: values.reshape((slots, *values.shape[2:])),
            buffer_state.experience,
        )
        held = written_rows(buffer_state)
        row_of_slot, env_of_slot = jnp.arange(slots) % ring, jnp.arange(slots) // ring
        stored = row_of_slot < held
        # A row's draw is keyed by the row (its env and its index in that
        # env's ring), not by its slot: the slots' count is the ring's
        # length, which a replay sized to its call sets, so a resumed run's
        # shorter first-leg ring would otherwise shuffle the same rows into
        # another order than one call does.
        row_identity = row_of_slot * n_envs + env_of_slot
        n_batches = n_envs * held // self.batch_size
        continuous = context.env_args.continuous

        def cloning_step(actor_state: Any, picked: jax.Array, key: jax.Array) -> tuple:
            picked_rows = as_read(jax.tree.map(lambda values: values[picked], rows))
            batch = as_replay_batch(picked_rows)
            labels = picked_rows["a_expert"]  # the expert's action at each row
            stepping = agent_state.replace(actor_state=actor_state)

            def compute_cloning_loss(params: Any) -> tuple[jax.Array, jax.Array, Any]:
                policy, _ = get_pi(actor_state, params, batch.obs)
                if continuous:
                    # Normal is per dimension; the Gaussian's are their sums.
                    log_likelihood = policy.log_prob(labels).sum(-1)
                    entropy = policy.entropy().sum(-1)
                    policy_mean = policy.mean()
                else:
                    log_likelihood = policy.log_prob(labels[..., 0])
                    entropy = policy.entropy()
                    policy_mean = None
                bonus = fit.entropy_weight * jnp.mean(entropy)
                loss = -jnp.mean(log_likelihood) - bonus
                return loss, loss, policy_mean

            actor_state, loss, grad_norm, metrics = actor_step(
                stepping,
                compute_cloning_loss,
                context.extensions,
                key,
                context.total_timesteps,
                batch,
            )
            return actor_state, loss, (grad_norm, metrics)

        def run_epoch(epoch: jax.Array, carry: tuple) -> tuple:
            shuffle_key, batches_key = split_carry_first(fold_index(fit_key, epoch), 1)
            draws = jax.vmap(
                lambda identity: jax.random.uniform(fold_index(shuffle_key, identity))
            )(row_identity)
            # The stored rows first, in the order of their draws: a uniform
            # shuffle of exactly those rows.
            order = jnp.argsort(jnp.where(stored, draws, jnp.inf))

            def run_batch(index: jax.Array, carry: tuple) -> tuple:
                actor_state, loss_sum, _ = carry
                start = index * self.batch_size
                picked = jax.lax.dynamic_slice_in_dim(order, start, self.batch_size)
                key = fold_index(batches_key, index)
                actor_state, loss, last = cloning_step(actor_state, picked, key)
                return actor_state, loss_sum + loss, last

            return jax.lax.fori_loop(0, n_batches, run_batch, carry)

        no_step = jax.eval_shape(
            lambda actor_state: cloning_step(
                actor_state, jnp.zeros(self.batch_size, jnp.int32), fit_key
            )[2],
            agent_state.actor_state,
        )
        initial = (agent_state.actor_state, jnp.zeros(()), unevaluated(no_step))
        actor_state, loss_sum, (grad_norm, metrics) = jax.lax.fori_loop(
            0, fit.epochs, run_epoch, initial
        )
        steps = fit.epochs * n_batches
        mean_loss = jnp.where(steps > 0, loss_sum / jnp.maximum(steps, 1), jnp.nan)
        policy_aux = PolicyAuxiliaries(imitation_loss=mean_loss, grad_norm=grad_norm)
        return agent_state.replace(actor_state=actor_state), policy_aux, metrics

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
    pretrain_context = frozenset({"env_args", "seed_index"})
    stores_expert_actions = True


def choose_learner_action(policy: Any, continuous: bool) -> jax.Array:
    """The actor's deterministic action: the squashed mean, or the most
    likely action as an int (n, 1) column, as a discrete replay keeps it."""
    if continuous:
        return policy.mean()
    return policy.mode()[..., None].astype(jnp.int32)


def sample_learner_action(policy: Any, continuous: bool, key: jax.Array) -> jax.Array:
    """The actor's sampled action: a Gaussian sample, which the env clips to
    its bounds as the reference's policy.predict does, or a sampled index
    as an int (n, 1) column."""
    action = policy.sample(seed=key)
    if continuous:
        return action
    return action[..., None].astype(jnp.int32)


# -- State and logged metrics -------------------------------------------------
@struct.dataclass(kw_only=True)
class DAggerState(BaseAgentState):
    """DAgger carries its actor and collector; its critic is unused."""


@struct.dataclass
class PolicyAuxiliaries:
    imitation_loss: jax.Array  # an IterationFit's: the mean over its steps
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
