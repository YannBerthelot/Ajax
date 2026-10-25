"""Truncated Quantile Critics (Kuznetsov, Shvechikov, Grishin and Vetrov,
ICML 2020), for continuous actions: SAC with distributional critics whose
largest pooled target quantiles are dropped.

One update on a replay batch, in the order of the authors' code
(bayesgroup/tqc_pytorch, Software Heritage
swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16, tqc/trainer.py), each
line one method below or one of the equations under it::

    a' ~ pi(. | s'), log pi(a' | s')                  sample_next_actions (SAC's)
    every target critic's quantiles at (s', a'), pooled and sorted, without
    the top_quantiles_to_drop_per_net * num_critics largest  truncated_quantiles
    y = scale r + gamma (1 - d) (z - alpha log pi(a' | s')) for each kept z
                                                         compute_tqc_td_target
    Z_i <- argmin quantile Huber loss of every critic's quantiles to y
                                                    update_critic, quantile_huber_loss
    Z_target <- tau Z + (1 - tau) Z_target
    actor <- argmin E[alpha log pi(a | s) - mean of every quantile at (s, a)]
                                       update_actor, compute_soft_policy_loss (SAC's)
    alpha <- argmin log alpha sg(-log pi(a | s) - H_target)
                                                 compute_temperature_loss (SAC's)

The actor reads the critics from before their step (update). The policy acts
from the first step, with no uniform warm-up, as the authors' code does.
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
    BELLMAN_CONTEXT,
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
    compute_soft_bellman_target,
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
from ajax.networks.utils import TORCH_LINEAR
from ajax.state import LoadedTrainState, Transition
from ajax.types import Architecture


@dataclass(kw_only=True, eq=False, repr=False)
class TQC(
    EntropyTemperature,
    PolicyUpdateStart,
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    RecurrentReplayAgent,
):
    """TQC: SAC with num_critics distributional critics, each predicting
    n_quantiles quantiles of the return, whose largest pooled target
    quantiles are dropped (the module's update).

    Defaults are the authors' code's (main.py, tqc/): one Adam at 3e-4 for
    each of the actor, the critics and the temperature, with torch's eps
    1e-8 (trainer.py:25-27, torch 1.3.0); critics of three 512-unit layers
    and an actor of two of 256 (structures.py:94, :108); five critics of 25
    quantiles, two dropped per critic (main.py:92-94); a target entropy of
    -1 per action dimension (main.py:45); training from the 257th step, the
    policy acting from the first (main.py:65).

    Args:
        num_critics: The distributional critics (n_nets).
        n_quantiles: The quantiles each critic predicts.
        top_quantiles_to_drop_per_net: The pooled target quantiles dropped
            per critic. The paper tunes it per task (Appendix B, Table 6: 5
            on Hopper, 0 on HalfCheetah, 2 on Walker2d, Ant and Humanoid).
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("256", "relu", "256", "relu")
    critic_architecture: Architecture = ("512", "relu", "512", "relu", "512", "relu")
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    num_critics: int = 5
    n_quantiles: int = 25
    top_quantiles_to_drop_per_net: int = 2
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = 257

    def check(self) -> None:
        n_drop = self.top_quantiles_to_drop_per_net * self.num_critics
        if (
            self.n_quantiles < 1
            or not 0 <= n_drop < self.num_critics * self.n_quantiles
        ):
            raise ValueError(
                "TQC keeps some of its num_critics * n_quantiles pooled target"
                " quantiles: need n_quantiles >= 1 and 0 <="
                " top_quantiles_to_drop_per_net < n_quantiles, got"
                f" n_quantiles={self.n_quantiles},"
                f" top_quantiles_to_drop_per_net={self.top_quantiles_to_drop_per_net}."
            )

    def actor_critic_args(self) -> dict[str, Any]:
        # The authors' layers are torch's nn.Linear at its default
        # initialisation, biases included (structures.py:36-58, "TODO:
        # initialization"). Recurrent TQC (memory set) keeps Ajax's, as
        # recurrent SAC, REDQ and ASAC do until Ni et al.'s is audited.
        args: dict[str, Any] = {"squash": True}
        if self.memory is not None:
            return args
        init = ("encoder_kernel_init", "mean_kernel_init", "log_std_kernel_init")
        args.update(dict.fromkeys((*init, "value_kernel_init"), TORCH_LINEAR))
        args.update({"log_std_init": None})
        return args

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # torch's Adam (trainer.py:25-27, torch 1.3.0)

    def network_settings(self) -> dict[str, Any]:
        # The critics' quantile heads; the tanh log-det is exact, as the
        # authors' (structures.py:142), Squash's default.
        return {"critic_quantiles": self.n_quantiles}

    def init(self, key: jax.Array, context: TrainContext) -> TQCState:
        """A squashed Gaussian actor, num_critics quantile critics with their
        targets, and the temperature."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=True,
            action_value=True,
            squash=True,
            num_critics=self.num_critics,
        )
        collector_state = context.make_collector(collector_key)
        return TQCState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha=create_alpha_train_state(**to_state_dict(self.alpha_args)),
            collector_state=collector_state,
        )

    def update(
        self, agent_state: TQCState, context: TrainContext
    ) -> tuple[TQCState, AuxiliaryLogs]:
        """One update on a replay batch: the critics' step and the target
        update, then the actor's and the temperature's steps, each kept from
        its start.

        The actor step reads the critics from before their step, as the
        authors' code builds its actor loss (trainer.py:59-61) before the
        critics' optimizer steps (:64-66). It differentiates that loss after
        them; the torch it pins (1.3.0) steps parameters in place through
        .data, which autograd does not track, so that backward pass would
        read the stepped critics' weights (from memory, unverified). Ajax
        differentiates the loss it computed, at the critics before their
        step: the order is logged as an upstream bug, which Ajax does not
        follow (owner, 2026-10-23), and the conformance check runs their
        update with the actor's backward before the critics' step.
        """
        sample_key, agent_state = agent_state.get_keys_and_new_state()
        batch, carries = sample_replay(
            agent_state,
            context.buffer,
            sample_key,
            context.recurrent,
            self.burn_in,
            self.stored_state,
        )
        critics_before = agent_state.critic_state.params
        agent_state, value_aux, value_metrics = self.update_critic(
            agent_state, batch, carries, context
        )
        agent_state = update_steps.update_target_networks(agent_state, self.tau)
        critic_state = agent_state.critic_state
        timestep = agent_state.collector_state.timestep
        stepped, policy_aux, log_probs, policy_metrics = self.update_actor(
            agent_state.replace(
                critic_state=critic_state.replace(params=critics_before)
            ),
            batch,
            carries,
            context,
        )
        stepped = stepped.replace(critic_state=critic_state)
        # The actor step runs from the first update, since its log-probs feed
        # the temperature's, and is kept from policy_update_start.
        agent_state, (policy_aux, policy_metrics) = update_steps.run_actor_if(
            self.policy_update_due(timestep),
            lambda _: (stepped, (policy_aux, policy_metrics)),
            agent_state,
        )
        tempered, temperature_aux = update_temperature(
            agent_state,
            log_probs,
            self.target_entropy_at(timestep),
            compute_temperature_loss,
        )
        agent_state = update_steps.run_if(
            self.temperature_update_due(timestep), lambda _: tempered, agent_state
        )
        return agent_state, AuxiliaryLogs(
            temperature=temperature_aux,
            policy=policy_aux,
            value=value_aux,
            extensions=merge_metrics(value_metrics, policy_metrics),
        )

    def update_critic(
        self,
        agent_state: TQCState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[TQCState, ValueAuxiliaries, dict]:
        """The quantile critics' step on the truncated target."""
        critic_key, agent_state = agent_state.get_keys_and_new_state()
        # A truncated row bootstraps on its final observation (bootstrap_obs).
        dones = batch.terminated
        target_q, next_log_probs = compute_tqc_td_target(
            agent_state.actor_state,
            agent_state.critic_state,
            critic_key,
            batch.next_obs,
            dones,
            batch.reward,
            self.gamma,
            get_alpha(agent_state),
            self.top_quantiles_to_drop_per_net * self.num_critics,
            self.reward_scale,
            carries,
            context.executed,
            batch.next_a_expert,
        )

        def critic_loss(
            params: FrozenDict, target_q: jax.Array
        ) -> tuple[jax.Array, Any]:
            quantiles = q_values(
                agent_state.critic_state,
                params,
                batch.obs,
                batch.action,
                carries,
                quantiles=True,
            )
            loss = quantile_huber_loss(quantiles, target_q)
            spread = jnp.max(target_q, axis=-1) - jnp.min(target_q, axis=-1)
            return loss, ValueAuxiliaries(
                critic_loss=loss,
                target_q=target_q.mean().flatten(),
                log_probs=next_log_probs.mean().flatten(),
                target_spread=spread.mean().flatten(),
            )

        critic_state, aux, _, metrics = update_steps.critic_step(
            agent_state,
            batch,
            target_q,
            critic_loss,
            context.extensions,
            critic_key,
            context.total_timesteps,
            rewards=batch.reward,
            dones=dones,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
        )
        return agent_state.replace(critic_state=critic_state), aux, metrics

    def update_actor(
        self,
        agent_state: TQCState,
        batch: Transition,
        carries: Optional[RecurrentCarries],
        context: TrainContext,
    ) -> tuple[TQCState, PolicyAuxiliaries, jax.Array, dict]:
        """The actor step on alpha log pi(a|s) - mean Z(s, a), the mean over
        every critic's quantiles (trainer.py:61), the critics scoring a
        residual policy's executed actions; also returns its sample's
        log-probs, which the temperature step reads."""
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
                jnp.mean,
                transform,
            )
            return loss, (loss, log_probs, q_values), policy.mean()

        actor_state, (loss, log_probs, q_mean), _, metrics = update_steps.actor_step(
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
            policy_loss=loss, log_pi=log_probs.mean(), q_mean=q_mean.mean()
        )
        return agent_state.replace(actor_state=actor_state), aux, log_probs, metrics

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "TQC"
    # No warmup: TQC acts with its policy from its first step, as its
    # authors' code does (DeclaredAgent.uniform_steps), so no warm-up chooser.
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "distributional_critic",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
        }
    )
    supported_extension_phases = (
        RecurrentReplayAgent.supported_extension_phases.difference(
            {"choose_warmup_action"}
        )
    )
    extension_context = RecurrentReplayAgent.extension_context.union(BELLMAN_CONTEXT)
    stores_expert_actions = True
    supported_buffer_fields = frozenset({"raw_next_obs", "return_to_go"})


# -- TQC's equations, in the order update calls them -------------------------
def compute_tqc_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    n_drop: int,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """r + gamma (1 - d)(z - alpha log pi(a'|s')) for every kept target
    quantile z at a' ~ pi(.|s') (trainer.py:45-53), executed (executed, a
    residual policy's). Returns the held target atoms and the next actions'
    log-probs."""
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, rng, carries, executed, next_a_expert
    )
    next_quantiles = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
        quantiles=True,
    )
    target = compute_soft_bellman_target(
        truncated_quantiles(next_quantiles, n_drop),
        log_probs,
        rewards,
        dones,
        gamma,
        alpha,
        reward_scale,
    )
    return jax.lax.stop_gradient(target), log_probs


def truncated_quantiles(quantiles: jax.Array, n_drop: int) -> jax.Array:
    """Every critic's quantiles (the ensemble axis first, the quantiles
    last) pooled per row and sorted, without the n_drop largest
    (trainer.py:48-50)."""
    pooled = jnp.moveaxis(quantiles, 0, -2)
    pooled = pooled.reshape(pooled.shape[:-2] + (-1,))
    return jnp.sort(pooled, axis=-1)[..., : pooled.shape[-1] - n_drop]


def quantile_huber_loss(quantiles: jax.Array, targets: jax.Array) -> jax.Array:
    """Every critic's quantiles (num_critics, ..., M) against every target
    atom (..., K): the quantile Huber loss, threshold 1, at the quantile
    fractions (2m + 1) / 2M, averaged over critics, rows, quantiles and atoms
    (functions.py:22-32)."""
    delta = targets[None, ..., None, :] - quantiles[..., :, None]
    abs_delta = jnp.abs(delta)
    huber = jnp.where(abs_delta > 1.0, abs_delta - 0.5, 0.5 * delta**2)
    quantile_count = quantiles.shape[-1]
    tau = (
        jnp.arange(quantile_count, dtype=jnp.float32) / quantile_count
        + 1 / 2 / quantile_count
    )
    return jnp.mean(jnp.abs(tau[:, None] - (delta < 0)) * huber)


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class TQCState(SoftActorCriticState):
    """TQC's state: a soft actor-critic's, nothing more."""


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_mean: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array  # the kept target quantiles' mean
    log_probs: jax.Array
    # The kept target quantiles' spread, max - min per row: the return
    # distribution's width the critics learn.
    target_spread: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics
