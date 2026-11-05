"""Contextualized Hybrid Ensemble Q-learning (Cramer, Frauenknecht, Sabirov
and Trimpe, RLC 2024; Data-Science-in-Mechanical-Engineering/cheq@adedba2):
the executed action blends the expert's and the RL agent's, with a weight that
falls as the critics disagree, and the actor and the critics read the weight::

    a_mix = (1 - lambda_t) a_prior + lambda_t a_RL                       eq. 2
    u_t = std_e Q_e(s_t, a_RL, lambda_t)                                 eq. 4
    lambda_{t+1} = lambda_min + clip((u_t - u_max) / (u_min - u_max), 0, 1)
                   * (lambda_max - lambda_min)                           eq. 5

:func:`blend`, :func:`disagreement` and :func:`adapted_weight` are those three.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from flax import struct

from ajax.controller import ObserveInputs, observation_layout
from ajax.environments.interaction import get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.observe import ExpertStateInput, PIDTermsInput
from ajax.networks.networks import action_value_input, predict_value

#: The warm-up's weights, an even spread over its band without a key.
GOLDEN_RATIO_FRACTION = 0.6180339887


def blend(prior: jax.Array, rl_action: jax.Array, weight: jax.Array) -> jax.Array:
    """Eq. 2: the regularised mix, the RL action in full at weight 1."""
    return (1.0 - weight) * prior + weight * rl_action


def disagreement(
    critic_state: Any, observations: jax.Array, actions: jax.Array
) -> jax.Array:
    """Eq. 4: the critics' standard deviation at (s, a), one per row."""
    x = action_value_input(observations, actions)
    return jnp.std(predict_value(critic_state, critic_state.params, x), axis=0)


def adapted_weight(
    uncertainty: jax.Array,
    u_low: float,
    u_high: float,
    lambda_min: float,
    lambda_max: float,
) -> jax.Array:
    """Eq. 5: lambda_max below u_low, lambda_min above u_high, linear between."""
    falling = (uncertainty - u_high) / (u_low - u_high)
    return lambda_min + jnp.clip(falling, 0.0, 1.0) * (lambda_max - lambda_min)


def track(tracked: jax.Array, value: jax.Array, rate: float) -> jax.Array:
    """tracked, moved by rate towards value: a running average."""
    return tracked + rate * (value - tracked)


def interquartile_range(values: jax.Array) -> jax.Array:
    """The spread of the middle half of values, which one trip's cost among
    a batch's rows leaves as it is, as it would not a standard deviation."""
    low, high = jnp.quantile(values, jnp.array([0.25, 0.75]))
    return high - low


def row_masks(key: jax.Array, rows: jax.Array, critics: int, rate: float) -> jax.Array:
    """Per critic and row, whether the critic trains on the row, (critics, B,
    1): Bernoulli(rate), drawn from the row's own bits and key, so a row keeps
    its masks every time it is sampled."""
    bits = jax.lax.bitcast_convert_type(rows, jnp.uint32)
    odd_weights = 2 * jnp.arange(rows.shape[-1], dtype=jnp.uint32) + 1
    seeds = jnp.sum(bits * odd_weights, axis=-1, dtype=jnp.uint32)

    def draw(seed: jax.Array) -> jax.Array:
        return jax.random.bernoulli(jax.random.fold_in(key, seed), rate, (critics,))

    return jax.vmap(draw, out_axes=1)(seeds)[..., None]


@struct.dataclass
class HeldWeight:
    """The weight each env acts on, and whether its episode has started."""

    weight: jax.Array  # (n_envs, 1)
    started: jax.Array  # (n_envs, 1)


@struct.dataclass
class CHEQState:
    value_spread: jax.Array  # the bootstrapped values' interquartile range
    uncertainty: jax.Array  # the last step's mean u_t / value_spread
    weight: jax.Array  # lambda_t tracked: a log reads no episode's first step
    mask_key: jax.Array  # the rows' masks' (row_masks)


def relative_uncertainty(
    critic_state: Any, ext_state: CHEQState, observations: Any, actions: Any
) -> jax.Array:
    """Eq. 4 as a share of the value spread; past u_high before any target."""
    tiny = jnp.finfo(ext_state.value_spread.dtype).tiny
    spread = jnp.maximum(ext_state.value_spread, tiny)
    return disagreement(critic_state, observations, actions) / spread


@dataclass(frozen=True)
class CHEQ(Extension):
    """The weight's column on every observation, the blend executed, the RL
    action stored.

    The agent's expert is the control prior. Each observation carries lambda_t,
    raw, so the actor and the critics read it. A row stores the RL action a_RL,
    the uniform draw in the warm-up and the actor's sample after, and its next
    observation holds the row's lambda_t (Algorithm 2), so training sees a_RL
    in the hybrid MDP, not a_mix. Before adaptation_start each episode runs on
    one weight of the warm-up's band, up to warmup_lambda_max (their [0.2, 0.3]
    for 5e3 steps). From it, lambda_t starts each episode at lambda_min and
    follows eq. 5 after every update.

    **Thresholds.** u_low and u_high are shares of the value spread, the
    interquartile range of the values the targets bootstrap, tracked at
    scale_rate, not Q units: the paper's 0.03 and 0.15 are Q units of one task,
    and these tasks' rewards span 1e-4 to 6e4 per step. Scaling or shifting the
    rewards leaves u / spread as it is, and one trip's cost in a batch does not
    inflate it. The defaults are the paper's numbers over a spread of 6, an
    estimate from its rewards (0.08 to 0.28 per step at gamma 0.99), not a
    measurement: a starting point to tune per task.

    **The prior's memory.** The paper assumes a prior "time-invariant and
    without an internal state". A PID with an integrator breaks that: a_prior
    then depends on the error history, so (s, a_RL, lambda) is not Markov. So a
    stateful expert needs PIDTermsInput or ExpertStateInput in the stack: its
    memory joins the observation the actor and the critics read, and the hybrid
    MDP is Markov again. ExpertStateInput holds the memory itself.
    PIDTermsInput holds the error's plain integral, which parts from it where
    the expert's anti-windup holds its integrator (four_tank's PI, while its
    pumps saturate). The expert's memory goes on advancing on the observed
    error under the blend, and restarts with each episode.

    Departures from the paper. Evaluation has no state to carry lambda in:
    its observation holds lambda_max, which is the actor's input if evaluated
    alone; eval_action reads eq. 4 and 5 there, at the current state rather
    than the last, then the actor again at that weight, in place of lambda
    carried along the episode. Masks seed on a row's bits (row_masks), not a
    stored mask, and each critic's loss keeps the batch's size as its divisor.
    The warm-up's band is walked per episode, not per step. Inherited from the
    shared backbone, and gone on REDQ: SAC's actor loss reads the critics'
    minimum (theirs, the mean), its target every critic (theirs, a random
    pair), and its critic steps share a batch. eq. 4 is the population
    deviation; their code's torch.std is the sample's (a factor sqrt(E / (E -
    1)), which the thresholds absorb). Their configs run 3 critics for the
    paper's E = 5.
    """

    lambda_min: float = 0.2
    lambda_max: float = 1.0
    u_low: float = 0.005
    u_high: float = 0.025
    warmup_lambda_max: float = 0.3
    adaptation_start: int = 5_000
    mask_rate: float = 0.8
    scale_rate: float = 0.01
    # Where the observation holds the weight, bound by bind_to_agent.
    weight_column: int = -1
    observer_index: int = -1
    name: str = "cheq"
    action_slot: ClassVar[str] = "post_warmup"
    observes_normalised: ClassVar[bool] = False
    #: It tells evaluation (no timestep, progress above 0) from the run's first
    #: observation (no timestep, progress 0), so the collector keeps the horizon.
    reads_progress: ClassVar[bool] = True
    requires: ClassVar[frozenset[str]] = frozenset(
        {
            "agent_expert",
            "continuous_actions",
            "critic_ensemble",
            "obs_networks",
            "q_critic",
            "replay",
            "scalar_target",
        }
    )
    context: ClassVar[frozenset[str]] = frozenset({"env_args", "expert_policy"})
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset({"next_values"}),
        "critic_loss": frozenset({"actions", "critic_params", "critic_state"}),
    }

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_min <= self.lambda_max <= 1.0:
            raise ValueError("CHEQ needs 0 <= lambda_min <= lambda_max <= 1.")
        if not 0.0 <= self.u_low < self.u_high:
            raise ValueError("CHEQ needs 0 <= u_low < u_high.")

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def bind_to_agent(self, **agent_context: Any) -> "CHEQ":
        """Find the weight's column; refuse an expert whose memory the
        observations do not hold (the class docstring)."""
        env_args = agent_context["env_args"]
        memory = (PIDTermsInput, ExpertStateInput)
        if hasattr(agent_context["expert_policy"], "init_state") and not any(
            isinstance(observer, memory) for observer in env_args.observers
        ):
            raise ValueError(
                "CHEQ's prior has no memory in the paper, and this expert has one:"
                " add PIDTermsInput or ExpertStateInput to the extensions, so the"
                " actor and the critics read it."
            )
        width = get_state_action_shapes(env_args.env)[0][-1]
        layout = observation_layout(width, env_args.observers, env_args)
        names = [observer.name for observer in env_args.observers]
        return dataclasses.replace(
            self,
            weight_column=layout.slice(self.name).start,
            observer_index=names.index(self.name),
        )

    def init_state(self, agent_state: Any, rng: jax.Array) -> CHEQState:
        del agent_state
        return CHEQState(
            jnp.zeros(()), jnp.zeros(()), jnp.full((), self.lambda_min), rng
        )

    # -- the weight's column -----------------------------------------------
    def observation_width(self, env_args: Any) -> int:
        del env_args
        return 1

    def init_observer(self, n_envs: int) -> HeldWeight:
        weight = jnp.full((n_envs, 1), self.lambda_min)
        return HeldWeight(weight, jnp.zeros((n_envs, 1), jnp.bool_))

    def observe(
        self, inputs: ObserveInputs, state: HeldWeight
    ) -> tuple[jax.Array, Any]:
        """The weight the env acts on next: its episode's, chosen at the
        episode's first observation. Evaluation holds the actor's own column
        (the class docstring). The run's first observation and an expert
        prefill's rows, at progress 0, hold lambda_min."""
        if inputs.timestep is None:
            evaluating = inputs.progress is None or inputs.progress > 0
            weight = jnp.where(evaluating, self.lambda_max, state.weight)
            return weight, HeldWeight(weight, jnp.ones_like(state.started))
        band = self.warmup_lambda_max - self.lambda_min
        walked = jnp.mod(inputs.timestep * GOLDEN_RATIO_FRACTION, 1.0)
        warming = inputs.timestep < self.adaptation_start
        first = self.lambda_min + jnp.where(warming, walked * band, 0.0)
        weight = jnp.where(state.started, state.weight, first)
        return weight, HeldWeight(weight, jnp.ones_like(state.started))

    def read_weight(self, observations: jax.Array) -> jax.Array:
        column = self.weight_column
        return observations[..., column : column + 1]

    # -- acting ------------------------------------------------------------
    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        """The blend of the chosen action and the expert's, the chosen one
        (uniform in the warm-up) going to the replay."""
        del agent_state, ext_state, rng, context
        rl_action = obs["env_action"]
        obs["replay_action"] = rl_action
        weight = self.read_weight(obs["observations"])
        return blend(obs["expert_action"], rl_action, weight)

    def eval_action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        """The blend at the weight eq. 4 and 5 give at this state, with the
        actor's action at that weight (the class docstring)."""
        del rng, context
        critic, actor = agent_state.critic_state, agent_state.actor_state
        observations = obs["observations"]
        weight = self.weight_at(
            relative_uncertainty(
                critic, ext_state, observations, obs["executed_action"]
            )
        )
        told = observations.at[..., self.weight_column].set(weight[..., 0])
        pi, _ = get_pi(actor, actor.params, told)
        return blend(obs["expert_action"], pi.mean(), weight)

    def weight_at(self, uncertainty: jax.Array) -> jax.Array:
        return adapted_weight(
            uncertainty, self.u_low, self.u_high, self.lambda_min, self.lambda_max
        )

    # -- learning ----------------------------------------------------------
    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        target: jax.Array,
        context: ExtensionContext,
    ) -> tuple[jax.Array, dict, Any]:
        """The target unchanged, the spread of what it bootstraps tracked."""
        del agent_state, context
        spread = interquartile_range(batch["next_values"])
        tracked = track(ext_state.value_spread, spread, self.scale_rate)
        tracked = jnp.where(ext_state.value_spread > 0.0, tracked, spread)
        return target, {}, ext_state.replace(value_spread=tracked)

    def critic_loss(
        self,
        agent_state: Any,
        ext_state: CHEQState,
        batch: Any,
        context: ExtensionContext,
    ) -> Any:
        """The masked rows taken back out of the agent's mean squared error."""
        del agent_state, context
        if self.mask_rate >= 1.0:
            return 0.0
        x = action_value_input(batch["observations"], batch["actions"])
        values = predict_value(batch["critic_state"], batch["critic_params"], x)
        kept = row_masks(ext_state.mask_key, x, values.shape[0], self.mask_rate)
        return -jnp.mean(jnp.where(kept, 0.0, (values - batch["targets"]) ** 2))

    def post_update(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        """lambda_{t+1} from the newest row (s_t, a_RL, lambda_t), written to
        the next observation and its env's held weight; a new episode keeps
        its start."""
        collector = agent_state.collector_state
        replay = collector.buffer_state
        rows = replay.experience
        newest = (replay.current_index - 1) % rows["obs"].shape[1]
        observations, actions = rows["obs"][:, newest], rows["action"][:, newest]
        critic = agent_state.critic_state
        uncertainty = relative_uncertainty(critic, ext_state, observations, actions)
        held = self.read_weight(collector.last_obs)
        ended = jnp.logical_or(collector.last_terminated, collector.last_truncated)
        keep = jnp.logical_or(ended, context.step < self.adaptation_start)
        weight = jnp.where(keep.reshape(held.shape), held, self.weight_at(uncertainty))
        states = list(collector.observer_states)
        index = self.observer_index
        states[index] = states[index].replace(weight=weight)
        last_obs = collector.last_obs.at[..., self.weight_column].set(weight[..., 0])
        collector = collector.replace(last_obs=last_obs, observer_states=tuple(states))
        return (
            agent_state.replace(collector_state=collector),
            ext_state.replace(
                uncertainty=jnp.mean(uncertainty),
                weight=track(ext_state.weight, jnp.mean(weight), self.scale_rate),
            ),
        )

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: CHEQState,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        """The weight the envs have been acting on, the disagreement behind
        it, and the spread that scales it."""
        del agent_state, rng, context
        return {
            "cheq/weight": ext_state.weight,
            "cheq/uncertainty": ext_state.uncertainty,
            "cheq/value_spread": ext_state.value_spread,
        }


__all__ = ["CHEQ", "adapted_weight", "blend", "disagreement", "row_masks"]
