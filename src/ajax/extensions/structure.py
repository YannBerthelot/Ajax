"""What RL can change around a controller it keeps.

Most expert features only call the expert, expert(memory, obs). Three need its
law, and ExpertLaw is all they ask of it: one pure step(params, memory, obs) ->
(action, memory) for one env, with params and memory pytrees, as TargetGym's
Controller holds it, and the names of what they touch.

* OuterLoopReference: the policy moves the target the expert reads.
* GainScheduledLaw, under ExpertGains: the policy sets multipliers on its gains.
* BumplessTracking: the expert's memory follows the action executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.controller import call_expert
from ajax.extensions.base import Extension


def path_name(path: tuple) -> str:
    """A pytree leaf's dotted name, "tecs.time_const"."""
    return ".".join(str(getattr(key, "name", key)) for key in path)


def scale_gains(params: Any, names: tuple[str, ...], factors: jax.Array) -> Any:
    """params with each leaf of names multiplied by its factor."""

    def scaled(path: tuple, leaf: Any) -> Any:
        name = path_name(path)
        return leaf * factors[names.index(name)] if name in names else leaf

    return jax.tree_util.tree_map_with_path(scaled, params)


def conditioned_memory(
    step: Callable,
    params: Any,
    memory: Any,
    obs: jax.Array,
    action: jax.Array,
    channels: tuple[int, ...],
) -> Any:
    """The memory after a step that executed action, for one env: Hanus,
    Kinnaert and Henrotte (Automatica 1987), for a controller x+ = A x + B e,
    u = C x + D e, with the state fed the realisable error, the one at which
    the controller would have output the executed action::

        realisable error = e + inverse(D) (u_executed - u)

    The paper is paywalled: the equations are as MERL TR2019-054 restates them.

    D is the law's derivative in the readings it compares with its targets, so
    a cascade's is the whole cascade's. The error moves through the reading,
    not the target as the paper has it: a law biasing on its target (four_tank's
    steady voltages, cstr's steady temperature) has that bias stay put, and may
    not be defined at a moved one. D's pseudo-inverse leaves a law sitting at
    its own limit, D = 0, to its own anti-windup, and the action a law chose
    itself shifts nothing, so that step is the law's own. One Newton step,
    exact for a law affine in its readings.
    """
    columns = jnp.asarray(channels)

    def stepped(shift: jax.Array) -> tuple[jax.Array, Any]:
        return step(params, memory, obs.at[columns].add(shift))

    no_shift = jnp.zeros(columns.shape, obs.dtype)
    sensitivity = jax.jacfwd(lambda shift: stepped(shift)[0])(no_shift)
    shift = jnp.linalg.pinv(sensitivity) @ (action - stepped(no_shift)[0])
    return stepped(shift)[1]


def batched(tree: Any, n_envs: int) -> Any:
    """tree repeated along a new leading axis, one entry per env."""

    def repeat(leaf: Any) -> jax.Array:
        leaf = jnp.asarray(leaf)
        return jnp.broadcast_to(leaf, (n_envs, *leaf.shape))

    return jax.tree.map(repeat, tree)


@dataclass(frozen=True, eq=False)
class ExpertLaw:
    """An expert as its law, batched over envs, for an agent's expert_policy.

    step, params and memory as the module docstring says, in the env's units.
    reading_channels are the observation columns its errors read, one per
    loop; gain_names the dotted leaves of params its gains are.
    """

    step: Callable
    params: Any
    memory: Any
    reading_channels: tuple[int, ...] = ()
    gain_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        known = [
            path_name(path)
            for path, _ in jax.tree_util.tree_flatten_with_path(self.params)[0]
        ]
        unknown = [name for name in self.gain_names if name not in known]
        if unknown:
            raise ValueError(f"The law's params have no gains {unknown}.")

    @classmethod
    def of(cls, controller: Any, **names: Any) -> "ExpertLaw":
        """The law of a controller with step, params and init_state."""
        step, params = controller.step, controller.params
        return cls(step, params, controller.init_state, **names)

    def act(self, params: Any, memory: Any, obs: jax.Array) -> tuple[jax.Array, Any]:
        """One env's step, its action a column: battery's law returns a number,
        which the agent's (n_envs, n_actions) action does not take."""
        action, memory = self.step(params, memory, obs)
        return jnp.ravel(action), memory

    def init_state(self, n_envs: int) -> Any:
        return batched(self.memory, n_envs)

    def __call__(self, state: Any, obs: jax.Array) -> tuple[jax.Array, Any]:
        return jax.vmap(self.act, (None, 0, 0))(self.params, state, obs)

    def track(self, state: Any, obs: jax.Array, action: jax.Array) -> Any:
        """The memory after the executed action (conditioned_memory)."""
        if not self.reading_channels:
            raise ValueError("Tracking an action needs the law's reading_channels.")

        def one_env(memory: Any, reading: jax.Array, executed: jax.Array) -> Any:
            return conditioned_memory(
                self.act,
                self.params,
                memory,
                reading,
                executed,
                self.reading_channels,
            )

        return jax.vmap(one_env)(state, obs, action)


@struct.dataclass
class GainLayerState:
    """A gain-scheduled law's state per env: the law's memory now and at the
    last step, the last observation and executed action, the multipliers'
    logarithms, and whether the episode has had a step."""

    memory: Any
    last_memory: Any
    last_obs: jax.Array
    last_action: jax.Array
    log_factors: jax.Array
    started: jax.Array


@dataclass(frozen=True, eq=False)
class GainScheduledLaw:
    """A law whose gains take multipliers each step, through a velocity form
    (Bloor et al., arXiv 2408.13566, eq. 10): for ExpertGains, which bounds the
    multipliers (max_factor), and an agent's expert_policy.

    The paper's layer is a PID. These laws are not all PIDs, so the form is
    theirs, held at the step's gains: the change in the action is the law's
    change from the last step to this one, both at this step's gains::

        u_t = u_(t-1) + f(g_t; x_t) - f(g_t; x_(t-1))

    On a PID that is eq. 10, and a new gain alone moves nothing. At constant
    gains it is the law itself, with its limits and anti-windup. It is written
    u_t = f(g_t; x_t) + bias, so at unit multipliers bias is 0 and the action
    is the law's, to float32 rounding where it divides by a gain (XLA divides
    by one number through its reciprocal, not by one per env). bias_retention
    scales the bias each step: at 1 this is eq. 10, below it a law at its own
    limit, or holding an inner integrator, cannot stay held off that limit by
    an old bias.

    Beyond the paper, whose gains jump each step: log multipliers move at most
    max_log_rate a step. The memory they hold joins the state, so the policy
    reads what its action will move from through ExpertStateInput. The paper
    trains by particle swarm; Ajax's SAC here has no published result.
    """

    law: ExpertLaw
    obs_width: int
    max_log_rate: float = 0.05
    bias_retention: float = 0.99
    action_limits: tuple[float, float] = (-1.0, 1.0)

    @property
    def learnable_fields(self) -> tuple[str, ...]:
        return self.law.gain_names

    @property
    def anchor_gains(self) -> jax.Array:
        """Ones: the gains ExpertGains sets here are the multipliers."""
        return jnp.ones(len(self.law.gain_names))

    def init_state(self, n_envs: int) -> GainLayerState:
        law = self.law
        obs = jnp.zeros(self.obs_width)

        def action_at(memory: Any, reading: jax.Array) -> jax.Array:
            return law.act(law.params, memory, reading)[0]

        action = jax.eval_shape(action_at, law.memory, obs)
        return batched(
            GainLayerState(
                memory=law.memory,
                last_memory=law.memory,
                last_obs=obs,
                last_action=jnp.zeros(action.shape, action.dtype),
                log_factors=jnp.zeros(len(law.gain_names)),
                started=jnp.zeros((), jnp.bool_),
            ),
            n_envs,
        )

    def __call__(self, state: GainLayerState, obs: jax.Array) -> Any:
        """The law at the multipliers it holds."""
        return self.step_with_gains(state, obs, jnp.exp(state.log_factors))

    def step_with_gains(
        self, state: GainLayerState, obs: jax.Array, factors: jax.Array
    ) -> tuple[jax.Array, GainLayerState]:
        """One step at the multipliers factors, (n_envs, n_gains), as far as
        the rate limit lets them move."""
        law = self.law
        change = jnp.log(factors) - state.log_factors
        change = jnp.clip(change, -self.max_log_rate, self.max_log_rate)
        log_factors = state.log_factors + change

        def one_env(log_factor: jax.Array, memory: Any, last_memory: Any, *rest: Any):
            last_obs, reading = rest
            params = scale_gains(law.params, law.gain_names, jnp.exp(log_factor))
            held = law.act(params, last_memory, last_obs)[0]
            return held, *law.act(params, memory, reading)

        held, action, memory = jax.vmap(one_env)(
            log_factors, state.memory, state.last_memory, state.last_obs, obs
        )
        bias = self.bias_retention * (state.last_action - held)
        bias = jnp.where(state.started[:, None], bias, 0.0)
        action = jnp.clip(action + bias, *self.action_limits)
        return action, GainLayerState(
            memory,
            state.memory,
            obs,
            action,
            log_factors,
            jnp.ones_like(state.started),
        )


@dataclass(frozen=True)
class OuterLoopReference(Extension):
    """Outer-loop control (Arroyo and Giraldo, arXiv 2008.13732): the policy
    moves the target the agent's expert reads, and the unchanged expert acts.
    The actions are one per column of target_channels, each scaled by its
    max_change, in the target's units. The replay and the critics hold the
    policy's actions, and a warm-up draws uniform changes.

    Departures from the paper: the policy outputs a change, not the reference
    itself, so a change of 0 is the expert to the bit and the policy starts
    near it; and the reward stays the task's, without the paper's penalties on
    fast references (eq. 7).
    """

    target_channels: tuple[int, ...]
    max_change: tuple[float, ...]
    name: str = "outer_loop_reference"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "obs_networks", "reward_learning"}
    )

    def __post_init__(self) -> None:
        # Tuples: the extension is a static, hashed argument.
        object.__setattr__(self, "target_channels", tuple(self.target_channels))
        object.__setattr__(self, "max_change", tuple(self.max_change))
        if len(self.target_channels) != len(self.max_change):
            raise ValueError("OuterLoopReference takes one max_change per channel.")
        if not self.max_change or min(self.max_change) <= 0:
            raise ValueError("OuterLoopReference's max_change is positive.")

    def policy_action_dim(self, env_args: Any, expert: Any) -> int:
        del env_args, expert
        return len(self.target_channels)

    def actuate(self, actions: jax.Array, inputs: Any) -> tuple[jax.Array, Any]:
        changes = jnp.clip(actions, -1.0, 1.0) * jnp.asarray(self.max_change)
        columns = jnp.asarray(self.target_channels)
        moved = inputs.raw_obs.at[..., columns].add(changes)
        return call_expert(inputs.expert, inputs.expert_state, moved)

    def policy_action_for_expert(self, policy_actions: jax.Array) -> jax.Array:
        # No change leaves the target, and the expert, as they are.
        return jnp.zeros_like(policy_actions)


@dataclass(frozen=True)
class BumplessTracking(Extension):
    """The agent's expert's memory follows the action executed: without it the
    memory advances on the error whoever acts, so its integral winds up while
    a learner holds the plant. It makes no difference to what the arm chooses,
    so it stacks on any switching or blending arm, and leaving it out is the
    ablation. The agent's expert needs track(memory, obs, action) -> memory,
    which ExpertLaw gives.
    """

    name: str = "bumpless_tracking"
    tracks_executed_action: ClassVar[bool] = True
    # An agent differentiating through the expert's memory (APG) has no use for
    # a memory that a pseudo-inverse rewrites, and a world model's imagined steps
    # (obs_networks) execute no action.
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "obs_networks", "off_policy"}
    )

    def bind_to_agent(self, **agent_context: Any) -> "BumplessTracking":
        expert: Optional[Any] = agent_context.get("expert_policy")
        if not hasattr(expert, "track"):
            raise ValueError(
                "BumplessTracking needs an expert with track(memory, obs, action):"
                " give the agent an ExpertLaw with its reading_channels."
            )
        return self


__all__ = [
    "BumplessTracking",
    "ExpertLaw",
    "GainScheduledLaw",
    "OuterLoopReference",
    "conditioned_memory",
]
