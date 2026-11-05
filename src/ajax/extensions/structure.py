"""What RL can change around a controller it keeps.

* OuterLoopReference: the policy moves the target the expert reads.
* GainScheduledLaw, under ExpertGains: the policy sets multipliers on its gains.
* BumplessTracking: the expert's memory follows the action applied.

Each asks only what it uses of a stateful expert, expert(memory, obs) ->
(action, memory): the first and the last which columns to move, the second its
gains through step_with_gains. TargetGym's batched controllers have that form,
and its envs name the columns (obs_target_index, obs_value_index). ExpertLaw gives
gains to a law that form does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
from flax import struct

from ajax.controller import call_expert
from ajax.extensions.base import Extension
from ajax.networks.memory import tree_where_reset


def conditioned_memory(
    expert: Any,
    memory: Any,
    obs: jax.Array,
    action: jax.Array,
    channels: tuple[int, ...],
) -> Any:
    """The batched expert's memory after a step that executed action: Hanus,
    Kinnaert and Henrotte (Automatica 1987), for a controller x+ = A x + B e,
    u = C x + D e, with the state fed the realisable error, the one at which
    the controller would have output the executed action::

        realisable error = e + inverse(D) (u_executed - u)

    The paper is paywalled: the equations are as MERL TR2019-054 restates them.

    D is the expert's derivative in the readings it compares with its targets,
    so a cascade's is the whole cascade's. The error moves through the reading,
    not the target as the paper has it: a law biasing on its target (four_tank's
    steady voltages, cstr's steady temperature) has that bias stay put, and may
    not be defined at a moved one. D's pseudo-inverse leaves a law sitting at
    its own limit, D = 0, to its own anti-windup. One Newton step, exact for a
    law affine in its readings.
    """
    columns = jnp.asarray(channels)

    def one_env(env_memory: Any, env_obs: jax.Array, executed: jax.Array) -> Any:
        def stepped(shift: jax.Array) -> tuple[jax.Array, Any]:
            batch = jax.tree.map(lambda leaf: leaf[None], env_memory)
            actions, after = expert(batch, env_obs.at[columns].add(shift)[None])
            return actions[0], jax.tree.map(lambda leaf: leaf[0], after)

        no_shift = jnp.zeros(columns.shape, env_obs.dtype)
        gap = executed - stepped(no_shift)[0]
        sensitivity = jax.jacfwd(lambda shift: stepped(shift)[0])(no_shift)
        return stepped(jnp.linalg.pinv(sensitivity) @ gap)[1]

    return jax.vmap(one_env)(memory, obs, action)


def path_name(path: tuple) -> str:
    """A pytree leaf's dotted name, "tecs.time_const"."""
    return ".".join(str(getattr(key, "name", key)) for key in path)


@dataclass(frozen=True, eq=False)
class ExpertLaw:
    """An expert with step_with_gains, from its law: step(params, memory, obs)
    -> (action, memory) for one env, with params and memory pytrees, as
    TargetGym's Controller holds it, and gain_names the dotted leaves of params
    that are gains.

    TargetGym's batched form has step_with_gains too, for a law whose flat gains
    are registered. Most are not, TECS nests its gains, and that form makes
    every param an array, which wind_turbine's law cannot take to numpy.
    """

    step: Callable
    params: Any
    memory: Any
    gain_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.gain_names).difference(self._leaves())
        if unknown:
            raise ValueError(f"The law's params have no gains {sorted(unknown)}.")

    def _leaves(self) -> dict[str, Any]:
        leaves = jax.tree_util.tree_flatten_with_path(self.params)[0]
        return {path_name(path): leaf for path, leaf in leaves}

    @classmethod
    def of(cls, controller: Any, gain_names: tuple[str, ...]) -> "ExpertLaw":
        """The law of a controller with step, params and init_state."""
        step, params = controller.step, controller.params
        return cls(step, params, controller.init_state, gain_names)

    @property
    def learnable_fields(self) -> tuple[str, ...]:
        return self.gain_names

    @property
    def anchor_gains(self) -> jax.Array:
        leaves = self._leaves()
        return jnp.array([float(leaves[name]) for name in self.gain_names])

    def init_state(self, n_envs: int) -> Any:
        return batched(self.memory, n_envs)

    def __call__(self, state: Any, obs: jax.Array) -> tuple[jax.Array, Any]:
        return jax.vmap(self.step, (None, 0, 0))(self.params, state, obs)

    def step_with_gains(
        self, state: Any, obs: jax.Array, gains: jax.Array
    ) -> tuple[jax.Array, Any]:
        """A step at gains, (n_envs, n_gains), the other params left as they are."""

        def one_env(env_gains: jax.Array, memory: Any, reading: jax.Array) -> Any:
            def gained(path: tuple, leaf: Any) -> Any:
                name = path_name(path)
                if name not in self.gain_names:
                    return leaf
                return env_gains[self.gain_names.index(name)]

            params = jax.tree_util.tree_map_with_path(gained, self.params)
            return self.step(params, memory, reading)

        return jax.vmap(one_env)(gains, state, obs)


def batched(tree: Any, n_envs: int) -> Any:
    """tree repeated along a new leading axis, one entry per env."""

    def repeat(leaf: Any) -> jax.Array:
        leaf = jnp.asarray(leaf)
        return jnp.broadcast_to(leaf, (n_envs, *leaf.shape))

    return jax.tree.map(repeat, tree)


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
    multipliers (max_factor), and an agent's expert_policy. expert is any with
    step_with_gains, learnable_fields, anchor_gains and init_state.

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
    an old bias. Near a law's own limit the difference is of limited outputs, so
    a higher gain can move the action less, where eq. 10 on a PID would not.

    Beyond the paper, whose gains jump each step: log multipliers move at most
    max_log_rate a step. The memory they hold joins the state, so the policy
    reads what its action will move from through ExpertStateInput. The paper
    trains by particle swarm; Ajax's SAC here has no published result.
    """

    expert: Any
    obs_width: int
    max_log_rate: float = 0.05
    bias_retention: float = 0.99
    action_limits: tuple[float, float] = (-1.0, 1.0)

    @property
    def learnable_fields(self) -> tuple[str, ...]:
        return self.expert.learnable_fields

    @property
    def anchor_gains(self) -> jax.Array:
        """Ones: the gains ExpertGains sets here are the multipliers."""
        return jnp.ones(len(self.learnable_fields))

    def init_state(self, n_envs: int) -> GainLayerState:
        memory = self.expert.init_state(n_envs)
        obs = jnp.zeros((n_envs, self.obs_width))
        action = jax.eval_shape(self.expert, memory, obs)[0]
        return GainLayerState(
            memory=memory,
            last_memory=memory,
            last_obs=obs,
            last_action=jnp.zeros(action.shape, action.dtype),
            log_factors=jnp.zeros((n_envs, len(self.learnable_fields))),
            started=jnp.zeros((n_envs,), jnp.bool_),
        )

    def __call__(self, state: GainLayerState, obs: jax.Array) -> Any:
        """The law at the multipliers it holds."""
        return self.step_with_gains(state, obs, jnp.exp(state.log_factors))

    def step_with_gains(
        self, state: GainLayerState, obs: jax.Array, factors: jax.Array
    ) -> tuple[jax.Array, GainLayerState]:
        """One step at the multipliers factors, (n_envs, n_gains), as far as
        the rate limit lets them move."""
        change = jnp.log(factors) - state.log_factors
        change = jnp.clip(change, -self.max_log_rate, self.max_log_rate)
        log_factors = state.log_factors + change
        gains = self.expert.anchor_gains * jnp.exp(log_factors)
        held, _ = self.expert.step_with_gains(state.last_memory, state.last_obs, gains)
        action, memory = self.expert.step_with_gains(state.memory, obs, gains)
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
    """The agent's expert's memory follows the action applied: without it the
    memory advances on the error whoever acts, so its integral winds up while
    a learner holds the plant. It changes nothing the arm chooses, so it stacks
    on any switching or blending arm, and leaving it out is the ablation.

    reading_channels are the columns the expert's errors read, TargetGym's
    obs_value_index. What it follows is the actuator's report, the env's
    info["applied_action"], or the commanded action where the env gives none or
    no env is stepped (a deployed policy). A step the expert chose itself stays
    its own, its anti-windup with it, whatever the actuator made of it: that is
    the expert the study scores, and its discrepancy is not a takeover.
    """

    reading_channels: tuple[int, ...]
    name: str = "bumpless_tracking"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert", "obs_networks"})

    def __post_init__(self) -> None:
        # A tuple: the extension is a static, hashed argument.
        object.__setattr__(self, "reading_channels", tuple(self.reading_channels))
        if not self.reading_channels:
            raise ValueError("BumplessTracking needs the expert's reading_channels.")

    def bind_to_agent(self, **agent_context: Any) -> "BumplessTracking":
        if not hasattr(agent_context.get("expert_policy"), "init_state"):
            raise ValueError("BumplessTracking needs a stateful expert_policy.")
        return self

    def track_memory(
        self,
        expert: Any,
        memory: Any,
        obs: jax.Array,
        command: jax.Array,
        applied: jax.Array,
    ) -> Any:
        own_action, own_memory = expert(memory, obs)
        tracked = conditioned_memory(
            expert, memory, obs, applied, self.reading_channels
        )
        chose_itself = jnp.all(command == own_action, axis=-1)
        return tree_where_reset(chose_itself, own_memory, tracked)


__all__ = [
    "BumplessTracking",
    "ExpertLaw",
    "GainScheduledLaw",
    "OuterLoopReference",
    "conditioned_memory",
]
