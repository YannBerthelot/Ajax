"""How a trained agent acts: the part of an evaluation step before the
environment steps, which evaluate runs on its rollout (ajax.evaluate), and
the policy object a user acts with (agent.policy), which acts as the
evaluation does."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ajax.controller import (
    ActingState,
    Controller,
    ObserveInputs,
    call_expert,
    init_observers,
    observe_features,
    restart_observers,
)
from ajax.environments.action_stage import choose_in_evaluation
from ajax.environments.interaction import get_pi
from ajax.environments.utils import ExpertInAgentUnits, agent_action_to_env
from ajax.networks.distributions import latent
from ajax.networks.memory import tree_where_reset, zeros_carry_like
from ajax.random_keys import derive_key, split_carry_first
from ajax.schedule import progress
from ajax.utils import online_normalize


def normalised_reward(x: jax.Array, norm_info: Optional[Any]) -> jax.Array:
    """``x``, a reward or a sum of rewards in the env's units, in the
    agent's: divided by the scale the training env's reward normaliser
    holds (``norm_info.reward``, frozen), as collection divides each
    reward; unchanged without one."""
    stats = getattr(norm_info, "reward", None)
    if stats is None:
        return x
    return online_normalize(
        x, stats.count, stats.mean, stats.mean_2, train=False, shift=False
    )[0]


def frozen_progress(agent_state: Any) -> Any:
    """The training progress an evaluation's features hold: the collector's
    as it stands (None where it keeps no horizon)."""
    collector = getattr(agent_state, "collector_state", None)
    if collector is None or collector.max_timesteps is None:
        return None
    return progress(collector.timestep, collector.max_timesteps)


def get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous):
    """Return a function mapping (obs, done, hidden) → (action, entropy, hidden).

    In recurrent mode the hidden state is an explicit input/output so the
    eval loop can thread it through its carry (the live carry on
    ``actor_state`` belongs to the training envs and has the wrong batch
    size here). Non-recurrent mode passes ``hidden`` through untouched.
    """

    def fn(obs: jax.Array, done: Optional[jax.Array], hidden):
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        state = actor_state.replace(hidden_state=hidden) if recurrent else actor_state
        pi, new_state = get_pi(state, actor_state.params, obs, done, recurrent)
        action = pi.mean() if continuous else pi.mode()
        entropy = latent(pi).entropy()
        if recurrent:
            # drop the single-step time axis added by get_pi
            action = action.squeeze(0)
            entropy = entropy.squeeze(0)
            return action, entropy, new_state.hidden_state
        return action, entropy, hidden

    return fn


class Acted(NamedTuple):
    """One acting step, per env, in agent units."""

    actions: jax.Array  # the executed action
    actor_actions: jax.Array  # the actor's (the gains under gain mode)
    entropy: jax.Array
    actor_hidden: Any  # the actor's memory after the step
    expert_state: Any  # the tracked expert's state after the step
    actor_input: jax.Array  # what the actor read, which the critics score
    # 1 where a chooser handed the expert the step; None without choosers.
    handed: Optional[jax.Array]


def make_act(
    actor_state: Any,
    recurrent: bool,
    continuous: bool,
    expert_policy: Any = None,
    agent_state: Any = None,
    controller: Optional[Controller] = None,
    command: Optional[Any] = None,
) -> Callable[..., Acted]:
    """A pure function acting once, as the evaluation acts:
    act(obs, done, expert_obs, expert_state, actor_hidden, command_value,
    rng) -> Acted.

    It acts as the agent's controller says, on an ActingState: the actor's
    input (Controller.observe), its deterministic action, then the executed
    one (Controller.act). The expert acts in agent units on expert_obs, the
    environment's own observation. A command-conditioned actor reads
    command.actor_input(obs, command_value). The controller's choosers (an
    extension's eval_action phase) may then replace the executed action by
    the expert's (choose_in_evaluation), on keys derived from rng, so they
    move no other draw.
    """
    controller = Controller() if controller is None else controller
    # The statistics the agent's controller normalises with, as they stand.
    obs_norm = (
        agent_state.collector_state.obs_norm_info if controller.normalize_obs else None
    )
    actor = get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous)

    def act(
        obs: jax.Array,
        done: jax.Array,
        expert_obs: Optional[jax.Array],
        expert_state: Any,
        actor_hidden: Any,
        command_value: Any,
        rng: jax.Array,
    ) -> Acted:
        acting = ActingState(
            obs=obs,
            done=done,
            timestep=None,
            expert_state=expert_state,
            carry=actor_hidden,
            obs_norm=obs_norm,
        )
        observations = actor_input = controller.observe(acting)
        if command is not None:
            actor_input = command.actor_input(actor_input, command_value)

        actor_actions, entropy, new_actor_hidden = actor(
            actor_input, done if recurrent else None, actor_hidden
        )
        actions, new_expert_state = controller.act(
            actor_actions, acting, expert_policy, expert_obs
        )
        handed = None
        if controller.choosers:
            expert_action = None
            if expert_policy is not None:
                expert_action, _ = call_expert(expert_policy, expert_state, expert_obs)
            choice = {
                "executed_action": actions,
                "expert_action": expert_action,
                "observations": observations,  # the critics' input
                "raw_obs": expert_obs,
            }
            choice, handed = choose_in_evaluation(
                agent_state, acting, controller.choosers, choice, rng
            )
            actions = choice["executed_action"]
        return Acted(
            actions,
            actor_actions,
            entropy,
            new_actor_hidden,
            new_expert_state,
            actor_input,
            handed,
        )

    return act


class PolicyCarry(NamedTuple):
    """What a policy carries from one step to the next, per env; None where
    it carries nothing."""

    memory: Any = None  # the actor's memory, or a world model's state
    expert_state: Any = None  # a stateful expert's
    observer_states: Any = None  # the observation features'
    command: Any = None  # a command-conditioned actor's (UDRL)
    first: Any = None  # True where the next observation starts an episode
    key: Any = None  # the stream act draws a step's key from when given none


class PolicyParts(NamedTuple):
    """How an agent acts, on one seed's state: initial(n_envs) is the carry
    at the start of an episode, step(obs, carry, starts, reward, key) ->
    (action, carry) acts once in the environment's units, starts True where
    obs starts an episode. reads_reward: step reads the reward the last
    action earned."""

    initial: Callable[[int], PolicyCarry]
    step: Callable[..., tuple[jax.Array, PolicyCarry]]
    reads_reward: bool = False


class Policy:
    """A trained agent's policy (agent.policy), acting in the environment's
    units exactly as the agent's evaluation acts:

        carry = policy.reset(n_envs)
        action, carry = policy.act(obs, carry, done, reward)

    obs is a batch of the environment's own observations, one row per env.
    done is True where obs starts a new episode, the last action having
    ended one (None: nowhere); the memory, the expert's state, the
    observation features and the command restart there. reward is what the
    last action earned, which a command-conditioned agent (UDRL) reads. key
    is the step's key for the policy's draws (a chooser's, a world model's
    sampling); None draws it from the carry's own stream. act is compiled
    once, its state an argument of the program, not a constant in it.
    """

    def __init__(self, build: Callable[[Any], PolicyParts], agent_state: Any):
        self.agent_state = agent_state
        self.build = build
        self.reads_reward = build(agent_state).reads_reward

        def step(agent_state, obs, carry, done, reward, key):
            starts = jnp.logical_or(carry.first, done)
            parts = build(agent_state)
            action, stepped = parts.step(obs, carry, starts, reward, key)
            return action, stepped._replace(first=jnp.zeros_like(starts), key=carry.key)

        self._step = jax.jit(step)

    def reset(self, n_envs: int) -> PolicyCarry:
        """The carry of n_envs envs whose next observation starts an episode."""
        carry = self.build(self.agent_state).initial(n_envs)
        return carry._replace(
            first=jnp.ones(n_envs, bool),
            key=derive_key(self.agent_state.eval_rng, "policy"),
        )

    @property
    def has_memory(self) -> bool:
        """Whether the policy carries anything from one step to the next."""
        carry = self.build(self.agent_state).initial(1)
        return any(jnp.size(leaf) for leaf in jax.tree.leaves(carry))

    def act(
        self,
        obs: Any,
        carry: PolicyCarry,
        done: Any = None,
        reward: Any = None,
        key: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, PolicyCarry]:
        """The action at obs, in the environment's units, and the next carry."""
        if self.reads_reward and done is not None and reward is None:
            raise ValueError(
                "This policy reads the reward the last action earned (UDRL's"
                " command counts it down): pass reward with done."
            )
        n_envs = carry.first.shape[0]
        done = jnp.zeros(n_envs, bool) if done is None else jnp.asarray(done, bool)
        if reward is None:
            reward = jnp.zeros(n_envs, jnp.float32)
        if key is None:
            stream, key = split_carry_first(carry.key, 1)
            carry = carry._replace(key=stream)
        obs, reward = jnp.asarray(obs), jnp.asarray(reward, jnp.float32)
        return self._step(self.agent_state, obs, carry, done, reward, key)


def evaluation_policy(
    env_args: Any,
    agent_state: Any,
    expert_policy: Any = None,
    controller: Optional[Controller] = None,
    recurrent: bool = False,
    command: Optional[Any] = None,
    norm_info: Optional[Any] = None,
) -> PolicyParts:
    """The policy evaluate runs (make_act), acting on the environment's own
    observations: it appends the run's features to them, as collection
    does, and maps the executed action onto the environment's bounds. The
    expert's state, the features' and the command restart with each
    episode; the command counts down each reward, in the agent's units
    (norm_info, normalised_reward)."""
    env, env_params = env_args.env, env_args.env_params
    expert = ExpertInAgentUnits.wrap(expert_policy, env, env_params)
    controller = Controller() if controller is None else controller
    actor_state = agent_state.actor_state
    act = make_act(
        actor_state,
        recurrent,
        env_args.continuous,
        expert,
        agent_state,
        controller,
        command,
    )
    observers, progress_now = controller.observers, frozen_progress(agent_state)
    stateful = hasattr(expert, "init_state")

    def initial(n_envs: int) -> PolicyCarry:
        return PolicyCarry(
            memory=(
                zeros_carry_like(actor_state.hidden_state, n_envs)
                if recurrent
                else None
            ),
            expert_state=expert.init_state(n_envs) if stateful else None,
            observer_states=init_observers(observers, n_envs) if observers else None,
            command=None if command is None else command.initial(agent_state, n_envs),
        )

    def step(obs, carry, starts, reward, key):
        n_envs, raw_obs = obs.shape[0], obs.astype(jnp.float32)
        observer_states, expert_state = carry.observer_states, carry.expert_state
        if observers:
            observer_states, expert_state = restart_observers(
                starts, observers, observer_states, expert, expert_state, n_envs
            )
            inputs = ObserveInputs(
                raw_obs,
                expert=expert,
                expert_state=expert_state if stateful else None,
                progress=progress_now,
            )
            obs, observer_states = observe_features(
                obs, inputs, observers, observer_states
            )
        elif stateful:
            fresh = expert.init_state(n_envs)
            expert_state = tree_where_reset(starts, fresh, expert_state)
        value = carry.command
        if command is not None:
            # The last reward counts the command down; a new episode restarts it.
            reward = normalised_reward(reward, norm_info)
            value = command.update(value, reward, jnp.zeros_like(starts))
            value = tree_where_reset(
                starts, command.initial(agent_state, n_envs), value
            )
        done = starts.astype(jnp.int8)  # as the evaluation's carry holds it
        acted = act(obs, done, raw_obs, expert_state, carry.memory, value, key)
        carry = PolicyCarry(
            acted.actor_hidden, acted.expert_state, observer_states, value
        )
        return agent_action_to_env(acted.actions, env, env_params), carry

    return PolicyParts(initial, step, reads_reward=command is not None)
