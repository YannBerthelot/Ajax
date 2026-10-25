"""One-shot pre-training research features as composable :class:`Extension`s.

CloningPretrain, MCPretrain and BellmanPretrain run once, before the
training loop (the ``pretrain`` phase, skipped on resume); PhiRefresh
refreshes MCPretrain's critic during training.

* :class:`CloningPretrain`     — behaviour cloning of the actor on an
  expert's rollouts.
* :class:`MCPretrain`          — regress a frozen expert critic on
  Monte-Carlo returns, then (optionally) lightly nudge the online critic
  toward it.
* :class:`BellmanPretrain`     — Bellman-bootstrapped critic pre-training
  on the expert buffer.
* :class:`PhiRefresh`          — periodic self-consistent refresh of the
  frozen expert critic during training (the ``post_update`` phase).
* :class:`PriorData`           — a transition dataset in the replay, and the
  agent's own updates on it before acting.
* ExpertPrefill                — the agent's expert's transitions in the
  replay before training.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.cloning import cloning_columns, pre_train
from ajax.agents.loop import (
    count_in_normalisers,
    seed_replay,
    with_actor_metrics_slot,
)
from ajax.buffers.dataset import TransitionDataset
from ajax.buffers.utils import get_batch_from_buffer
from ajax.controller import policy_action_dim, reads_agent_expert
from ajax.environments.interaction import (
    collect_experience_from_expert_policy,
    replay_row,
    reward_statistics,
)
from ajax.environments.utils import raw_obs_columns
from ajax.extensions.base import Extension, ExtensionContext, ExtensionStack
from ajax.extensions.expert import expert_next_actions, reads_stored_expert_actions
from ajax.networks.networks import phi_star_critic_state, predict_value
from ajax.state import EnvironmentConfig, LoadedTrainState, Transition
from ajax.types import BufferType, NormalizationInfo


@struct.dataclass
class MCPretrainAux:
    """Diagnostics logged after MC critic pretraining."""

    initial_loss: jax.Array
    final_loss: jax.Array
    q_expert_mean: jax.Array
    q_expert_min: jax.Array
    q_expert_max: jax.Array
    v_min: jax.Array
    v_max: jax.Array


@struct.dataclass
class PretrainLosses:
    """A pretraining fit's first and last loss, kept in its extension's state
    for the evaluations' log. NaN before the fit, when it takes no step, and
    in a run resumed from a checkpoint saved before they were kept
    (``added_fields``, :func:`ajax.checkpoint.restore_into`)."""

    first_loss: jax.Array
    last_loss: jax.Array
    added_fields: ClassVar[frozenset[str]] = frozenset({"first_loss", "last_loss"})

    @classmethod
    def unknown(cls) -> Any:
        """Every field NaN."""
        nan = jnp.full((), jnp.nan, jnp.float32)
        return cls(**{field.name: nan for field in dataclasses.fields(cls)})


@struct.dataclass
class MCPretrainResults(PretrainLosses):
    """MCPretrain's fit: its first and last regression loss, and the mean
    over the fit's rows of ``φ*``'s value at the expert's action (the
    smallest of its critics')."""

    q_expert_mean: jax.Array
    added_fields: ClassVar[frozenset[str]] = PretrainLosses.added_fields.union(
        {"q_expert_mean"}
    )


def _first_and_last(losses: jax.Array) -> PretrainLosses:
    """A fit's first and last loss, from its losses in step order."""
    if losses.shape[0] == 0:
        return PretrainLosses.unknown()
    return PretrainLosses(first_loss=losses[0], last_loss=losses[-1])


@partial(
    jax.jit,
    static_argnames=[
        "gamma",
        "reward_scale",
        "n_steps",
        "expert_policy",
        "n_mc_steps",
        "n_mc_episodes",
        "mode",
        "env_args",
        "batch_size",
    ],
)
def pretrain_critic_mc(
    agent_state,
    expert_critic_state: LoadedTrainState,
    expert_policy: Callable,
    mode: str,
    env_args: EnvironmentConfig,
    gamma: float,
    reward_scale: float,
    n_mc_steps: int = 10_000,
    n_mc_episodes: int = 500,
    n_steps: int = 5_000,
    batch_size: int = 256,
) -> Tuple[Any, FrozenDict, jax.Array, jax.Array, MCPretrainAux, LoadedTrainState]:
    """Pre-train critic using Monte Carlo returns from expert trajectories.

    Unlike Bellman pretraining (bootstraps from an untrained critic → biased),
    MC returns G_t = Σ γ^k r_{t+k} are unbiased estimates of V^expert(s).
    The critic starts with accurate Q(s, a_expert) near the target from step 1.

    Collection strategy: single call with n_mc_steps * n_mc_episodes // n_envs
    timesteps. The n_envs parallel environments reset to different initial
    states on each episode boundary, giving the same state-space coverage
    as separate per-seed rollouts — without any mapping over traced keys
    (which fails inside the outer vmap over seeds).
    Total transitions ≈ n_mc_steps * n_mc_episodes regardless of n_envs.

    Returns: (agent_state, frozen_expert_params, obs_batched, action_batched,
              MCPretrainAux, expert_critic_state_trained)
    """
    n_total_steps = max(1, (n_mc_steps * n_mc_episodes) // env_args.n_envs)
    all_transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=agent_state.rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_total_steps,
    )

    rewards = all_transitions.reward * reward_scale
    dones = jnp.logical_or(
        all_transitions.terminated, all_transitions.truncated
    ).astype(jnp.float32)

    def mc_scan(carry, x):
        reward, done = x
        mc_return = reward + gamma * carry * (1.0 - done)
        return mc_return, mc_return

    _, mc_returns = jax.lax.scan(
        mc_scan,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], dones[::-1]),
    )
    mc_returns = mc_returns[::-1]

    T, n_envs = rewards.shape[:2]
    obs_flat = all_transitions.obs.reshape(T * n_envs, -1)
    action_flat = all_transitions.action.reshape(T * n_envs, -1)
    mc_flat = mc_returns.reshape(T * n_envs, 1)

    n_total = obs_flat.shape[0]
    n_batches = n_total // batch_size
    obs_batched = obs_flat[: n_batches * batch_size].reshape(n_batches, batch_size, -1)
    action_batched = action_flat[: n_batches * batch_size].reshape(
        n_batches, batch_size, -1
    )
    mc_batched = mc_flat[: n_batches * batch_size].reshape(n_batches, batch_size, 1)

    # Supervised regression: Q(s, a_expert) → MC return
    def mc_loss_fn(critic_params, obs, actions, targets):
        q_preds = predict_value(
            critic_state=expert_critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, actions], axis=-1),
        )
        return jnp.mean((q_preds - targets) ** 2)

    def regression_step(carry, batch):
        expert_critic_state, step = carry
        obs_b, action_b, mc_b = batch
        loss, grads = jax.value_and_grad(mc_loss_fn)(
            expert_critic_state.params,
            obs_b,
            action_b,
            mc_b,
        )
        new_expert_critic_state = expert_critic_state.apply_gradients(grads=grads)
        return (new_expert_critic_state, step + 1), loss

    initial_loss, _ = jax.value_and_grad(mc_loss_fn)(
        expert_critic_state.params,
        obs_batched[0],
        action_batched[0],
        mc_batched[0],
    )

    n_passes = max(1, n_steps // n_batches)
    batches = (obs_batched, action_batched, mc_batched)

    def one_pass(carry, _):
        return jax.lax.scan(regression_step, carry, batches)

    (expert_critic_state, _), loss_history = jax.lax.scan(
        one_pass, (expert_critic_state, 0), None, length=n_passes
    )
    final_loss = loss_history[-1, -1]

    expert_critic_state = expert_critic_state.soft_update(tau=1.0)

    frozen_expert_params = jax.lax.stop_gradient(expert_critic_state.params)

    # The value range (OnlineBC's weight, ValueBox's threshold) and the
    # diagnostics over every regression row: one batch may hold a few
    # timesteps only (envs in lockstep).
    q_preds_final = predict_value(
        critic_state=expert_critic_state,
        critic_params=frozen_expert_params,
        x=jnp.concatenate([obs_batched, action_batched], axis=-1).reshape(
            n_batches * batch_size, -1
        ),
    )
    q_for_stats = jnp.min(q_preds_final, axis=0)
    v_min = q_for_stats.min()
    v_max = q_for_stats.max()

    return (
        agent_state,
        frozen_expert_params,
        obs_batched,
        action_batched,
        MCPretrainAux(
            initial_loss=initial_loss,
            final_loss=final_loss,
            q_expert_mean=q_for_stats.mean(),
            q_expert_min=q_for_stats.min(),
            q_expert_max=q_for_stats.max(),
            v_min=v_min,
            v_max=v_max,
        ),
        expert_critic_state,
    )


@partial(jax.jit, static_argnames=["n_steps", "lr_scale"])
def pretrain_critic_online_light(
    agent_state,
    obs_batched: jax.Array,
    action_batched: jax.Array,
    n_steps: int = 500,
    lr_scale: float = 0.1,
):
    """Weak supervised nudge of the online critic toward φ*.

    Goal: reduce seed-to-seed variance at initialization.
    NOT a full MC pretrain — just reduces starting point spread.
    Target network stays random and untouched.
    """

    def light_loss_fn(critic_params, obs, actions):
        v_expert = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=phi_star_critic_state(agent_state.critic_state),
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([obs, actions], axis=-1),
                ),
                axis=0,
            )
        )
        q_online = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, actions], axis=-1),
        )
        return jnp.mean((q_online - v_expert) ** 2)

    def step(carry, batch):
        agent_state = carry
        obs_b, action_b = batch
        loss, grads = jax.value_and_grad(light_loss_fn)(
            agent_state.critic_state.params, obs_b, action_b
        )
        grads = jax.tree.map(lambda g: g * lr_scale, grads)
        new_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
        return agent_state.replace(critic_state=new_critic_state), loss

    agent_state, _ = jax.lax.scan(
        step, agent_state, (obs_batched[:n_steps], action_batched[:n_steps])
    )
    return agent_state


def refresh_phi_star(
    agent_state,
    buffer: BufferType,
    phi_refresh_steps: int,
    gamma: float,
    reward_scale: float,
    ext: "PhiRefresh",
) -> Any:
    """Periodic self-consistent φ* refresh using expert-flagged buffer transitions.
    ``ext``'s expert gives the action at s' (expert_next_actions).

    Target: r + γ * min_k Q_φ*(s′, π*(s′))  — φ* supervises its own bootstraps.
    Non-expert transitions are masked out, so the gradient comes only from
    (s, a_expert, r, s') rows where EDGE fired the expert action.
    """
    buffer_state = agent_state.collector_state.buffer_state
    reward_stats = reward_statistics(agent_state.collector_state.env_state)
    expert_critic_state = agent_state.expert_critic_state

    # Three keys: the first once drew a diagnostic batch; splitting three
    # keeps the refresh's key and the agent's stream unchanged.
    _, refresh_key, new_rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=new_rng)

    def refresh_step(carry, _):
        expert_critic_state, step_key = carry
        sample_key, step_key = jax.random.split(step_key)

        batch = get_batch_from_buffer(buffer, buffer_state, sample_key, reward_stats)
        obs, next_obs, actions = batch.obs, batch.next_obs, batch.action

        expert_mask = batch.is_expert[..., 0]
        rewards = batch.reward * reward_scale
        dones = batch.terminated.astype(jnp.float32)

        # The expert's action at s': stored (the agent's expert, from its
        # state), or its call on the raw next observations.
        stored = batch.next_a_expert if ext.reads_stored_expert else None
        a_expert_next = expert_next_actions(ext, batch.raw_next_obs, stored)
        q_next = predict_value(
            critic_state=expert_critic_state,
            critic_params=expert_critic_state.target_params,
            x=jnp.concatenate([next_obs, a_expert_next], axis=-1),
        )
        target = jax.lax.stop_gradient(
            rewards + gamma * (1.0 - dones) * jnp.min(q_next, axis=0)
        )

        def loss_fn(params):
            q_preds = predict_value(
                critic_state=expert_critic_state,
                critic_params=params,
                x=jnp.concatenate([obs, actions], axis=-1),
            )
            mse_per = jnp.mean((q_preds - target) ** 2, axis=(0, 2))
            n_expert = expert_mask.sum() + 1e-6
            return (mse_per * expert_mask).sum() / n_expert

        _, grads = jax.value_and_grad(loss_fn)(expert_critic_state.params)
        new_expert_critic_state = expert_critic_state.apply_gradients(grads=grads)
        return (new_expert_critic_state, step_key), None

    (new_expert_critic_state, _), _ = jax.lax.scan(
        refresh_step, (expert_critic_state, refresh_key), None, length=phi_refresh_steps
    )
    new_expert_critic_state = new_expert_critic_state.soft_update(tau=1.0)
    return agent_state.replace(
        expert_critic_state=new_expert_critic_state,
        expert_critic_params=jax.lax.stop_gradient(new_expert_critic_state.params),
    )


@dataclass(frozen=True)
class CloningPretrain(Extension):
    """Behaviour cloning of the actor on an expert's rollouts, before training.

    Rolls ``expert_policy`` out for ``n_steps`` steps per env, observing as
    the agent does, then fits the actor to its actions on observations
    standardised with the dataset's statistics
    (:func:`ajax.agents.cloning.pre_train`: ``epochs`` passes in batches of
    ``batch_size``, at the actor's learning rate). The agent's observation
    statistics start from the dataset's. ``skip_actor`` leaves the actor as
    initialised; after cloning, ``reset_log_std`` puts its log-std (head
    or parameter) back at its initial values and ``reset_actor_head`` also
    its output head (the mean layer, or a PID actor's gains). Evaluations
    log the mean loss of cloning's first and last epoch
    (``cloning/first_loss``, ``cloning/last_loss``).

    The fields after ``reset_actor_head`` are the agent's context, filled by
    :meth:`bind_to_agent` (SAC, REDQ, TD3 and APO do).
    """

    expert_policy: Callable
    n_steps: int = 100_000
    epochs: int = 10
    batch_size: int = 64
    skip_actor: bool = False
    reset_log_std: bool = False
    reset_actor_head: bool = False
    # Agent context, bound by bind_to_agent.
    env_args: Any = None
    actor_optimizer_args: Any = None
    name: str = "cloning_pretrain"
    requires: ClassVar[frozenset[str]] = frozenset({"actor", "continuous_actions"})
    context: ClassVar[frozenset[str]] = frozenset({"actor_optimizer_args", "env_args"})
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def bind_to_agent(self, **agent_context: Any) -> "CloningPretrain":
        """Copy the agent's ``env_args`` (its observation features
        included) and ``actor_optimizer_args``. Raises ``ValueError`` when
        a feature reads the agent's expert and ``expert_policy`` is another
        expert, or when another pretrain-phase extension comes before it in
        the stack."""
        optimizer = agent_context.get("actor_optimizer_args")
        if self.actor_optimizer_args is not None or optimizer is None:
            return self
        # Cloning seeds the observation statistics the critic reads too: a
        # critic pretrained before it (MCPretrain's nudge, BellmanPretrain)
        # would then be queried on inputs it was not fitted on.
        stack = agent_context.get("extensions", ())
        ahead = next((stack[:i] for i, e in enumerate(stack) if e is self), ())
        if any("pretrain" in e.implemented_phases() for e in ahead):
            raise ValueError(
                "CloningPretrain seeds the observation statistics the critic"
                " reads: put it before the other pretraining extensions."
            )
        env_args = agent_context["env_args"]
        if reads_agent_expert(env_args) and self.expert_policy is not agent_context.get(
            "expert_policy"
        ):
            raise ValueError(
                "CloningPretrain clones on the agent's observations, whose"
                " features read its expert: give CloningPretrain the agent's"
                " expert_policy."
            )
        return dataclasses.replace(
            self, env_args=env_args, actor_optimizer_args=optimizer
        )

    def init_state(self, agent_state: Any, rng: jax.Array) -> PretrainLosses:
        del agent_state, rng
        return PretrainLosses.unknown()

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, PretrainLosses]:
        """Clone the actor on ``context.rng`` (:meth:`clone`) and keep its losses."""
        del ext_state
        return self.clone(agent_state, context.rng)

    def clone(self, agent_state: Any, key: jax.Array) -> tuple[Any, PretrainLosses]:
        """``agent_state`` with its actor cloned on rollouts drawn with
        ``key`` and its observation statistics seeded, and the mean loss of
        cloning's first and last epoch (NaN with ``skip_actor``); raises
        ``ValueError`` when the agent did not bind its context."""
        if self.actor_optimizer_args is None:
            raise ValueError(
                "CloningPretrain needs the agent's env and actor optimizer; the"
                " agent did not bind them (SAC, REDQ, TD3 and APO do)."
            )
        env = self.env_args.env
        dataset = collect_experience_from_expert_policy(
            self.expert_policy,
            rng=key,
            env_args=self.env_args,
            mode=self.env_args.mode,
            n_timesteps=self.n_steps,
        )
        actor_state, metrics, obs_mean, obs_std = pre_train(
            rng=key,
            actor_state=agent_state.actor_state,
            dataset=dataset,
            actor_lr=self.actor_optimizer_args.learning_rate,
            actor_epochs=self.epochs,
            actor_batch_size=self.batch_size,
            skip_actor=self.skip_actor,
            reset_log_std=self.reset_log_std,
            reset_actor_head=self.reset_actor_head,
            raw_columns=raw_obs_columns(env),
            **cloning_columns(self.env_args),
        )
        losses = (
            PretrainLosses.unknown()
            if self.skip_actor
            else _first_and_last(metrics["actor_loss"])
        )
        # The actor was just trained on inputs standardised with the
        # dataset's statistics: the agent's running observation statistics
        # start from them, so that get_pi / predict_value standardise its
        # inputs online as cloning did. An agent without running statistics
        # gets them here (online collection then updates them from this
        # baseline); without them the cloned actor would be queried on raw
        # observations.
        new_state = agent_state.replace(actor_state=actor_state)
        if self.skip_actor and agent_state.collector_state.obs_norm_info is None:
            return new_state, losses  # no actor cloned: nothing to keep consistent
        # Agent-side statistics have a leading axis of size 1
        # (init_agent_obs_norm).
        n_samples = float(dataset.obs.reshape(-1, dataset.obs.shape[-1]).shape[0])
        var = (obs_std**2).reshape(1, -1)
        seeded = NormalizationInfo(
            count=jnp.full((1, 1), n_samples),
            mean=obs_mean.reshape(1, -1),
            mean_2=var * n_samples,
            var=var,
            returns=None,
        )
        seeded_state = new_state.replace(
            collector_state=new_state.collector_state.replace(obs_norm_info=seeded),
            actor_state=new_state.actor_state.replace(obs_norm_info=seeded),
            critic_state=new_state.critic_state.replace(obs_norm_info=seeded),
        )
        return seeded_state, losses

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del agent_state, rng, context
        return {
            "cloning/first_loss": ext_state.first_loss,
            "cloning/last_loss": ext_state.last_loss,
        }


@dataclass(frozen=True)
class MCPretrain(Extension):
    """Monte-Carlo critic pre-training.

    Regresses a frozen expert critic ``φ*`` on unbiased MC returns from
    expert rollouts, persists ``φ*`` + the value range ``v_min/v_max`` on
    the agent state, and — when ``use_online_light`` is set — applies a
    weak supervised nudge of the online critic toward ``φ*`` to reduce
    seed-to-seed variance at initialisation.

    Building ``φ*`` needs the agent's env, critic network and optimiser
    config, ``gamma`` and ``reward_scale``: the fields after the
    hyperparameters, which the agent fills through :meth:`bind_to_agent`
    (only SAC does). ``φ*`` trains with that optimiser config as it is, so
    the agent's ``weight_decay`` (like its gradient clipping and learning-
    rate schedule) also decays ``φ*`` while it fits the MC returns.
    ``use_phi_refresh`` keeps ``φ*``'s optimiser state for a
    :class:`PhiRefresh` in the same stack. Evaluations log the value range
    (``mc_pretrain/v_min``, ``mc_pretrain/v_max``) and the fit
    (``mc_pretrain/first_loss``, ``mc_pretrain/last_loss``,
    ``mc_pretrain/q_expert_mean``: :class:`MCPretrainResults`).
    """

    expert_policy: Callable
    n_mc_steps: int = 10_000
    n_mc_episodes: int = 100
    n_steps: int = 5_000
    use_online_light: bool = True
    online_light_steps: int = 500
    online_light_lr_scale: float = 0.1
    # Agent context, bound by bind_to_agent.
    env_args: Any = None
    network_args: Any = None
    critic_optimizer_args: Any = None
    num_critics: int = 2
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    use_phi_refresh: bool = False
    name: str = "mc_pretrain"
    requires: ClassVar[frozenset[str]] = frozenset({"critic", "discounted"})
    provides: ClassVar[frozenset[str]] = frozenset({"phi_star"})
    context: ClassVar[frozenset[str]] = frozenset(
        {"critic_optimizer_args", "env_args", "gamma", "network_args", "reward_scale"}
    )
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def bind_to_agent(self, **agent_context: Any) -> "MCPretrain":
        """Copy the agent's context (``env_args``, ``network_args``,
        ``critic_optimizer_args``, ``num_critics``, ``gamma``,
        ``reward_scale``) onto a new instance, and keep
        ``φ*``'s optimiser when the stack (``extensions``) holds a
        :class:`PhiRefresh`."""
        env_args = agent_context.get("env_args", None)
        if self.env_args is not None or env_args is None:
            return self
        if reads_agent_expert(env_args) and self.expert_policy is not agent_context.get(
            "expert_policy"
        ):
            raise ValueError(
                "MCPretrain fits phi* on the agent's observations, whose features"
                " read its expert: its MC rollout must be that expert's. Give"
                " MCPretrain the agent's expert_policy."
            )
        return dataclasses.replace(
            self,
            env_args=env_args,
            network_args=agent_context.get("network_args"),
            critic_optimizer_args=agent_context.get("critic_optimizer_args"),
            num_critics=agent_context.get("num_critics", self.num_critics),
            gamma=agent_context.get("gamma"),
            reward_scale=agent_context.get("reward_scale"),
            use_phi_refresh=any(
                isinstance(e, PhiRefresh) for e in agent_context.get("extensions", ())
            ),
        )

    def init_state(self, agent_state: Any, rng: jax.Array) -> MCPretrainResults:
        del agent_state, rng
        return MCPretrainResults.unknown()

    def pretrain(
        self,
        agent_state: Any,
        ext_state: Any,
        context: ExtensionContext,
    ) -> tuple[Any, MCPretrainResults]:
        """Build the frozen expert critic ``φ*`` and persist it on ``agent_state``:
        a fresh critic regressed on unbiased MC returns sets
        ``expert_critic_params`` / ``expert_v_min`` / ``expert_v_max``,
        then (``use_online_light``) the online critic is nudged toward it.
        The fit's results are the extension's state.

        Raises ``ValueError`` when the agent did not bind its context.
        """
        del ext_state
        needed = ("env_args", "network_args", "critic_optimizer_args", "gamma")
        unbound = [f for f in (*needed, "reward_scale") if getattr(self, f) is None]
        if unbound:
            raise ValueError(
                f"MCPretrain needs the agent context {unbound} to build phi*;"
                " the agent did not bind it (only SAC supports MCPretrain)."
            )

        from ajax.networks.networks import get_initialized_critic

        rng = context.rng
        # The online critic's input: the observation and its features.
        expert_critic_state = get_initialized_critic(
            key=rng,
            env_config=self.env_args,
            critic_optimizer_config=self.critic_optimizer_args,
            network_config=self.network_args,
            num_critics=self.num_critics,
        )
        (
            agent_state,
            frozen_expert_params,
            mc_obs_batched,
            mc_action_batched,
            mc_aux,
            expert_critic_state_trained,
        ) = pretrain_critic_mc(
            agent_state=agent_state,
            expert_critic_state=expert_critic_state,
            expert_policy=self.expert_policy,
            mode=self.env_args.mode,
            env_args=self.env_args,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            n_mc_steps=self.n_mc_steps,
            n_mc_episodes=self.n_mc_episodes,
            n_steps=self.n_steps,
        )
        agent_state = agent_state.replace(
            expert_critic_params=frozen_expert_params,
            expert_v_min=mc_aux.v_min,
            expert_v_max=mc_aux.v_max,
            # Keep φ* optimizer state alive for periodic refresh (None when disabled)
            expert_critic_state=(
                expert_critic_state_trained if self.use_phi_refresh else None
            ),
        )
        jax.debug.print(
            "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
            "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
            i=mc_aux.initial_loss,
            f=mc_aux.final_loss,
            qm=mc_aux.q_expert_mean,
            qn=mc_aux.q_expert_min,
            qx=mc_aux.q_expert_max,
        )
        if self.use_online_light:
            agent_state = pretrain_critic_online_light(
                agent_state,
                mc_obs_batched,
                mc_action_batched,
                n_steps=self.online_light_steps,
                lr_scale=self.online_light_lr_scale,
            )
            jax.debug.print(
                "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                n=self.online_light_steps,
                s=self.online_light_lr_scale,
            )
        return agent_state, MCPretrainResults(
            first_loss=mc_aux.initial_loss,
            last_loss=mc_aux.final_loss,
            q_expert_mean=mc_aux.q_expert_mean,
        )

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del rng, context
        return {
            "mc_pretrain/v_min": agent_state.expert_v_min,
            "mc_pretrain/v_max": agent_state.expert_v_max,
            "mc_pretrain/first_loss": ext_state.first_loss,
            "mc_pretrain/last_loss": ext_state.last_loss,
            "mc_pretrain/q_expert_mean": ext_state.q_expert_mean,
        }


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "gamma",
        "reward_scale",
        "n_steps",
        "buffer",
        "update_value_fn",
        "update_target_fn",
    ],
)
def pretrain_critic_bellman(
    agent_state,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    buffer: BufferType,
    n_steps: int,
    update_value_fn: Callable,
    update_target_fn: Callable,
):
    """``n_steps`` Bellman critic updates (the agent's ``update_value_fn``)
    on replay batches, each followed by a target update at tau 5e-4.
    Returns the agent state and each update's critic loss."""

    def critic_pretrain_step(carry, _):
        agent_state = carry
        sample_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        collector = agent_state.collector_state
        batch = get_batch_from_buffer(
            buffer,
            collector.buffer_state,
            sample_key,
            reward_statistics(collector.env_state),
        )
        # The plain Bellman target: no extension's target or loss term.
        agent_state, aux, _ = update_value_fn(
            agent_state,
            batch,
            recurrent,
            gamma,
            ExtensionStack(),
            reward_scale=reward_scale,
        )
        agent_state = update_target_fn(agent_state, tau=5e-4)
        return agent_state, aux.critic_loss

    return jax.lax.scan(critic_pretrain_step, agent_state, None, length=n_steps)


@dataclass(frozen=True)
class BellmanPretrain(Extension):
    """Bellman-bootstrapped critic pre-training.

    ``n_steps`` critic updates on the replay buffer before training, which
    a seeder fills first (ExpertPrefill or PriorData, which pretrain before
    it). Bootstraps from an untrained critic, so the pre-trained values are
    biased (:class:`MCPretrain` regresses unbiased returns instead); kept
    for ablation parity.
    Evaluations log the critic loss of its first and last update
    (``bellman_pretrain/first_loss``, ``bellman_pretrain/last_loss``).

    The fields after ``n_steps`` are the agent's context, filled by
    :meth:`bind_to_agent`: a replay agent with a discounted Q critic and
    target critics gives its critic step
    (ajax.agents.families.ReplayAgent.extension_values). It needs the
    target critics, which each step's Polyak update moves.
    """

    n_steps: int = 5_000
    # Agent context, bound by bind_to_agent.
    buffer: Any = None
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    recurrent: bool = False
    update_value_fn: Optional[Callable] = None
    update_target_fn: Optional[Callable] = None
    name: str = "bellman_pretrain"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"seeded_replay", "critic", "replay", "target_critic"}
    )
    context: ClassVar[frozenset[str]] = frozenset(
        {
            "buffer",
            "gamma",
            "recurrent",
            "reward_scale",
            "update_target_fn",
            "update_value_fn",
        }
    )

    def bind_to_agent(self, **agent_context: Any) -> "BellmanPretrain":
        """Copy the agent's buffer, ``gamma``, ``reward_scale``, ``recurrent``
        and critic updates (``update_value_fn``, ``update_target_fn``). It
        requires a seeded replay: without a seeder the replay is empty when
        it runs."""
        if self.update_value_fn is not None or "update_value_fn" not in agent_context:
            return self
        return dataclasses.replace(
            self,
            buffer=agent_context["buffer"],
            gamma=agent_context["gamma"],
            reward_scale=agent_context["reward_scale"],
            recurrent=agent_context["recurrent"],
            update_value_fn=agent_context["update_value_fn"],
            update_target_fn=agent_context["update_target_fn"],
        )

    def init_state(self, agent_state: Any, rng: jax.Array) -> PretrainLosses:
        del agent_state, rng
        return PretrainLosses.unknown()

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, PretrainLosses]:
        """Pre-train the critic and keep its losses; raises ``ValueError``
        when unbound."""
        del ext_state, context
        if self.update_value_fn is None:
            raise ValueError(
                "BellmanPretrain needs the agent's buffer and critic updates;"
                " the agent did not bind them."
            )
        agent_state, losses = pretrain_critic_bellman(
            agent_state,
            recurrent=self.recurrent,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            buffer=self.buffer,
            n_steps=self.n_steps,
            update_value_fn=self.update_value_fn,
            update_target_fn=self.update_target_fn,
        )
        jax.debug.print("[Bellman pretrain] done ({n} steps)", n=self.n_steps)
        return agent_state, _first_and_last(losses)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del agent_state, rng, context
        return {
            "bellman_pretrain/first_loss": ext_state.first_loss,
            "bellman_pretrain/last_loss": ext_state.last_loss,
        }


@dataclass(frozen=True)
class PhiRefresh(Extension):
    """Periodic self-consistent refresh of the frozen expert critic.

    Every ``interval`` steps, run ``steps`` self-consistent Bellman
    updates on expert-flagged buffer transitions, keeping ``φ*`` aligned
    with the live data distribution.
    Requires :class:`MCPretrain` (which creates the refreshable ``φ*``;
    refused without it).

    ``buffer``, ``gamma`` and ``reward_scale`` come from the agent through
    :meth:`bind_to_agent` (only SAC binds them); unbound, ``post_update``
    raises ``ValueError``.
    """

    expert_policy: Callable
    interval: int = 500
    steps: int = 20
    buffer: Any = None
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "phi_refresh"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"discounted", "phi_star", "q_critic", "replay"}
    )
    context: ClassVar[frozenset[str]] = frozenset({"buffer", "gamma", "reward_scale"})
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def bind_to_agent(self, **agent_context: Any) -> "PhiRefresh":
        """Copy the agent's ``buffer``, ``gamma`` and ``reward_scale``, and
        whether its expert is the agent's (:func:`reads_stored_expert_actions`)."""
        buffer = agent_context.get("buffer", None)
        gamma = agent_context.get("gamma", None)
        reward_scale = agent_context.get("reward_scale", None)
        if self.buffer is not None or buffer is None:
            return self
        return dataclasses.replace(
            self,
            buffer=buffer,
            gamma=gamma,
            reward_scale=reward_scale,
            reads_stored_expert=reads_stored_expert_actions(self, agent_context),
        )

    def post_update(
        self,
        agent_state: Any,
        ext_state: Any,
        context: ExtensionContext,
    ) -> tuple[Any, Any]:
        del context
        if self.buffer is None or self.gamma is None or self.reward_scale is None:
            raise ValueError(
                "PhiRefresh needs the agent's buffer, gamma and reward_scale to"
                " refresh phi*; the agent did not bind them (only SAC supports"
                " PhiRefresh)."
            )
        interval = self.interval
        steps = self.steps
        gamma = self.gamma
        reward_scale = self.reward_scale
        buffer = self.buffer

        def _do_refresh(s: Any) -> Any:
            return refresh_phi_star(
                s,
                buffer,
                steps,
                gamma,
                reward_scale,
                self,
            )

        new_state = jax.lax.cond(
            agent_state.collector_state.timestep % interval == 0,
            _do_refresh,
            lambda s: s,
            operand=agent_state,
        )
        return new_state, ext_state


def collect_and_store_expert_transitions(
    expert_policy: Callable,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    buffer_state: Any,
    rng: jax.Array,
    n_steps: int,
) -> Any:
    """Collect n_steps expert steps per env (collect_prefill_transitions)
    and store them in the replay (store_expert_transitions)."""
    transitions = collect_prefill_transitions(expert_policy, env_args, rng, n_steps)
    return store_expert_transitions(transitions, env_args, buffer, buffer_state)


def collect_prefill_transitions(
    expert_policy: Callable, env_args: EnvironmentConfig, rng: jax.Array, n_steps: int
) -> Transition:
    """n_steps expert steps per env, as the replay keeps them: with the
    observations' features (the training progress holds 0 before
    training)."""
    # The next observations the live collector stores: after a termination
    # the reset one, which ASAC's differential target continues from.
    return collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=env_args.mode,
        env_args=env_args,
        n_timesteps=n_steps,
        bootstrap_next_obs=True,
    )


def store_expert_transitions(
    transitions: Transition,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    buffer_state: Any,
) -> Any:
    """Store the expert's transitions, time along their first axis, in the
    replay row by row as the live collector does (replay_row): one step of
    every env per add (the buffer's add batch is the env axis), with the
    expert's actions at s and s' as a_expert and next_a_expert, and
    is_expert 1."""
    schema = buffer_state.experience
    if "actor_carry" in schema:
        raise ValueError(
            "The expert prefill cannot store a recurrent actor's carry"
            " (actor_carry, stored_state=True): that carry exists only along"
            " the agent's own rollouts."
        )
    # A row stores what collection stores when the expert acts. Under an
    # action transform, the policy's action is the one at which it executes
    # the expert's own (policy_action_for_expert); gain mode stores it as
    # the action too, residual RL stores the executed action, the expert's.
    transform = next(iter(env_args.transforms), None)
    stores_choice = (
        transform is not None and "actuate" in transform.implemented_phases()
    )
    choose_for_expert = None
    if transform is not None and (stores_choice or "policy_action" in schema):
        choose_for_expert = transform.policy_action_for_expert

    def add_step(buffer_state: Any, step: Any) -> tuple[Any, None]:
        chosen = step.action
        if choose_for_expert is not None:
            # Shaped like the policy's actions at this step.
            policy_actions = jnp.zeros(
                (*step.reward.shape[:-1], policy_action_dim(env_args))
            )
            chosen = choose_for_expert(policy_actions)
        row = replay_row(
            schema,
            obs=step.obs,
            action=chosen if stores_choice else step.action,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            next_obs=step.next_obs,
            raw_obs=step.raw_obs,
            is_expert=jnp.ones_like(step.reward),
            a_expert=step.a_expert,
            next_a_expert=step.next_a_expert,
            raw_next_obs=step.raw_next_obs,
            policy_action=chosen,
        )
        # The replay's dtypes: the expert collector keeps the env's.
        row = {name: value.astype(schema[name].dtype) for name, value in row.items()}
        return buffer.add(buffer_state, row), None

    buffer_state, _ = jax.lax.scan(add_step, buffer_state, transitions)
    return buffer_state


#: The prefill's own stream off the pretrain key. CloningPretrain and
#: MCPretrain roll the expert out on split(key, n), and under the
#: partitionable threefry split(key, n)[i] equals fold_in(key, i): a small
#: tag would reuse one of their keys, a 32-bit one no realistic n reaches.
EXPERT_PREFILL_KEY_TAG = 0x70726566


@dataclass(frozen=True)
class ExpertPrefill(Extension):
    """The agent's expert acts n_steps steps in every env before training,
    and its transitions go to the replay as collection writes its own
    (store_expert_transitions): with the agent's features at
    training progress 0, is_expert 1, and under an action transform the
    policy action that executes the expert's own. The rows count once in
    the observation and reward normalisers, as PriorData's and collection's
    own do (count_in_normalisers), and in no timestep, so learning_starts
    counts live steps.

    It needs the agent's expert, a replay and an update that stays correct
    on actions the policy did not choose: SAC, REDQ, ASAC, TQC, TD3, CrossQ
    and IQL take it. It runs before the other pretraining, which may read
    its rows (ExtensionStack.apply_pretrain)."""

    n_steps: int
    # The agent's expert, in agent units, bound by bind_to_agent.
    expert: Any = None
    name: str = "expert_prefill"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "off_policy", "replay"}
    )
    provides: ClassVar[frozenset[str]] = frozenset({"seeded_replay"})
    context: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    pretrain_context: ClassVar[frozenset[str]] = frozenset({"buffer", "env_args"})

    def bind_to_agent(self, **agent_context: Any) -> "ExpertPrefill":
        if self.expert is not None or "expert_policy" not in agent_context:
            return self
        return dataclasses.replace(self, expert=agent_context["expert_policy"])

    def replay_rows(self, n_envs: int) -> int:
        return self.n_steps * n_envs

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        if self.expert is None:
            raise ValueError(
                "ExpertPrefill rolls out the agent's expert_policy, which the"
                " agent did not bind."
            )
        key = jax.random.fold_in(context.rng, EXPERT_PREFILL_KEY_TAG)
        env_args = context.env_args
        transitions = collect_prefill_transitions(
            self.expert, env_args, key, self.n_steps
        )
        collector = agent_state.collector_state
        buffer_state = store_expert_transitions(
            transitions, env_args, context.buffer, collector.buffer_state
        )
        collector = collector.replace(buffer_state=buffer_state)
        agent_state = agent_state.replace(collector_state=collector)
        agent_state = count_in_normalisers(
            agent_state,
            transitions.obs,
            transitions.reward,
            transitions.terminated,
            transitions.truncated,
            env_args,
        )
        return agent_state, ext_state


@dataclass(frozen=True)
class PriorData(Extension):
    """Prior data in the replay, and the agent's own updates on it before
    acting (offline-to-online RL).

    Adds the dataset's rows to the agent's replay, where they stay
    (ajax.buffers.dataset.fill_replay), and to its observation and reward
    normalisers, each row counted once as collection counts its own; then
    runs offline_updates of the agent's update on that replay: its critic,
    actor and target steps and its extensions' loss terms, but no
    post_update between them. offline_updates=0 is RLPD's prior data in the
    replay (Ball et al. 2023) without its symmetric sampling. Ajax's warm-up
    acts uniformly: pair it with learning_starts=0.

    Needs the replay and update an off-policy loop hands the pretrain phase
    (pretrain_context): SAC, TD3, REDQ, ASAC, TQC, CrossQ, IQL and DQN take
    it, every other agent refuses it when built.
    """

    dataset: TransitionDataset
    offline_updates: int = 0
    name: str = "prior_data"
    requires: ClassVar[frozenset[str]] = frozenset({"off_policy", "replay"})
    provides: ClassVar[frozenset[str]] = frozenset({"seeded_replay"})
    pretrain_context: ClassVar[frozenset[str]] = frozenset({"buffer", "update"})

    def replay_rows(self, n_envs: int) -> int:
        del n_envs
        return len(self.dataset.reward)

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        agent_update = context.update
        if context.buffer is None or agent_update is None:
            raise ValueError(
                "PriorData adds its rows to the agent's replay: give it to an"
                " agent replaying transitions (SAC, TD3, REDQ, ASAC, TQC,"
                " CrossQ, IQL, DQN)."
            )
        agent_state = seed_replay(
            agent_state, context.buffer, self.dataset, context.env_args
        )

        def update(agent_state: Any, _: None) -> tuple[Any, None]:
            return agent_update(agent_state, None)[0], None

        # The scan's carry holds, from the first update, the slot an update
        # keeping its actor's last metrics writes.
        agent_state = with_actor_metrics_slot(
            agent_state, lambda state: agent_update(state, None)[0]
        )
        agent_state, _ = jax.lax.scan(
            update, agent_state, None, length=self.offline_updates
        )
        return agent_state, ext_state


__all__ = [
    "BellmanPretrain",
    "CloningPretrain",
    "ExpertPrefill",
    "MCPretrain",
    "PhiRefresh",
    "PriorData",
]
