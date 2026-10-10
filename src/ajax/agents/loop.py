"""The training loop the actor-critic agents share.

An agent's ``make_train`` builds a :class:`TrainLoop` and hands it what
makes the agent its algorithm: how to initialise its state and how to
update it. The loop owns everything else, the same for every agent::

    fresh run:  state = init(key, pretrain_key), then every extension's
                initial state and one-shot pretraining
    resumed:    state = the checkpointed state
    iterate:    collect one step per env (off-policy) or a rollout (on-policy)
                if timestep >= learning_starts (on-policy: always):
                    state, aux = update(state, experience)
                    fold the extensions' post_update
                else:
                    aux = the update's metrics, filled with NaN
                evaluate and log every log_frequency steps

The result is :func:`ajax.perf_utils.build_resumable_train`'s jitted
``train(key, index, initial_state, resume_from_state)``, which
:meth:`ajax.agents.base.ActorCritic.train` vmaps over seeds. It returns
``(state, None)`` without a logging config. With one it evaluates every
``log_frequency`` steps, whether or not a backend (W&B, TensorBoard) records
the logs, and returns ``(state, evaluations)``: every logged key's values at
the evaluations, in order, padded to the most a call makes with -1 (integer
leaves, the timestep) and NaN. They are written into rows the scan carries,
not stacked per iteration, which would cost ``seeds x iterations`` memory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.environments.interaction import (
    collect_experience,
    preallocate_last_rollout,
    should_use_uniform_sampling,
)
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import Extension, ExtensionStack
from ajax.log import compose_eval_metrics, evaluate_and_log, unevaluated
from ajax.logging.wandb_logging import LoggingConfig, start_async_logging, vmap_log
from ajax.perf_utils import build_resumable_train
from ajax.state import EnvironmentConfig
from ajax.utils import fill_with_nan

#: ``init(key, pretrain_key) -> agent_state``: a fresh agent state. Any
#: one-shot pretraining (behaviour cloning) draws on ``pretrain_key``, from
#: which the loop also derives the extensions' keys.
Init = Callable[[jax.Array, jax.Array], Any]
#: ``update(agent_state, experience) -> (agent_state, aux)``: one update on
#: the experience just collected (a replay agent samples its buffer instead).
Update = Callable[[Any, Any], tuple[Any, Any]]
#: ``update(agent_state, rollout, start) -> (agent_state, aux)``: one update
#: on the rollout just collected from the state ``start`` (a recurrent agent
#: replays the rollout from the carries it started with).
RolloutUpdate = Callable[[Any, Any, Any], tuple[Any, Any]]
#: ``iteration(agent_state, index) -> (agent_state, metrics)``: one scan
#: iteration; ``metrics`` are what it logged (the sentinel when it did not).
Iteration = Callable[[Any, Any], tuple[Any, dict]]


def record(rows: dict, count: jax.Array, metrics: dict) -> tuple[dict, jax.Array]:
    """Write ``metrics`` into row ``count`` of ``rows`` when they are an
    evaluation (their timestep is not the -1 sentinel), else nothing.

    Returns the rows and the count of evaluations written.
    """
    past_end = jax.tree.leaves(rows)[0].shape[0]
    if not past_end:  # a call too short to evaluate
        return rows, count
    evaluated = metrics["timestep"] >= 0
    row = jnp.where(evaluated, count, past_end)  # out of bounds: dropped
    rows = jax.tree.map(lambda r, m: r.at[row].set(m, mode="drop"), rows, metrics)
    return rows, count + evaluated


def gradient_step(train_state: Any, loss_fn: Callable) -> tuple[Any, Any]:
    """One optimiser step down ``loss_fn(params) -> (loss, aux)``.

    Returns the stepped train state and the loss's ``aux``.
    """
    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=grads), aux


def critic_step(
    agent_state: Any,
    batch: Any,
    target_q: jax.Array,
    value_loss: Callable[[Any, jax.Array], tuple[jax.Array, Any]],
    extension_stack: ExtensionStack,
    key: jax.Array,
    total_timesteps: int,
    *,
    rewards: jax.Array,
    gamma: Optional[float],
    reward_scale: float,
) -> tuple[Any, Any]:
    """A replay agent's critic step on ``batch`` (a :class:`Transition`).

    Folds the extensions' ``on_target`` into ``target_q`` (held constant),
    then steps the critic down ``value_loss(params, target_q) -> (loss,
    aux)`` plus their ``critic_loss`` terms. ``rewards``, ``gamma`` (None
    for an average-reward agent) and ``reward_scale`` are what the target
    used. Returns the stepped critic state and ``aux``.
    """
    step = agent_state.collector_state.timestep
    target_batch = {
        "observations": batch.obs,
        "actions": batch.action,
        "next_observations": batch.next_obs,
        "rewards": rewards,
        "dones": jnp.logical_or(batch.terminated, batch.truncated),
        "gamma": gamma,
        "reward_scale": reward_scale,
    }
    target_q = jax.lax.stop_gradient(
        extension_stack.fold_on_target(
            agent_state, target_batch, target_q, step, key, total_timesteps
        )
    )
    critic_state = agent_state.critic_state

    def loss_fn(params: Any) -> tuple[jax.Array, Any]:
        loss, aux = value_loss(params, target_q)
        loss_batch = {
            "observations": batch.obs,
            "actions": batch.action,
            "critic_params": params,
            "critic_state": critic_state,
        }
        extra = extension_stack.fold_critic_loss(
            agent_state, loss_batch, step, key, total_timesteps
        )
        return loss + extra, aux

    return gradient_step(critic_state, loss_fn)


@dataclasses.dataclass(frozen=True, eq=False)
class TrainLoop:
    """One run's environment, budget, extensions and logging (compared and
    hashed by identity: the environment's parameters are arrays)."""

    env_args: EnvironmentConfig
    total_timesteps: int
    num_episode_test: int
    stack: ExtensionStack
    log: bool
    log_fn: Callable
    log_frequency: Optional[int]

    @classmethod
    def create(
        cls,
        env_args: EnvironmentConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Optional[LoggingConfig] = None,
        extensions: Sequence[Extension] = (),
    ) -> TrainLoop:
        """The loop of one ``make_train`` call; starts the logging worker
        when ``logging_config`` names a backend."""
        log_frequency = None
        if logging_config is not None:
            if logging_config.backend:
                start_async_logging()
            log_frequency = logging_config.log_frequency
        return cls(
            env_args=env_args,
            total_timesteps=total_timesteps,
            num_episode_test=num_episode_test,
            stack=ExtensionStack(extensions),
            log=logging_config is not None,
            log_fn=partial(vmap_log, run_ids=run_ids),
            log_frequency=log_frequency,
        )

    @property
    def mode(self) -> str:
        return "gymnax" if check_env_is_gymnax(self.env_args.env) else "brax"

    def n_rollouts(self, n_steps: int) -> int:
        """The iterations of :meth:`on_policy` for ``n_steps`` rollouts: the
        budget in whole rollouts, plus one (as PPO always ran)."""
        return self.total_timesteps // (self.env_args.n_envs * n_steps) + 1

    @property
    def n_evaluations(self) -> int:
        """The most evaluations one call makes: ``ajax.log``'s gate counts
        each one as a log point, so the k-th comes at timestep ``k *
        log_frequency`` or later, and none comes past the budget."""
        if not self.log or not self.log_frequency:
            return 0
        return self.total_timesteps // self.log_frequency

    # -- the steps of an iteration -----------------------------------------
    def collect_kwargs(self, recurrent: bool, **kwargs: Any) -> dict:
        """:func:`collect_experience`'s arguments for this run."""
        common = {"recurrent": recurrent, "mode": self.mode, "env_args": self.env_args}
        return common | kwargs

    def post_update(self, agent_state: Any) -> Any:
        """Fold the extensions' ``post_update`` on a fresh key (no key is
        drawn without extensions)."""
        if not self.stack:
            return agent_state
        key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        return self.stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            key,
            self.total_timesteps,
        )

    def maybe_update(
        self,
        agent_state: Any,
        learning_starts: int,
        update: Callable[[Any], tuple[Any, Any]],
        aux_cls: type,
    ) -> tuple[Any, Any]:
        """``update`` and fold ``post_update`` once ``learning_starts`` steps
        are collected; before, the metrics are ``aux_cls`` filled with NaN
        (which the logger drops)."""

        def do_update(agent_state: Any) -> tuple[Any, Any]:
            agent_state, aux = update(agent_state)
            # One (1,)-shaped leaf per metric: the metric-flattening contract.
            aux = jax.tree.map(lambda x: jnp.reshape(x, (1,)), aux)
            return self.post_update(agent_state), aux

        def skip_update(agent_state: Any) -> tuple[Any, Any]:
            return agent_state, fill_with_nan(aux_cls)

        ready = agent_state.collector_state.timestep >= learning_starts
        return jax.lax.cond(ready, do_update, skip_update, agent_state)

    def evaluate_and_log(
        self,
        agent_state: Any,
        aux: Any,
        index: Any,
        recurrent: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        """Evaluate and log every ``log_frequency`` steps
        (:func:`ajax.log.evaluate_and_log`), the extensions' eval metrics
        included; returns the state and the metrics."""
        return evaluate_and_log(
            agent_state,
            aux,
            index,
            self.mode,
            self.env_args,
            self.num_episode_test,
            recurrent,
            self.log,
            self.log_fn,
            self.log_frequency,
            self.total_timesteps,
            extra_eval_metrics=compose_eval_metrics(
                None, self.stack, self.total_timesteps
            ),
            **kwargs,
        )

    # -- the train function ------------------------------------------------
    def train(
        self,
        init: Init,
        iteration: Iteration,
        num_updates: int,
        last_rollout: Optional[tuple[int, dict]] = None,
    ) -> Callable:
        """The resumable train function: ``init`` (fresh runs only), then
        ``num_updates`` scan iterations ``iteration(agent_state, index)``,
        recording the evaluations when logging (see the module docstring).

        ``last_rollout``, ``(length, collect_kwargs)``, pre-allocates
        ``agent_state.last_rollout`` for an agent exposing its rollouts.
        """

        def init_fn(key: jax.Array, index: Any) -> Any:
            del index
            init_key, pretrain_key = jax.random.split(key)
            agent_state = init(init_key, pretrain_key)
            agent_state = self.stack.fold_init(
                agent_state, pretrain_key, self.total_timesteps
            )
            if last_rollout is not None:
                length, collect_kwargs = last_rollout
                agent_state = preallocate_last_rollout(
                    agent_state, length, **collect_kwargs
                )
            return agent_state

        def make_scan_fn(_state: Any, _resume: bool, _key: Any, index: Any) -> Any:
            if not self.log:
                return lambda agent_state, _: (iteration(agent_state, index)[0], None)

            def body(carry: Any, _: Any) -> tuple[Any, None]:
                agent_state, (rows, count) = carry
                agent_state, metrics = iteration(agent_state, index)
                return (agent_state, record(rows, count, metrics)), None

            return body

        def empty_rows(agent_state: Any, index: Any) -> tuple[dict, jax.Array]:
            shapes = jax.eval_shape(lambda s, i: iteration(s, i)[1], agent_state, index)
            return unevaluated(shapes, (self.n_evaluations,)), jnp.asarray(0)

        train = build_resumable_train(
            init_fn=init_fn,
            make_scan_fn=make_scan_fn,
            num_updates=num_updates,
            carry_out=empty_rows if self.log else None,
        )
        if not self.log:
            return train

        def train_with_evaluations(*args: Any, **kwargs: Any) -> tuple[Any, dict]:
            agent_state, (rows, _) = train(*args, **kwargs)
            return agent_state, rows

        return train_with_evaluations

    def off_policy(
        self,
        init: Init,
        update: Update,
        aux_cls: type,
        learning_starts: int,
        *,
        recurrent: bool = False,
        expose_rollout: bool = False,
        after_collect: Optional[Callable[[Any, Any], Any]] = None,
        collect_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
    ) -> Callable:
        """Train on one environment step per env per iteration.

        Each iteration collects a step (uniform actions before
        ``learning_starts``), applies ``after_collect(agent_state,
        transition)`` when given, then from ``learning_starts`` runs
        ``update(agent_state, transition)``. ``expose_rollout`` keeps the
        step on ``agent_state.last_rollout`` as a ``T = 1`` rollout.
        """
        collect = self.collect_kwargs(recurrent, **(collect_kwargs or {}))

        def iteration(agent_state: Any, index: Any) -> tuple[Any, dict]:
            timestep = agent_state.collector_state.timestep
            uniform = should_use_uniform_sampling(timestep, learning_starts)
            agent_state, transition = collect_experience(
                agent_state, None, uniform=uniform, **collect
            )
            if expose_rollout:
                rollout = jax.tree.map(lambda x: x[None], transition)
                agent_state = agent_state.replace(last_rollout=rollout)
            if after_collect is not None:
                agent_state = after_collect(agent_state, transition)
            agent_state, aux = self.maybe_update(
                agent_state,
                learning_starts,
                lambda agent_state: update(agent_state, transition),
                aux_cls,
            )
            return self.evaluate_and_log(
                agent_state, aux, index, recurrent, **(eval_kwargs or {})
            )

        return self.train(
            init,
            iteration,
            self.total_timesteps // self.env_args.n_envs,
            last_rollout=(1, collect) if expose_rollout else None,
        )

    def on_policy(
        self,
        init: Init,
        update: RolloutUpdate,
        n_steps: int,
        *,
        recurrent: bool = False,
        expose_rollout: bool = False,
        collect_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
    ) -> Callable:
        """Train on an ``n_steps`` rollout per env per iteration.

        Each iteration collects the rollout from the state ``start``
        (:func:`collect_experience` with ``collect_kwargs``), then runs
        ``update(agent_state, rollout, start)`` and folds ``post_update``;
        ``expose_rollout`` keeps the rollout on ``agent_state.last_rollout``.
        The budget is :meth:`n_rollouts`.
        """
        collect = self.collect_kwargs(recurrent, **(collect_kwargs or {}))

        def iteration(start: Any, index: Any) -> tuple[Any, dict]:
            agent_state, rollout = jax.lax.scan(
                partial(collect_experience, **collect),
                start,
                xs=None,
                length=n_steps,
            )
            if expose_rollout:
                agent_state = agent_state.replace(last_rollout=rollout)
            agent_state, aux = update(agent_state, rollout, start)
            agent_state = self.post_update(agent_state)
            return self.evaluate_and_log(
                agent_state, aux, index, recurrent, **(eval_kwargs or {})
            )

        return self.train(
            init,
            iteration,
            self.n_rollouts(n_steps),
            last_rollout=(n_steps, collect) if expose_rollout else None,
        )


__all__ = ["TrainLoop", "critic_step", "gradient_step"]
