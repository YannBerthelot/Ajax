"""The training loop the agents share.

An agent's ``make_train`` builds a :class:`TrainLoop` and hands it what
makes the agent its algorithm: how to initialise its state and how to
update it. The loop owns everything else, the same for every agent::

    fresh run:  state = init(key), then every extension's initial state
                and one-shot pretraining
    resumed:    state = the checkpointed state
    iterate:    collect one step per env (off-policy) or a rollout (on-policy)
                if timestep >= learning_starts (on-policy: always):
                    state, aux = update(state, experience)
                    apply the extensions' post_update
                else:
                    aux = the update's metrics, filled with NaN
                evaluate and log every log_frequency steps

An off-policy agent that acts, trains and evaluates its own way (the world
models: a planner or a filtered latent acts through the row collector, a
train ratio schedules the updates) supplies those steps too, and the loop
runs them in the same order (:meth:`TrainLoop.off_policy`)::

    iterate:    state, experience = collect(state, tick)
                repeat n_updates(tick, state) times:
                    state, _ = update(state, experience)
                    apply the extensions' post_update
                evaluate and log every evaluation.every ticks

A replay agent given a dataset trains offline instead
(:meth:`TrainLoop.offline`)::

    fresh run:  the replay and the normalisers start from the dataset
    iterate:    state, aux = update(state, None)   # no environment step
                apply the extensions' post_update
                evaluate and log every log_frequency updates

The result is :func:`ajax.perf_utils.build_resumable_train`'s jitted
``train(key, index, initial_state, resume_from_state)``, which
:meth:`ajax.agents.base.ActorCritic.train` vmaps over seeds. It returns
``(state, None)`` without a logging config, the evaluation not even traced
(tracing alone rebuilds the evaluation env). With one it evaluates every
``log_frequency`` steps, whether or not a backend (W&B, TensorBoard) records
the logs, and returns ``(state, evaluations)``: every logged key's values at
the evaluations, in order, padded to the most a call makes with -1 (integer
leaves, the timestep) and NaN. They are written into rows the scan carries,
not stacked per iteration, which would cost ``seeds x iterations`` memory.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.agents.update_steps import apply_post_update, run_if
from ajax.buffers.dataset import TransitionDataset, fill_replay
from ajax.controller import Controller, make_run_controller, reads_progress
from ajax.environments.create import strip_ajax_wrappers
from ajax.environments.interaction import (
    collect_experience,
    expert_collect_kwargs,
    force_reset,
    preallocate_last_rollout,
    reward_statistics,
    should_use_uniform_sampling,
    with_reward_statistics,
)
from ajax.environments.obs_norm import update_obs_norm
from ajax.environments.utils import raw_obs_columns
from ajax.extensions.base import Extension, ExtensionStack
from ajax.log import (
    compose_eval_metrics,
    evaluate_and_log,
    maybe_eval_and_log,
    unevaluated,
)
from ajax.logging.wandb_logging import LoggingConfig, start_async_logging, vmap_log
from ajax.perf_utils import build_resumable_train
from ajax.state import EnvironmentConfig
from ajax.types import NormalizationInfo
from ajax.utils import online_normalize

#: The extension phases the loop applies for every agent: a fresh run's
#: one-shot pretrain, post_update after each update, eval_metrics at each
#: evaluation, observe wherever the shared collector and evaluation meet an
#: observation, action in the shared collector's action stage and
#: eval_action in the shared evaluation (an agent rolling out on its own
#: removes those three). An agent adds the phases its own update applies
#: (ActorCritic.supported_extension_phases).
LOOP_PHASES = frozenset(
    {"pretrain", "post_update", "eval_metrics", "observe", "action", "eval_action"}
)


#: ``init(key) -> agent_state``: a fresh agent state. One-shot pretraining
#: (behaviour cloning) is an extension's, on a key the loop splits off.
Init = Callable[[jax.Array], Any]
#: ``update(agent_state, experience) -> (agent_state, aux)``: one update on
#: the experience just collected (a replay agent samples its buffer instead).
Update = Callable[[Any, Any], tuple[Any, Any]]
#: update(agent_state, rollout, agent_state_at_rollout_start) -> (agent_state,
#: aux): one update on the rollout just collected from the agent state
#: agent_state_at_rollout_start (a recurrent agent replays the rollout from
#: the memories it started with).
RolloutUpdate = Callable[[Any, Any, Any], tuple[Any, Any]]
#: ``collect(agent_state, tick) -> (agent_state, experience)``: an agent's
#: own acting and storing of one step per env on the absolute tick ``tick``;
#: ``experience`` is what its updates read (a replay agent: the tick, which
#: locates the replay's newest rows).
Collect = Callable[[Any, Any], tuple[Any, Any]]
#: ``iteration(agent_state, index, tick) -> (agent_state, metrics)``: one
#: scan iteration, ``tick`` the absolute, unbatched scan index (offset on
#: resume by ``ActorCritic.resume_iteration_offset``); ``metrics`` are what
#: it logged (the sentinel when it did not; ``None`` without logging).
Iteration = Callable[[Any, Any, Any], tuple[Any, Optional[dict]]]


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """An agent's own evaluation, every ``every`` ticks (``None``: never).

    ``metrics(agent_state, aux)`` are its training metrics (``timestep``
    included), updated with its policy's ``evaluate(agent_state, key)``
    when it has one and the extensions' ``eval_metrics``;
    ``after_log(agent_state, logged)`` runs on every tick after the gate.
    """

    metrics: Callable[[Any, Any], dict]
    every: Optional[int]
    evaluate: Optional[Callable[[Any, jax.Array], dict]] = None
    after_log: Optional[Callable[[Any, jax.Array], Any]] = None

    def count(self, n_ticks: int) -> int:
        """The most evaluations ``n_ticks`` consecutive ticks make."""
        return -(-n_ticks // self.every) if self.every else 0


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


def fresh_state(
    init: Init,
    extensions: ExtensionStack,
    total_steps: int,
    key: Any,
    **context: Any,
) -> Any:
    """A fresh run's state: ``init(init_key)`` on ``split(key)``, then the
    extensions' initial states and one-shot pretraining on
    ``extension_key`` (:meth:`ExtensionStack.apply_init`; ``context``: the
    pretraining's static context fields)."""
    init_key, extension_key = jax.random.split(key)
    env_args = context.get("env_args")
    state = with_obs_norm(init(init_key), env_args)
    state = with_horizon(state, env_args, total_steps)
    return extensions.apply_init(state, extension_key, total_steps, **context)


def with_actor_metrics_slot(agent_state: Any, one_update: Callable[[Any], Any]) -> Any:
    """agent_state with the slot its update keeps the actor's last metrics
    in (BaseAgentState.last_actor_metrics, update_steps.run_actor_if):
    their structure, read from the trace of one_update (a state to the
    state after one update), NaN until the actor's first step, so the scan
    carry keeps one structure from the first update. A state that has the
    slot keeps it."""
    if agent_state.last_actor_metrics is not None:
        return agent_state
    kept = jax.eval_shape(one_update, agent_state).last_actor_metrics
    return agent_state.replace(last_actor_metrics=unevaluated(kept))


def with_horizon(agent_state: Any, env_args: Any, total_steps: int) -> Any:
    """A state whose collector keeps the run's horizon ``total_steps``,
    which a feature reading the training progress needs
    (:func:`ajax.controller.reads_progress`); a resumed run measures its
    progress against its own call's."""
    if env_args is None or not reads_progress(env_args):
        return agent_state
    horizon = jnp.asarray(total_steps, jnp.int32)
    collector = agent_state.collector_state.replace(max_timesteps=horizon)
    return agent_state.replace(collector_state=collector)


def with_obs_norm(agent_state: Any, env_args: Any = None) -> Any:
    """An agent normalising its own observations (the collector's
    ``obs_norm_info``) with those statistics mirrored on its networks, which
    apply them; collection refreshes the mirror at every step, and setting
    it at init keeps the scan carry's structure fixed. An agent storing
    normalised observations (``env_args.store_normalized_obs``) gets no
    mirror: its networks read them as stored."""
    collector = getattr(agent_state, "collector_state", None)
    stats = None if collector is None else collector.obs_norm_info
    if stats is None or getattr(env_args, "store_normalized_obs", False):
        return agent_state
    return agent_state.replace(
        actor_state=agent_state.actor_state.replace(obs_norm_info=stats),
        critic_state=agent_state.critic_state.replace(obs_norm_info=stats),
    )


def seed_replay(
    agent_state: Any, buffer: Any, data: TransitionDataset, env_args: EnvironmentConfig
) -> Any:
    """``agent_state`` with ``data``'s rows in its replay, where they stay
    (:func:`ajax.buffers.dataset.fill_replay`), and in the normalisers it
    has (count_in_normalisers)."""
    collector = agent_state.collector_state
    collector = collector.replace(
        buffer_state=fill_replay(buffer, collector.buffer_state, data)
    )
    agent_state = agent_state.replace(collector_state=collector)
    return count_in_normalisers(
        agent_state, data.obs, data.reward, data.terminated, data.truncated, env_args
    )


def count_in_normalisers(
    agent_state: Any,
    obs: Any,
    rewards: Any,
    terminated: Any,
    truncated: Any,
    env_args: EnvironmentConfig,
) -> Any:
    """The agent state with rows written to its replay before training
    counted in the normalisers it has, each row once, as collection counts
    its own: their observations in its own observation statistics, mirrored
    on its networks, and in the env's reward statistics the reward or, with
    the normaliser's gamma, the discounted return of the episode so far (its
    running return left as it is). Time runs along the first axis of the
    rewards and the episode ends, the envs along the others; the
    observations keep their columns last. A normaliser the agent lacks
    costs nothing."""
    collector = agent_state.collector_state
    env = env_args.env
    if collector.obs_norm_info is not None:
        obs = jnp.asarray(obs)
        _, obs_stats = update_obs_norm(
            obs.reshape(-1, obs.shape[-1]),
            collector.obs_norm_info,
            raw_obs_columns(env),
        )
        collector = collector.replace(obs_norm_info=obs_stats)
    stats = reward_statistics(collector.env_state)
    if stats is not None:
        gamma = strip_ajax_wrappers(env)[1].get("gamma") or 0.0
        ends = jnp.logical_or(jnp.asarray(terminated), jnp.asarray(truncated))

        # The reward normaliser's running return, as the env's normaliser
        # steps through the rows: restarted after each episode's end.
        def add_reward(running: jax.Array, row: tuple) -> tuple[jax.Array, jax.Array]:
            reward, end = row
            running = reward + gamma * running
            return jnp.where(end, 0.0, running), running

        rewards = jnp.asarray(rewards, jnp.float32)
        _, returns = jax.lax.scan(
            add_reward, jnp.zeros_like(rewards[0]), (rewards, ends)
        )
        returns = returns.reshape(-1, 1)
        _, count, mean, mean_2, var = online_normalize(
            returns,
            stats.count,
            stats.mean,
            stats.mean_2,
            shift=False,
            returns=returns,
        )
        stats = NormalizationInfo(var, count, mean, mean_2, stats.returns)
        env_state = with_reward_statistics(collector.env_state, stats)
        collector = collector.replace(_env_state=env_state)
    return with_obs_norm(agent_state.replace(collector_state=collector))


@dataclasses.dataclass(frozen=True, eq=False)
class TrainLoop:
    """One call's environment, budget, extensions and logging (compared and
    hashed by identity: the environment's parameters are arrays).

    A call trains from ``start_timestep`` (a resumed run's timestep, else 0)
    to ``end_timestep`` (``None``: the horizon); ``total_timesteps`` is the
    run's horizon: every schedule (the extensions' ``total_steps``, the
    training fraction, the logging's end) runs over the whole run, so a run
    split into calls follows the schedules of one call; the call's own
    iterations come from its :attr:`budget`.
    """

    env_args: EnvironmentConfig
    total_timesteps: int
    num_episode_test: int
    extensions: ExtensionStack
    log: bool
    log_fn: Callable
    log_frequency: Optional[int]
    start_timestep: int = 0
    end_timestep: Optional[int] = None
    #: The agent's ``expert_policy``, whose actions collection stores.
    expert_policy: Any = None
    #: A transition dataset to train on alone (:meth:`offline`).
    offline_data: Optional[TransitionDataset] = None
    #: Evaluate only in the last fifth of training (LoggingConfig.sweep, for
    #: hyperparameter sweeps).
    sweep: bool = False

    @classmethod
    def create(
        cls,
        env_args: EnvironmentConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Optional[LoggingConfig] = None,
        extensions: Sequence[Extension] = (),
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        expert_policy: Any = None,
        offline_data: Optional[TransitionDataset] = None,
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
            extensions=ExtensionStack(extensions),
            log=logging_config is not None,
            log_fn=partial(vmap_log, run_ids=run_ids),
            log_frequency=log_frequency,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            expert_policy=expert_policy,
            offline_data=offline_data,
            sweep=logging_config is not None and logging_config.sweep,
        )

    @property
    def mode(self) -> str:
        return self.env_args.mode

    @property
    def budget(self) -> int:
        """The timesteps this call trains for."""
        end = self.total_timesteps if self.end_timestep is None else self.end_timestep
        return end - self.start_timestep

    def warn_if_partial(self, per_iteration: int, unit: str) -> None:
        """Warn when the budget is not a whole number of iterations of
        ``per_iteration`` env steps (``unit`` names them): the iterations
        stop at a multiple of it, not at the call's end, so the run trains
        another number of steps than asked and an evaluation due at that end
        never happens (its row stays NaN)."""
        if self.budget % per_iteration:
            end = self.start_timestep + self.budget
            warnings.warn(
                f"n_timesteps={self.budget} is not a multiple of the"
                f" {per_iteration} env steps of one iteration ({unit}): the run"
                f" stops at a multiple of {per_iteration}, so an evaluation due"
                f" at timestep {end} does not happen (NaN)."
                f" Pass a multiple of {per_iteration}.",
                UserWarning,
                stacklevel=2,
            )

    def n_rollouts(self, n_steps: int, timesteps: Optional[int] = None) -> int:
        """The iterations of :meth:`on_policy` for ``n_steps`` rollouts over
        ``timesteps`` (``None``: the budget, warning when it is not whole
        rollouts): in whole rollouts, plus one (as PPO always ran)."""
        per_rollout = self.env_args.n_envs * n_steps
        if timesteps is None:
            self.warn_if_partial(per_rollout, "n_envs x n_steps")
            timesteps = self.budget
        return timesteps // per_rollout + 1

    @property
    def n_evaluations(self) -> int:
        """The most evaluations one call ending at its budget makes
        (:meth:`evaluations_until`)."""
        return self.evaluations_until(self.start_timestep + self.budget)

    def evaluations_until(self, timestep: int) -> int:
        """The most evaluations one call reaching ``timestep`` makes:
        ``ajax.log``'s gate counts each one as a log point, so the k-th of
        the run comes at timestep ``k * log_frequency`` or later, and none
        comes past the horizon."""
        if not self.log or not self.log_frequency:
            return 0
        f = self.log_frequency
        return min(self.total_timesteps, timestep) // f - self.start_timestep // f

    # -- the steps of an iteration -----------------------------------------
    def collect_kwargs(self, recurrent: bool, **kwargs: Any) -> dict:
        """collect_experience's arguments for this run. With the agent's
        expert, its action pipeline also stores the expert's actions from the
        expert's tracked state, the run's action extensions act on its
        result, gain mode actuates the chosen action through the expert, and
        the default pipeline runs the run's warm-up chooser
        (expert_collect_kwargs); without it, the action extensions acting
        with an expert of their own (IBRL)."""
        common = {"recurrent": recurrent, "mode": self.mode, "env_args": self.env_args}
        args = dict(common)
        args.update(kwargs)
        pipeline = kwargs.get("action_pipeline")
        n_envs = self.env_args.n_envs
        if self.expert_policy is None:
            stage = expert_collect_kwargs(
                None,
                pipeline,
                recurrent,
                n_envs,
                extensions=self.extensions,
                total_timesteps=self.total_timesteps,
            )
            args.update(stage)
            return args
        # The observation features read the expert the collector tracks.
        expert = {"expert": self.expert_policy} if self.env_args.observers else {}
        transform = next(iter(self.env_args.transforms), None)
        warmups = self.env_args.warmups
        args.update(
            expert_collect_kwargs(
                self.expert_policy,
                pipeline,
                recurrent,
                n_envs,
                transform,
                warmups,
                self.extensions,
                self.total_timesteps,
            )
        )
        args.update(expert)
        return args

    def post_update(self, agent_state: Any) -> Any:
        """:func:`apply_post_update` at the collector's timestep."""
        return apply_post_update(
            self.extensions,
            agent_state,
            agent_state.collector_state.timestep,
            self.total_timesteps,
        )

    def maybe_update(
        self,
        agent_state: Any,
        learning_starts: int,
        update: Callable[[Any], tuple[Any, Any]],
        timestep: Optional[jax.Array] = None,
    ) -> tuple[Any, Any]:
        """update and apply post_update once timestep, by default the steps
        collected, reaches learning_starts; before, the metrics have the
        update's structure, filled with NaN (which the logger drops)."""

        def do_update(agent_state: Any) -> tuple[Any, Any]:
            agent_state, aux = update(agent_state)
            # One (1,)-shaped leaf per metric: the metric-flattening contract.
            aux = jax.tree.map(lambda x: jnp.reshape(x, (1,)), aux)
            return self.post_update(agent_state), aux

        if timestep is None:
            timestep = agent_state.collector_state.timestep
        return run_if(timestep >= learning_starts, do_update, agent_state)

    def repeat_update(
        self, agent_state: Any, n: Any, update: Callable[[Any], tuple[Any, Any]]
    ) -> Any:
        """``n`` times ``update`` then ``post_update``; the updates keep what
        they log on the state. An ``n`` derived from unbatched values (the
        tick, static hyperparameters) keeps the loop one ``while`` under the
        seed ``vmap``; one read from the per-seed state runs it to the
        largest count, each seed's state frozen after its own."""
        return jax.lax.fori_loop(
            0,
            n,
            lambda _, agent_state: self.post_update(update(agent_state)[0]),
            agent_state,
        )

    def evaluate_and_log(
        self,
        agent_state: Any,
        aux: Any,
        index: Any,
        recurrent: bool = False,
        agent_metrics: Optional[Callable[[Any], dict]] = None,
        **kwargs: Any,
    ) -> tuple[Any, Optional[dict]]:
        """Evaluate and log every log_frequency steps
        (ajax.log.evaluate_and_log), the agent's family's agent_metrics and
        the extensions' eval metrics included; returns the state and the
        metrics (None without logging), only in the last fifth of training
        in a sweep. An agent storing normalised observations evaluates with
        them normalised by its running statistics, unless its own controller
        says how it acts."""
        controller = make_run_controller(
            self.env_args, kwargs.get("controller"), self.extensions
        )
        if controller != Controller():
            kwargs.update({"controller": controller})
        if self.sweep:
            kwargs.setdefault("sweep", True)
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
                self.extensions, self.total_timesteps, agent_metrics
            ),
            **kwargs,
        )

    def evaluate_every(
        self, agent_state: Any, aux: Any, index: Any, tick: Any, evaluation: Evaluation
    ) -> tuple[Any, Optional[dict]]:
        """An agent's own evaluation and logs on the ticks ``t`` with ``(t +
        1) % evaluation.every == 0`` (:func:`ajax.log.maybe_eval_and_log`,
        the extensions' eval metrics included); returns the state and the
        metrics (``None`` without logging)."""
        logs = agent_state.n_logs
        agent_state, metrics = maybe_eval_and_log(
            agent_state,
            aux,
            index,
            tick,
            metrics_fn=evaluation.metrics,
            evaluate_fn=evaluation.evaluate or (lambda *_: {}),
            extra_eval_metrics=compose_eval_metrics(
                self.extensions, self.total_timesteps
            ),
            log=self.log,
            log_fn=self.log_fn,
            every=evaluation.every,
        )
        if evaluation.after_log is not None:
            agent_state = evaluation.after_log(agent_state, agent_state.n_logs != logs)
        return agent_state, metrics

    # -- the train function ------------------------------------------------
    def continued(self, agent_state: Any) -> Any:
        """A resumed state, its training progress measured against the
        run's horizon (a fresh state is built with it)."""
        return with_horizon(agent_state, self.env_args, self.total_timesteps)

    def train(
        self,
        init: Init,
        iteration: Iteration,
        num_updates: int,
        last_rollout: Optional[tuple[int, dict]] = None,
        evaluations: Optional[int] = None,
        update: Optional[Update] = None,
        buffer: Any = None,
        one_update: Optional[Callable[[Any], Any]] = None,
        resume: Optional[Callable[[Any], Any]] = None,
        returns_iterations: bool = False,
    ) -> Callable:
        """The resumable train function: init (fresh runs only), then
        num_updates scan iterations iteration(agent_state, index, tick),
        recording the evaluations when logging (see the module docstring):
        n_evaluations rows, or evaluations.

        last_rollout, a (length, collect_kwargs) pair, pre-allocates
        agent_state.last_rollout for an agent exposing its rollouts.
        The extensions' pretraining gets the env, update and the replay
        buffer (ajax.extensions.base.ExtensionContext).
        one_update, a state to the state after one update, is given by an
        update keeping its actor's last metrics: a fresh or resumed state gets
        their slot (with_actor_metrics_slot). resume transforms a resumed
        state after continued (APG rebuilds its optimiser). With
        returns_iterations the train function returns every iteration's
        output, stacked, logging or not, instead of the rows.
        """
        rows = self.n_evaluations if evaluations is None else evaluations
        records = self.log and not returns_iterations

        def with_slot(agent_state: Any) -> Any:
            if one_update is None:
                return agent_state
            return with_actor_metrics_slot(agent_state, one_update)

        context = {"env_args": self.env_args, "update": update, "buffer": buffer}

        def init_fn(key: jax.Array, index: Any) -> Any:
            del index
            agent_state = fresh_state(
                init, self.extensions, self.total_timesteps, key, **context
            )
            if last_rollout is not None:
                length, collect_kwargs = last_rollout
                agent_state = preallocate_last_rollout(
                    agent_state, length, **collect_kwargs
                )
            return with_slot(agent_state)

        def make_scan_fn(_state: Any, _resume: bool, _key: Any, index: Any) -> Any:
            if returns_iterations:
                return lambda agent_state, tick: iteration(agent_state, index, tick)
            if not self.log:
                return lambda agent_state, tick: (
                    iteration(agent_state, index, tick)[0],
                    None,
                )

            def body(carry: Any, tick: Any) -> tuple[Any, None]:
                agent_state, (rows, count) = carry
                agent_state, metrics = iteration(agent_state, index, tick)
                assert metrics is not None  # logging: always the metrics
                return (agent_state, record(rows, count, metrics)), None

            return body

        def empty_rows(agent_state: Any, index: Any) -> tuple[dict, jax.Array]:
            shapes = jax.eval_shape(
                lambda s, i: iteration(s, i, jnp.asarray(0))[1], agent_state, index
            )
            return unevaluated(shapes, (rows,)), jnp.asarray(0)

        def resume_transform(agent_state: Any, _key: Any) -> Any:
            agent_state = with_slot(self.continued(agent_state))
            return agent_state if resume is None else resume(agent_state)

        train = build_resumable_train(
            init_fn=init_fn,
            make_scan_fn=make_scan_fn,
            num_updates=num_updates,
            resume_transform=resume_transform,
            carry_out=empty_rows if records else None,
        )
        if not records:
            return train

        def train_with_evaluations(*args: Any, **kwargs: Any) -> tuple[Any, dict]:
            agent_state, (rows, _) = train(*args, **kwargs)
            return agent_state, rows

        return train_with_evaluations

    def off_policy(
        self,
        init: Init,
        update: Update,
        learning_starts: int = 0,
        *,
        uniform_steps: Optional[int] = None,
        recurrent: bool = False,
        expose_rollout: bool = False,
        after_collect: Optional[Callable[[Any, Any], Any]] = None,
        collect_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
        collect: Optional[Collect] = None,
        n_updates: Optional[Callable[[Any, Any], Any]] = None,
        evaluation: Optional[Evaluation] = None,
        num_ticks: Optional[int] = None,
        keeps_actor_metrics: bool = False,
    ) -> Callable:
        """Train on one environment step per env per iteration (a tick).

        Each tick collects a step (uniform actions before ``uniform_steps``,
        by default ``learning_starts``; 0: the policy's from the first step,
        as TQC's authors act), applies ``after_collect(agent_state,
        transition)`` when given, then from ``learning_starts`` runs
        ``update(agent_state, transition)`` (without a replay, from the step
        collected at learning_starts), then evaluates the actor
        (:meth:`evaluate_and_log`). ``expose_rollout`` keeps the step on
        ``agent_state.last_rollout`` as a ``T = 1`` rollout. The budget is
        :attr:`budget` ``// n_envs`` ticks.

        An agent acting, training and evaluating its own way (the module
        docstring) supplies those steps: ``collect`` replaces the collection
        and the arguments configuring it; ``n_updates(tick, agent_state)``
        updates (of the absolute, unbatched tick: a train ratio, a burst;
        or of the collected state: the env steps the tick took) replace the
        ``learning_starts`` gate (:meth:`repeat_update`); ``evaluation``
        replaces the actor's evaluation (:meth:`evaluate_every`);
        ``num_ticks`` the budget.

        ``keeps_actor_metrics``: the update keeps its actor's last metrics on
        the state (update_steps.run_actor_if), which then gets their slot.
        With :attr:`offline_data` it trains offline instead (:meth:`offline`).
        """
        collect_args = self.collect_kwargs(recurrent, **(collect_kwargs or {}))
        if self.offline_data is not None:
            if collect is not None or n_updates is not None or evaluation is not None:
                raise ValueError(
                    "Offline training runs the agent's update on a replay of the"
                    " dataset; an agent collecting its own way has no such loop."
                )
            return self.offline(
                init,
                update,
                collect_args["buffer"],
                recurrent,
                eval_kwargs,
                keeps_actor_metrics,
            )

        def collect_step(agent_state: Any, tick: Any) -> tuple[Any, Any]:
            if collect is not None:
                return collect(agent_state, tick)
            timestep = agent_state.collector_state.timestep
            warmup = learning_starts if uniform_steps is None else uniform_steps
            uniform = should_use_uniform_sampling(timestep, warmup)
            agent_state, transition = collect_experience(
                agent_state, None, uniform=uniform, **collect_args
            )
            if expose_rollout:
                rollout = jax.tree.map(lambda x: x[None], transition)
                agent_state = agent_state.replace(last_rollout=rollout)
            if after_collect is not None:
                agent_state = after_collect(agent_state, transition)
            return agent_state, transition

        # An update without a replay learns from its step alone, so its gate
        # reads the timestep the step was collected at: it never learns from
        # a warm-up step. A replay's gate reads the steps it holds.
        learns_from_its_step = collect_args.get("buffer") is None

        def iteration(
            agent_state: Any, index: Any, tick: Any
        ) -> tuple[Any, Optional[dict]]:
            collected_at = agent_state.collector_state.timestep
            agent_state, experience = collect_step(agent_state, tick)

            def step(agent_state: Any) -> tuple[Any, Any]:
                return update(agent_state, experience)

            aux = None
            if n_updates is None:
                agent_state, aux = self.maybe_update(
                    agent_state,
                    learning_starts,
                    step,
                    collected_at if learns_from_its_step else None,
                )
            else:
                agent_state = self.repeat_update(
                    agent_state, n_updates(tick, agent_state), step
                )
            if evaluation is not None:
                return self.evaluate_every(agent_state, aux, index, tick, evaluation)
            return self.evaluate_and_log(
                agent_state, aux, index, recurrent, **(eval_kwargs or {})
            )

        if num_ticks is None:
            self.warn_if_partial(self.env_args.n_envs, "n_envs")
            num_ticks = self.budget // self.env_args.n_envs
        return self.train(
            init,
            iteration,
            num_ticks,
            last_rollout=(1, collect_args) if expose_rollout else None,
            evaluations=None if evaluation is None else evaluation.count(num_ticks),
            update=update,
            buffer=collect_args.get("buffer"),
            one_update=(
                (lambda state: update(*collect_step(state, jnp.asarray(0)))[0])
                if keeps_actor_metrics
                else None
            ),
        )

    def offline(
        self,
        init: Init,
        update: Update,
        buffer: Any,
        recurrent: bool = False,
        eval_kwargs: Optional[dict] = None,
        keeps_actor_metrics: bool = False,
    ) -> Callable:
        """Train on :attr:`offline_data` alone, with no environment step.

        A fresh run's replay and normalisers start from the dataset's rows
        (:func:`seed_replay`). Each tick runs ``update(agent_state, None)``
        on that replay, applies ``post_update`` and counts the update in the
        collector's timestep: the schedules run over the updates, and the
        evaluations (:meth:`evaluate_and_log`) come every ``log_frequency``
        updates and log the updates done as their timestep. The budget is
        :attr:`budget` updates.
        """
        data = self.offline_data
        assert data is not None
        # The rows the pretraining writes (ExpertPrefill) come after the
        # dataset's: past the replay's capacity its ring would overwrite them.
        n_envs = self.env_args.n_envs
        added = sum(extension.replay_rows(n_envs) for extension in self.extensions)

        def seeded(key: jax.Array) -> Any:
            agent_state = init(key)
            experience = agent_state.collector_state.buffer_state.experience
            replay_envs, rows_per_env = jax.tree.leaves(experience)[0].shape[:2]
            capacity = replay_envs * rows_per_env
            if len(data.reward) + added > capacity:
                raise ValueError(
                    f"The dataset's {len(data.reward)} rows and the {added} the"
                    " extensions write before training outnumber the replay's"
                    f" {capacity}, which would overwrite dataset rows: raise"
                    " buffer_size or write fewer."
                )
            return seed_replay(agent_state, buffer, data, self.env_args)

        def iteration(
            agent_state: Any, index: Any, tick: Any
        ) -> tuple[Any, Optional[dict]]:
            del tick
            agent_state, aux = update(agent_state, None)
            # One (1,)-shaped leaf per metric, as maybe_update hands them.
            aux = jax.tree.map(lambda x: jnp.reshape(x, (1,)), aux)
            agent_state = self.post_update(agent_state)
            collector = agent_state.collector_state
            collector = collector.replace(timestep=collector.timestep + 1)
            agent_state = agent_state.replace(collector_state=collector)
            return self.evaluate_and_log(
                agent_state,
                aux,
                index,
                recurrent,
                phase="offline",
                **(eval_kwargs or {}),
            )

        return self.train(
            seeded,
            iteration,
            self.budget,
            update=update,
            buffer=buffer,
            one_update=(
                (lambda state: update(state, None)[0]) if keeps_actor_metrics else None
            ),
        )

    def forced_reset(
        self, agent_state: Any, n_steps: int, resets_per_eval: int, num_evals: int
    ) -> Any:
        """brax's ``num_resets_per_eval``: every env reset
        (:func:`ajax.environments.interaction.force_reset`) every
        ``reset_every`` updates, ``resets_per_eval`` times per evaluation
        after the first of ``num_evals``.

        Playground envs draw a fresh initial state at every episode end by
        default (build_env_from_id(fresh_reset=True)); with fresh_reset=False
        their auto-reset replays the first reset state, and without these
        resets a run sees only n_envs initial conditions.
        """
        # The cadence divides the whole training: the updates of one call
        # over the run, from 0 to its horizon.
        total_n_updates = self.n_rollouts(n_steps, self.total_timesteps)
        num_evals_after_init = max(int(num_evals) - 1, 1)
        reset_every = jnp.maximum(
            1,
            jnp.ceil(total_n_updates / (num_evals_after_init * resets_per_eval)).astype(
                jnp.int32
            ),
        )
        should_reset = (agent_state.n_updates % reset_every == 0) & (
            agent_state.n_updates > 0
        )
        return jax.lax.cond(
            should_reset,
            lambda s: force_reset(s, self.env_args, self.mode, self.expert_policy),
            lambda s: s,
            agent_state,
        )

    def on_policy(
        self,
        init: Init,
        update: RolloutUpdate,
        n_steps: int,
        *,
        recurrent: bool = False,
        expose_rollout: bool = False,
        forced_resets: tuple[int, int] = (0, 1),
        collect_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
    ) -> Callable:
        """Train on an n_steps rollout per env per iteration.

        Each iteration collects the rollout from the agent state
        agent_state_at_rollout_start (collect_experience with
        collect_kwargs), then runs update(agent_state, rollout,
        agent_state_at_rollout_start), resets every env on brax's
        cadence when forced_resets, (resets_per_eval, num_evals),
        asks for any (forced_reset), and applies post_update;
        expose_rollout keeps the rollout on agent_state.last_rollout.
        The budget is n_rollouts; the last one may reach past the
        call's end, and evaluates there when the horizon is further.
        """
        collect = self.collect_kwargs(recurrent, **(collect_kwargs or {}))

        def iteration(
            agent_state_at_rollout_start: Any, index: Any, tick: Any
        ) -> tuple[Any, Optional[dict]]:
            del tick
            agent_state, rollout = jax.lax.scan(
                partial(collect_experience, **collect),
                agent_state_at_rollout_start,
                xs=None,
                length=n_steps,
            )
            if expose_rollout:
                agent_state = agent_state.replace(last_rollout=rollout)
            agent_state, aux = update(
                agent_state, rollout, agent_state_at_rollout_start
            )
            if forced_resets[0] > 0:
                agent_state = self.forced_reset(agent_state, n_steps, *forced_resets)
            agent_state = self.post_update(agent_state)
            return self.evaluate_and_log(
                agent_state, aux, index, recurrent, **(eval_kwargs or {})
            )

        n_rollouts = self.n_rollouts(n_steps)
        reach = self.start_timestep + n_rollouts * self.env_args.n_envs * n_steps
        return self.train(
            init,
            iteration,
            n_rollouts,
            last_rollout=(n_steps, collect) if expose_rollout else None,
            evaluations=self.evaluations_until(reach),
        )


__all__ = [
    "LOOP_PHASES",
    "Evaluation",
    "TrainLoop",
    "count_in_normalisers",
    "fresh_state",
    "seed_replay",
]
