"""The families of the agents whose iteration, or whose replay, is their
own (ajax.agents.families describes the families): OwnIterationAgent (UDRL,
APG) and OwnReplayAgent (the world models, DreamerV3 and TD-MPC2).
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import jax

from ajax.agents.base import run_horizon, shared_counters
from ajax.agents.families import DeclaredAgent, TrainContext
from ajax.agents.loop import LOOP_PHASES, Evaluation
from ajax.agents.sweep import Sweep
from ajax.agents.sweep import check as check_sweep
from ajax.agents.sweep import configurations as sweep_configurations
from ajax.environments.row_collector import resume_tick
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import Extension
from ajax.policy import PolicyParts
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


@dataclass(kw_only=True, eq=False, repr=False)
class OwnIterationAgent(DeclaredAgent):
    """An agent whose iteration is its own (TrainLoop.train): it acts and
    learns its way, then the loop evaluates on its cadence, logs and
    resumes. UDRL collects a segment and fits its replay of segments
    (Schmidhuber, 2019); APG takes a gradient step through a differentiable
    rollout.

    An agent writes init, iterate (applying the extensions' post_update
    where its order puts it), report (the training metrics an evaluation
    logs), evaluate and count_iteration_steps. It may take options per train
    call (configure_call), and return every iteration's output instead of
    the evaluations (returns_iterations, make_iteration_output). It takes no
    dataset: train_offline refuses it.
    """

    # The loop's phases but observe, action, eval_action and track_memory, which
    # no shared collector or evaluation meets; an agent adds the phases its
    # iteration applies.
    supported_extension_phases = LOOP_PHASES.difference(
        {"observe", "action", "eval_action", "track_memory"}
    )
    #: Whether train returns every iteration's output (make_iteration_output),
    #: stacked, logging or not, instead of the evaluations.
    returns_iterations = False

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def iterate(self, agent_state: Any, context: TrainContext) -> tuple[Any, Any]:
        """One iteration, the extensions' post_update applied: (agent_state,
        aux, which report and make_iteration_output read)."""
        raise NotImplementedError

    def report(self, agent_state: Any, aux: Any) -> dict:
        """The training metrics an evaluation logs, timestep included."""
        raise NotImplementedError

    def evaluate(
        self,
        agent_state: Any,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """The policy's evaluation metrics, on num_episode_test episodes."""
        raise NotImplementedError

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """How the agent acts: its own, as its evaluation is."""
        raise NotImplementedError

    def count_iteration_steps(self, context: TrainContext) -> tuple[int, str]:
        """The env steps one iteration takes, and how a warning names them."""
        raise NotImplementedError

    def configure_call(
        self, context: TrainContext, n_iterations: int
    ) -> tuple[TrainContext, Optional[Callable[[Any], Any]]]:
        """The context of one train call of n_iterations iterations, and how
        a state it resumes from is transformed (None: as it is). An agent
        taking options per call declares them as keywords here."""
        del n_iterations
        return context, None

    def make_iteration_output(self, aux: Any, evaluations: Optional[dict]) -> Any:
        """What an iteration returns: its evaluations, which the loop records
        (None without logging); with returns_iterations, the agent's own."""
        del aux
        return evaluations

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        extensions: Sequence[Extension] = (),
        **call_options: Any,
    ) -> Callable:
        """The train function ActorCritic.train builds: the loop over init
        and iterate, call_options going to configure_call. It takes no
        dataset (train_offline refuses it)."""
        # An option configure_call does not take is refused before
        # make_context starts the logging worker.
        options = set(inspect.signature(self.configure_call).parameters)
        options = options.difference({"context", "n_iterations"})
        unknown = sorted(set(call_options).difference(options))
        if unknown:
            raise TypeError(f"{self.name}.train takes no {', '.join(unknown)}.")
        loop, context = self.make_context(
            total_timesteps,
            num_episode_test,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            extensions=extensions,
            run_ids=run_ids,
            logging_config=logging_config,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )
        steps, unit = self.count_iteration_steps(context)
        loop.warn_if_partial(steps, unit)
        n_iterations = max(loop.budget // steps, 1)
        context, resume = self.configure_call(context, n_iterations, **call_options)
        evaluation = Evaluation(
            metrics=self.report,
            every=loop.log_frequency and max(loop.log_frequency // steps, 1),
            evaluate=partial(
                self.evaluate, context=context, num_episode_test=num_episode_test
            ),
        )

        def log(agent_state: Any, aux: Any, index: Any, tick: Any) -> tuple[Any, Any]:
            agent_state, evaluations = loop.evaluate_every(
                agent_state, aux, index, tick, evaluation
            )
            return agent_state, self.make_iteration_output(aux, evaluations)

        def iteration(agent_state: Any, index: Any, tick: Any) -> tuple[Any, Any]:
            agent_state, aux = self.iterate(agent_state, context)
            if self.returns_iterations:  # every iteration's output, evaluated
                return log(agent_state, aux, index, tick)
            return agent_state, aux

        return loop.train(
            partial(self.init, context=context),
            iteration,
            n_iterations,
            evaluations=evaluation.count(n_iterations),
            resume=resume,
            returns_iterations=self.returns_iterations,
            log=log,
            due=evaluation.ticks,
        )


@dataclass(kw_only=True, eq=False, repr=False)
class OwnReplayAgent(DeclaredAgent):
    """An agent storing rows in a replay of its own (TrainLoop.off_policy's
    hooks): each tick its row collector acts and stores one row per env, it
    runs the updates its schedule gives the absolute tick, and the loop
    evaluates on its cadence, logs and resumes. DreamerV3 trains on windows
    of the replay at a train ratio (Hafner et al., 2023).

    It is no ReplayAgent: its replay holds rows, not transitions, sized for
    each call (prepare_replay), and it takes no dataset. It assumes the row
    collector (ajax.environments.row_collector): a resumed run starts at the
    tick of the rows it counted, and each env copy may carry its own system.

    An agent writes prepare_replay, init, collect, count_updates, update,
    report and evaluate; it may count its ticks (count_ticks), restart what
    it logs after each log (make_after_log) and check a call (check_run).
    """

    # The own-iteration family's phases: no shared collector or evaluation
    # meets an observation.
    supported_extension_phases = OwnIterationAgent.supported_extension_phases
    pretrain_context = frozenset({"env_args", "seed_index", "update"})
    supports_batched_env_params = True

    def prepare_replay(
        self, start_tick: int, end_tick: int, agent_state: Any
    ) -> tuple[Any, Any]:
        """The replay of a call training from tick start_tick to end_tick,
        and the state it resumes from (None for a fresh run), its replay
        moved into that replay's shapes. The agent records the sizes."""
        raise NotImplementedError

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def collect(
        self, agent_state: Any, tick: jax.Array, context: TrainContext
    ) -> tuple[Any, Any]:
        """Act and store one row per env at the absolute tick: (agent_state,
        what update reads)."""
        raise NotImplementedError

    def count_updates(
        self, agent_state: Any, tick: jax.Array, context: TrainContext
    ) -> jax.Array:
        """The updates run after the tick's rows (an int32 count)."""
        raise NotImplementedError

    def update(
        self, agent_state: Any, experience: Any, context: TrainContext
    ) -> tuple[Any, Any]:
        """One update on the replay; the loop folds post_update after it."""
        raise NotImplementedError

    def report(self, agent_state: Any, aux: Any) -> dict:
        """The training metrics an evaluation logs, timestep included."""
        raise NotImplementedError

    def evaluate(
        self,
        agent_state: Any,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """The policy's evaluation metrics, on num_episode_test episodes."""
        raise NotImplementedError

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """How the agent acts: its own, as its evaluation is."""
        raise NotImplementedError

    def count_ticks(self, n_timesteps: int, start_tick: int) -> int:
        """The ticks of n_timesteps from start_tick: one row per env each."""
        del start_tick
        return n_timesteps // self.env_args.n_envs

    def steps_before(self, tick: Any) -> Any:
        """The timesteps the absolute ticks before tick take: one row per
        env each."""
        return tick * self.env_args.n_envs

    def make_after_log(self) -> Optional[Callable[[Any, jax.Array], Any]]:
        """What runs on every tick after the logging gate (Evaluation's
        after_log). None by default: an identity would still trace the gate."""
        return None

    def check_run(self, agent_state: Any, end_tick: int) -> None:
        """Check the state a call ended with, at tick end_tick."""
        del agent_state, end_tick

    def value_estimate(self) -> Optional[ValueEstimate]:
        """None: the agent evaluates itself (evaluate), so the shared
        evaluation's critic estimate is never compared; a world model's
        critic reads its latent state, not the observation that estimate
        passes."""
        return None

    def resume_iteration_offset(self, initial_state: Any) -> int:
        """The absolute tick of a resumed run, equal across seeds
        (row_collector.resume_tick), so that every schedule of the tick
        continues instead of restarting."""
        return resume_tick(initial_state.collector_state, self.env_args.n_envs)

    def train(
        self,
        seed: Union[int, Sequence[int]] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Any = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Any = None,
        total_timesteps: Optional[int] = None,
        sweep: Optional[Sweep] = None,
        devices: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """ActorCritic.train on the call's replay (prepare_replay), which
        continues the replay of the uninterrupted run when it resumes."""
        state = initial_state
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]
        # Refused before prepare_replay records the replay's sizes on the
        # agent: a refused call leaves the agent as it was.
        found = sweep_configurations(sweep)
        if found:
            check_sweep(type(self).__name__, self.sweepable_fields(), list(found[0]))
        start = 0
        if state is not None:
            counters = shared_counters(state, self.per_seed_update_counts)
            start = int(counters.get("timestep", 0))
        run_horizon(start, n_timesteps, total_timesteps)
        start_tick = 0 if state is None else self.resume_iteration_offset(state)
        end_tick = start_tick + self.count_ticks(n_timesteps, start_tick)
        replay, state = self.prepare_replay(start_tick, end_tick, state)
        agent_state, out = super().train(
            seed=seed,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            on_ids_ready=on_ids_ready,
            initial_state=state,
            total_timesteps=total_timesteps,
            sweep=sweep,
            devices=devices,
            replay=replay,
            start_tick=start_tick,
            **kwargs,
        )
        self.check_run(agent_state, end_tick)
        return agent_state, out

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        extensions: Sequence[Extension] = (),
        *,
        replay: Any,
        start_tick: int = 0,
    ) -> Callable:
        """The train function ActorCritic.train builds: off_policy's ticks of
        the agent's collect, count_updates and update on the call's replay,
        evaluating every log_frequency timesteps. It takes no dataset
        (train_offline refuses it)."""
        loop, context = self.make_context(
            total_timesteps,
            num_episode_test,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            extensions=extensions,
            run_ids=run_ids,
            logging_config=logging_config,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )
        context = dataclasses.replace(context, buffer=replay)
        loop.warn_if_partial(env_args.n_envs, "n_envs")
        log_frequency = loop.log_frequency
        evaluation = Evaluation(
            metrics=self.report,
            every=log_frequency,
            evaluate=partial(
                self.evaluate, context=context, num_episode_test=num_episode_test
            ),
            after_log=self.make_after_log(),
            steps_before=self.steps_before,
        )

        def count_updates(tick: jax.Array, agent_state: Any) -> jax.Array:
            return self.count_updates(agent_state, tick, context)

        return loop.off_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            collect=partial(self.collect, context=context),
            n_updates=count_updates,
            evaluation=evaluation,
            num_ticks=self.count_ticks(loop.budget, start_tick),
        )
