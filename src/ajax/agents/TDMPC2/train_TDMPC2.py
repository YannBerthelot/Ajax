"""TD-MPC2 single-task online training: the paper-era loop, on device.

The reference loop (``nicklashansen/tdmpc2@b67b21c:tdmpc2/trainer/
online_trainer.py:67-117``, unchanged at 5f6fade; tdmpc2_spec §4.C) is::

    step = 0
    while step <= steps:
        if done: (eval every eval_freq steps); buffer.add(episode); reset
        action = agent.act(obs, t0) if step > seed_steps else env.rand_act()
        obs, reward, done = env.step(action)
        if step >= seed_steps:
            for _ in range(seed_steps if step == seed_steps else 1):
                agent.update(buffer)
        step += 1

Ajax runs it on the shared off-policy loop
(:meth:`ajax.agents.loop.TrainLoop.off_policy`), one ``lax.scan`` over
*ticks* (``DESIGN.md`` §2, §4.4): one obs-aligned row per env per tick,
``T + 1`` ticks per fixed-length episode (``T`` stepping ticks, then the held
tick ``i mod (T + 1) == T`` that emits the final observation and resets the
envs with a fresh key). One tick::

    row = collect_row(tick)        # collect: uniform random actions in the
    buffer.add(row, tick)          # seed phase, the MPPI planner afterwards;
                                   # the episode ring, in place
    repeat schedule.n_updates(tick) times:
        agent.update(buffer.sample(fresh key), fresh noise)   # update
        extensions.post_update
    every log_frequency env steps: evaluate + log (with a logging_config)

Every schedule is static arithmetic on the absolute, unbatched tick
(:class:`Schedule`), so it stays a real ``cond`` / ``while`` under the seed
``vmap`` and continues across resumes (the tick is offset by
``ActorCritic.resume_iteration_offset``). With one env it reproduces b67b21c
exactly: random actions at env steps ``0 .. S`` (S + 1 of them), a burst of
``S`` updates right after step ``S`` with ``floor(S / T)`` episodes in the
buffer, then one update per env step. ``n_envs > 1`` is a generalisation
(deviation T7), and so is another number of updates per env step
(``updates_per_step``, deviation T28).

``n_timesteps`` counts env (agent) steps, summed over envs (``DESIGN.md`` §2);
logs add ``env_frames = env steps * action_repeat``, the papers' x-axis.
Terminations are refused (deviation T10): the static collector counts
off-schedule ``done``s, the loop counts the true terminations on scheduled
episode ends (``is_terminal`` rows), and
:meth:`ajax.agents.TDMPC2.TDMPC2.TDMPC2.train` raises after the run if either
is non-zero; the paper-era TD targets have no termination factor.

The episodic mode (``config.episodic``, deviation T29) takes terminating
tasks: the dynamic reset mode, the variable-length replay
(:class:`~ajax.agents.TDMPC2.buffer.EpisodicBuffer`) and
:class:`EpisodicSchedule`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp

from ajax.agents.loop import Evaluation, TrainLoop
from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.buffer import (
    EpisodeBuffer,
    EpisodeBufferState,
    EpisodicBuffer,
    EpisodicBufferState,
)
from ajax.agents.TDMPC2.planner import plan_for_envs
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2State
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.environments.utils import get_action_dim
from ajax.evaluate import evaluate_policy
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import FloatOrCallable


@dataclasses.dataclass(frozen=True)
class Schedule:
    """The static seed-phase / burst / update schedule (``DESIGN.md`` §4.4).

    On the absolute tick ``i`` (period ``P = T + 1``; stepping ticks
    ``i mod P < T`` advance every env by one step, the held tick does not):

    * the *seed tick* is the stepping tick of per-env step
      ``k_S = max(S div n, T)``, the first stepping tick after which the
      total env steps ``n (k + 1)`` exceed ``S`` and at least one round
      (``n`` episodes) is committed. For one env ``k_S = S`` (``S >= T`` is
      the reference's ``max(1000, 5 T)``): steps ``0 .. S`` act randomly
      (``online_trainer.py:97-100``);
    * actions are uniform random on every tick up to the seed tick (and on
      held ticks, whose action is discarded, so the planner never runs
      there);
    * ``n_updates(i)`` is ``S`` on the seed tick (the pretraining burst,
      ``online_trainer.py:105-112``), :attr:`updates_per_tick` on later
      stepping ticks and 0 otherwise.

    The methods accept Python ints, NumPy arrays and traced (unbatched)
    ticks.

    Attributes:
        n_envs: lockstep envs ``n``.
        episode_length: ``T`` agent steps.
        seed_steps: ``S`` (the reference's ``max(1000, 5 T)`` by default).
        updates_per_step: ``r``, updates per env step after the burst (the
            reference's 1; deviation T28).
    """

    n_envs: int
    episode_length: int
    seed_steps: int
    updates_per_step: float = 1.0

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.episode_length < 1 or self.seed_steps < 0:
            raise ValueError(
                "need n_envs >= 1, episode_length >= 1 and seed_steps >= 0, got"
                f" {self.n_envs}, {self.episode_length}, {self.seed_steps}"
            )
        if not self.updates_per_step > 0:
            raise ValueError(
                f"updates_per_step must be > 0, got {self.updates_per_step}"
            )

    @property
    def updates_per_tick(self) -> int:
        """Updates per stepping tick after the burst: ``max(1, round(n r))``
        (Python's ``round``, halves to even); ``n`` for the reference's
        ``r = 1``. At least one, so a small ``r`` never stops training."""
        return max(1, round(self.n_envs * self.updates_per_step))

    @property
    def period(self) -> int:
        """Ticks per episode, ``T + 1``."""
        return self.episode_length + 1

    def tick_of_step(self, step: int) -> int:
        """Tick of per-env env step ``step`` (``(step div T) P + step mod T``).

        Also the number of ticks, from tick 0, that take ``step`` env steps
        per env (a trailing held tick included after a whole episode).
        """
        t = self.episode_length
        return (step // t) * self.period + step % t

    def steps_before(self, tick: int) -> int:
        """Per-env env steps taken by ticks ``0 .. tick - 1``."""
        return (tick // self.period) * self.episode_length + min(
            tick % self.period, self.episode_length
        )

    def num_ticks(self, n_timesteps: int, start_tick: int = 0) -> int:
        """Ticks of a run of ``n_timesteps`` env steps (``n_timesteps div n``
        per env) starting at absolute tick ``start_tick`` (a resume).

        A split run covers exactly the ticks of an uninterrupted one: the run
        ends on the tick of its last env step, plus the held tick when that
        step ends an episode.
        """
        end_step = self.steps_before(start_tick) + n_timesteps // self.n_envs
        return self.tick_of_step(end_step) - start_tick

    @property
    def seed_step(self) -> int:
        """``k_S``: the per-env env step of the seed tick. At least ``T``, so
        the seed tick follows the first committed round: the sampler never
        draws from an empty buffer (committed rounds only grow with the
        tick)."""
        return max(self.seed_steps // self.n_envs, self.episode_length)

    @property
    def seed_tick(self) -> int:
        """The last random-action stepping tick, followed by the burst."""
        return self.tick_of_step(self.seed_step)

    def is_held(self, tick: Any) -> Any:
        return tick % self.period == self.episode_length

    def random_phase(self, tick: Any) -> Any:
        return (tick <= self.seed_tick) | self.is_held(tick)

    def n_updates(self, tick: Any) -> Any:
        after = jnp.where(tick > self.seed_tick, self.updates_per_tick, 0)
        stepping = jnp.where(tick == self.seed_tick, self.seed_steps, after)
        return jnp.where(self.is_held(tick), 0, stepping).astype(jnp.int32)


@dataclasses.dataclass(frozen=True)
class EpisodicSchedule:
    """The episodic mode's schedule (deviation T29).

    The dynamic reset mode holds an env for the tick of its final
    observation, so ticks are not env steps. With ``k_S = max(S div n, T)``:
    random actions on ticks ``0 .. k_S`` (the seed phase counts rows, held
    ones included; at least ``T + 1`` ticks, so every env has finished an
    episode); the burst of ``S`` on tick ``k_S``, then, on a tick that
    stepped ``stepped > 0`` envs, ``max(1, round(stepped r))`` updates
    (``r = updates_per_step``, T28: one per env step by default); a run of
    ``n_timesteps`` rows lasts ``n_timesteps div n`` ticks.
    """

    n_envs: int
    episode_length: int
    seed_steps: int
    updates_per_step: float = 1.0

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.episode_length < 1 or self.seed_steps < 0:
            raise ValueError(
                "need n_envs >= 1, episode_length >= 1 and seed_steps >= 0, got"
                f" {self.n_envs}, {self.episode_length}, {self.seed_steps}"
            )
        if not self.updates_per_step > 0:
            raise ValueError(
                f"updates_per_step must be > 0, got {self.updates_per_step}"
            )

    @property
    def seed_tick(self) -> int:
        return max(self.seed_steps // self.n_envs, self.episode_length)

    def num_ticks(self, n_timesteps: int, start_tick: int = 0) -> int:
        del start_tick  # every tick is a row per env
        return n_timesteps // self.n_envs

    def random_phase(self, tick: Any) -> Any:
        return tick <= self.seed_tick

    def n_updates(self, tick: Any, stepped: Any) -> Any:
        # jnp.round, like Python's round in Schedule, rounds halves to even.
        per_tick = jnp.maximum(1, jnp.round(stepped * self.updates_per_step))
        after = jnp.where((tick > self.seed_tick) & (stepped > 0), per_tick, 0)
        return jnp.where(tick == self.seed_tick, self.seed_steps, after).astype(
            jnp.int32
        )


def _policy(
    agent_state: TDMPC2State, config: TDMPC2Config, gamma: float, eval_mode: bool
) -> Callable:
    return partial(
        plan_for_envs,
        wm_params=agent_state.world_model_state.params,
        pi_params=agent_state.actor_state.params,
        config=config,
        gamma=gamma,
        eval_mode=eval_mode,
    )


def init_TDMPC2(
    key: jax.Array,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    buffer: Union[EpisodeBuffer, EpisodicBuffer],
    *,
    gamma: float,
    learning_rate: FloatOrCallable,
    enc_lr_scale: float,
    pi_eps: float,
    weight_decay: float = 0.0,
) -> TDMPC2State:
    """Fresh networks and optimizers (:func:`core.create_update_state`), a zero
    planner carry, reset envs (every env's first row is ``is_first``) and an
    empty buffer."""
    model_key, collector_key, rng, eval_rng = jax.random.split(key, 4)
    obs_dim, action_dim = buffer.obs_dim, buffer.action_dim
    learner = core.create_update_state(
        model_key,
        config,
        obs_dim,
        action_dim,
        learning_rate=learning_rate,
        enc_lr_scale=enc_lr_scale,
        pi_eps=pi_eps,
        weight_decay=weight_decay,
    )
    prev_mean = jnp.zeros((env_args.n_envs, config.horizon, action_dim))
    collector_state = init_row_collector_state(
        collector_key, env_args, policy_carry=prev_mean
    )
    return TDMPC2State(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=learner.actor_state,
        world_model_state=learner.world_model_state,
        q_scale=learner.q_scale,
        pi_gradnorm_sq=learner.pi_gradnorm_sq,
        collector_state=collector_state,
        buffer_state=buffer.init(),
        update_metrics=core.make_zero_update_metrics(
            learner, config, gamma, obs_dim, action_dim
        ),
        n_terminations=jnp.zeros((), jnp.int32),
        n_updates=jnp.zeros((), jnp.int32),
        n_logs=jnp.zeros((), jnp.int32),
    )


def evaluate_tdmpc2(
    agent_state: TDMPC2State,
    key: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    gamma: float,
    num_episodes: int,
) -> dict[str, jax.Array]:
    """Mean return and length of ``num_episodes`` evaluation episodes.

    :func:`ajax.evaluate.evaluate_policy` (a rebuilt env with the training
    action repeat and episode length) with the planner in ``eval_mode``: no
    final exploration noise, everything else stochastic as in the reference
    (``online_trainer.py:27-48``, tdmpc2_spec 4.21). The planner carry starts
    at zeros, separate from the training one (deviation T9). ``key`` is
    fresh at each evaluation (:func:`ajax.log.evaluation_keys`): each starts
    from new initial states, as the reference's resets of its running env
    do.
    """
    action_dim = get_action_dim(env_args.env, env_args.env_params)
    ret, length = evaluate_policy(
        env_args,
        _policy(agent_state, config, gamma, eval_mode=True),
        lambda m: jnp.zeros((m, config.horizon, action_dim)),
        num_episodes,
        key,
    )
    return {"Eval/episodic mean reward": ret, "Eval/mean episodic length": length}


def train_metrics(
    agent_state: TDMPC2State, aux: Any, *, action_repeat: int
) -> dict[str, Any]:
    """The house keys, ``env_frames`` and the last update's quantities."""
    del aux
    cs = agent_state.collector_state
    metrics = {
        "timestep": cs.timestep,
        "env_frames": cs.env_steps * action_repeat,
        "Train/episodic mean reward": cs.episodic_mean_return,
        "Train/n_updates": agent_state.n_updates,
        "Train/offschedule_dones": cs.n_offschedule_dones,
        "Train/terminations": agent_state.n_terminations,
    }
    metrics.update({f"Train/{k}": v for k, v in agent_state.update_metrics.items()})
    return metrics


def collect(
    agent_state: TDMPC2State,
    tick: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    schedule: Union[Schedule, EpisodicSchedule],
    buffer: Union[EpisodeBuffer, EpisodicBuffer],
    gamma: float,
) -> tuple[TDMPC2State, jax.Array]:
    """Act and store one row per env on the absolute, unbatched tick
    ``tick``; returns the state and ``tick``, which locates the buffer's
    newest rows for the tick's updates (:func:`update`). The episodic mode
    collects in the dynamic reset mode, counting rows."""
    cs = agent_state.collector_state
    collector_state, row = collect_row(
        cs,
        tick,
        _policy(agent_state, config, gamma, eval_mode=False),
        env_args=env_args,
        reset_mode="dynamic" if config.episodic else "static",
        episode_length=schedule.episode_length,  # the static mode's
        random_phase=schedule.random_phase(tick),
        timestep_unit="rows" if config.episodic else "env_steps",
    )
    replay = agent_state.buffer_state
    if isinstance(buffer, EpisodicBuffer):
        assert isinstance(replay, EpisodicBufferState)
        # The row's index in its episode: a final row (the env held) is
        # its episode's last, after last_episode_length steps.
        assert cs.step_in_episode is not None
        step = jnp.where(cs.is_last, cs.last_episode_length, cs.step_in_episode)
        replay = buffer.add(replay, row, step, tick)
    else:
        assert isinstance(replay, EpisodeBufferState)
        replay = buffer.add(replay, row.obs, row.action, row.reward, tick)
    agent_state = agent_state.replace(
        collector_state=collector_state,
        buffer_state=replay,
        # True terminations on scheduled episode ends (the off-schedule
        # ones are the collector's n_offschedule_dones); every one in the
        # episodic mode.
        n_terminations=agent_state.n_terminations
        + jnp.sum(row.is_terminal, dtype=jnp.int32),
    )
    return agent_state, tick


def update(
    agent_state: TDMPC2State,
    tick: jax.Array,
    *,
    config: TDMPC2Config,
    buffer: Union[EpisodeBuffer, EpisodicBuffer],
    gamma: float,
) -> tuple[TDMPC2State, None]:
    """One ``agent.update(buffer)`` after tick ``tick``'s rows: a fresh batch
    and fresh noise, then :func:`core.update` (tdmpc2_spec §2), its logged
    quantities kept on ``update_metrics``. The loop folds ``post_update``
    after it."""
    rng, sample_key, noise_key = jax.random.split(agent_state.rng, 3)
    replay = agent_state.buffer_state
    if isinstance(buffer, EpisodicBuffer):
        assert isinstance(replay, EpisodicBufferState)
        batch = buffer.sample(
            replay, sample_key, tick, config.batch_size, config.horizon
        )
    else:
        assert isinstance(replay, EpisodeBufferState)
        batch = buffer.sample(
            replay, sample_key, tick, config.batch_size, config.horizon
        )
    noise = core.draw_update_noise(
        noise_key, config, config.batch_size, buffer.action_dim
    )
    agent_state, metrics = core.update(
        agent_state.replace(rng=rng), batch, noise, config=config, gamma=gamma
    )
    agent_state = agent_state.replace(
        n_updates=agent_state.n_updates + 1, update_metrics=metrics
    )
    return agent_state, None


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    model_config: TDMPC2Config,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    *,
    gamma: float,
    schedule: Union[Schedule, EpisodicSchedule],
    buffer: Union[EpisodeBuffer, EpisodicBuffer],
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
    weight_decay: float = 0.0,
    start_tick: int = 0,
    extensions: Sequence = (),
) -> Callable:
    """TD-MPC2's train function on :meth:`TrainLoop.off_policy`.

    ``total_timesteps`` counts env steps (summed over envs), the run's
    horizon (:class:`TrainLoop`); the scan runs :meth:`Schedule.num_ticks`
    ticks of the call's budget from ``start_tick``, the absolute tick a
    resumed run starts at (the ``iteration_offset`` the agent's ``train``
    passes, :meth:`ajax.agents.TDMPC2.TDMPC2.TDMPC2.resume_iteration_offset`;
    0 for a fresh run). ``schedule`` and ``buffer`` are the agent's (the
    buffer sized for the whole run so far,
    :meth:`ajax.agents.TDMPC2.TDMPC2.TDMPC2._replay_buffer`).
    ``logging_config.log_frequency`` counts env steps and is converted to
    ticks with the static map from the start of the run (exact when it is a
    multiple of ``T * n_envs``, as the reference's ``eval_freq`` 50,000 is of
    DMC's 500). The optimizers are built by :func:`core.create_update_state`
    from ``learning_rate``, ``enc_lr_scale``, ``pi_eps`` and ``weight_decay``;
    the base class's actor / critic optimizer and network configs do not
    apply. The episodic mode's updates follow the envs each tick stepped
    (those not held: the collector sets ``is_first`` exactly after a held
    tick) and wait for a sampleable episode.
    """
    del actor_optimizer_args, critic_optimizer_args, network_args
    config = model_config
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    )
    # The schedule counts each env's steps, n_timesteps // n_envs.
    loop.warn_if_partial(env_args.n_envs, "n_envs")

    def init(key: jax.Array) -> TDMPC2State:
        return init_TDMPC2(
            key,
            env_args,
            config,
            buffer,
            gamma=gamma,
            learning_rate=learning_rate,
            enc_lr_scale=enc_lr_scale,
            pi_eps=pi_eps,
            weight_decay=weight_decay,
        )

    models: dict[str, Any] = {"config": config, "buffer": buffer, "gamma": gamma}
    every = loop.log_frequency and max(schedule.num_ticks(loop.log_frequency), 1)

    def n_updates(tick: jax.Array, agent_state: TDMPC2State) -> jax.Array:
        if isinstance(schedule, Schedule):
            return schedule.n_updates(tick)
        replay = agent_state.buffer_state
        assert isinstance(buffer, EpisodicBuffer)
        assert isinstance(replay, EpisodicBufferState)
        held = agent_state.collector_state.is_first  # held this tick
        stepped = env_args.n_envs - jnp.sum(held, dtype=jnp.int32)
        ready = buffer.num_episodes(replay, tick, config.horizon)
        return jnp.where(ready > 0, schedule.n_updates(tick, stepped), 0)

    return loop.off_policy(
        init,
        partial(update, **models),
        collect=partial(collect, env_args=env_args, schedule=schedule, **models),
        n_updates=n_updates,
        evaluation=Evaluation(
            metrics=partial(train_metrics, action_repeat=env_args.action_repeat),
            every=every,
            evaluate=partial(
                evaluate_tdmpc2,
                env_args=env_args,
                config=config,
                gamma=gamma,
                num_episodes=num_episode_test,
            ),
        ),
        num_ticks=schedule.num_ticks(loop.budget, start_tick),
    )


__all__ = [
    "EpisodicSchedule",
    "Schedule",
    "collect",
    "evaluate_tdmpc2",
    "init_TDMPC2",
    "make_train",
    "train_metrics",
    "update",
]
