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
(deviation T7).

``n_timesteps`` counts env (agent) steps, summed over envs (``DESIGN.md`` §2);
logs add ``env_frames = env steps * action_repeat``, the papers' x-axis.
Terminations are refused (deviation T10): the static collector counts
off-schedule ``done``s, the loop counts the true terminations on scheduled
episode ends (``is_terminal`` rows), and
:meth:`ajax.agents.TDMPC2.TDMPC2.TDMPC2.train` raises after the run if either
is non-zero; the paper-era TD targets have no termination factor.
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
from ajax.agents.TDMPC2.buffer import EpisodeBuffer
from ajax.agents.TDMPC2.planner import draw_plan_noise, plan
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2State
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.environments.utils import get_action_dim
from ajax.evaluate import evaluate_policy
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import FloatOrCallable

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


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
      ``online_trainer.py:105-112``), ``n`` on later stepping ticks (UTD 1
      per env step) and 0 otherwise.

    The methods accept Python ints, NumPy arrays and traced (unbatched)
    ticks.

    Attributes:
        n_envs: lockstep envs ``n``.
        episode_length: ``T`` agent steps.
        seed_steps: ``S`` (the reference's ``max(1000, 5 T)`` by default).
    """

    n_envs: int
    episode_length: int
    seed_steps: int

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.episode_length < 1 or self.seed_steps < 0:
            raise ValueError(
                "need n_envs >= 1, episode_length >= 1 and seed_steps >= 0, got"
                f" {self.n_envs}, {self.episode_length}, {self.seed_steps}"
            )

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
        after = jnp.where(tick > self.seed_tick, self.n_envs, 0)
        stepping = jnp.where(tick == self.seed_tick, self.seed_steps, after)
        return jnp.where(self.is_held(tick), 0, stepping).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


def planner_policy(
    carry: jax.Array,
    obs: jax.Array,
    is_first: jax.Array,
    key: jax.Array,
    *,
    wm_params: Any,
    pi_params: Any,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
    eval_mode: bool,
    task: Optional[core.TaskContext] = None,
) -> tuple[jax.Array, jax.Array, None]:
    """The row collector's policy: one MPPI decision per env.

    :func:`ajax.agents.TDMPC2.planner.plan` vmapped over the leading env
    axis of ``obs [m, obs_dim]``, ``is_first [m]`` (the reference's ``t0``)
    and the warm-start carry ``prev_mean [m, H, A]``, each env with its own
    draws. Returns ``(action [m, A], new prev_mean, None)``. ``task`` is
    the multi-task conditioning of one task shared by the envs (M8,
    :func:`ajax.agents.TDMPC2.multitask.task_planner_policy`); ``None`` is
    the single-task planner.
    """
    n, action_dim = carry.shape[0], carry.shape[-1]
    noise = jax.vmap(lambda k: draw_plan_noise(k, config, action_dim))(
        jax.random.split(key, n)
    )
    decide = partial(plan, config=config, gamma=gamma, eval_mode=eval_mode, task=task)
    action, prev_mean, _ = jax.vmap(decide, in_axes=(None, None, 0, 0, 0, 0))(
        wm_params, pi_params, obs, carry, is_first, noise
    )
    return action, prev_mean, None


def _policy(
    agent_state: TDMPC2State, config: TDMPC2Config, gamma: float, eval_mode: bool
) -> Callable:
    return partial(
        planner_policy,
        wm_params=agent_state.world_model_state.params,
        pi_params=agent_state.actor_state.params,
        config=config,
        gamma=gamma,
        eval_mode=eval_mode,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def update_metrics_zeros(
    learner: core.TDMPC2UpdateState,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
    obs_dim: int,
    action_dim: int,
    task: Optional[core.TaskContext] = None,
) -> dict[str, jax.Array]:
    """Zeros with the structure of :func:`core.update`'s logged quantities
    (``task``: a batch's multi-task conditioning, M8; ``None``: single
    task)."""
    h, b = config.horizon, config.batch_size
    batch = core.TDMPC2Batch(
        obs=jnp.zeros((h + 1, b, obs_dim)),
        action=jnp.zeros((h, b, action_dim)),
        reward=jnp.zeros((h, b)),
    )
    noise = core.draw_update_noise(jax.random.PRNGKey(0), config, b, action_dim)

    def logged(state: Any) -> dict[str, jax.Array]:
        _, metrics = core.update(
            state, batch, noise, config=config, gamma=gamma, task=task
        )
        return metrics

    shapes = jax.eval_shape(logged, learner)
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)


def init_TDMPC2(
    key: jax.Array,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    buffer: EpisodeBuffer,
    *,
    gamma: float,
    learning_rate: FloatOrCallable,
    enc_lr_scale: float,
    pi_eps: float,
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
        update_metrics=update_metrics_zeros(
            learner, config, gamma, obs_dim, action_dim
        ),
        n_terminations=jnp.zeros((), jnp.int32),
        n_updates=jnp.zeros((), jnp.int32),
        n_logs=jnp.zeros((), jnp.int32),
    )


# ---------------------------------------------------------------------------
# Evaluation + logging
# ---------------------------------------------------------------------------


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
    folded with the number of evaluations so far (``agent_state.n_logs``):
    each evaluation starts from fresh initial states, as the reference's
    resets of its running env do, deterministically per seed.
    """
    action_dim = get_action_dim(env_args.env, env_args.env_params)
    ret, length = evaluate_policy(
        env_args,
        _policy(agent_state, config, gamma, eval_mode=True),
        lambda m: jnp.zeros((m, config.horizon, action_dim)),
        num_episodes,
        jax.random.fold_in(key, agent_state.n_logs),
    )
    return {"Eval/episodic mean reward": ret, "Eval/mean episodic length": length}


def train_metrics(
    agent_state: TDMPC2State, aux: Any, *, action_repeat: int
) -> dict[str, jax.Array]:
    """The house keys, ``env_frames`` and the last update's quantities."""
    del aux
    cs = agent_state.collector_state
    metrics = {
        "timestep": cs.timestep,
        "env_frames": cs.timestep * action_repeat,
        "Train/episodic mean reward": cs.episodic_mean_return,
        "Train/n_updates": agent_state.n_updates,
        "Train/offschedule_dones": cs.n_offschedule_dones,
        "Train/terminations": agent_state.n_terminations,
    }
    metrics.update({f"Train/{k}": v for k, v in agent_state.update_metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
# One tick
# ---------------------------------------------------------------------------


def collect(
    agent_state: TDMPC2State,
    tick: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    schedule: Schedule,
    buffer: EpisodeBuffer,
    gamma: float,
) -> tuple[TDMPC2State, jax.Array]:
    """Act and store one row per env on the absolute, unbatched tick
    ``tick``; returns the state and ``tick``, which locates the buffer's
    newest rows for the tick's updates (:func:`update`)."""
    collector_state, row = collect_row(
        agent_state.collector_state,
        tick,
        _policy(agent_state, config, gamma, eval_mode=False),
        env_args=env_args,
        reset_mode="static",
        episode_length=schedule.episode_length,
        random_phase=schedule.random_phase(tick),
        timestep_unit="env_steps",
    )
    agent_state = agent_state.replace(
        collector_state=collector_state,
        buffer_state=buffer.add(
            agent_state.buffer_state, row.obs, row.action, row.reward, tick
        ),
        # True terminations on scheduled episode ends (the off-schedule
        # ones are the collector's n_offschedule_dones).
        n_terminations=agent_state.n_terminations
        + jnp.sum(row.is_terminal, dtype=jnp.int32),
    )
    return agent_state, tick


def update(
    agent_state: TDMPC2State,
    tick: jax.Array,
    *,
    config: TDMPC2Config,
    buffer: EpisodeBuffer,
    gamma: float,
) -> tuple[TDMPC2State, None]:
    """One ``agent.update(buffer)`` after tick ``tick``'s rows: a fresh batch
    and fresh noise, then :func:`core.update` (tdmpc2_spec §2), its logged
    quantities kept on ``update_metrics``. The loop folds ``post_update``
    after it."""
    rng, sample_key, noise_key = jax.random.split(agent_state.rng, 3)
    batch = buffer.sample(
        agent_state.buffer_state, sample_key, tick, config.batch_size, config.horizon
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: TDMPC2Config,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    *,
    gamma: float,
    schedule: Schedule,
    buffer: EpisodeBuffer,
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
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
    from ``learning_rate``, ``enc_lr_scale`` and ``pi_eps``; the base class's
    actor / critic optimizer and network configs do not apply.
    """
    del actor_optimizer_args, critic_optimizer_args, network_args
    config = agent_config
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
    )

    def init(key: jax.Array, pretrain_key: jax.Array) -> TDMPC2State:
        del pretrain_key  # no pretraining of its own
        return init_TDMPC2(
            key,
            env_args,
            config,
            buffer,
            gamma=gamma,
            learning_rate=learning_rate,
            enc_lr_scale=enc_lr_scale,
            pi_eps=pi_eps,
        )

    models = {"config": config, "buffer": buffer, "gamma": gamma}
    every = loop.log_frequency and max(schedule.num_ticks(loop.log_frequency), 1)
    return loop.off_policy(
        init,
        partial(update, **models),
        collect=partial(collect, env_args=env_args, schedule=schedule, **models),
        n_updates=schedule.n_updates,
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
    "Schedule",
    "collect",
    "evaluate_tdmpc2",
    "init_TDMPC2",
    "make_train",
    "planner_policy",
    "train_metrics",
    "update",
    "update_metrics_zeros",
]
