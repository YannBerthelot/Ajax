"""TD-MPC2 (Hansen, Su, Wang, ICLR 2024; arXiv:2310.16828v2), single-task online.

The paper-era algorithm (nicklashansen/tdmpc2@b67b21c = 5f6fade for the
agent; docs/world_models/DESIGN.md section 4, tdmpc2_spec sections 1-4,
version choices and deviations in docs/world_models/deviations.md): a
decoder-free world model (encoder, latent dynamics, two-hot reward and Q
ensemble) and a policy prior trained on sub-trajectories of a whole-episode
replay (core, buffer), acting by MPPI planning in latent space (planner).

One tick of the own-replay family's loop is the reference's online loop
(b67b21c tdmpc2/trainer/online_trainer.py:67-117, unchanged at 5f6fade;
tdmpc2_spec section 4.C) on obs-aligned rows (DESIGN.md sections 2, 4.4),
each line a method or class below::

    act: uniform random up to the seed tick and on held ticks, else
        the MPPI planner, its warm start reset where an episode starts  collect
    store one row per env in the episode ring, in place                collect
    the burst of seed_steps updates on the seed tick, then updates_per_step
        per env step (deviation T28)                       FixedLengthSchedule
    each update on a fresh batch with fresh noise                       update
    every log_frequency env steps: evaluate and log           report, evaluate

T + 1 ticks make an episode of T agent steps: T stepping ticks, then the
held tick that emits the final observation and resets the envs with a fresh
key. With one env this is b67b21c exactly: random actions at env steps 0 ..
S, a burst of S updates right after step S, then one update per env step;
more envs are a generalisation (deviation T7). n_timesteps counts env (agent)
steps, summed over envs; logs add env_frames = env steps * action_repeat,
the papers' x-axis. The paper's DMC protocol, the replay's memory and the
evaluation's cost: docs/algorithms/tdmpc2.md.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.agents.families import OwnReplayAgent, TrainContext
from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.buffer import (
    EpisodeBuffer,
    EpisodeBufferState,
    EpisodicBuffer,
    EpisodicBufferState,
    make_episode_buffer,
    replay_capacity,
)
from ajax.agents.TDMPC2.planner import plan_for_envs
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.controller import feature_width
from ajax.environments.row_collector import (
    RowCollectorState,
    collect_row,
    init_row_collector_state,
)
from ajax.environments.utils import (
    agent_episode_length,
    get_action_dim,
    get_state_action_shapes,
)
from ajax.evaluate import evaluate_policy
from ajax.normalizers import RunningScale
from ajax.policy import PolicyParts, row_collector_policy
from ajax.state import BaseAgentState, LoadedTrainState
from ajax.types import FloatOrCallable

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=False, repr=False)
class TDMPC2(OwnReplayAgent):
    """TD-MPC2 for continuous control, single task, online (the module's tick).

    Every hyperparameter is the one of the same name in
    nicklashansen/tdmpc2@5f6fade:tdmpc2/config.yaml (lr is learning_rate),
    with its paper default; setup builds model_config (TDMPC2Config) from
    them, the widths left None taking the model_size preset's (tdmpc2_spec
    1.20). learning_rate is a float or a schedule, the encoder's
    learning_rate * enc_lr_scale (5f6fade:tdmpc2/tdmpc2.py:21-28);
    weight_decay is Ajax's (0 keeps the reference's plain Adam). Left None,
    two derive from the episode length T in agent steps
    (ajax.environments.utils.agent_episode_length), into resolved_gamma and
    resolved_seed_steps: gamma, clip((T/denom - 1) / (T/denom),
    discount_min, discount_max) (tdmpc2_spec 2.20, tdmpc2.py:36-49), and
    seed_steps, max(1000, 5 T) (4.3, envs/__init__.py:81).

    The replay (buffer, the b67b21c episode ring) keeps min(buffer_size,
    total) env steps, total being the run's when a train call ends (the
    reference's clamp to the run length, deviation T26): every call sizes it
    (prepare_replay), moving a resumed run's episodes into it, so a run
    trained in chunks replays what the uninterrupted run does.

    Continuous actions only. The env must be fixed-length and
    non-terminating (all DMC tasks): a termination raises ValueError after
    train (deviation T10). episodic=True takes terminating tasks instead (the
    latest code's episodic mode, nicklashansen/tdmpc2@eece801; deviation
    T29): a termination head, 1 - terminated in the targets and the planner,
    a replay of variable-length episodes and n_timesteps counting rows (env
    steps plus one per finished episode). Observations and rewards are the
    env's raw values, as in the reference.

    Extensions: pretrain, post_update (after every update, step = env steps),
    eval_metrics (with init_state) and observe (the observation features,
    which the row collector and the evaluation append and the encoder reads).
    """

    #: Simulator steps per episode of a brax or playground env.
    episode_length: int = 1000
    # The size (tdmpc2_spec 1.20): a preset of 1, 5, 19, 48 or 317 M
    # parameters, or explicit widths overriding it.
    model_size: int = 5
    enc_dim: Optional[int] = None
    mlp_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    num_enc_layers: Optional[int] = None
    num_q: Optional[int] = None
    simnorm_dim: int = 8
    num_bins: int = 101
    vmax: float = 10.0
    dropout: float = 0.01
    # One update (tdmpc2_spec section 2).
    horizon: int = 3
    rho: float = 0.5
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    learning_rate: FloatOrCallable = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    pi_eps: float = 1e-5
    tau: float = 0.01
    entropy_coef: float = 1e-4
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    # The replay and the schedule (tdmpc2_spec section 4): seed_steps random
    # steps, their burst of updates, then updates_per_step per env step (the
    # reference's 1, which config.yaml does not expose; deviation T28).
    batch_size: int = 256
    buffer_size: int = 1_000_000
    seed_steps: Optional[int] = None
    updates_per_step: float = 1.0
    # The discount (tdmpc2_spec 2.20).
    gamma: Optional[float] = None
    discount_denom: float = 5
    discount_min: float = 0.95
    discount_max: float = 0.995
    # The MPPI planner (tdmpc2_spec section 3).
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    # The latest code's episodic mode (deviation T29).
    episodic: bool = False
    termination_coef: float = 1.0

    def setup(self) -> None:
        self.model_config = TDMPC2Config.from_arguments(self.config)
        env, params = self.env_args.env, self.env_args.env_params
        obs_shape, _ = get_state_action_shapes(env)
        if len(obs_shape) != 1:
            raise ValueError(
                f"TD-MPC2 takes flat state observations, got shape {obs_shape}"
            )
        # The env's columns, then the observation features the collector appends.
        self.obs_dim = int(obs_shape[0]) + feature_width(self.env_args)
        self.action_dim = int(get_action_dim(env, params))
        # T, in agent steps: never the episode_length argument directly.
        self.agent_episode_length = agent_episode_length(
            env, params, self.action_repeat
        )
        if self.agent_episode_length < self.horizon:
            raise ValueError(
                f"episodes of T = {self.agent_episode_length} agent steps are too"
                f" short for horizon {self.horizon}: a training slice needs"
                " horizon + 1 <= T + 1 rows"
            )
        #: The discount used: gamma, or derived from T (class docstring).
        self.resolved_gamma = (
            core.discount_from_episode_length(
                self.agent_episode_length,
                self.discount_denom,
                self.discount_min,
                self.discount_max,
            )
            if self.gamma is None
            else float(self.gamma)
        )
        if not 0.0 < self.resolved_gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.resolved_gamma}")
        #: The seed phase used: seed_steps, or derived from T.
        self.resolved_seed_steps = (
            max(1000, 5 * self.agent_episode_length)
            if self.seed_steps is None
            else int(self.seed_steps)
        )
        schedule = {
            "n_envs": self.n_envs,
            "episode_length": self.agent_episode_length,
            "seed_steps": self.resolved_seed_steps,
        }
        ratio = float(self.updates_per_step)
        self.schedule: FixedLengthSchedule | EpisodicSchedule = (
            EpisodicSchedule(**schedule, updates_per_step=ratio)
            if self.episodic
            else FixedLengthSchedule(**schedule, updates_per_step=ratio)
        )
        # The episodic mode's updates wait on each seed's ready episodes
        # (count_updates), so its seeds may update unequally; the paper-era
        # mode's follow the tick, in step.
        self.per_seed_update_counts = self.episodic
        #: The replay's capacity and bytes per seed of the latest train call.
        self.replay_capacity: Optional[int] = None
        self.replay_bytes_per_seed: Optional[int] = None
        # The values actually used, next to the constructor arguments (whose
        # None defaults are resolved above), for the loggers' run config.
        self.config.update(
            {
                "agent_episode_length": self.agent_episode_length,
                "resolved_gamma": self.resolved_gamma,
                "resolved_seed_steps": self.resolved_seed_steps,
            }
        )
        self.config.update(self.model_config.resolved(self.action_dim))

    def run_capabilities(self) -> frozenset[str]:
        """The shared run's capabilities, with warmup_steps: whatever
        seed_steps, its seed phase acts uniformly for at least one episode
        (both schedules' seed tick is at least the episode length), where
        the shared rule reads learning_starts."""
        return super().run_capabilities().union({"warmup_steps"})

    def prepare_replay(
        self, start_tick: int, end_tick: int, agent_state: Optional[TDMPC2State]
    ) -> tuple[EpisodeBuffer | EpisodicBuffer, Optional[TDMPC2State]]:
        """The replay of a call ending at tick end_tick, for the env steps
        (rows in the episodic mode) the run has taken by then, and the state
        it resumes from, its carried episodes moved into that replay (adopt).
        Records the replay's capacity and bytes, also in the run config."""
        # Per env, the rows the replay stores by then: env steps, or every
        # tick's row in the episodic mode (held ones included).
        per_env = (
            end_tick
            if isinstance(self.schedule, EpisodicSchedule)
            else self.schedule.steps_before(end_tick)
        )
        capacity = replay_capacity(self.buffer_size, per_env * self.n_envs)
        buffer = make_episode_buffer(
            capacity,
            self.n_envs,
            self.agent_episode_length,
            self.obs_dim,
            self.action_dim,
            self.episodic,
        )
        self.replay_capacity = capacity
        self.replay_bytes_per_seed = buffer.nbytes
        self.config.update(
            {"replay_capacity": capacity, "replay_bytes_per_seed": buffer.nbytes}
        )
        logger.info(
            "TD-MPC2 replay: capacity %d env steps, %.1f MB per seed",
            capacity,
            buffer.nbytes / 1e6,
        )
        if agent_state is not None:
            # The carried ring is of this buffer's kind (episodic or not).
            replay = buffer.adopt(agent_state.buffer_state, start_tick)  # type: ignore[arg-type]
            agent_state = agent_state.replace(buffer_state=replay)
        return buffer, agent_state

    def init(self, key: jax.Array, context: TrainContext) -> TDMPC2State:
        """Fresh networks and optimizers (core.create_update_state), a zero
        planner carry, reset envs (every env's first row is is_first) and an
        empty replay."""
        model_key, collector_key, training_key, evaluation_key = jax.random.split(
            key, 4
        )
        config = self.model_config
        learner = core.create_update_state(
            model_key,
            config,
            self.obs_dim,
            self.action_dim,
            learning_rate=self.learning_rate,
            enc_lr_scale=float(self.enc_lr_scale),
            pi_eps=float(self.pi_eps),
            weight_decay=float(self.weight_decay),
        )
        # The planner's warm-start mean per env (the reference's prev_mean).
        planner_carry = jnp.zeros(
            (context.env_args.n_envs, config.horizon, self.action_dim)
        )
        collector_state = init_row_collector_state(
            collector_key, context.env_args, policy_carry=planner_carry
        )
        return TDMPC2State(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=learner.actor_state,
            world_model_state=learner.world_model_state,
            q_scale=learner.q_scale,
            pi_gradnorm_sq=learner.pi_gradnorm_sq,
            collector_state=collector_state,
            buffer_state=context.buffer.init(),
            update_metrics=core.make_zero_update_metrics(
                learner, config, self.resolved_gamma, self.obs_dim, self.action_dim
            ),
            n_terminations=jnp.zeros((), jnp.int32),
            n_updates=jnp.zeros((), jnp.int32),
            n_logs=jnp.zeros((), jnp.int32),
        )

    def collect(
        self, agent_state: TDMPC2State, tick: jax.Array, context: TrainContext
    ) -> tuple[TDMPC2State, jax.Array]:
        """Act and store one row per env on the absolute, unbatched tick;
        returns the state and the tick, which locates the replay's newest rows
        for the tick's updates. The episodic mode collects in the dynamic
        reset mode, counting rows."""
        previous_collector = agent_state.collector_state
        collector_state, row = collect_row(
            previous_collector,
            tick,
            self.make_planner_policy(agent_state, self.resolved_gamma, eval_mode=False),
            env_args=context.env_args,
            reset_mode="dynamic" if self.episodic else "static",
            episode_length=self.schedule.episode_length,  # the static mode's
            random_phase=self.schedule.random_phase(tick),
            timestep_unit="rows" if self.episodic else "env_steps",
        )
        replay, buffer = agent_state.buffer_state, context.buffer
        if isinstance(buffer, EpisodicBuffer):
            assert isinstance(replay, EpisodicBufferState)
            # The row's index in its episode: a final row (the env held) is
            # its episode's last, after last_episode_length steps.
            assert previous_collector.step_in_episode is not None
            step = jnp.where(
                previous_collector.is_last,
                previous_collector.last_episode_length,
                previous_collector.step_in_episode,
            )
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

    def count_updates(
        self, agent_state: TDMPC2State, tick: jax.Array, context: TrainContext
    ) -> jax.Array:
        """The schedule's updates of the tick. The episodic mode's follow the
        envs the tick stepped (those not held: the collector sets is_first
        exactly after a held tick) and wait for a sampleable episode."""
        if isinstance(self.schedule, FixedLengthSchedule):
            return self.schedule.n_updates(tick)
        replay, buffer = agent_state.buffer_state, context.buffer
        assert isinstance(buffer, EpisodicBuffer)
        assert isinstance(replay, EpisodicBufferState)
        held = agent_state.collector_state.is_first  # held this tick
        stepped = context.env_args.n_envs - jnp.sum(held, dtype=jnp.int32)
        ready = buffer.num_episodes(replay, tick, self.model_config.horizon)
        return jnp.where(ready > 0, self.schedule.n_updates(tick, stepped), 0)

    def update(
        self, agent_state: TDMPC2State, tick: jax.Array, context: TrainContext
    ) -> tuple[TDMPC2State, None]:
        """The reference's agent.update(buffer) after the tick's rows: a fresh
        batch and fresh noise, then core.update (tdmpc2_spec section 2), its
        logged quantities kept on update_metrics. The loop folds post_update
        after it."""
        config, buffer = self.model_config, context.buffer
        sample_key, noise_key, agent_state = agent_state.get_keys_and_new_state(2)
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
            agent_state,
            batch,
            noise,
            config=config,
            gamma=self.resolved_gamma,
        )
        agent_state = agent_state.replace(
            n_updates=agent_state.n_updates + 1, update_metrics=metrics
        )
        return agent_state, None

    def evaluate(
        self,
        agent_state: TDMPC2State,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """Mean return and length of num_episode_test episodes of the planner
        in eval_mode (no final exploration noise, everything else stochastic:
        online_trainer.py:27-48, tdmpc2_spec 4.21) on a rebuilt env, its carry
        from zeros, apart from the training one (deviation T9). The key is
        fresh at each evaluation (ajax.log.evaluation_keys), so each starts
        from new initial states, as the reference's resets of its env do."""
        horizon = self.model_config.horizon
        mean_return, mean_length = evaluate_policy(
            context.env_args,
            self.make_planner_policy(agent_state, self.resolved_gamma, eval_mode=True),
            lambda n_envs: jnp.zeros((n_envs, horizon, self.action_dim)),
            num_episode_test,
            key,
            agent_state=agent_state,
        )
        return {
            "Eval/episodic mean reward": mean_return,
            "Eval/mean episodic length": mean_length,
        }

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """As its evaluation acts: the planner in eval_mode, its carry from
        zeros."""
        horizon = self.model_config.horizon

        def parts(agent_state: TDMPC2State) -> PolicyParts:
            return row_collector_policy(
                self.env_args,
                agent_state,
                self.make_planner_policy(agent_state, self.resolved_gamma, True),
                lambda n_envs: jnp.zeros((n_envs, horizon, self.action_dim)),
            )

        return parts

    def report(self, agent_state: TDMPC2State, aux: Any) -> dict:
        """The house keys, env_frames and the last update's quantities."""
        del aux
        collector_state = agent_state.collector_state
        metrics = {
            "timestep": collector_state.timestep,
            "env_frames": collector_state.env_steps * self.env_args.action_repeat,
            "Train/episodic mean reward": collector_state.episodic_mean_return,
            "Train/n_updates": agent_state.n_updates,
            "Train/offschedule_dones": collector_state.n_offschedule_dones,
            "Train/terminations": agent_state.n_terminations,
        }
        metrics.update(
            {
                f"Train/{name}": value
                for name, value in agent_state.update_metrics.items()
            }
        )
        return metrics

    def count_ticks(self, n_timesteps: int, start_tick: int) -> int:
        # The run ends on the tick of its last env step (the schedule's).
        return self.schedule.num_ticks(n_timesteps, start_tick)

    def check_run(self, agent_state: TDMPC2State, end_tick: int) -> None:
        """Refuse a paper-era run whose env terminated (deviation T10): the
        paper-era TD-MPC2 bootstraps through every episode end, so a
        terminating task would train on wrong targets; and an episodic run
        that never updated."""
        if not self.episodic:
            refuse_terminations(agent_state, self.agent_episode_length)
        elif end_tick > self.schedule.seed_tick + self.agent_episode_length:
            # By then every episode running at the seed tick has ended, so
            # an episode of horizon + 1 rows would have been drawn from.
            refuse_untrained(agent_state, self.horizon)

    def make_planner_policy(
        self, agent_state: TDMPC2State, gamma: float, eval_mode: bool
    ) -> Callable:
        """plan_for_envs with the agent's current parameters and the gamma
        its caller passes bound: the row collector's policy and the
        evaluation's."""
        return partial(
            plan_for_envs,
            wm_params=agent_state.world_model_state.params,
            pi_params=agent_state.actor_state.params,
            config=self.model_config,
            gamma=gamma,
            eval_mode=eval_mode,
        )

    def extension_values(self, configs: dict[str, Any]) -> dict[str, Any]:
        # The discount used, over the gamma field (None where it is derived).
        del configs
        return {"gamma": self.resolved_gamma}

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "TDMPC2"
    capabilities = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "next_action_target",
            "off_policy",
            "policy_density",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
            "warmup",
        }
    )
    # Its row collector and evaluation append the observation features.
    supported_extension_phases = OwnReplayAgent.supported_extension_phases.union(
        {"observe"}
    )


def refuse_untrained(state: Any, horizon: int) -> None:
    """Raise if an episodic run never updated: the replay only draws
    episodes of at least horizon + 1 rows (the reference's strict-length
    slices), and every episode ended within horizon agent steps (the state's
    per-seed update counts)."""
    n_updates = np.asarray(jax.device_get(state.n_updates)).reshape(-1)
    if np.any(n_updates == 0):
        raise ValueError(
            f"TD-MPC2 never updated (updates per seed {n_updates.tolist()}):"
            f" a training slice needs an episode of horizon + 1 = {horizon + 1}"
            f" rows, but every episode terminated within {horizon} agent steps."
            " Lower the horizon or lengthen the episodes."
        )


def refuse_terminations(state: Any, episode_length: int) -> None:
    """Raise if a paper-era run's env ended an episode off its
    episode_length-step schedule or terminated on its last step (the state's
    per-seed counters; deviation T10)."""
    offschedule = np.asarray(
        jax.device_get(state.collector_state.n_offschedule_dones)
    ).reshape(-1)
    terminations = np.asarray(jax.device_get(state.n_terminations)).reshape(-1)
    if np.any(offschedule > 0) or np.any(terminations > 0):
        raise ValueError(
            "TD-MPC2 needs fixed-length, non-terminating episodes, but the env"
            f" ended episodes off the T = {episode_length}-step"
            f" schedule {offschedule.tolist()} times and terminated on its"
            f" last step {terminations.tolist()} times (per seed):"
            " terminations are not supported (paper-era TD-MPC2; deviation"
            " T10)."
        )


# -- The tick's schedules --------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FixedLengthSchedule:
    """The paper-era mode's seed phase, burst and updates (DESIGN.md section
    4.4), for n lockstep envs, episodes of T agent steps, S seed steps and r
    updates per env step after the burst (the reference's 1; deviation T28).

    On the absolute tick i (period P = T + 1; stepping ticks i mod P < T
    advance every env by one step, the held tick does not):

    - the seed tick is the stepping tick of per-env step k_S = max(S div n,
      T), the first after which the total env steps n (k + 1) exceed S and at
      least one round (n episodes) is committed. For one env k_S = S (S >= T
      is the reference's max(1000, 5 T)): steps 0 .. S act randomly
      (online_trainer.py:97-100);
    - actions are uniform random on every tick up to the seed tick, and on
      held ticks, whose action is discarded, so the planner never runs there;
    - n_updates(i) is S on the seed tick (the pretraining burst,
      online_trainer.py:105-112), updates_per_tick on later stepping ticks
      and 0 otherwise.

    The methods take Python ints, NumPy arrays and traced (unbatched) ticks.
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
        """Updates per stepping tick after the burst: max(1, round(n r))
        (Python's round, halves to even); n for the reference's r = 1. At
        least one, so a small r never stops training."""
        return max(1, round(self.n_envs * self.updates_per_step))

    @property
    def period(self) -> int:
        """Ticks per episode, T + 1."""
        return self.episode_length + 1

    def tick_of_step(self, step: int) -> int:
        """The tick of per-env env step step, (step div T) P + step mod T:
        also the number of ticks from tick 0 that take step env steps per env
        (a trailing held tick included after a whole episode)."""
        length = self.episode_length
        return (step // length) * self.period + step % length

    def steps_before(self, tick: int) -> int:
        """Per-env env steps taken by ticks 0 .. tick - 1."""
        return (tick // self.period) * self.episode_length + min(
            tick % self.period, self.episode_length
        )

    def num_ticks(self, n_timesteps: int, start_tick: int = 0) -> int:
        """Ticks of a run of n_timesteps env steps (n_timesteps div n per
        env) from the absolute tick start_tick (a resume). A split run covers
        exactly the ticks of an uninterrupted one: the run ends on the tick of
        its last env step, plus the held tick when that step ends an
        episode."""
        end_step = self.steps_before(start_tick) + n_timesteps // self.n_envs
        return self.tick_of_step(end_step) - start_tick

    @property
    def seed_step(self) -> int:
        """k_S: the per-env env step of the seed tick. At least T, so the seed
        tick follows the first committed round: the sampler never draws from
        an empty replay (committed rounds only grow with the tick)."""
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
    observation, so ticks are not env steps. With k_S = max(S div n, T):
    random actions on ticks 0 .. k_S (the seed phase counts rows, held ones
    included; at least T + 1 ticks, so every env has finished an episode);
    the burst of S on tick k_S, then, on a tick that stepped stepped > 0
    envs, max(1, round(stepped r)) updates (r = updates_per_step, T28: one
    per env step by default); a run of n_timesteps rows lasts n_timesteps
    div n ticks.
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
        # jnp.round, like Python's round in FixedLengthSchedule, rounds halves
        # to even.
        per_tick = jnp.maximum(1, jnp.round(stepped * self.updates_per_step))
        after = jnp.where((tick > self.seed_tick) & (stepped > 0), per_tick, 0)
        return jnp.where(tick == self.seed_tick, self.seed_steps, after).astype(
            jnp.int32
        )


# -- State ---------------------------------------------------------------------
@struct.dataclass(kw_only=True)
class TDMPC2State(BaseAgentState):
    """The TD-MPC2 agent's state (DESIGN.md section 4.5), keyword-only so
    the narrowed critic_state default can sit before the new required fields.
    It carries the four attributes of state.TDMPC2UpdateState, so core.update
    takes it directly.

    Attributes:
        critic_state: None; the Q ensemble belongs to the world model.
        collector_state: the row collector's state; its policy_carry is the
            planner's per-env warm-start mean prev_mean [n_envs, H, A], its
            timestep counts env (agent) steps.
        buffer_state: the episode ring (buffer.EpisodeBuffer; the episodic
            mode's buffer.EpisodicBuffer).
        update_metrics: the last update's logged quantities, kept across
            ticks without updates (held ticks, seed phase) for logging, as
            the reference keeps its last train_metrics.
        n_terminations: true terminations on scheduled episode ends
            (is_terminal rows), which with the collector's
            n_offschedule_dones make the agent refuse terminating tasks after
            train (deviation T10); every termination in the episodic mode.
    """

    critic_state: Optional[LoadedTrainState] = None  # type: ignore[assignment]
    world_model_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array
    collector_state: RowCollectorState
    buffer_state: EpisodeBufferState | EpisodicBufferState
    update_metrics: dict[str, jax.Array]
    n_terminations: jax.Array
