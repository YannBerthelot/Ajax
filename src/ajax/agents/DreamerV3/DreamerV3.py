"""DreamerV3 (Hafner, Pasukonis, Ba, Lillicrap, arXiv:2301.04104v2 / Nature 2025).

The paper-era recipe of danijar/dreamerv3@2411f7d with the upstream
replay-context fix 29eb964 (docs/world_models/DESIGN.md section 6,
docs/world_models/deviations.md section 1), for vector observations, in
float32 and synchronous (deviations D1-D6): a world model (block-GRU RSSM,
encoder, decoder, reward and continue heads) learned from replayed
sequences, an actor and a critic learned in its imagination, one joint
gradient per update and LaProp with adaptive gradient clipping.

One tick of the own-replay family's loop is the reference's driver loop
(embodied/run/train.py with embodied/core/driver.py and when.Ratio) made
synchronous and seeded (dreamerv3_spec Algorithm K; DESIGN.md sections
6.4-6.6), each line a method or function below::

    filter the observation into the posterior, then sample the actor
        (no deterministic mode; dreamerv3_spec 7.1, 7.3)   world_model.observe_and_act
    store one row per env, its action zeroed at is_last and its posterior
        latent kept for the replay context                            collect
    after the training start, the updates the train ratio gives the tick
        (deviations D3, D26)                                         TrainRatio
        each on a fresh batch, online queue first, with fresh noise,
        then the latents written back (deviation D1)                   update
    every log_frequency rows: the training episodes' returns (the paper's
        score), sampled evaluation episodes, the mean training metrics
                                                              report, evaluate

n_timesteps counts rows, the reference's step: one per env per tick, reset
rows included; env_frames = rows * action_repeat is logged beside it. The
paper's DMC protocol and the replay's memory: docs/algorithms/dreamerv3.md.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.agents.DreamerV3.learner import (
    LearnerState,
    draw_train_noise,
    init_learner,
    list_train_metrics,
    train_step,
)
from ajax.agents.DreamerV3.replay import (
    ReplayState,
    StreamReplay,
    count_held_rows,
    count_replay_bytes,
    grow_rings,
)
from ajax.agents.DreamerV3.state import DreamerV3Config, LearningRate
from ajax.agents.DreamerV3.world_model import make_initial_carry, observe_and_act
from ajax.agents.families import OwnReplayAgent, TrainContext
from ajax.buffers.ring import count_ring_rows, size_ring
from ajax.controller import feature_width
from ajax.environments.row_collector import (
    RowCollectorState,
    collect_row,
    init_row_collector_state,
)
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.evaluate import evaluate_policy
from ajax.log import MetricsAccumulator
from ajax.normalizers import ReturnNormalizer
from ajax.policy import PolicyParts, row_collector_policy
from ajax.state import BaseAgentState, EnvironmentConfig, LoadedTrainState

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=False, repr=False)
class DreamerV3(OwnReplayAgent):
    """DreamerV3, paper-era (2411f7d + 29eb964), for vector observations and
    either action space (the module's tick).

    Every hyperparameter of the world model, the actor-critic and the
    optimiser (weight_decay included) is the DreamerV3Config field of the
    same name and default, documented there; setup builds model_config from
    them, the widths left None taking the model_size preset's. The env must
    not normalise observations or rewards.
    """

    #: The reference's run.num_envs.
    n_envs: int = 16
    #: Simulator steps per episode of a brax or playground env.
    episode_length: int = 1000
    # The size (configs.yaml size12m ... size400m; dreamerv3_spec 2.1): a
    # preset of the model dimension d, or explicit widths overriding it.
    model_size: str = "12m"
    units: Optional[int] = None
    deter: Optional[int] = None
    hidden: Optional[int] = None
    classes: Optional[int] = None
    stoch: int = 32
    blocks: int = 8
    # The world model (configs.yaml dyn, enc, dec, rewhead, conhead,
    # rssm_loss and loss_scales).
    enc_layers: int = 3
    dec_layers: int = 3
    rew_layers: int = 1
    con_layers: int = 1
    bins: int = 255
    free_nats: float = 1.0
    unimix: float = 0.01
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    rec_scale: float = 1.0
    rew_scale: float = 1.0
    con_scale: float = 1.0
    # The actor-critic (29eb964 configs.yaml:127-146).
    actor_layers: int = 3
    critic_layers: int = 3
    actor_unimix: float = 0.01
    actor_scale: float = 1.0
    critic_scale: float = 1.0
    repval_scale: float = 0.3
    imag_horizon: int = 15
    return_horizon: float = 333
    lam: float = 0.95
    repval_lam: float = 0.95
    actent: float = 3e-4
    slowreg: float = 1.0
    slow_rate: float = 0.02
    retnorm_rate: float = 0.01
    retnorm_limit: float = 1.0
    minstd: float = 0.1
    maxstd: float = 1.0
    # LaProp with adaptive gradient clipping (configs.yaml opt).
    learning_rate: LearningRate = 4e-5
    agc: float = 0.3
    agc_pmin: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-20
    warmup: int = 1000
    # The schedule and the replay (configs.yaml run.train_ratio, batch_size,
    # batch_length with replay_context 1; Table 4's capacity): an update every
    # batch_size * batch_length / train_ratio rows after the training start,
    # on batch_size windows of batch_length + 1 rows; replay_capacity rows
    # kept over all envs.
    train_ratio: float = 512
    batch_size: int = 16
    batch_length: int = 64
    replay_capacity: int = 5_000_000
    #: Not a keyword: 1 - 1 / return_horizon (setup), the discount of the
    #: continue target and of the replay critic.
    gamma: float = dataclasses.field(init=False)

    def setup(self) -> None:
        # The reference reads a float horizon, whatever the caller passed.
        arguments = dict(self.config)
        arguments.update({"return_horizon": float(self.return_horizon)})
        self.model_config = DreamerV3Config.from_arguments(arguments)
        # The values actually used, next to the constructor arguments (whose
        # None widths the preset resolves), for the loggers' run config.
        self.config.update(self.model_config.resolved())
        self.gamma = self.model_config.gamma
        self.schedule = TrainRatio(
            n_envs=self.n_envs,
            batch_size=self.batch_size,
            batch_length=self.batch_length,
            train_ratio=self.train_ratio,
        )
        self.spec = measure_env(self.env_args)  # refuses what it cannot encode
        #: Rows per env the capacity keeps (in rows: deviation D27).
        self.capacity_rows = -(-int(self.replay_capacity) // self.n_envs)
        if self.capacity_rows < self.schedule.min_ring_rows():
            raise ValueError(
                f"replay_capacity={self.replay_capacity} keeps"
                f" {self.capacity_rows} rows per env, fewer than the"
                f" {self.schedule.min_ring_rows()} that hold the first batch's"
                f" {self.batch_size} windows of {self.batch_length + 1} rows."
            )
        #: The ring length C (rows per env) of the latest train() call.
        self.replay_rows_per_env: Optional[int] = None

    def prepare_replay(
        self, start_tick: int, end_tick: int, agent_state: Optional[DreamerV3State]
    ) -> tuple[StreamReplay, Optional[DreamerV3State]]:
        """The replay of a call ending at tick end_tick, one ring of C rows
        per env (count_ring_rows), and the state it resumes from (DESIGN.md
        sections 5.5, 5.6, 6.3). A fresh run sizes the ring from its own
        length and records it, but for a 0-tick one (a checkpoint skeleton),
        which takes the ring the latest run recorded (the minimal ring before
        any) so that its shapes are that run's. A resumed run grows the
        state's ring to the uninterrupted run's (size_ring) and records it.
        """
        rows = count_ring_rows(
            end_tick, self.capacity_rows, self.schedule.min_ring_rows()
        )
        if agent_state is not None:
            replay_state = agent_state.replay_state
            held = count_held_rows(replay_state)
            rows = size_ring(held, start_tick, rows, "replay_capacity")
            if rows > held:
                logger.info(
                    "DreamerV3 replay: growing the state's ring from %d to %d"
                    " rows per env for the resumed run, which has seen %d.",
                    held,
                    rows,
                    start_tick,
                )
                replay_state = grow_rings(replay_state, start_tick, rows)
                agent_state = agent_state.replace(replay_state=replay_state)
            self._record_replay_rows(rows)
        elif end_tick > 0:
            self._record_replay_rows(rows)
        elif self.replay_rows_per_env is not None:
            rows = self.replay_rows_per_env
        replay = StreamReplay(
            n_envs=self.n_envs,
            capacity=rows,
            batch_size=self.batch_size,
            batch_length=self.batch_length,
        )
        return replay, agent_state

    def _record_replay_rows(self, rows: int) -> None:
        self.replay_rows_per_env = rows
        self.config.update(
            replay_rows_per_env=rows, replay_bytes_per_seed=self.replay_bytes_per_seed
        )
        logger.info(
            "DreamerV3 replay: %d rows per env x %d envs, %.3f GB per seed"
            " (the replay rings; the training step's working set comes on top)",
            rows,
            self.n_envs,
            self.replay_bytes_per_seed / 1e9,
        )

    @property
    def replay_bytes_per_seed(self) -> int:
        """Bytes of the resolved replay rings of one seed (0 before)."""
        if self.replay_rows_per_env is None:
            return 0
        spec = self.spec
        return count_replay_bytes(
            self.n_envs,
            self.replay_rows_per_env,
            spec.obs_dim,
            spec.action_dim,
            spec.discrete,
            self.model_config,
        )

    def init(self, key: jax.Array, context: TrainContext) -> DreamerV3State:
        """A fresh agent: parameters, optimizers, collector, empty replay."""
        learner_key, collector_key, training_key, evaluation_key = jax.random.split(
            key, 4
        )
        config, spec, replay = self.model_config, self.spec, context.buffer
        learner = init_learner(
            learner_key, config, spec.obs_dim, spec.action_dim, spec.discrete
        )
        carry = make_initial_carry(
            config, context.env_args.n_envs, spec.action_dim, spec.discrete
        )
        # An abstract trace of the training step, cached per configuration.
        metric_names = list_train_metrics(
            config,
            spec.obs_dim,
            spec.action_dim,
            spec.discrete,
            self.batch_size,
            self.batch_length,
        )
        return DreamerV3State(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=learner.actor_state,
            critic_state=learner.critic_state,
            world_model_state=learner.world_model_state,
            retnorm=learner.retnorm,
            collector_state=init_row_collector_state(
                collector_key, context.env_args, policy_carry=carry
            ),
            replay_state=replay.init(
                spec.obs_dim,
                spec.action_dim,
                spec.discrete,
                config.deter,
                config.stoch,
                config.classes,
            ),
            train_metrics=MetricsAccumulator.zeros(metric_names),
            n_updates=jnp.zeros((), jnp.int32),
            n_logs=jnp.zeros((), jnp.int32),
        )

    def collect(
        self, agent_state: DreamerV3State, tick: jax.Array, context: TrainContext
    ) -> tuple[DreamerV3State, jax.Array]:
        """Act and store one row per env (driver.py:55-81,
        replay.py:97-144).

        tick is the absolute, unbatched tick index (the scan input, offset
        on resume): the row of every env written now is the tick-th of its
        stream. Returns the state and tick, which locates the replay's newest
        rows for the tick's updates (update, after the tick's rows:
        deviation D3).
        """
        collector_state, row = collect_row(
            agent_state.collector_state,
            tick,
            self.make_policy(agent_state),
            env_args=context.env_args,
            reset_mode="dynamic",
            timestep_unit="rows",
        )
        agent_state = agent_state.replace(
            collector_state=collector_state,
            replay_state=context.buffer.add(agent_state.replay_state, row, tick),
        )
        return agent_state, tick

    def count_updates(
        self, agent_state: DreamerV3State, tick: jax.Array, context: TrainContext
    ) -> jax.Array:
        # The train ratio reads the absolute tick alone (TrainRatio).
        del agent_state, context
        return self.schedule.updates_in_tick(tick)

    def update(
        self, agent_state: DreamerV3State, tick: jax.Array, context: TrainContext
    ) -> tuple[DreamerV3State, None]:
        """One training step on a fresh batch, then the write-back.

        tick is the absolute, unbatched tick whose rows were added last: the
        replay holds tick + 1 rows per env. The batch pops the online queue
        and fills up uniformly (StreamReplay.sample); the learner's step
        (Algorithm J: replay context, joint gradient, LaProp, slow critic)
        uses fresh noise; its posterior latents of the trained rows are
        written back at once (synchronous, deviation D1). The loop folds the
        extensions' post_update phase after it.
        """
        config, spec, replay = self.model_config, self.spec, context.buffer
        rows = jnp.asarray(tick, jnp.int32) + 1
        sample_key, noise_key, agent_state = agent_state.get_keys_and_new_state(2)
        replay_state, index = replay.sample(agent_state.replay_state, rows, sample_key)
        batch = replay.gather(
            replay_state, index, spec.action_dim if spec.discrete else None
        )
        noise = draw_train_noise(
            noise_key,
            config,
            replay.batch_size,
            replay.batch_length,
            spec.action_dim,
            spec.discrete,
        )
        trained = LearnerState(
            world_model_state=agent_state.world_model_state,
            actor_state=agent_state.actor_state,
            critic_state=agent_state.critic_state,
            retnorm=agent_state.retnorm,
        )
        learner, entries, metrics = train_step(trained, batch, noise, config=config)
        agent_state = agent_state.replace(
            world_model_state=learner.world_model_state,
            actor_state=learner.actor_state,
            critic_state=learner.critic_state,
            retnorm=learner.retnorm,
            replay_state=replay.write_back(replay_state, index, entries),
            n_updates=agent_state.n_updates + 1,
            train_metrics=agent_state.train_metrics.add(metrics),
        )
        return agent_state, None

    def evaluate(
        self,
        agent_state: DreamerV3State,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """Sampled evaluation episodes from a zero carry (is_first first).

        The reference's policy samples in every mode (dreamerv3_spec 7.3);
        ajax.evaluate.evaluate_policy runs one episode in each of
        num_episode_test freshly reset envs with the training action repeat,
        from key: fresh at each evaluation (ajax.log.evaluation_keys),
        so each draws new initial states and action noise.
        """
        config, spec = self.model_config, self.spec
        mean_return, mean_length = evaluate_policy(
            context.env_args,
            self.make_policy(agent_state),
            lambda n_envs: make_initial_carry(
                config, n_envs, spec.action_dim, spec.discrete
            ),
            num_episode_test,
            key,
            agent_state=agent_state,
        )
        return {
            "Eval/episodic mean reward": mean_return,
            "Eval/mean episodic length": mean_length,
        }

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """As its evaluation acts: the row collector's sampling policy, from
        a zero carry."""
        config, spec = self.model_config, self.spec

        def parts(agent_state: DreamerV3State) -> PolicyParts:
            return row_collector_policy(
                self.env_args,
                agent_state,
                self.make_policy(agent_state),
                lambda n_envs: make_initial_carry(
                    config, n_envs, spec.action_dim, spec.discrete
                ),
            )

        return parts

    def report(self, agent_state: DreamerV3State, aux: Any) -> dict:
        """The logged training metrics: house keys, env_frames, losses.

        Train/episodic mean reward is the rolling mean of the training
        episodes' returns, the reference's score (dreamerv3_spec 7.3, 7.7);
        env_frames = rows * action_repeat the reference's x-axis, its step
        clock (rows, reset rows included) times the repeat (2411f7d
        dreamerv3/main.py:124-132, embodied/core/logger.py:32;
        DESIGN.md section 2): one row per episode above the simulator's
        frames; Train/n_updates the training steps so far; Train/<metric>
        the mean of each training metric over the updates since the last log.
        """
        del aux
        timestep = agent_state.collector_state.timestep
        metrics = {
            "timestep": timestep,
            "env_frames": timestep * self.env_args.action_repeat,
            "Train/n_updates": agent_state.n_updates,
            "Train/episodic mean reward": agent_state.collector_state.episodic_mean_return,
        }
        metrics.update(
            {
                f"Train/{name}": value
                for name, value in agent_state.train_metrics.mean().items()
            }
        )
        return metrics

    def make_after_log(self) -> Callable[[DreamerV3State, jax.Array], DreamerV3State]:
        return self.restart_train_metrics

    def restart_train_metrics(
        self, agent_state: DreamerV3State, logged: jax.Array
    ) -> DreamerV3State:
        """The training metrics' means restart after each log (train.py:113-121)."""
        return agent_state.replace(
            train_metrics=agent_state.train_metrics.reset_where(logged)
        )

    def make_policy(self, agent_state: DreamerV3State) -> Callable:
        """observe_and_act with the agent's current parameters bound: the row
        collector's policy and the evaluation's."""
        return partial(
            observe_and_act,
            agent_state.world_model_state.params,
            agent_state.actor_state.params,
            config=self.model_config,
            action_dim=self.spec.action_dim,
            discrete=self.spec.discrete,
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "DreamerV3"
    capabilities = frozenset(
        {
            "actor",
            "discounted",
            "off_policy",
            "policy_density",
            "replay",
            "v_critic",
            "reward_learning",
        }
    )
    # Its row collector and evaluation append the observation features
    # (observe); they run no action stage or evaluation chooser yet.
    supported_extension_phases = OwnReplayAgent.supported_extension_phases.union(
        {"observe"}
    )
    extension_context = frozenset({"env_args", "gamma"})
    action_space = "any"


# -- The tick's schedule and the env's sizes ------------------------------------
@dataclass(frozen=True)
class TrainRatio:
    """When to train: 2411f7d's training-start gate and when.Ratio.

    The reference (embodied/run/train.py:26-28, :69-91;
    embodied/core/when.py:26-42) runs, after every transition (one
    env's row: step += 1, replay.add, then train_step), nothing
    while len(replay) < batch_size, and otherwise Ratio(step) updates
    with Ratio = when.Ratio(train_ratio / (batch_size * batch_length)).
    Ratio returns 1 on its first call; after it, the updates up to
    transition t number 1 + floor((t - t0) r), t0 the first call,
    r the ratio (prev advances by repeats / r).

    Lockstep envs make both static: transition (tick i, env w) is step
    s = i n_envs + w + 1, and a worker's stream yields its first item at
    its batch_length + 1-th row, so len(replay) = s - batch_length
    n_envs from then on and the gate opens at t0 = batch_length n_envs +
    batch_size (16 envs: step 1040, the last env of tick 64). The updates
    of tick i are the closed form's difference between the ends of ticks
    i and i - 1, all run after the tick's rows are added (deviation
    D3). The closed form is Ratio in exact arithmetic, with r the
    exact fraction of train_ratio. The reference keeps prev in
    float64, exact when 1 / r is (a power of two: every reference
    configuration, train_ratio 32 ... 1024 over 16 x 64); otherwise
    its rounding sometimes runs an update one transition late, and the
    closed form is then at most one update ahead (deviation D26). It is
    evaluated on the absolute, unbatched tick (DESIGN.md principle 2), so
    the update loop stays a real loop under the seed vmap and resumes
    correctly.

    The agent's replay ring is sized so that the gate's batch_size items
    fit before any row is overwritten (min_ring_rows), which keeps
    len(replay) as above until the gate opens.
    """

    n_envs: int
    batch_size: int
    batch_length: int
    train_ratio: float

    def __post_init__(self) -> None:
        for name in ("n_envs", "batch_size", "batch_length"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not self.train_ratio > 0:
            raise ValueError(f"train_ratio must be > 0, got {self.train_ratio}")
        ratio = self.ratio
        if ratio.numerator * ratio.denominator >= 2**31:
            raise ValueError(
                f"train_ratio={self.train_ratio} gives the update ratio {ratio},"
                " whose numerator times denominator does not fit the int32"
                " schedule arithmetic; use a ratio with a small denominator"
                " (e.g. an integer train_ratio)."
            )

    @property
    def ratio(self) -> Fraction:
        """Updates per transition, train_ratio / (batch_size * batch_length),
        as an exact fraction (of the float's binary value)."""
        return Fraction(self.train_ratio) / (self.batch_size * self.batch_length)

    @property
    def first_update_step(self) -> int:
        """t0: the transition at which len(replay) reaches batch_size."""
        return self.batch_length * self.n_envs + self.batch_size

    @property
    def first_update_tick(self) -> int:
        """The tick whose transitions include t0."""
        return (self.first_update_step - 1) // self.n_envs

    def min_ring_rows(self) -> int:
        """Rows per env the ring needs to hold batch_size items at once."""
        return self.batch_length + -(-self.batch_size // self.n_envs)

    def updates_until(self, steps: jax.Array) -> jax.Array:
        """Updates run up to and including transition steps (int32).

        0 before t0, then 1 + floor((steps - t0) r), the ratio r an exact
        fraction: computed as (since_first // denominator) numerator plus
        ((since_first % denominator) numerator) // denominator, so that no
        intermediate exceeds the number of updates or numerator times
        denominator (int32)."""
        numerator, denominator = self.ratio.numerator, self.ratio.denominator
        since_first = jnp.asarray(steps, jnp.int32) - self.first_update_step
        later = (since_first // denominator) * numerator + (
            (since_first % denominator) * numerator
        ) // denominator
        return jnp.where(since_first >= 0, 1 + later, 0)

    def updates_in_tick(self, tick: jax.Array) -> jax.Array:
        """Updates to run after tick tick's rows are added (int32)."""
        tick = jnp.asarray(tick, jnp.int32)
        return self.updates_until((tick + 1) * self.n_envs) - self.updates_until(
            tick * self.n_envs
        )

    def total_updates(self, ticks: int) -> int:
        """Updates after the first ticks ticks (host-side Python int)."""
        since_first = ticks * self.n_envs - self.first_update_step
        return 0 if since_first < 0 else 1 + int(since_first * self.ratio)


class EnvSpec(NamedTuple):
    """The env's sizes as the agent reads them (static)."""

    obs_dim: int
    action_dim: int  # dimension, or number of discrete actions
    discrete: bool


def measure_env(env_args: EnvironmentConfig) -> EnvSpec:
    """The env's sizes for the model. Observations of rank 1 or 2 are
    flattened, as the reference's vector encoder concatenates its keys'
    features (DictConcat, dreamerv3_spec.md 2.9); images (rank 3) need the
    reference's CNN encoder, out of scope here (DESIGN.md), and raise. The
    run's observation features (env_args.observers) widen the observation:
    the row collector stores them with it, so the encoder reads them and the
    decoder reconstructs them, as any other column."""
    obs_shape, _ = get_state_action_shapes(env_args.env)
    if len(obs_shape) > 2:
        raise ValueError(
            f"DreamerV3 takes vector observations (rank 1 or 2), got shape"
            f" {tuple(obs_shape)}: image observations need the CNN encoder,"
            " which is out of scope (docs/world_models/DESIGN.md)."
        )
    if len(obs_shape) == 2 and feature_width(env_args):
        raise ValueError(
            "DreamerV3 cannot take observation features with the rank-2"
            f" observation {tuple(obs_shape)} yet: the collector appends them on"
            " its last axis, which the encoder's flat width does not count."
        )
    return EnvSpec(
        obs_dim=int(np.prod(obs_shape)) + feature_width(env_args),
        action_dim=int(get_action_dim(env_args.env, env_args.env_params)),
        discrete=not env_args.continuous,
    )


# -- State ---------------------------------------------------------------------
@struct.dataclass(kw_only=True)
class DreamerV3State(BaseAgentState):
    """The DreamerV3 agent's state (docs/world_models/DESIGN.md section 6.6).
    Beside BaseAgentState's fields, critic_state's target_params is the slow
    critic, its step counting the slow critic's updates (update_slow_critic
    reads it).

    Attributes:
        collector_state: the row collector's state; its policy_carry is the
            acting posterior and previous action (PolicyCarry).
        world_model_state: the world model (enc, rssm, dec, rew, con) and
            its LaProp instance.
        retnorm: the return normaliser.
        replay_state: the replay rings and the online-queue counter
            (ReplayState).
        train_metrics: the training metrics since the last log.

    The schedule needs no state of its own: it is a function of the
    absolute tick, the scan input.
    """

    collector_state: RowCollectorState
    world_model_state: LoadedTrainState
    retnorm: ReturnNormalizer
    replay_state: ReplayState
    train_metrics: MetricsAccumulator


__all__ = ["DreamerV3", "DreamerV3State", "EnvSpec", "TrainRatio", "measure_env"]
