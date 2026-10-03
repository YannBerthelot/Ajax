"""The DreamerV3 training loop: act, store, train at the train ratio, log.

The paper-era driver loop (``danijar/dreamerv3@2411f7d:embodied/run/
train.py`` with ``embodied/core/driver.py`` and ``when.Ratio``) made
synchronous and seeded: dreamerv3_spec Algorithm K, ``docs/world_models/
DESIGN.md`` sections 6.4-6.6, deviations D1-D3 and D6. One scan iteration
(a *tick*) is one vector step of the ``n_envs`` workers:

1. **Act** (:func:`policy_step`; dreamerv3_spec 7.1, 2411f7d
   ``dreamerv3/agent.py:145-180`` = ``29eb964:dreamerv3/agent.py:129-164``):
   the posterior filter -- one RSSM observe step from the carried ``(h, z)``
   and previous action, reset where ``is_first`` -- then a **sample** of the
   actor (the reference has no deterministic mode, dreamerv3_spec 7.3). The
   carry keeps the raw sample as the previous action; the row keeps the
   posterior latent for the replay context.
2. **Store**: the row collector (:mod:`ajax.environments.row_collector`, in
   its dynamic reset mode: the stream replay takes any episode boundary)
   emits one row per env -- the driver's transition, its action zeroed at
   ``is_last`` (``driver.py:67-75``) -- and maps the action to the env's
   bounds; the replay stores the raw action (:mod:`.replay`).
3. **Train**: :meth:`TrainRatio.updates_in_tick` updates, each on a fresh
   batch (online queue first, then uniform), with fresh noise
   (:func:`~ajax.agents.DreamerV3.learner.train_step`), followed by the
   latent write-back and the Extension ``post_update`` fold.
4. **Log** (:func:`ajax.log.maybe_eval_and_log`) every
   ``logging_config.log_frequency`` rows: the training-episode returns of
   the stochastic policy (the reference's score, dreamerv3_spec 7.3), sampled
   evaluation episodes (:func:`ajax.evaluate.evaluate_policy`, from a zero
   carry), the mean training metrics since the last log, ``env_frames``.

``n_timesteps`` counts **rows** (the reference's ``step``, reset rows
included); the collector's ``timestep`` advances by ``n_envs`` per tick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.DreamerV3.distributions import draw_action_noise
from ajax.agents.DreamerV3.learner import (
    LearnerState,
    draw_train_noise,
    init_learner,
    train_step,
)
from ajax.agents.DreamerV3.networks import (
    RSSM,
    Actor,
    RSSMState,
    WorldModel,
    encode_action,
    features,
    initial_state,
)
from ajax.agents.DreamerV3.replay import StreamReplay, context_batch
from ajax.agents.DreamerV3.state import (
    DreamerV3AgentConfig,
    DreamerV3Config,
    DreamerV3State,
    MetricsAccumulator,
)
from ajax.agents.DreamerV3.world_model import ReplayContextBatch, draw_posterior_noise
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.evaluate import evaluate_policy
from ajax.extensions.base import Extension, ExtensionStack
from ajax.log import compose_eval_metrics, maybe_eval_and_log
from ajax.logging.wandb_logging import LoggingConfig, start_async_logging, vmap_log
from ajax.perf_utils import build_resumable_train, final_aux_fori
from ajax.state import EnvironmentConfig

# ---------------------------------------------------------------------------
# Schedule: 2411f7d train.py + when.Ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainRatio:
    """When to train: 2411f7d's training-start gate and ``when.Ratio``.

    The reference (``embodied/run/train.py:26-28``, ``:69-91``;
    ``embodied/core/when.py:26-42``) runs, after **every transition** (one
    env's row: ``step += 1``, ``replay.add``, then ``train_step``), nothing
    while ``len(replay) < batch_size``, and otherwise ``Ratio(step)`` updates
    with ``Ratio = when.Ratio(train_ratio / (batch_size * batch_length))``.
    ``Ratio`` returns 1 on its first call; after it, the updates up to
    transition ``t`` number ``1 + floor((t - t0) r)``, ``t0`` the first call,
    ``r`` the ratio (``prev`` advances by ``repeats / r``).

    Lockstep envs make both static: transition ``(tick i, env w)`` is step
    ``s = i n_envs + w + 1``, and a worker's stream yields its first item at
    its ``batch_length + 1``-th row, so ``len(replay) = s - batch_length
    n_envs`` from then on and the gate opens at ``t0 = batch_length n_envs +
    batch_size`` (16 envs: step 1040, the last env of tick 64). The updates
    of tick ``i`` are the closed form's difference between the ends of ticks
    ``i`` and ``i - 1``, all run after the tick's rows are added (deviation
    D3). The closed form is ``Ratio`` in exact arithmetic, with ``r`` the
    exact fraction of ``train_ratio``. The reference keeps ``prev`` in
    float64, exact when ``1 / r`` is (a power of two: every reference
    configuration, ``train_ratio`` 32 ... 1024 over ``16 x 64``); otherwise
    its rounding sometimes runs an update one transition late, and the
    closed form is then at most one update ahead (deviation D26). It is
    evaluated on the absolute, unbatched tick (``DESIGN.md`` principle 2), so
    the update loop stays a real loop under the seed ``vmap`` and resumes
    correctly.

    The agent's replay ring is sized so that the gate's ``batch_size`` items
    fit before any row is overwritten (:meth:`min_ring_rows`), which keeps
    ``len(replay)`` as above until the gate opens.
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
        """Updates per transition, ``train_ratio / (batch_size * batch_length)``,
        as an exact fraction (of the float's binary value)."""
        return Fraction(self.train_ratio) / (self.batch_size * self.batch_length)

    @property
    def first_update_step(self) -> int:
        """``t0``: the transition at which ``len(replay)`` reaches ``batch_size``."""
        return self.batch_length * self.n_envs + self.batch_size

    @property
    def first_update_tick(self) -> int:
        """The tick whose transitions include ``t0``."""
        return (self.first_update_step - 1) // self.n_envs

    def min_ring_rows(self) -> int:
        """Rows per env the ring needs to hold ``batch_size`` items at once."""
        return self.batch_length + -(-self.batch_size // self.n_envs)

    def updates_until(self, steps: jax.Array) -> jax.Array:
        """Updates run up to and including transition ``steps`` (int32).

        ``0`` before ``t0``, then ``1 + floor((steps - t0) p / q)`` with ``r
        = p / q``, computed as ``(x // q) p + ((x % q) p) // q`` so that no
        intermediate exceeds the number of updates or ``p q`` (int32).
        """
        p, q = self.ratio.numerator, self.ratio.denominator
        x = jnp.asarray(steps, jnp.int32) - self.first_update_step
        later = (x // q) * p + ((x % q) * p) // q
        return jnp.where(x >= 0, 1 + later, 0)

    def updates_in_tick(self, tick: jax.Array) -> jax.Array:
        """Updates to run after tick ``tick``'s rows are added (int32)."""
        tick = jnp.asarray(tick, jnp.int32)
        return self.updates_until((tick + 1) * self.n_envs) - self.updates_until(
            tick * self.n_envs
        )

    def total_updates(self, ticks: int) -> int:
        """Updates after the first ``ticks`` ticks (host-side Python int)."""
        x = ticks * self.n_envs - self.first_update_step
        return 0 if x < 0 else 1 + int(x * self.ratio)


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


class PolicyCarry(NamedTuple):
    """The acting state of ``n`` envs (the reference's ``(lat, act)`` carry).

    ``deter [n, D]`` and the one-hot ``stoch [n, S, C]`` of the last
    posterior; ``prevact`` the last action, the raw sample: ``[n, A]``
    float32 (continuous) or ``[n]`` int32 (discrete index). The reference
    carries it unmasked even at ``is_last`` (the observe step masks it at the
    next ``is_first``; dreamerv3_spec 7.1).
    """

    deter: jax.Array
    stoch: jax.Array
    prevact: jax.Array


def initial_policy_carry(
    config: DreamerV3Config, n: int, action_dim: int, discrete: bool
) -> PolicyCarry:
    """Zeros (``Agent.init_policy``, ``29eb964:dreamerv3/agent.py:114-118``)."""
    state = initial_state(config, (n,))
    prevact = (
        jnp.zeros((n,), jnp.int32)
        if discrete
        else jnp.zeros((n, action_dim), jnp.float32)
    )
    return PolicyCarry(state.deter, state.stoch, prevact)


def policy_step(
    world_model_params: dict,
    actor_params: dict,
    carry: PolicyCarry,
    obs: jax.Array,
    is_first: jax.Array,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    action_dim: int,
    discrete: bool,
) -> tuple[jax.Array, PolicyCarry, tuple[jax.Array, jax.Array]]:
    """One acting step of ``n`` envs: filter ``obs``, then sample the actor.

    ``Agent.policy`` (``29eb964:dreamerv3/agent.py:129-164``; dreamerv3_spec
    7.1): encode the observation, one-hot the previous action, one RSSM
    observe step (Algorithm B; ``deter``, ``stoch`` and the previous action
    are zeroed where ``is_first``) with a **sampled** posterior, then
    ``a ~ pi(concat(h, z))``, sampled in every mode. Returns the action (raw
    sample, or its index for a discrete space), the new carry and, for the
    replay context, the posterior ``(deter [n, D], stoch [n, S])`` as class
    indices (``agent.py:142-144``). The signature after ``key`` is the row
    collector's :class:`~ajax.environments.row_collector.PolicyFn` once the
    parameters are bound.
    """
    post_key, action_key = jax.random.split(key)
    n = obs.shape[0]
    obs = obs.reshape(n, -1)
    model = WorldModel(config, obs.shape[-1])
    tokens = model.apply({"params": world_model_params}, obs, method=WorldModel.encode)
    prevact = encode_action(carry.prevact, action_dim if discrete else None)
    state, _ = RSSM(config).apply(
        {"params": world_model_params["rssm"]},
        RSSMState(carry.deter, carry.stoch),
        tokens,
        prevact,
        is_first,
        draw_posterior_noise(post_key, config, (n,)),
        method=RSSM.observe_step,
    )
    policy = Actor(config, action_dim, discrete).apply(
        {"params": actor_params}, features(state.deter, state.stoch)
    )
    sample = policy.sample(draw_action_noise(action_key, (n, action_dim), discrete))
    action = jnp.argmax(sample, -1).astype(jnp.int32) if discrete else sample
    latent = (state.deter, jnp.argmax(state.stoch, -1).astype(jnp.int32))
    return action, PolicyCarry(state.deter, state.stoch, action), latent


def bind_policy(
    agent_state: DreamerV3State,
    config: DreamerV3Config,
    action_dim: int,
    discrete: bool,
) -> Callable:
    """:func:`policy_step` with the agent's current parameters bound."""
    return partial(
        policy_step,
        agent_state.world_model_state.params,
        agent_state.actor_state.params,
        config=config,
        action_dim=action_dim,
        discrete=discrete,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class EnvSpec(NamedTuple):
    """The env's sizes as the agent reads them (static)."""

    obs_dim: int
    action_dim: int  # dimension, or number of discrete actions
    discrete: bool


def env_spec(env_args: EnvironmentConfig) -> EnvSpec:
    obs_shape, _ = get_state_action_shapes(env_args.env)
    return EnvSpec(
        obs_dim=int(np.prod(obs_shape)),
        action_dim=int(get_action_dim(env_args.env, env_args.env_params)),
        discrete=not env_args.continuous,
    )


def learner_state(agent_state: DreamerV3State) -> LearnerState:
    """The part of the agent state a training step updates."""
    return LearnerState(
        world_model_state=agent_state.world_model_state,
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        retnorm=agent_state.retnorm,
    )


@lru_cache(maxsize=32)
def train_metric_keys(
    config: DreamerV3Config, spec: EnvSpec, batch_size: int, batch_length: int
) -> tuple[str, ...]:
    """Names of :func:`~ajax.agents.DreamerV3.learner.train_step`'s metrics
    (abstract evaluation on shapes; nothing is computed).

    Cached: the abstract trace of the training step takes seconds, and its
    arguments (a frozen dataclass, a named tuple, ints) are hashable.
    """
    f32, length = jnp.float32, batch_length + 1

    def shape(*dims: int, dtype: Any = f32) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(dims, dtype)

    key = shape(2, dtype=jnp.uint32)
    learner = jax.eval_shape(
        lambda k: init_learner(k, config, spec.obs_dim, spec.action_dim, spec.discrete),
        key,
    )
    flag = shape(batch_size, length, dtype=bool)
    batch = ReplayContextBatch(
        obs=shape(batch_size, length, spec.obs_dim),
        action=shape(batch_size, length, spec.action_dim),
        reward=shape(batch_size, length),
        is_first=flag,
        is_last=flag,
        is_terminal=flag,
        context_deter=shape(batch_size, config.deter),
        context_stoch=shape(batch_size, config.stoch, dtype=jnp.int32),
    )
    noise = jax.eval_shape(
        lambda k: draw_train_noise(
            k, config, batch_size, batch_length, spec.action_dim, spec.discrete
        ),
        key,
    )
    metrics = jax.eval_shape(
        lambda s, x, n: train_step(s, x, n, config=config)[2], learner, batch, noise
    )
    return tuple(metrics)


def init_dreamer(
    key: jax.Array,
    env_args: EnvironmentConfig,
    config: DreamerV3Config,
    spec: EnvSpec,
    replay: StreamReplay,
    metric_keys: Sequence[str],
) -> DreamerV3State:
    """A fresh agent: parameters, optimizers, collector, empty replay."""
    learner_key, collector_key, rng, eval_rng = jax.random.split(key, 4)
    learner = init_learner(
        learner_key, config, spec.obs_dim, spec.action_dim, spec.discrete
    )
    carry = initial_policy_carry(
        config, env_args.n_envs, spec.action_dim, spec.discrete
    )
    return DreamerV3State(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=learner.actor_state,
        critic_state=learner.critic_state,
        world_model_state=learner.world_model_state,
        retnorm=learner.retnorm,
        collector_state=init_row_collector_state(
            collector_key, env_args, policy_carry=carry
        ),
        replay_state=replay.init(
            spec.obs_dim,
            spec.action_dim,
            spec.discrete,
            config.deter,
            config.stoch,
            config.classes,
        ),
        train_metrics=MetricsAccumulator.zeros(metric_keys),
        n_updates=jnp.zeros((), jnp.int32),
        n_logs=jnp.zeros((), jnp.int32),
    )


# ---------------------------------------------------------------------------
# One update (Algorithm J)
# ---------------------------------------------------------------------------


def update(
    _: jax.Array,
    agent_state: DreamerV3State,
    *,
    rows: jax.Array,
    config: DreamerV3Config,
    spec: EnvSpec,
    replay: StreamReplay,
    extension_stack: Optional[ExtensionStack],
    total_timesteps: int,
) -> tuple[DreamerV3State, None]:
    """One training step on a fresh batch, then the write-back.

    ``rows`` is the number of rows per env in the replay (unbatched). The
    batch pops the online queue and fills up uniformly
    (:meth:`~ajax.agents.DreamerV3.replay.StreamReplay.sample`); the
    learner's step (Algorithm J: replay context, joint gradient, LaProp,
    slow critic) uses fresh noise; its posterior latents of the trained rows
    are written back at once (synchronous, deviation D1); the Extension
    ``post_update`` phase is folded after the update.
    """
    rng, sample_key, noise_key, post_key = jax.random.split(agent_state.rng, 4)
    replay_state, index = replay.sample(agent_state.replay_state, rows, sample_key)
    batch = context_batch(
        replay.gather(replay_state, index), spec.action_dim if spec.discrete else None
    )
    noise = draw_train_noise(
        noise_key,
        config,
        replay.batch_size,
        replay.batch_length,
        spec.action_dim,
        spec.discrete,
    )
    learner, entries, metrics = train_step(
        learner_state(agent_state), batch, noise, config=config
    )
    agent_state = agent_state.replace(
        rng=rng,
        world_model_state=learner.world_model_state,
        actor_state=learner.actor_state,
        critic_state=learner.critic_state,
        retnorm=learner.retnorm,
        replay_state=replay.write_back(replay_state, index, entries),
        n_updates=agent_state.n_updates + 1,
        train_metrics=agent_state.train_metrics.add(metrics),
    )
    if extension_stack is not None:
        agent_state = extension_stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            post_key,
            total_timesteps,
        )
    return agent_state, None


# ---------------------------------------------------------------------------
# Evaluation and logging
# ---------------------------------------------------------------------------


def evaluate_dreamer(
    agent_state: DreamerV3State,
    key: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: DreamerV3Config,
    spec: EnvSpec,
    num_episode_test: int,
) -> dict:
    """Sampled evaluation episodes from a zero carry (``is_first`` first).

    The reference's policy samples in every mode (dreamerv3_spec 7.3);
    :func:`ajax.evaluate.evaluate_policy` runs one episode in each of
    ``num_episode_test`` freshly reset envs with the training action repeat.
    """
    mean_return, mean_length = evaluate_policy(
        env_args,
        bind_policy(agent_state, config, spec.action_dim, spec.discrete),
        lambda n: initial_policy_carry(config, n, spec.action_dim, spec.discrete),
        num_episode_test,
        key,
    )
    return {
        "Eval/episodic mean reward": mean_return,
        "Eval/mean episodic length": mean_length,
    }


def train_metrics(agent_state: DreamerV3State, aux: Any, *, action_repeat: int) -> dict:
    """The logged training metrics: house keys, ``env_frames``, losses.

    ``Train/episodic mean reward`` is the rolling mean of the training
    episodes' returns, the reference's score (dreamerv3_spec 7.3, 7.7);
    ``env_frames = rows * action_repeat`` the reference's x-axis, its step
    clock (rows, reset rows included) times the repeat (2411f7d
    ``dreamerv3/main.py:124-132``, ``embodied/core/logger.py:32``;
    ``DESIGN.md`` section 2): one row per episode above the simulator's
    frames; ``Train/n_updates`` the training steps so far; ``Train/<metric>``
    the mean of each training metric over the updates since the last log.
    """
    del aux
    timestep = agent_state.collector_state.timestep
    metrics = {
        "timestep": timestep,
        "env_frames": timestep * action_repeat,
        "Train/n_updates": agent_state.n_updates,
        "Train/episodic mean reward": agent_state.collector_state.episodic_mean_return,
    }
    metrics.update(
        {f"Train/{k}": v for k, v in agent_state.train_metrics.mean().items()}
    )
    return metrics


# ---------------------------------------------------------------------------
# One tick (Algorithm K)
# ---------------------------------------------------------------------------


def training_iteration(
    agent_state: DreamerV3State,
    tick: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: DreamerV3Config,
    spec: EnvSpec,
    replay: StreamReplay,
    schedule: TrainRatio,
    extension_stack: Optional[ExtensionStack],
    total_timesteps: int,
    index: Any,
    log_kwargs: dict,
) -> tuple[DreamerV3State, dict]:
    """One vector step: act and store, train at the ratio, maybe log.

    ``tick`` is the absolute, unbatched tick index (the scan input, offset
    on resume): the row of every env written now is the ``tick``-th of its
    stream.
    """
    # Act and store (driver.py:55-81; replay.py:97-144).
    collector_state, row = collect_row(
        agent_state.collector_state,
        tick,
        bind_policy(agent_state, config, spec.action_dim, spec.discrete),
        env_args=env_args,
        reset_mode="dynamic",
        timestep_unit="rows",
    )
    agent_state = agent_state.replace(
        collector_state=collector_state,
        replay_state=replay.add(agent_state.replay_state, row, tick),
    )

    # Train (train.py:80-91, when.Ratio), after the tick's rows (D3).
    agent_state, _ = final_aux_fori(
        partial(
            update,
            rows=jnp.asarray(tick, jnp.int32) + 1,
            config=config,
            spec=spec,
            replay=replay,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
        ),
        agent_state,
        schedule.updates_in_tick(tick),
    )

    # Log (train.py:113-121): the training metrics' means since the last log.
    n_logs = agent_state.n_logs
    agent_state, metrics = maybe_eval_and_log(
        agent_state, None, index, tick, **log_kwargs
    )
    agent_state = agent_state.replace(
        train_metrics=agent_state.train_metrics.reset_where(
            agent_state.n_logs != n_logs
        )
    )
    return agent_state, metrics


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: Any,
    critic_optimizer_args: Any,
    network_args: Any,
    agent_config: DreamerV3AgentConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    *,
    config: DreamerV3Config,
    replay_rows_per_env: int,
    extensions: Sequence[Extension] = (),
    **_unused: Any,
) -> Callable:
    """The per-seed train function (``build_resumable_train``).

    ``total_timesteps`` counts rows: ``total_timesteps // n_envs`` ticks.
    ``replay_rows_per_env`` is the ring length ``C`` the agent resolved.
    The optimizer, network and actor/critic-split arguments of the shared
    :meth:`ActorCritic.train` are unused: :class:`DreamerV3Config` holds
    every network and optimizer hyperparameter.
    """
    del actor_optimizer_args, critic_optimizer_args, network_args
    n_envs = env_args.n_envs
    num_ticks = total_timesteps // n_envs
    spec = env_spec(env_args)
    replay = StreamReplay(
        n_envs=n_envs,
        capacity=replay_rows_per_env,
        batch_size=agent_config.batch_size,
        batch_length=agent_config.batch_length,
    )
    schedule = TrainRatio(
        n_envs=n_envs,
        batch_size=agent_config.batch_size,
        batch_length=agent_config.batch_length,
        train_ratio=agent_config.train_ratio,
    )
    if replay.capacity < schedule.min_ring_rows():
        raise ValueError(
            f"A ring of {replay.capacity} rows per env cannot hold the"
            f" {agent_config.batch_size} items the first update needs"
            f" (at least {schedule.min_ring_rows()} rows per env)."
        )
    metric_keys = train_metric_keys(
        config, spec, agent_config.batch_size, agent_config.batch_length
    )
    extension_stack = (
        ExtensionStack(extensions).bind_to_agent(
            env_args=env_args,
            agent_config=agent_config,
            gamma=config.gamma,
            total_timesteps=total_timesteps,
        )
        if extensions
        else None
    )

    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)
    if log:
        start_async_logging()
    log_kwargs = {
        "metrics_fn": partial(train_metrics, action_repeat=env_args.action_repeat),
        "evaluate_fn": partial(
            evaluate_dreamer,
            env_args=env_args,
            config=config,
            spec=spec,
            num_episode_test=num_episode_test,
        ),
        "extra_eval_metrics": compose_eval_metrics(
            None, extension_stack, total_timesteps
        ),
        "log": log,
        "log_fn": log_fn,
        "log_frequency": (
            logging_config.log_frequency if logging_config is not None else None
        ),
        "per_update": n_envs,
    }

    def init_fn(key, index):
        agent_state = init_dreamer(key, env_args, config, spec, replay, metric_keys)
        return agent_state.replace(index=index)

    def init_transform(agent_state, key):
        if extension_stack is None:
            return agent_state
        ext_key, pre_key = jax.random.split(key)
        agent_state = extension_stack.fold_init_states(agent_state, ext_key)
        return extension_stack.fold_pretrain(
            agent_state, jnp.asarray(0), pre_key, total_timesteps
        )

    def make_scan_fn(_agent_state, _resume, _key, index):
        return partial(
            training_iteration,
            env_args=env_args,
            config=config,
            spec=spec,
            replay=replay,
            schedule=schedule,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            index=index,
            log_kwargs=log_kwargs,
        )

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_ticks,
        init_transform=init_transform,
    )


__all__ = [
    "EnvSpec",
    "PolicyCarry",
    "TrainRatio",
    "bind_policy",
    "env_spec",
    "initial_policy_carry",
    "make_train",
    "policy_step",
    "training_iteration",
    "update",
]
