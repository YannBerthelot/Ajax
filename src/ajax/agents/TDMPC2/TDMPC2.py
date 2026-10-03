"""TD-MPC2 (Hansen, Su, Wang, ICLR 2024; arXiv:2310.16828v2), single-task online.

The paper-era algorithm (``nicklashansen/tdmpc2@b67b21c`` = ``5f6fade`` for
the agent; ``docs/world_models/DESIGN.md`` §4, tdmpc2_spec §1-§4, version
choices and deviations in ``docs/world_models/deviations.md``): a decoder-free
world model (encoder, latent dynamics, two-hot reward and Q ensemble) and a
policy prior trained on sub-trajectories of a whole-episode replay buffer
(:mod:`.core`, :mod:`.buffer`), acting by MPPI planning in latent space
(:mod:`.planner`), in the reference's online loop (:mod:`.train_TDMPC2`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional

import jax
import numpy as np
from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.TDMPC2.buffer import EpisodeBuffer, replay_capacity
from ajax.agents.TDMPC2.core import discount_from_episode_length
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.agents.TDMPC2.train_TDMPC2 import Schedule, make_train
from ajax.environments.utils import (
    agent_episode_length,
    get_action_dim,
    get_state_action_shapes,
)
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import BaseAgentState
from ajax.types import EnvType, FloatOrCallable

logger = logging.getLogger(__name__)


class TDMPC2(ActorCritic):
    """TD-MPC2 for continuous control, single task, online.

    Paper DMC protocol: ``action_repeat=2, episode_length=1000`` (simulator
    steps, so ``T = 500`` agent steps, ``gamma = 0.99``, ``seed_steps =
    2500``), ``model_size=5``, ``n_envs=1``; ``n_timesteps`` counts agent
    steps, so the paper's "1M env steps" is ``n_timesteps=500_000`` (logs
    carry ``env_frames``, tdmpc2_spec 4.22).

    Static hyperparameters (they fix shapes or trace-time structure): the
    architecture (``model_size`` preset of ``{1, 5, 19, 48, 317}`` M
    parameters filling the ``None`` widths ``enc_dim``, ``mlp_dim``,
    ``latent_dim``, ``num_enc_layers``, ``num_q``; an explicit width
    overrides it; tdmpc2_spec 1.20), ``simnorm_dim``, ``num_bins``, ``vmax``
    (two-hot bins over ``[-vmax, vmax]`` in symlog space; the reference's
    ``vmin = -vmax``, T24), ``horizon``, ``batch_size``, ``buffer_size``,
    ``seed_steps`` and the planner sizes ``iterations`` (+2 when the action
    dimension is >= 20), ``num_samples``, ``num_elites``, ``num_pi_trajs``.
    Every other hyperparameter is a static Python float as well, except
    ``learning_rate``, which accepts a float or a schedule
    ``Callable[[int], float]`` (the world model's encoder runs at
    ``learning_rate * enc_lr_scale``, ``5f6fade:tdmpc2/tdmpc2.py:21-28``).

    Derived from the episode length ``T`` in agent steps
    (:func:`ajax.environments.utils.agent_episode_length`: brax / playground
    ``episode_length // action_repeat``, gymnax the env's
    ``max_steps_in_episode``): ``gamma=None`` is
    ``clip((T/denom - 1) / (T/denom), discount_min, discount_max)``
    (tdmpc2_spec 2.20, ``tdmpc2.py:36-49``) and ``seed_steps=None`` is
    ``max(1000, 5 T)`` (4.3, ``envs/__init__.py:81``).

    Replay (:mod:`ajax.agents.TDMPC2.buffer`, the b67b21c episode buffer):
    ``buffer_size`` env steps, stored in a ring sized at every :meth:`train`
    call for the env steps the run has taken when the call ends,
    ``min(buffer_size, total)`` (the reference's clamp to the run length,
    which saves memory without changing what is replayed; deviation T26).
    A resumed run -- the same agent continuing, or a new one restoring a
    checkpoint into its ``n_timesteps=0`` skeleton (``ajax.checkpoint``) --
    moves the carried episodes into the ring of its own call
    (:meth:`~ajax.agents.TDMPC2.buffer.EpisodeBuffer.adopt`), so a run
    trained in chunks replays exactly what the uninterrupted run does. The
    last call's capacity and bytes are :attr:`replay_capacity` and
    :attr:`replay_bytes_per_seed` (also in :attr:`config`, hence the run
    config of the loggers). Memory per seed: ``R * n_envs * (T + 1) *
    (obs_dim + A + 1) * 4`` bytes with ``R = ceil(capacity / (T n_envs)) +
    1``: 124 MB for DMC walker (obs 24, A 6, T 500) at the full
    1,000,000-step capacity; the seed ``vmap`` multiplies it. With a
    ``logging_config``, :meth:`train` also returns the per-tick metrics: 4
    bytes per key (18, plus the extensions' metrics) per tick per seed, a
    tick being one env step of every env (``T + 1`` ticks per episode);
    without one it returns no metrics.

    Evaluation (with a ``logging_config``): every ``log_frequency`` env
    steps, ``num_episode_test`` episodes of the planner in ``eval_mode`` on
    a rebuilt env with the training action repeat and episode length, each
    evaluation from fresh initial states (``online_trainer.py:27-48``;
    deviation T9). One evaluation makes ``num_episode_test * T`` planner
    decisions (batched over the episodes; on CPU each costs roughly half a
    training env step, which plans and updates). The paper protocol is
    ``log_frequency = 50_000`` with ``num_episode_test = 10`` (tdmpc2_spec
    4.21); :class:`~ajax.logging.wandb_logging.LoggingConfig`'s default
    ``log_frequency`` of 1000 makes a T = 500 run spend about twice as long
    evaluating as training.

    Continuous actions only. The env must be fixed-length and
    non-terminating (all DMC tasks): a termination, off the ``T``-step
    schedule or on its last step, raises ``ValueError`` after :meth:`train`
    (deviation T10). Observations and rewards are the
    env's raw values (no normalisation, as in the reference).

    Extensions: ``pretrain``, ``post_update`` (folded after every update,
    ``step`` = env steps) and ``eval_metrics`` (with ``init_state``); the
    other phases are rejected at construction.
    """

    name: str = "TDMPC2"
    supported_extension_phases: frozenset = frozenset(
        {"pretrain", "post_update", "eval_metrics"}
    )

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        model_size: int = 5,
        enc_dim: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_enc_layers: Optional[int] = None,
        num_q: Optional[int] = None,
        simnorm_dim: int = 8,
        num_bins: int = 101,
        vmax: float = 10.0,
        dropout: float = 0.01,
        horizon: int = 3,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        learning_rate: FloatOrCallable = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        pi_eps: float = 1e-5,
        tau: float = 0.01,
        entropy_coef: float = 1e-4,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        seed_steps: Optional[int] = None,
        gamma: Optional[float] = None,
        discount_denom: float = 5,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        iterations: int = 6,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        episode_length: int = 1000,
        action_repeat: int = 1,
        env_params: Optional[EnvParams] = None,
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Args:
            env_id: env id (gymnax, brax or mujoco_playground) or a prebuilt
                gymnax env.
            n_envs: lockstep envs; each stepping tick runs ``n_envs``
                updates (UTD 1 per env step; deviation T7 for ``> 1``).
            episode_length: simulator steps per episode on brax /
                playground (ignored on gymnax, which uses its params'
                ``max_steps_in_episode``).
            action_repeat: simulator steps per agent step (brax /
                playground; the paper's DMC protocol uses 2).
            env_params: gymnax env params (one system).
            extensions: composable research features (see
                :mod:`ajax.extensions`).

        Every other argument is the hyperparameter of the same name in
        ``nicklashansen/tdmpc2@5f6fade:tdmpc2/config.yaml`` (``lr`` is
        ``learning_rate``), with its paper default; see the class docstring
        for the static ones and the derived defaults.
        """
        self.config = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }
        self.config.update({"algo_name": "TDMPC2"})

        # The base class's actor / critic optimizer and network configs do
        # not apply: core.create_update_state builds TD-MPC2's optimizers.
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            env_params=env_params,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        if not self.env_args.continuous:
            raise ValueError("TD-MPC2 only supports continuous action spaces.")

        self.agent_config = TDMPC2Config.from_model_size(
            model_size,
            enc_dim=enc_dim,
            mlp_dim=mlp_dim,
            latent_dim=latent_dim,
            num_enc_layers=num_enc_layers,
            num_q=num_q,
            simnorm_dim=simnorm_dim,
            num_bins=num_bins,
            vmax=vmax,
            dropout=dropout,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            horizon=horizon,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            grad_clip_norm=grad_clip_norm,
            tau=tau,
            iterations=iterations,
            num_samples=num_samples,
            num_elites=num_elites,
            num_pi_trajs=num_pi_trajs,
            min_std=min_std,
            max_std=max_std,
            temperature=temperature,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
        env, params = self.env_args.env, self.env_args.env_params
        obs_shape, _ = get_state_action_shapes(env)
        if len(obs_shape) != 1:
            raise ValueError(
                f"TD-MPC2 takes flat state observations, got shape {obs_shape}"
            )
        self.obs_dim: int = int(obs_shape[0])
        self.action_dim: int = int(get_action_dim(env, params))
        # T, in agent steps: never the episode_length argument directly.
        self.agent_episode_length: int = agent_episode_length(
            env, params, action_repeat
        )
        if self.agent_episode_length < horizon:
            raise ValueError(
                f"episodes of T = {self.agent_episode_length} agent steps are too"
                f" short for horizon {horizon}: a training slice needs"
                " horizon + 1 <= T + 1 rows"
            )
        self.gamma: float = (
            discount_from_episode_length(
                self.agent_episode_length, discount_denom, discount_min, discount_max
            )
            if gamma is None
            else float(gamma)
        )
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        self.seed_steps: int = (
            max(1000, 5 * self.agent_episode_length)
            if seed_steps is None
            else int(seed_steps)
        )
        if self.seed_steps < 0:
            raise ValueError(f"seed_steps must be >= 0, got {self.seed_steps}")
        self.schedule = Schedule(
            n_envs=n_envs,
            episode_length=self.agent_episode_length,
            seed_steps=self.seed_steps,
        )
        self.learning_rate = learning_rate
        self.enc_lr_scale = float(enc_lr_scale)
        self.pi_eps = float(pi_eps)
        # Sized at every train() call (see the class docstring).
        self.replay_capacity: Optional[int] = None
        self.replay_bytes_per_seed: Optional[int] = None
        # The values actually used, next to the constructor arguments (whose
        # None defaults are resolved above), for the loggers' run config.
        config = self.agent_config
        self.config.update(
            {
                "agent_episode_length": self.agent_episode_length,
                "resolved_gamma": self.gamma,
                "resolved_seed_steps": self.seed_steps,
                "resolved_iterations": config.planning_iterations(self.action_dim),
                **{
                    f"resolved_{name}": getattr(config, name)
                    for name in (
                        "enc_dim",
                        "mlp_dim",
                        "latent_dim",
                        "num_enc_layers",
                        "num_q",
                    )
                },
            }
        )

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            gamma=self.gamma,
            seed_steps=self.seed_steps,
            learning_rate=self.learning_rate,
            enc_lr_scale=self.enc_lr_scale,
            pi_eps=self.pi_eps,
            extensions=tuple(self.extension_stack.extensions),
        )

    def resume_iteration_offset(self, initial_state: BaseAgentState) -> int:
        """The absolute tick a resumed run starts at: the rows the collector
        emitted per env (one per tick), equal across seeds."""
        rows = np.asarray(jax.device_get(initial_state.collector_state.rows))
        rows = rows.reshape(-1)
        n_envs = self.env_args.n_envs
        if rows.size == 0 or np.any(rows != rows[0]) or rows[0] % n_envs:
            raise ValueError(
                "cannot resume: the collector row counts differ across seeds or"
                f" are not a multiple of n_envs={n_envs} ({rows.tolist()})"
            )
        return int(rows[0]) // n_envs

    def _replay_buffer(self, total_timesteps: int) -> EpisodeBuffer:
        """The ring of a call that ends after ``total_timesteps`` env steps of
        the run; records its capacity and size (class docstring)."""
        capacity = replay_capacity(self.agent_config.buffer_size, total_timesteps)
        buffer = EpisodeBuffer.create(
            capacity=capacity,
            n_envs=self.env_args.n_envs,
            episode_length=self.agent_episode_length,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
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
        return buffer

    def train(
        self,
        seed: int | Sequence[int] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Optional[BaseAgentState] = None,
        **kwargs: Any,
    ) -> Any:
        """:meth:`ActorCritic.train`; ``n_timesteps`` counts env (agent) steps.

        Sizes the replay ring for the env steps the run will have taken when
        this call ends and, on a resume, moves the carried episodes into it
        (class docstring). Returns ``(state, metrics)``: the per-tick metrics
        with a ``logging_config`` (NaN on the ticks that do not log), else
        ``None`` (nothing is evaluated or logged). Raises ``ValueError``
        after the run if any env terminated (deviation T10): the paper-era
        TD-MPC2 bootstraps through every episode end, so a terminating task
        would train on wrong targets.
        """
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            initial_state = initial_state[0]  # (state, metrics) from train()
        # The run's tick count depends on where it starts (it ends on the
        # tick of its last env step); the base class passes the same offset
        # to the scan.
        start_tick = (
            0 if initial_state is None else self.resume_iteration_offset(initial_state)
        )
        end_tick = start_tick + self.schedule.num_ticks(n_timesteps, start_tick)
        buffer = self._replay_buffer(
            self.schedule.steps_before(end_tick) * self.env_args.n_envs
        )
        if initial_state is not None:
            initial_state = initial_state.replace(
                buffer_state=buffer.adopt(initial_state.buffer_state, start_tick)
            )
        result: Any = super().train(
            seed=seed,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            on_ids_ready=on_ids_ready,
            initial_state=initial_state,
            capacity=self.replay_capacity,
            start_tick=start_tick,
            **kwargs,
        )
        state = result[0]
        offschedule = np.asarray(
            jax.device_get(state.collector_state.n_offschedule_dones)
        ).reshape(-1)
        terminations = np.asarray(jax.device_get(state.n_terminations)).reshape(-1)
        if np.any(offschedule > 0) or np.any(terminations > 0):
            raise ValueError(
                "TD-MPC2 needs fixed-length, non-terminating episodes, but the env"
                f" ended episodes off the T = {self.agent_episode_length}-step"
                f" schedule {offschedule.tolist()} times and terminated on its"
                f" last step {terminations.tolist()} times (per seed):"
                " terminations are not supported (paper-era TD-MPC2; deviation"
                " T10)."
            )
        return result
