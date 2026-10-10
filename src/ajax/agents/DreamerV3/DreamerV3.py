"""DreamerV3 (Hafner, Pasukonis, Ba, Lillicrap, arXiv:2301.04104v2 / Nature 2025).

The paper-era recipe of ``danijar/dreamerv3@2411f7d`` with the upstream
replay-context fix ``29eb964`` (``docs/world_models/DESIGN.md`` section 6,
``docs/world_models/deviations.md`` section 1), for vector observations, in
float32 and synchronous (deviations D1-D6): a world model (block-GRU RSSM,
encoder, decoder, reward and continue heads) learned from replayed
sequences, an actor and a critic learned in its imagination, one joint
gradient per update and LaProp with adaptive gradient clipping.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.DreamerV3.replay import bytes_per_row, grow_rings
from ajax.agents.DreamerV3.state import (
    DreamerV3AgentConfig,
    DreamerV3Config,
    DreamerV3State,
    LearningRate,
)
from ajax.agents.DreamerV3.train_DreamerV3 import TrainRatio, env_spec, make_train
from ajax.environments.row_collector import resume_tick
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import BaseAgentState
from ajax.types import EnvType

logger = logging.getLogger(__name__)


class DreamerV3(ActorCritic):
    """DreamerV3, paper-era (2411f7d + 29eb964), for vector observations.

    Discrete and continuous actions. ``n_timesteps`` (of :meth:`train`)
    counts **rows**, the reference's ``step``: one per env per vector step,
    reset rows included (``DESIGN.md`` section 2); ``env_frames = rows *
    action_repeat`` is logged next to it.

    **Paper DMC-proprio protocol** (Table 2 p.19; dreamerv3_spec 7.5): the
    model size and train ratio of 2411f7d's ``dmc_proprio`` preset
    (``configs.yaml:233-236``), its action repeat (``env.dmc.repeat: 2``,
    ``:84``) and Table 2's budget of 500K env frames, i.e. 250K rows at
    repeat 2 (the preset itself sets ``run.steps: 3e5``, ``:237``;
    dreamerv3_spec Open question 1)::

        DreamerV3(
            task,
            model_size="12m",
            n_envs=16,
            train_ratio=512,
            action_repeat=2,
            episode_length=1000,
        )
        agent.train(seed=..., n_timesteps=250_000)  # 500K env frames

    on the mujoco_playground ports of the DMC tasks (deviation E18), whose
    episodes are then 500 agent steps (1000 simulator steps). The score is
    ``Train/episodic mean reward``, the returns of the stochastic policy's
    training episodes (dreamerv3_spec 7.3, 7.7).

    Args:
        env_id: env id (gymnax, brax or mujoco_playground) or a prebuilt
            env. The env must not normalise observations or rewards.
        n_envs: parallel envs (``run.num_envs``, 16).
        model_size: preset ``'1m'`` (later code only, for small runs),
            ``'12m'``, ``'25m'``, ``'50m'``, ``'100m'``, ``'200m'``,
            ``'400m'``: model dimension ``d`` with ``units = hidden = d``,
            ``deter = 8 d``, ``classes = d / 16``
            (:data:`~ajax.agents.DreamerV3.state.MODEL_SIZES`).
        units, deter, hidden, classes: explicit widths, overriding the
            preset's.
        stoch, blocks, enc_layers, dec_layers, rew_layers, con_layers,
        actor_layers, critic_layers, bins: the remaining network sizes
            (:class:`~ajax.agents.DreamerV3.state.DreamerV3Config`).
        free_nats, unimix, actor_unimix: world-model free bits and the
            latents' and discrete actor's uniform mixtures.
        dyn_scale, rep_scale, rec_scale, rew_scale, con_scale, actor_scale,
        critic_scale, repval_scale: loss scales (Table 4).
        train_ratio: replayed steps trained per collected row (an update
            every ``batch_size * batch_length / train_ratio`` rows after the
            training start; dreamerv3_spec 6.1-6.2).
        batch_size, batch_length: ``B`` windows of ``T`` trained rows (each
            window has one more row, the replay context; spec 5.8).
        imag_horizon: imagined steps ``H``.
        return_horizon: ``1 / (1 - gamma)``, the reference's ``horizon``
            (333): ``gamma = 1 - 1 / return_horizon`` scales the continue
            target and discounts the replay critic.
        lam, repval_lam: lambda of the imagination and replay returns.
        actent, slowreg, slow_rate: actor entropy scale, slow-critic
            regulariser weight and EMA rate.
        retnorm_rate, retnorm_limit: return-normaliser EMA rate and floor.
        minstd, maxstd: range of the continuous actor's standard deviation.
        learning_rate: a float or a schedule of the update count (the
            1000-update linear warmup multiplies it).
        agc, agc_pmin, beta1, beta2, eps, warmup: LaProp with adaptive
            gradient clipping (dreamerv3_spec 4.3-4.8).
        replay_capacity: rows kept, over all envs (Table 4: 5e6); see
            "Replay memory" below.
        episode_length: simulator steps per episode of brax /
            mujoco_playground envs (with ``action_repeat``, ``episode_length
            // action_repeat`` agent steps); gymnax envs keep their own.
        action_repeat: simulator steps per agent step (brax / playground).
        env_params: gymnax env parameters.
        extensions: Extensions folding the phases ``pretrain``,
            ``post_update`` (after every update) and ``eval_metrics``;
            others are rejected at construction.

    Static hyperparameters (fixing shapes or the program, so changing them
    recompiles): every width, depth and count, ``bins``, ``imag_horizon``,
    ``batch_size``, ``batch_length``, ``train_ratio``, ``replay_capacity``;
    the scalar loss and optimizer coefficients are Python floats of the
    frozen configuration as well.

    **Replay memory.** The replay is a ring of ``C = min(ceil(
    replay_capacity / n_envs), rows per env of the run)`` rows per env
    (capacity in rows: deviation D27), resolved by every :meth:`train` call
    (:meth:`_resolve_replay`: a run split into resumed calls keeps the
    replay of the uninterrupted run). Each row stores the observation, action,
    reward, flags and the posterior latent (``deter`` float32 and ``stoch``
    as uint8 class indices), about ``4 (obs_dim + action_dim + deter + 1) +
    stoch + 3`` bytes: 8.3 KB at 12m on a 24-dim observation, so the 250K
    rows of the DMC protocol take about 2.1 GB per seed
    (:attr:`replay_bytes_per_seed` gives the resolved figure). The training
    step needs a working set on top of it, mostly the imagination; it scales
    with ``B T (deter + stoch classes)``: about 1 GB at 12m (``B = 16``, ``T =
    64``, ``H = 15``), so about 3.2 GB per seed for the DMC protocol in all
    (XLA's memory analysis of the compiled run). The seed ``vmap``
    multiplies both.
    """

    name: str = "DreamerV3"
    supported_extension_phases: frozenset = frozenset(
        {"pretrain", "post_update", "eval_metrics"}
    )

    def __init__(
        self,
        env_id: Union[str, EnvType],
        n_envs: int = 16,
        model_size: str = "12m",
        units: Optional[int] = None,
        deter: Optional[int] = None,
        hidden: Optional[int] = None,
        classes: Optional[int] = None,
        stoch: int = 32,
        blocks: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        rew_layers: int = 1,
        con_layers: int = 1,
        actor_layers: int = 3,
        critic_layers: int = 3,
        bins: int = 255,
        free_nats: float = 1.0,
        unimix: float = 0.01,
        actor_unimix: float = 0.01,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        actor_scale: float = 1.0,
        critic_scale: float = 1.0,
        repval_scale: float = 0.3,
        train_ratio: float = 512,
        batch_size: int = 16,
        batch_length: int = 64,
        imag_horizon: int = 15,
        return_horizon: float = 333,
        lam: float = 0.95,
        repval_lam: float = 0.95,
        actent: float = 3e-4,
        slowreg: float = 1.0,
        slow_rate: float = 0.02,
        retnorm_rate: float = 0.01,
        retnorm_limit: float = 1.0,
        minstd: float = 0.1,
        maxstd: float = 1.0,
        learning_rate: LearningRate = 4e-5,
        agc: float = 0.3,
        agc_pmin: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup: int = 1000,
        replay_capacity: int = 5_000_000,
        episode_length: int = 1000,
        action_repeat: int = 1,
        env_params: Optional[EnvParams] = None,
        extensions: Sequence[Extension] = (),
    ) -> None:
        self.config = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }
        self.config.update({"algo_name": "DreamerV3"})
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            env_params=env_params,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        self.dreamer_config = DreamerV3Config.from_model_size(
            model_size,
            units=units,
            hidden=hidden,
            deter=deter,
            classes=classes,
            stoch=stoch,
            blocks=blocks,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            rew_layers=rew_layers,
            con_layers=con_layers,
            bins=bins,
            unimix=unimix,
            free_nats=free_nats,
            rec_scale=rec_scale,
            rew_scale=rew_scale,
            con_scale=con_scale,
            dyn_scale=dyn_scale,
            rep_scale=rep_scale,
            return_horizon=float(return_horizon),
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            actor_unimix=actor_unimix,
            minstd=minstd,
            maxstd=maxstd,
            imag_horizon=imag_horizon,
            lam=lam,
            repval_lam=repval_lam,
            actent=actent,
            slowreg=slowreg,
            slow_rate=slow_rate,
            retnorm_rate=retnorm_rate,
            retnorm_limit=retnorm_limit,
            actor_scale=actor_scale,
            critic_scale=critic_scale,
            repval_scale=repval_scale,
            learning_rate=learning_rate,
            agc=agc,
            agc_pmin=agc_pmin,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            warmup=warmup,
        )
        self.agent_config = DreamerV3AgentConfig(
            train_ratio=train_ratio,
            batch_size=batch_size,
            batch_length=batch_length,
            replay_capacity=replay_capacity,
        )
        self.schedule = TrainRatio(
            n_envs=n_envs,
            batch_size=batch_size,
            batch_length=batch_length,
            train_ratio=train_ratio,
        )
        self._capacity_rows = -(-int(replay_capacity) // n_envs)
        if self._capacity_rows < self.schedule.min_ring_rows():
            raise ValueError(
                f"replay_capacity={replay_capacity} keeps"
                f" {self._capacity_rows} rows per env, fewer than the"
                f" {self.schedule.min_ring_rows()} that hold the first batch's"
                f" {batch_size} windows of {batch_length + 1} rows."
            )
        #: Ring length ``C`` (rows per env) of the latest train() call.
        self.replay_rows_per_env: Optional[int] = None

    # ------------------------------------------------------------------ replay

    def _ring_rows(self, n_timesteps: int) -> int:
        """``C`` for a run of ``n_timesteps`` rows in all.

        ``min(ceil(replay_capacity / n_envs), rows per env of the run)``,
        never below the ring the first batch needs (a run too short to
        reach the first update keeps that minimal ring).
        """
        rows_per_env = int(n_timesteps) // self.env_args.n_envs
        return max(
            min(self._capacity_rows, rows_per_env), self.schedule.min_ring_rows()
        )

    def _resolve_replay(
        self, n_timesteps: int, state: Optional[DreamerV3State]
    ) -> tuple[int, Optional[DreamerV3State]]:
        """The ring length ``C`` of this :meth:`train` call, and the state to
        resume from (``DESIGN.md`` sections 5.5, 5.6, 6.3).

        * A fresh run of at least one tick sizes the ring from its own
          length (:meth:`_ring_rows`) and records it.
        * A fresh run of 0 ticks (a checkpoint skeleton) reuses the ring the
          agent's latest run recorded, so that its shapes are that run's
          (the minimal ring before any run); it records nothing.
        * A resumed run continues the uninterrupted run of all its rows: it
          needs the ring :meth:`_ring_rows` gives for the rows so far plus
          this call's. A shorter ring was sized by an earlier, shorter call
          and has overwritten nothing, so it grows to that length exactly
          (:func:`~ajax.agents.DreamerV3.replay.grow_rings`); a ring at
          least that long is kept. The result is recorded.
        """
        n_envs = self.env_args.n_envs
        ticks = int(n_timesteps) // n_envs
        if state is None:
            if ticks == 0:
                recorded = self.replay_rows_per_env
                return (self._ring_rows(0) if recorded is None else recorded), None
            rows = self._ring_rows(n_timesteps)
            self._record_replay_rows(rows)
            return rows, None
        replay_state = state.replay_state
        held = int(np.shape(replay_state.obs)[np.ndim(replay_state.popped) + 1])
        done = self.resume_iteration_offset(state)
        wanted = self._ring_rows((done + ticks) * n_envs)
        if wanted > held:
            if done > held:
                raise ValueError(
                    f"The state's replay keeps the last {held} of its {done}"
                    f" rows per env, but this agent's replay_capacity keeps"
                    f" {wanted} for the resumed run: resume with the"
                    " replay_capacity that produced the state."
                )
            logger.info(
                "DreamerV3 replay: growing the state's ring from %d to %d rows"
                " per env for the resumed run of %d rows per env.",
                held,
                wanted,
                done + ticks,
            )
            state = state.replace(replay_state=grow_rings(replay_state, done, wanted))
            held = wanted
        self._record_replay_rows(held)
        return held, state

    def _record_replay_rows(self, rows: int) -> None:
        self.replay_rows_per_env = rows
        logger.info(
            "DreamerV3 replay: %d rows per env x %d envs, %.3f GB per seed"
            " (the replay rings; the training step's working set comes on top)",
            rows,
            self.env_args.n_envs,
            self.replay_bytes_per_seed / 1e9,
        )

    @property
    def replay_bytes_per_seed(self) -> int:
        """Bytes of the resolved replay rings of one seed (0 before)."""
        if self.replay_rows_per_env is None:
            return 0
        spec = env_spec(self.env_args)
        c = self.dreamer_config
        row = bytes_per_row(
            spec.obs_dim, spec.action_dim, spec.discrete, c.deter, c.stoch, c.classes
        )
        return row * self.replay_rows_per_env * self.env_args.n_envs

    # ------------------------------------------------------------------ train

    def resume_iteration_offset(self, initial_state: BaseAgentState) -> int:
        """The absolute tick of a resumed run, equal across seeds
        (:func:`~ajax.environments.row_collector.resume_tick`), so every
        schedule (the training-start gate, the ratio, the online queue, the
        logging cadence) continues instead of restarting."""
        return resume_tick(initial_state.collector_state, self.env_args.n_envs)

    def train(
        self,
        seed: Union[int, Sequence[int]] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Optional[BaseAgentState] = None,
        **kwargs: Any,
    ) -> tuple[BaseAgentState, Any]:
        """Train for ``n_timesteps`` rows (``n_timesteps // n_envs`` ticks).

        Returns ``(state, evaluations)`` vmapped over seeds, as every agent
        on the shared loop (:mod:`ajax.agents.loop`): ``None`` without a
        logging config, else every logged key's values at the evaluations
        (every ``log_frequency`` rows). Resume with
        ``initial_state=`` a returned state (or ``(state, metrics)``): the
        run continues the uninterrupted run of all the rows, schedules and
        replay included (:meth:`_resolve_replay`).
        """
        state: Any = initial_state
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]
        rows, state = self._resolve_replay(n_timesteps, state)
        return super().train(
            seed=seed,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            on_ids_ready=on_ids_ready,
            initial_state=state,
            replay_rows_per_env=rows,
            **kwargs,
        )

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            config=self.dreamer_config,
            extensions=tuple(self.extension_stack.extensions),
        )


__all__ = ["DreamerV3"]
