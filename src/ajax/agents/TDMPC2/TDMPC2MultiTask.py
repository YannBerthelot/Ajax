"""TD-MPC2 multi-task (Hansen, Su, Wang, ICLR 2024), offline, paper-era code.

One TD-MPC2 agent trained offline on the pooled data of several tasks of
different observation and action dims (``nicklashansen/tdmpc2@b67b21c``
offline trainer, ``5f6fade`` agent; ``docs/world_models/DESIGN.md`` §7,
tdmpc2_spec 1.21-1.23, 2.18, 2.20, 3.9, 3.25, 4.15-4.20, 4.28): the
task-conditioned world model and policy of :mod:`.multitask`, trained by
:mod:`.train_TDMPC2MultiTask` on a :class:`~.dataset.MultiTaskDataset`.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.TDMPC2.dataset import MultiTaskDataset
from ajax.agents.TDMPC2.multitask import PAPER_TASK_DIM, TaskSet
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2MultiTaskState
from ajax.agents.TDMPC2.train_TDMPC2MultiTask import (
    evaluate_task,
    make_init,
    make_train,
    train_metrics,
)
from ajax.environments.create import prepare_env
from ajax.environments.utils import (
    agent_episode_length,
    get_action_dim,
    get_env_type,
    get_state_action_shapes,
)
from ajax.extensions.base import Extension, ExtensionStack, check_extension_phases
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    start_async_logging,
    stop_async_logging,
    vmap_log,
)
from ajax.state import EnvironmentConfig
from ajax.types import EnvType, FloatOrCallable

try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# The DMC normalisation of the paper's score: returns in [0, 1000] to [0, 100]
# (evaluate.py:91-94, spec 4.20).
DMC_SCORE_DIVISOR = 10.0


class TDMPC2MultiTask:
    """Multi-task TD-MPC2, trained offline on a pooled dataset (paper era).

    The reference's multi-task path is offline only (``train.py:52``;
    tdmpc2_spec 3.25, 4.1): the updates of the single-task agent, conditioned
    on each slice's task (:func:`ajax.agents.TDMPC2.multitask.update`: a
    learnable task embedding with max-norm 1, zero-padded observations,
    prefix action masks, per-task discounts), on batches of slices drawn
    uniformly over the dataset's pooled episodes, no planning during
    training. ``n_timesteps`` counts updates (the reference's iterations;
    paper: 10M at batch 1024, spec 4.B). The paper protocol is
    ``batch_size=1024`` (the default here) and ``task_dim=96`` for every
    model size (deviation T14).

    The dataset (:class:`~ajax.agents.TDMPC2.dataset.MultiTaskDataset`, e.g.
    :func:`~ajax.agents.TDMPC2.dataset.pool_tasks` of single-task runs'
    exports) fixes the tasks: task ``i`` has the dataset's ``obs_dims[i]``,
    ``action_dims[i]`` and episode length ``T_i``, whose discount is the
    reference's heuristic of ``T_i`` (``discount_denom``, ``discount_min``,
    ``discount_max``; spec 2.20). The model sees the padded observation and
    action spaces; the planner's iteration rule uses the padded action dim
    (spec 3.1). The dataset is one array set on the device, shared by every
    seed: the constructor places it there once (whatever built it: host
    NumPy arrays are transferred then, not at every chunk) and :meth:`train`
    passes it to its jitted, seed-vmapped program as an unbatched argument,
    so the seeds read the same copy (``dataset.nbytes``) and the program
    does not embed it.

    Logging (with a ``logging_config``): every ``log_frequency`` updates,
    the last update's quantities (``Train/...``, ``timestep`` = updates
    done), the extensions' ``eval_metrics`` and, with ``eval_envs``, an
    evaluation: a host loop over the tasks runs
    ``num_episode_test`` episodes of each in its own env (``eval_envs[i]``,
    built with ``episode_length`` simulator steps and ``action_repeat``; its
    dims and agent-step episode length must be task ``i``'s), the planner in
    ``eval_mode`` with ``t0`` on each episode's first step: the offline
    trainer's protocol (``offline_trainer.py:22-39``; deviation T16:
    ``evaluate.py``'s exploration noise is not reproduced). Logged per task
    ``Eval/<name>/episodic mean reward`` and length, their mean over tasks
    ``Eval/episodic mean reward`` and the normalised score ``Eval/normalized
    score``, the mean over tasks of ``return / 10`` (the paper's DMC
    normalisation, meaningful for DMC returns in [0, 1000]; spec 4.20). The
    first evaluation follows the first ``log_frequency`` updates (the
    reference's follows its first update). Each evaluation starts its
    episodes from fresh initial states. The embedding renorm of a look-up at
    evaluation is not persisted (deviation T15).

    Static hyperparameters, as :class:`~ajax.agents.TDMPC2.TDMPC2.TDMPC2`'s:
    the architecture (``model_size`` preset and explicit widths),
    ``task_dim``, ``simnorm_dim``, ``num_bins``, ``vmax``, ``horizon``,
    ``batch_size`` and the planner sizes. ``learning_rate`` accepts a float
    or a schedule (the task embedding is trained by the world-model
    optimizer at ``learning_rate``, the encoder at ``learning_rate *
    enc_lr_scale``).

    Extensions: ``pretrain``, ``post_update`` (folded after every update,
    ``step`` = updates done) and ``eval_metrics`` (with ``init_state``), the
    phases of the single-task agent; the others are rejected at
    construction.
    """

    name: str = "TDMPC2MultiTask"
    supported_extension_phases: frozenset = frozenset(
        {"pretrain", "post_update", "eval_metrics"}
    )

    def __init__(
        self,
        dataset: MultiTaskDataset,
        eval_envs: Optional[Sequence[str | EnvType]] = None,
        model_size: int = 5,
        enc_dim: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_enc_layers: Optional[int] = None,
        num_q: Optional[int] = None,
        task_dim: int = PAPER_TASK_DIM,
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
        batch_size: int = 1024,
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
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Args:
            dataset: the pooled multi-task dataset (it fixes the tasks).
            eval_envs: one env id or prebuilt gymnax env per task, in task-id
                order, for evaluation; ``None`` trains without evaluating.
            task_dim: task-embedding width (paper: 96; deviation T14).
            episode_length: simulator steps per episode of the brax /
                playground evaluation envs.
            action_repeat: simulator steps per agent step of the evaluation
                envs (the paper's DMC protocol: 2).
            extensions: composable research features (see
                :mod:`ajax.extensions`).

        Every other argument is the hyperparameter of the same name of
        :class:`~ajax.agents.TDMPC2.TDMPC2.TDMPC2` (``5f6fade``'s
        ``config.yaml``), with the paper's multi-task batch size 1024.
        """
        self.config = {
            k: v
            for k, v in locals().items()
            if k not in ("self", "__class__", "dataset", "eval_envs")
        }
        self.config.update({"algo_name": "TDMPC2MultiTask"})
        self.extension_stack = ExtensionStack(extensions)
        check_extension_phases(
            type(self).__name__, self.extension_stack, self.supported_extension_phases
        )
        dataset.check()
        if horizon > dataset.rows - 1:
            raise ValueError(
                f"the dataset's episodes of {dataset.rows} rows are too short for"
                f" horizon {horizon}: a slice needs horizon + 1 rows"
            )
        if task_dim < 1:
            raise ValueError(f"task_dim must be >= 1, got {task_dim}")
        self.dataset = jax.device_put(dataset)
        self.tasks = TaskSet.create(
            dataset.obs_dims,
            dataset.action_dims,
            dataset.episode_lengths,
            names=dataset.names,
            discount_denom=discount_denom,
            discount_min=discount_min,
            discount_max=discount_max,
        )
        self.task_dim = int(task_dim)
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
        )
        self.learning_rate = learning_rate
        self.enc_lr_scale = float(enc_lr_scale)
        self.pi_eps = float(pi_eps)
        self.action_repeat = int(action_repeat)
        self.eval_env_args: Optional[tuple[EnvironmentConfig, ...]] = (
            None
            if eval_envs is None
            else self._eval_envs(eval_envs, episode_length, action_repeat)
        )
        self.run_ids: list[str] = []
        self._train_fns: dict[tuple[int, int], Callable] = {}
        self._eval_fns: dict[tuple[int, int], Callable] = {}
        self._extra_eval_fns: dict[int, Callable] = {}
        self._init_fns: dict[int, Callable] = {}
        config = self.agent_config
        self.config.update(
            {
                "tasks": list(self.tasks.names),
                "num_tasks": self.tasks.num_tasks,
                "padded_obs_dim": self.tasks.obs_dim,
                "padded_action_dim": self.tasks.action_dim,
                "resolved_discounts": list(self.tasks.discounts),
                "resolved_iterations": config.planning_iterations(
                    self.tasks.action_dim
                ),
                "dataset_episodes": dataset.num_episodes,
                "dataset_rows": dataset.rows,
                "dataset_bytes": dataset.nbytes,
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

    # -- setup -----------------------------------------------------------
    def _eval_envs(
        self,
        eval_envs: Sequence[str | EnvType],
        episode_length: int,
        action_repeat: int,
    ) -> tuple[EnvironmentConfig, ...]:
        """One evaluation env per task, checked against the task's dims."""
        tasks = self.tasks
        if len(eval_envs) != tasks.num_tasks:
            raise ValueError(
                f"one eval env per task: {tasks.num_tasks} tasks, got"
                f" {len(eval_envs)} envs"
            )
        out = []
        for i, env_id in enumerate(eval_envs):
            env, env_params, _, continuous = prepare_env(
                env_id,
                episode_length=episode_length,
                n_envs=1,
                action_repeat=action_repeat,
            )
            if env_params is None and get_env_type(env) == "gymnax":
                env_params = env.default_params
            obs_shape, _ = get_state_action_shapes(env)
            dims = (
                tuple(obs_shape),
                get_action_dim(env, env_params) if continuous else None,
                agent_episode_length(env, env_params, action_repeat),
            )
            expected = (
                (tasks.obs_dims[i],),
                tasks.action_dims[i],
                tasks.episode_lengths[i],
            )
            if dims != expected:
                raise ValueError(
                    f"eval env {i} ({env_id!r}) has observation shape, action dim"
                    f" and episode length (agent steps) {dims}, but task"
                    f" {tasks.names[i]!r} has {expected} (continuous actions"
                    " only)"
                )
            out.append(
                EnvironmentConfig(
                    env=env,
                    env_params=env_params,
                    n_envs=1,
                    continuous=True,
                    action_repeat=action_repeat,
                )
            )
        return tuple(out)

    def resume_update_offset(self, initial_state: TDMPC2MultiTaskState) -> int:
        """The updates a resumed run has done, equal across seeds."""
        done = np.asarray(jax.device_get(initial_state.n_updates)).reshape(-1)
        if done.size == 0 or np.any(done != done[0]):
            raise ValueError(
                f"cannot resume: the update counts differ across seeds ({done})"
            )
        return int(done[0])

    # -- jitted programs -------------------------------------------------
    def _train_fn(self, num_updates: int, total: int) -> Callable:
        """The jitted, seed-vmapped program of a chunk (cached per length).

        ``(seeds, index, state, dataset) -> state``: the dataset is an
        argument, unbatched (``in_axes=None``), so every seed reads the one
        copy and the program does not embed it; the carried state is
        donated (as ``build_resumable_train`` donates ``initial_state``).
        """
        cache_key = (num_updates, total)
        if cache_key not in self._train_fns:
            train = make_train(
                self.agent_config,
                self.tasks,
                num_updates,
                task_dim=self.task_dim,
                learning_rate=self.learning_rate,
                enc_lr_scale=self.enc_lr_scale,
                pi_eps=self.pi_eps,
                extensions=self.extension_stack.extensions,
                total_timesteps=total,
            )

            def resume(seed: Any, index: Any, state: Any, dataset: Any) -> Any:
                return train(
                    jax.random.PRNGKey(seed),
                    index,
                    initial_state=state,
                    resume_from_state=True,
                    shared=dataset,
                )[0]

            self._train_fns[cache_key] = jax.jit(
                jax.vmap(resume, in_axes=(0, 0, 0, None)), donate_argnums=(2,)
            )
        return self._train_fns[cache_key]

    def _init(self, seeds: jax.Array, total: int) -> TDMPC2MultiTaskState:
        """Fresh states (:func:`make_init`: the extensions' ``init_state``
        and ``pretrain`` folded, ``total`` their ``total_steps``), one jitted
        program cached per ``total``: every chunk of a run is then the
        resumed path of a single training program (:meth:`_train_fn`)."""
        if total not in self._init_fns:
            init = make_init(
                self.agent_config,
                self.tasks,
                task_dim=self.task_dim,
                learning_rate=self.learning_rate,
                enc_lr_scale=self.enc_lr_scale,
                pi_eps=self.pi_eps,
                extensions=self.extension_stack.extensions,
                total_timesteps=total,
            )
            self._init_fns[total] = jax.jit(
                jax.vmap(lambda seed, index: init(jax.random.PRNGKey(seed), index))
            )
        return self._init_fns[total](seeds, jnp.arange(seeds.shape[0]))

    def _eval_fn(self, task: int, num_episodes: int) -> Callable:
        """Task ``task``'s seed-vmapped, jitted evaluation (cached)."""
        cache_key = (task, num_episodes)
        if cache_key not in self._eval_fns:
            assert self.eval_env_args is not None
            evaluate = partial(
                evaluate_task,
                env_args=self.eval_env_args[task],
                config=self.agent_config,
                tasks=self.tasks,
                task=task,
                num_episodes=num_episodes,
            )
            self._eval_fns[cache_key] = jax.jit(
                jax.vmap(lambda state: evaluate(state, state.eval_rng))
            )
        return self._eval_fns[cache_key]

    def evaluate(
        self, state: TDMPC2MultiTaskState, num_episodes: int = 10
    ) -> dict[str, np.ndarray]:
        """One evaluation of every task (host loop), per seed.

        ``state`` has a leading seed axis (as :meth:`train` returns it).
        Returns ``{key: [n_seeds]}``: per task ``Eval/<name>/episodic mean
        reward`` and ``Eval/<name>/mean episodic length``, their means over
        tasks and ``Eval/normalized score`` (class docstring). It does not
        advance ``state.n_logs``: :meth:`train` does after logging, so the
        next evaluation starts from new initial states.
        """
        if self.eval_env_args is None:
            raise ValueError("no eval_envs were given: nothing to evaluate on")
        out: dict[str, np.ndarray] = {}
        returns, lengths = [], []
        for task, name in enumerate(self.tasks.names):
            ret, length = jax.device_get(self._eval_fn(task, num_episodes)(state))
            out[f"Eval/{name}/episodic mean reward"] = np.asarray(ret)
            out[f"Eval/{name}/mean episodic length"] = np.asarray(length)
            returns.append(np.asarray(ret))
            lengths.append(np.asarray(length))
        out["Eval/episodic mean reward"] = np.mean(returns, axis=0)
        out["Eval/mean episodic length"] = np.mean(lengths, axis=0)
        out["Eval/normalized score"] = np.mean(
            np.asarray(returns) / DMC_SCORE_DIVISOR, axis=0
        )
        return out

    def _log_metrics(
        self, state: TDMPC2MultiTaskState, num_episodes: int, total: int
    ) -> dict[str, np.ndarray]:
        """``{key: [n_seeds]}`` of one log: the training metrics, the
        evaluation (with ``eval_envs``) and the extensions' ``eval_metrics``
        (``total`` is their ``total_steps``)."""
        out = {
            k: np.asarray(v)
            for k, v in jax.device_get(jax.vmap(train_metrics)(state)).items()
        }
        if self.eval_env_args is not None:
            out.update(self.evaluate(state, num_episodes))
        if self.extension_stack:
            extra = jax.device_get(self._extra_eval_fn(total)(state))
            out.update({k: np.asarray(v) for k, v in extra.items()})
        return out

    def _extra_eval_fn(self, total: int) -> Callable:
        """The extensions' ``eval_metrics`` fold, jitted, seed-vmapped
        (cached); ``step`` = updates done."""
        if total not in self._extra_eval_fns:
            stack = self.extension_stack

            def extra(s: TDMPC2MultiTaskState) -> dict:
                key = jax.random.fold_in(jax.random.split(s.eval_rng)[1], s.n_logs)
                return stack.fold_eval_metrics(s, s.n_updates, key, total)

            self._extra_eval_fns[total] = jax.jit(jax.vmap(extra))
        return self._extra_eval_fns[total]

    # -- training --------------------------------------------------------
    def train(
        self,
        seed: int | Sequence[int] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Optional[TDMPC2MultiTaskState] = None,
    ) -> tuple[TDMPC2MultiTaskState, Optional[dict[str, np.ndarray]]]:
        """Run ``n_timesteps`` more updates for every seed (vmapped).

        A fresh run initialises the networks (and folds the extensions'
        ``init_state`` and ``pretrain``); ``initial_state`` (a state this
        agent's ``train`` returned, or a checkpoint restored into the
        ``n_timesteps=0`` skeleton) continues a run; its buffers are donated
        to the first chunk (as ``ActorCritic.train`` donates them). The
        updates run as jitted scans; with a ``logging_config`` whose
        ``log_frequency`` (updates) is set, the scans stop every
        ``log_frequency`` updates of the run (an absolute cadence: a resumed
        run keeps it) to log the training metrics and, with ``eval_envs``,
        to evaluate every task (class docstring).

        Returns ``(state, history)``: the state with a leading seed axis and,
        when it logged, ``{key: [n_seeds, n_logs]}`` (training and evaluation
        metrics), else ``None``.
        """
        seeds = jnp.asarray([seed] if isinstance(seed, int) else list(seed))
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            initial_state = initial_state[0]
        if n_timesteps < 0:
            raise ValueError(f"n_timesteps must be >= 0, got {n_timesteps}")
        start = 0 if initial_state is None else self.resume_update_offset(initial_state)
        end = start + n_timesteps
        every = (
            int(logging_config.log_frequency)
            if logging_config is not None and logging_config.log_frequency
            else 0
        )

        if logging_config is not None:
            logging_config.config.update(self.config)
            new_id = wandb.util.generate_id if wandb is not None else _uuid
            self.run_ids = [new_id() for _ in range(seeds.shape[0])]
            for run_id, run_seed in zip(self.run_ids, seeds.tolist()):
                init_logging(run_id, logging_config, run_seed=int(run_seed))
            start_async_logging()
        else:
            self.run_ids = []
        if on_ids_ready is not None:
            on_ids_ready(self.run_ids)

        state = self._init(seeds, end) if initial_state is None else initial_state
        index = jnp.arange(seeds.shape[0])
        # Chunk ends: the run's multiples of `every` (absolute), then `end`.
        stops = (
            sorted({*range(start - start % every + every, end, every), end})
            if every
            else [end]
        )
        history: dict[str, list[np.ndarray]] = defaultdict(list)
        done = start
        t0 = time.time()
        for stop in stops:
            if stop > done:
                state = self._train_fn(stop - done, end)(
                    seeds, index, state, self.dataset
                )
                done = stop
            if every and done % every == 0 and done > start:
                metrics = self._log_metrics(state, num_episode_test, end)
                for k, v in metrics.items():
                    history[k].append(v)
                assert logging_config is not None
                for i in range(seeds.shape[0]):
                    vmap_log(
                        {k: v[i] for k, v in metrics.items()},
                        i,
                        self.run_ids,
                        logging_config,
                    )
                state = state.replace(n_logs=state.n_logs + 1)
        state = jax.block_until_ready(state)
        if logging_config is not None:
            stop_async_logging()
        logger.info(
            "TD-MPC2 multi-task: %d updates per seed in %.1f s",
            end - start,
            time.time() - t0,
        )
        if not history:
            return state, None
        return state, {k: np.stack(v, axis=-1) for k, v in history.items()}


def _uuid() -> str:
    return uuid.uuid4().hex


__all__ = ["DMC_SCORE_DIVISOR", "TDMPC2MultiTask"]
