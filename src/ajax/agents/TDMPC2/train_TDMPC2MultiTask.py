"""TD-MPC2 multi-task offline training: the paper-era offline trainer, on device.

The reference (``nicklashansen/tdmpc2@b67b21c:tdmpc2/trainer/offline_trainer.py
:41-92``, the same at 5f6fade; tdmpc2_spec 4.19, 4.20) loads the pooled
dataset into its episode buffer and runs::

    for i in range(steps):
        train_metrics = agent.update(buffer)  # the online update, batch 1024
        if i % eval_freq == 0:
            eval()  # every task: 10 episodes, act(t0=(t == 0), eval_mode=True)

There is no planning, no env and no collection during training. Ajax runs the
updates as one ``lax.scan`` per chunk of updates (:func:`make_train`, through
``build_resumable_train``), one seed per vmap lane, with the dataset an
*unbatched* argument shared by every seed (the ``shared`` input of
``build_resumable_train``, never closed over: :class:`~ajax.agents.TDMPC2.
dataset.MultiTaskDataset` can be gigabytes); evaluation runs on the host
between chunks, task after task, each task a jitted
:func:`~ajax.evaluate.evaluate_policy` of the masked, padded planner in that
task's env (:func:`evaluate_task`;
:class:`~ajax.agents.TDMPC2.TDMPC2MultiTask.TDMPC2MultiTask` drives the
chunks).

One update (:func:`update_step`)::

    batch, task = dataset.sample(key)   # uniform episode x uniform crop over
                                        # the pooled episodes (b67b21c)
    multitask.update(state, batch, task, noise)
    extensions.post_update(step = updates done)
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from ajax.agents.TDMPC2 import core, multitask
from ajax.agents.TDMPC2.dataset import MultiTaskDataset
from ajax.agents.TDMPC2.multitask import TaskSet
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2MultiTaskState
from ajax.agents.TDMPC2.train_TDMPC2 import update_metrics_zeros
from ajax.evaluate import evaluate_policy
from ajax.extensions.base import Extension, ExtensionStack
from ajax.perf_utils import build_resumable_train
from ajax.state import EnvironmentConfig
from ajax.types import FloatOrCallable


def init_TDMPC2MultiTask(
    key: jax.Array,
    config: TDMPC2Config,
    tasks: TaskSet,
    *,
    task_dim: int,
    learning_rate: FloatOrCallable,
    enc_lr_scale: float,
    pi_eps: float,
) -> TDMPC2MultiTaskState:
    """Fresh networks (embedding table included) and optimizers."""
    model_key, rng, eval_rng = jax.random.split(key, 3)
    learner = multitask.create_update_state(
        model_key,
        config,
        tasks,
        task_dim=task_dim,
        learning_rate=learning_rate,
        enc_lr_scale=enc_lr_scale,
        pi_eps=pi_eps,
    )
    # The logged quantities' structure, from a batch of any tasks.
    task = jnp.zeros((config.batch_size,), jnp.int32)
    return TDMPC2MultiTaskState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=learner.actor_state,
        world_model_state=learner.world_model_state,
        q_scale=learner.q_scale,
        pi_gradnorm_sq=learner.pi_gradnorm_sq,
        update_metrics=update_metrics_zeros(
            learner,
            config,
            tasks.discount(task),
            tasks.obs_dim,
            tasks.action_dim,
            task=tasks.context(task),
        ),
        n_updates=jnp.zeros((), jnp.int32),
        n_logs=jnp.zeros((), jnp.int32),
    )


def update_step(
    agent_state: TDMPC2MultiTaskState,
    dataset: MultiTaskDataset,
    *,
    config: TDMPC2Config,
    tasks: TaskSet,
    extension_stack: Optional[ExtensionStack],
    total_timesteps: int,
) -> TDMPC2MultiTaskState:
    """One ``agent.update(buffer)`` on a fresh batch with fresh noise, then
    the ``post_update`` fold (``step`` = updates done, ``total_steps`` = the
    run's updates when this ``train`` call ends)."""
    rng, sample_key, noise_key, post_key = jax.random.split(agent_state.rng, 4)
    batch, task = dataset.sample(sample_key, config.batch_size, config.horizon)
    noise = core.draw_update_noise(
        noise_key, config, config.batch_size, tasks.action_dim
    )
    agent_state, metrics = multitask.update(
        agent_state.replace(rng=rng), batch, task, noise, config=config, tasks=tasks
    )
    agent_state = agent_state.replace(
        n_updates=agent_state.n_updates + 1, update_metrics=metrics
    )
    if extension_stack:
        agent_state = extension_stack.fold_post_update(
            agent_state, agent_state.n_updates, post_key, total_timesteps
        )
    return agent_state


def make_init(
    config: TDMPC2Config,
    tasks: TaskSet,
    *,
    task_dim: int,
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
    extensions: Sequence[Extension] = (),
    total_timesteps: int = 0,
) -> Callable[[jax.Array, Any], TDMPC2MultiTaskState]:
    """``(key, index) -> state``: a fresh run's state.

    :func:`init_TDMPC2MultiTask`, then the extensions' ``init_state`` and
    ``pretrain`` (step 0) folds, ``total_timesteps`` being their
    ``total_steps`` (the single-task agent's ``init_transform``). The fresh
    path of :func:`make_train`, and the agent's separate initialisation
    program (one program then serves every chunk of a run, fresh or
    resumed).
    """
    extension_stack = ExtensionStack(extensions)

    def init(key: jax.Array, index: Any) -> TDMPC2MultiTaskState:
        agent_state = init_TDMPC2MultiTask(
            key,
            config,
            tasks,
            task_dim=task_dim,
            learning_rate=learning_rate,
            enc_lr_scale=enc_lr_scale,
            pi_eps=pi_eps,
        ).replace(index=index)
        return extension_stack.fold_init(agent_state, key, total_timesteps)

    return init


def make_train(
    config: TDMPC2Config,
    tasks: TaskSet,
    num_updates: int,
    *,
    task_dim: int,
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
    extensions: Sequence[Extension] = (),
    total_timesteps: int = 0,
) -> Callable:
    """The per-seed train function of ``num_updates`` updates
    (``build_resumable_train``).

    Call it with the dataset as ``shared=`` (vmapped with ``in_axes=None``
    over seeds). A fresh run starts from :func:`make_init`'s state (the
    extensions' ``init_state`` and ``pretrain`` folded); a resumed run
    continues from ``initial_state``. The scan carries the state only; the
    last update's quantities are ``state.update_metrics``.
    ``total_timesteps`` is the extensions' ``total_steps``.
    """
    extension_stack = ExtensionStack(extensions)
    init = make_init(
        config,
        tasks,
        task_dim=task_dim,
        learning_rate=learning_rate,
        enc_lr_scale=enc_lr_scale,
        pi_eps=pi_eps,
        extensions=extensions,
        total_timesteps=total_timesteps,
    )

    def make_scan_fn(
        _agent_state: Any,
        _resume: bool,
        _key: Any,
        _index: Any,
        *,
        shared: MultiTaskDataset,
    ) -> Callable:
        def body(agent_state: TDMPC2MultiTaskState, _: Any) -> tuple[Any, None]:
            agent_state = update_step(
                agent_state,
                shared,
                config=config,
                tasks=tasks,
                extension_stack=extension_stack,
                total_timesteps=total_timesteps,
            )
            return agent_state, None

        return body

    return build_resumable_train(
        init_fn=init, make_scan_fn=make_scan_fn, num_updates=num_updates
    )


def evaluate_task(
    agent_state: TDMPC2MultiTaskState,
    key: jax.Array,
    *,
    env_args: EnvironmentConfig,
    config: TDMPC2Config,
    tasks: TaskSet,
    task: int,
    num_episodes: int,
) -> tuple[jax.Array, jax.Array]:
    """Mean return and length of ``num_episodes`` episodes of task ``task``.

    The offline trainer's evaluation (``offline_trainer.py:22-39``, spec
    4.20; deviation T16): :func:`~ajax.evaluate.evaluate_policy` in the
    task's own env (``env_args``, its training action repeat and episode
    length) with the planner in ``eval_mode``, ``t0`` on each episode's
    first step and a zero warm start (:func:`multitask.task_planner_policy`:
    padded observations, the task's prefix of the padded action). The
    episodes start from fresh initial states, ``key`` being folded with the
    number of evaluations so far and the task id.
    """
    key = jax.random.fold_in(jax.random.fold_in(key, agent_state.n_logs), task)
    policy = partial(
        multitask.task_planner_policy,
        wm_params=agent_state.world_model_state.params,
        pi_params=agent_state.actor_state.params,
        config=config,
        tasks=tasks,
        task=task,
        eval_mode=True,
    )
    return evaluate_policy(
        env_args,
        policy,
        lambda m: jnp.zeros((m, config.horizon, tasks.action_dim)),
        num_episodes,
        key,
    )


def train_metrics(agent_state: TDMPC2MultiTaskState) -> dict[str, jax.Array]:
    """The update count (the run's clock, ``timestep``) and the last update's
    quantities (``offline_trainer.py:78-84``: ``iteration`` and
    ``train_metrics``)."""
    metrics = {
        "timestep": agent_state.n_updates,
        "Train/n_updates": agent_state.n_updates,
    }
    metrics.update({f"Train/{k}": v for k, v in agent_state.update_metrics.items()})
    return metrics


__all__ = [
    "evaluate_task",
    "init_TDMPC2MultiTask",
    "make_init",
    "make_train",
    "train_metrics",
    "update_step",
]
