"""Offline multi-task TD-MPC2 trainer (M8): smoke, update count, resume,
per-task evaluation, extensions, the update's wiring and the dataset shared
by the seeds.

A synthetic in-memory dataset of three toy tasks with different observation
and action dims (``docs/world_models/DESIGN.md`` §7, the CI gate) trains a
tiny model; the module-scoped runs (a logged run, its resumption, an
uninterrupted run of the same length) are asserted many ways.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2MultiTask
from ajax.agents.TDMPC2 import core, multitask, train_TDMPC2MultiTask
from ajax.agents.TDMPC2.core import TASK_EMB
from ajax.agents.TDMPC2.dataset import MultiTaskDataset, TaskEpisodes, pool_tasks
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.extensions.base import Extension, ExtensionContext
from ajax.logging.wandb_logging import LoggingConfig

from .toy_envs import TargetEnv

# The module (the package exports the class under the same name).
trainer_module = importlib.import_module("ajax.agents.TDMPC2.TDMPC2MultiTask")

T = 10
# (obs dim, action dim) per task: padded to 5 and 4.
DIMS = ((3, 2), (5, 1), (2, 4))
NAMES = ("reach", "push", "turn")
TINY = {
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "num_q": 2,
    "task_dim": 4,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 8,
    "num_pi_trajs": 4,
    "iterations": 2,
}
SEEDS = [0, 1]
FIRST, RESUMED, EVERY = 20, 15, 10


def make_dataset(episodes=(6, 4, 5), seed=0, episode_lengths=(T, T, T)):
    """Random episodes of ``T + 1`` rows of the three tasks in the schema's
    layout (each task's ``T`` may exceed the episodes' ``L - 1``)."""
    rng = np.random.default_rng(seed)
    tasks = []
    for n, (obs_dim, action_dim), name, length in zip(
        episodes, DIMS, NAMES, episode_lengths
    ):
        action = rng.uniform(-1, 1, (n, T + 1, action_dim)).astype(np.float32)
        action[:, -1] = 0.0
        reward = rng.uniform(0, 1, (n, T + 1)).astype(np.float32)
        reward[:, 0] = 0.0
        tasks.append(
            TaskEpisodes(
                obs=rng.normal(size=(n, T + 1, obs_dim)).astype(np.float32),
                action=action,
                reward=reward,
                episode_length=length,
                name=name,
            )
        )
    return pool_tasks(tasks)


def eval_envs():
    return [TargetEnv(o, a, T) for o, a in DIMS]


@dataclass(frozen=True)
class CounterExt(Extension):
    """Counts ``pretrain`` and ``post_update`` calls and their steps."""

    name: str = "counter"

    def init_state(self, agent_state, rng):
        return {
            "pretrain": jnp.asarray(0, jnp.int32),
            "updates": jnp.asarray(0, jnp.int32),
            "first_step": jnp.asarray(-1, jnp.int32),
            "last_step": jnp.asarray(-1, jnp.int32),
        }

    def pretrain(self, agent_state, ext_state, ctx: ExtensionContext):
        return agent_state, {**ext_state, "pretrain": ext_state["pretrain"] + 1}

    def post_update(self, agent_state, ext_state, ctx: ExtensionContext):
        first = ext_state["updates"] == 0
        return agent_state, {
            **ext_state,
            "updates": ext_state["updates"] + 1,
            "first_step": jnp.where(first, ctx.step, ext_state["first_step"]),
            "last_step": jnp.asarray(ctx.step, jnp.int32),
        }


@dataclass(frozen=True)
class MetricExt(Extension):
    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx: ExtensionContext):
        return {"ext_smoke/updates": jnp.asarray(ctx.step, jnp.float32)}


@dataclass(frozen=True)
class TargetTweak(Extension):
    name: str = "target-tweak"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target


@pytest.fixture(scope="module")
def runs():
    logged: list = []
    # (task, eval_mode) of every evaluation policy traced (once per program).
    eval_policies: set = set()
    task_planner_policy = multitask.task_planner_policy

    def policy_spy(*args, **kwargs):
        eval_policies.add((kwargs["task"], kwargs["eval_mode"]))
        return task_planner_policy(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            trainer_module,
            "vmap_log",
            lambda metrics, index, run_ids: logged.append((index, metrics)),
        )
        patch.setattr(trainer_module, "start_async_logging", lambda: None)
        patch.setattr(trainer_module, "stop_async_logging", lambda: None)
        patch.setattr(multitask, "task_planner_policy", policy_spy)
        dataset = make_dataset()
        agent = TDMPC2MultiTask(
            dataset,
            eval_envs=eval_envs(),
            extensions=[CounterExt(), MetricExt()],
            **TINY,
        )
        logging_config = LoggingConfig(
            config={}, use_wandb=False, use_tensorboard=False, log_frequency=EVERY
        )
        first, first_history = agent.train(
            seed=SEEDS,
            n_timesteps=FIRST,
            num_episode_test=2,
            logging_config=logging_config,
        )
        first_host = jax.device_get(first)  # train() donates the state it resumes
        first_logged = list(logged)
        resumed, resumed_history = agent.train(
            seed=SEEDS,
            n_timesteps=RESUMED,
            num_episode_test=2,
            logging_config=logging_config,
            initial_state=first,
        )
        resumed_logged = list(logged)
        # The uninterrupted run: one scan of 35 updates, logged once, by an
        # agent without eval envs (training and extension metrics only).
        full_agent = TDMPC2MultiTask(
            dataset, extensions=[CounterExt(), MetricExt()], **TINY
        )
        full, full_history = full_agent.train(
            seed=SEEDS,
            n_timesteps=FIRST + RESUMED,
            logging_config=LoggingConfig(
                config={},
                use_wandb=False,
                use_tensorboard=False,
                log_frequency=FIRST + RESUMED,
            ),
        )
    return SimpleNamespace(
        agent=agent,
        dataset=dataset,
        first=first_host,
        first_history=first_history,
        first_logged=first_logged,
        resumed=resumed,  # on the device, for the evaluation test
        resumed_history=resumed_history,
        resumed_logged=resumed_logged,
        full=jax.device_get(full),
        full_history=full_history,
        logged=logged,
        eval_policies=eval_policies,
    )


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_surface_follows_the_paper_multitask_protocol():
    """Batch 1024 and task_dim 96 by default (spec 4.19, deviation T14); the
    tasks, padded dims and per-task discounts come from the dataset; the
    planner iterates for the padded action dim."""
    dataset = make_dataset()
    agent = TDMPC2MultiTask(dataset)
    assert agent.agent_config == TDMPC2Config.from_model_size(5, batch_size=1024)
    assert agent.task_dim == 96
    # 10 evaluation episodes per task (offline_trainer.py:22-39, eval_episodes).
    for method, name in (
        (TDMPC2MultiTask.train, "num_episode_test"),
        (TDMPC2MultiTask.evaluate, "num_episodes"),
    ):
        assert inspect.signature(method).parameters[name].default == 10
    assert agent.tasks.names == NAMES
    assert (agent.tasks.obs_dim, agent.tasks.action_dim) == (5, 4)
    assert agent.tasks.discounts == pytest.approx((0.95,) * 3)  # T = 10
    assert agent.supported_extension_phases == {
        "pretrain",
        "post_update",
        "eval_metrics",
    }
    config = agent.config
    assert config["algo_name"] == "TDMPC2MultiTask"
    assert config["tasks"] == list(NAMES) and config["num_tasks"] == 3
    assert (config["padded_obs_dim"], config["padded_action_dim"]) == (5, 4)
    assert config["resolved_iterations"] == 6
    assert (
        config["dataset_episodes"] == 15 and config["dataset_bytes"] == dataset.nbytes
    )
    assert "dataset" not in config and "eval_envs" not in config


def test_each_task_discounts_with_its_own_episode_length():
    """Task ``i``'s discount is the heuristic of *its* ``T_i`` (spec 2.20),
    not of the longest task nor of the dataset's episode rows: here every
    task stores episodes of 11 rows but they last 10, 200 and 500 steps."""
    agent = TDMPC2MultiTask(make_dataset(episode_lengths=(10, 200, 500)), **TINY)
    assert agent.tasks.episode_lengths == (10, 200, 500)
    assert agent.tasks.discounts == tuple(
        core.discount_from_episode_length(t, 5, 0.95, 0.995) for t in (10, 200, 500)
    )
    assert agent.tasks.discounts == pytest.approx((0.95, 0.975, 0.99))
    assert agent.config["resolved_discounts"] == list(agent.tasks.discounts)


def test_the_dataset_is_placed_on_the_device_once():
    """A dataset of host (NumPy) arrays is transferred at construction, not
    by every chunk of training."""
    on_host = jax.tree.map(np.asarray, make_dataset())
    assert isinstance(on_host, MultiTaskDataset)
    assert isinstance(on_host.obs, np.ndarray)
    agent = TDMPC2MultiTask(on_host, **TINY)
    for leaf in jax.tree.leaves(agent.dataset):
        assert isinstance(leaf, jax.Array)
    assert agent.dataset.names == NAMES
    np.testing.assert_array_equal(agent.dataset.obs, on_host.obs)


def test_construction_is_validated():
    dataset = make_dataset()
    with pytest.raises(ValueError, match="on_target"):
        TDMPC2MultiTask(dataset, extensions=[TargetTweak()])
    with pytest.raises(ValueError, match="one eval env per task"):
        TDMPC2MultiTask(dataset, eval_envs=eval_envs()[:2])
    with pytest.raises(ValueError, match="eval env 1"):
        TDMPC2MultiTask(
            dataset,
            eval_envs=[TargetEnv(3, 2, T), TargetEnv(5, 2, T), TargetEnv(2, 4, T)],
        )
    with pytest.raises(ValueError, match="episode length"):
        TDMPC2MultiTask(dataset, eval_envs=[TargetEnv(o, a, T + 1) for o, a in DIMS])
    with pytest.raises(ValueError, match="horizon"):
        TDMPC2MultiTask(dataset, horizon=T + 1)
    with pytest.raises(ValueError, match="task_dim"):
        TDMPC2MultiTask(dataset, task_dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_smoke_counts_updates_and_keeps_everything_finite(runs):
    """``n_timesteps`` counts updates (``offline_trainer.py:73``): every seed
    makes exactly that many world-model and policy Adam steps."""
    for state, n in (
        (runs.first, FIRST),
        (jax.device_get(runs.resumed), FIRST + RESUMED),
    ):
        for count in (
            state.n_updates,
            state.world_model_state.step,
            state.actor_state.step,
        ):
            np.testing.assert_array_equal(count, [n, n])
        for tree in (state.world_model_state, state.actor_state, state.update_metrics):
            for leaf in jax.tree.leaves(tree):
                assert np.all(np.isfinite(leaf))
        assert np.all(state.update_metrics["total_loss"] > 0)
        assert state.world_model_state.params[TASK_EMB].shape == (2, 3, 4)
    # The seeds train differently.
    emb = runs.first.world_model_state.params[TASK_EMB]
    assert not np.allclose(emb[0], emb[1])
    # No env was ever stepped: the state has no collector.
    assert runs.first.collector_state is None


def test_update_step_updates_on_the_sampled_batch_and_its_tasks(monkeypatch):
    """One offline update (``offline_trainer.py:73``): the batch sampled from
    the dataset, *its* task ids (b67b21c ``buffer.py:28``) and fresh noise
    are what :func:`multitask.update` receives (a stub records them), with
    the agent's config and task set; then the update count advances."""
    agent = TDMPC2MultiTask(make_dataset(), **TINY)
    config, tasks, dataset = agent.agent_config, agent.tasks, agent.dataset
    received: dict = {}

    def update_stub(state, batch, task, noise, *, config, tasks):
        received.update(config=config, tasks=tasks)
        return state, {
            "batch": batch,
            "task": task,
            "noise": noise,
            "rng": state.rng,
        }

    monkeypatch.setattr(multitask, "update", update_stub)
    state = train_TDMPC2MultiTask.init_TDMPC2MultiTask(
        jax.random.PRNGKey(3),
        config,
        tasks,
        task_dim=agent.task_dim,
        learning_rate=3e-4,
        enc_lr_scale=0.3,
        pi_eps=1e-5,
    )
    new = jax.jit(
        partial(
            train_TDMPC2MultiTask.update_step,
            config=config,
            tasks=tasks,
            extension_stack=None,
            total_timesteps=1,
        )
    )(state, dataset)
    assert received == {"config": config, "tasks": tasks}
    rng, sample_key, noise_key, _ = jax.random.split(state.rng, 4)
    batch, task = dataset.sample(sample_key, config.batch_size, config.horizon)
    assert len(np.unique(np.asarray(task))) > 1  # a batch mixing the tasks
    noise = core.draw_update_noise(
        noise_key, config, config.batch_size, tasks.action_dim
    )
    got = new.update_metrics
    np.testing.assert_array_equal(got["task"], task)
    np.testing.assert_array_equal(got["rng"], rng)
    for a, b in zip(
        jax.tree.leaves((got["batch"], got["noise"])),
        jax.tree.leaves((batch, noise)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)
    assert int(new.n_updates) == 1


def test_a_fresh_train_function_starts_as_the_agent_does(monkeypatch):
    """``make_train``'s fresh path (``build_resumable_train``'s ``init_fn``)
    is the agent's: the same initial state, extensions' ``init_state`` and
    ``pretrain`` folded, then the same updates (with a stub update, which
    keeps the state, so only the initialisation, sampling and folds run)."""
    monkeypatch.setattr(
        multitask,
        "update",
        lambda state, batch, task, noise, **_: (state, state.update_metrics),
    )
    agent = TDMPC2MultiTask(make_dataset(), extensions=[CounterExt()], **TINY)
    seeds = jnp.asarray(SEEDS)
    index = jnp.arange(len(SEEDS))
    train = train_TDMPC2MultiTask.make_train(
        agent.agent_config,
        agent.tasks,
        2,
        task_dim=agent.task_dim,
        extensions=[CounterExt()],
        total_timesteps=2,
    )
    fresh = jax.jit(
        jax.vmap(
            lambda seed, i, d: train(jax.random.PRNGKey(seed), i, shared=d)[0],
            in_axes=(0, 0, None),
        )
    )(seeds, index, agent.dataset)
    resumed = agent._train_fn(2, 2)(seeds, index, agent._init(seeds, 2), agent.dataset)
    np.testing.assert_array_equal(fresh.ext_state[0]["pretrain"], [1, 1])
    np.testing.assert_array_equal(fresh.ext_state[0]["updates"], [2, 2])
    for a, b in zip(jax.tree.leaves(fresh), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)


def test_a_resumed_run_computes_the_uninterrupted_run(runs):
    """Chunks (evaluations between them) and a resume continue the same
    update stream: the same batches and draws as one scan of 35 updates."""
    resumed, full = jax.device_get(runs.resumed), runs.full
    for a, b in (
        (resumed.world_model_state.params, full.world_model_state.params),
        (resumed.world_model_state.target_params, full.world_model_state.target_params),
        (resumed.actor_state.params, full.actor_state.params),
        (resumed.q_scale.value, full.q_scale.value),
        (resumed.rng, full.rng),
    ):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_extension_hooks_follow_the_updates(runs):
    first = runs.first.ext_state[0]
    resumed = jax.device_get(runs.resumed.ext_state[0])
    np.testing.assert_array_equal(first["pretrain"], 1)
    np.testing.assert_array_equal(resumed["pretrain"], 1)  # not on resume
    np.testing.assert_array_equal(first["updates"], FIRST)
    np.testing.assert_array_equal(resumed["updates"], FIRST + RESUMED)
    np.testing.assert_array_equal(first["first_step"], 1)  # updates done
    np.testing.assert_array_equal(resumed["last_step"], FIRST + RESUMED)


def test_every_task_is_evaluated_and_logged_at_the_cadence(runs):
    """Every ``log_frequency`` updates (absolute: 10, 20, then 30 after the
    resume at 20), a host loop evaluates every task in its own env, logged
    per seed with the training metrics, the extensions' metrics and the
    normalised score ``mean(return / 10)``."""
    np.testing.assert_array_equal(runs.first_history["timestep"], [[10, 20]] * 2)
    np.testing.assert_array_equal(runs.resumed_history["timestep"], [[30]] * 2)
    np.testing.assert_array_equal(runs.first.n_logs, 2)
    np.testing.assert_array_equal(runs.resumed.n_logs, 3)
    assert len(runs.first_logged) == 2 * len(SEEDS)
    assert len(runs.resumed_logged) == 3 * len(SEEDS)
    for history in (runs.first_history, runs.resumed_history):
        returns = []
        for name in NAMES:
            np.testing.assert_array_equal(
                history[f"Eval/{name}/mean episodic length"], T
            )
            ret = history[f"Eval/{name}/episodic mean reward"]
            assert np.all((ret > -1.25 * T) & (ret <= T))  # 1 - mean((a - .5)^2)
            returns.append(ret)
        np.testing.assert_allclose(
            history["Eval/episodic mean reward"], np.mean(returns, 0), rtol=1e-6
        )
        np.testing.assert_allclose(
            history["Eval/normalized score"], np.mean(returns, 0) / 10, rtol=1e-6
        )
        np.testing.assert_array_equal(history["ext_smoke/updates"], history["timestep"])
        assert np.all(history["Train/total_loss"] > 0)
    index, entry = runs.resumed_logged[-1]
    assert index == 1 and int(entry["timestep"]) == 30
    assert {f"Eval/{name}/episodic mean reward" for name in NAMES} <= set(entry)


def test_every_task_is_evaluated_by_its_own_planner_in_eval_mode(runs):
    """The host loop evaluates each task with the planner conditioned on that
    task, in ``eval_mode`` (``offline_trainer.py:22-39``: ``act(...,
    eval_mode=True, task=task_idx)``)."""
    assert runs.eval_policies == {(0, True), (1, True), (2, True)}


def test_evaluation_does_not_persist_the_renorm(runs, monkeypatch):
    """Evaluating with a row above norm 1 renorms it for the decisions only:
    the state :meth:`train` returns keeps the row as it was (deviation T15;
    the reference writes it back at ``act``)."""
    monkeypatch.setattr(trainer_module, "vmap_log", lambda *args: None)
    monkeypatch.setattr(trainer_module, "start_async_logging", lambda: None)
    monkeypatch.setattr(trainer_module, "stop_async_logging", lambda: None)
    agent, state = runs.agent, runs.first  # 20 updates, on the host
    wm = state.world_model_state
    table = np.array(wm.params[TASK_EMB])  # [seed, task, task_dim]
    table[:, 2] *= 3.0 / np.linalg.norm(table[:, 2], axis=-1, keepdims=True)
    state = state.replace(
        world_model_state=wm.replace(params={**wm.params, TASK_EMB: table})
    )
    # No updates (the chunk only advances the clock): the evaluation at
    # update 30 is all that runs between the chunks.
    monkeypatch.setattr(
        agent,
        "_train_fn",
        lambda n, total: lambda seeds, index, s, dataset: s.replace(
            n_updates=s.n_updates + n
        ),
    )
    out, history = agent.train(
        seed=SEEDS,
        n_timesteps=EVERY,
        num_episode_test=2,
        logging_config=LoggingConfig(
            config={}, use_wandb=False, use_tensorboard=False, log_frequency=EVERY
        ),
        initial_state=state,
    )
    np.testing.assert_array_equal(history["timestep"], [[FIRST + EVERY]] * 2)
    assert np.all(np.isfinite(history["Eval/turn/episodic mean reward"]))
    np.testing.assert_array_equal(out.world_model_state.params[TASK_EMB], table)


def test_evaluation_plans_per_task_without_changing_the_state(runs):
    agent = runs.agent
    state = runs.resumed  # its evaluation programs are compiled
    out = agent.evaluate(state, num_episodes=2)
    again = agent.evaluate(state, num_episodes=2)
    for key in out:
        np.testing.assert_array_equal(out[key], again[key])
    assert set(out) >= {f"Eval/{name}/mean episodic length" for name in NAMES}
    # Each evaluation starts from fresh initial states (n_logs is folded in).
    later = agent.evaluate(state.replace(n_logs=state.n_logs + 1), num_episodes=2)
    assert not np.array_equal(
        later["Eval/reach/episodic mean reward"], out["Eval/reach/episodic mean reward"]
    )
    with pytest.raises(ValueError, match="no eval_envs"):
        TDMPC2MultiTask(runs.dataset, **TINY).evaluate(state)


def test_the_checkpoint_skeleton_has_no_updates():
    agent = TDMPC2MultiTask(make_dataset(), **TINY)
    skeleton, history = agent.train(seed=SEEDS, n_timesteps=0)
    np.testing.assert_array_equal(skeleton.n_updates, 0)
    assert history is None
    with pytest.raises(ValueError, match="update counts differ"):
        agent.resume_update_offset(
            skeleton.replace(n_updates=jnp.array([0, 1], jnp.int32))
        )


def test_without_eval_envs_the_training_metrics_are_logged(runs):
    """An agent without eval envs logs the training and extension metrics."""
    history = runs.full_history
    np.testing.assert_array_equal(history["timestep"], [[FIRST + RESUMED]] * 2)
    np.testing.assert_array_equal(history["ext_smoke/updates"], history["timestep"])
    assert not any(key.startswith("Eval/") for key in history)
    assert np.all(np.isfinite(history["Train/total_loss"]))
    assert len(runs.logged) == len(runs.resumed_logged) + len(SEEDS)
    np.testing.assert_array_equal(runs.full.n_logs, 1)


def test_the_dataset_enters_the_program_once_as_a_shared_argument():
    """The seed-vmapped training program takes the dataset as an unbatched
    argument: its size does not grow with the dataset (closing over it would
    embed the data as a constant) and no per-seed copy of it appears (a
    batched argument would make one)."""
    seeds = jnp.asarray(SEEDS)
    index = jnp.arange(len(SEEDS))

    def lowered(per_task):
        dataset = make_dataset((per_task,) * 3)
        agent = TDMPC2MultiTask(dataset, **TINY)
        state = jax.eval_shape(lambda: agent._init(seeds, 4))
        program = agent._train_fn(4, 4).lower(seeds, index, state, dataset)
        return dataset, program.as_text()

    small_dataset, small = lowered(4)
    large_dataset, large = lowered(400)
    assert large_dataset.nbytes > 90 * small_dataset.nbytes
    assert abs(len(large) - len(small)) < 1000
    for dataset, text in ((small_dataset, small), (large_dataset, large)):
        n, rows, obs_dim = dataset.obs.shape
        assert f"tensor<{n}x{rows}x{obs_dim}xf32>" in text  # the argument
        assert f"tensor<{len(SEEDS)}x{n}x{rows}x{obs_dim}xf32>" not in text
