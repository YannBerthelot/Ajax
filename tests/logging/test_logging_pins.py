"""Characterization pins: what a run with W&B and TensorBoard on logs today.

Pins current behavior, not verified against a spec. May encode bugs.

The pins hold the logging's restructure to today's calls. For a tiny PPO and
SAC run of two seeds, through train: every call the wandb module gets, in
order, per run, and from which process; each run's TensorBoard scalars and
config.json, the run's config holding a value that is neither JSON nor
picklable. The values are compared with the evaluations train returns
(a NaN metric is dropped from both sinks, docs/guide/logging.md), so the pins
do not depend on the numbers training produces.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from ajax import PPO, SAC
from ajax.logging.wandb_logging import LoggingConfig, load_scalars_from_tfevents
from tests.logging import fake_wandb

# These tests run the deprecated flags on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:LoggingConfig's use_wandb:FutureWarning"
)

NET = ("8", "relu")
SEEDS = [3, 5]
# A function, as an environment object in an agent's run configuration can
# be: not JSON, and not picklable.
USER_CONFIG = {"experiment": "pins", "reward": lambda step: 0.0}
# Each agent, its timesteps and its log frequency: 12 evaluations a run, so
# W&B gets a batch of 10 steps and, at shutdown, one of 2.
AGENTS: dict[str, tuple[Any, int, int]] = {
    "PPO": (
        lambda: PPO(
            "Pendulum-v1",
            n_envs=2,
            n_steps=4,
            num_minibatches=1,
            n_epochs=1,
            actor_architecture=NET,
            critic_architecture=NET,
        ),
        96,
        8,
    ),
    "SAC": (
        lambda: SAC(
            "Pendulum-v1",
            n_envs=2,
            learning_starts=16,
            batch_size=8,
            actor_architecture=NET,
            critic_architecture=NET,
        ),
        48,
        4,
    ),
}


def logged_rows(evaluations: dict, seed_index: int) -> list[tuple[int, dict]]:
    """A seed's (step, metrics) per evaluation, its NaN metrics left out."""
    rows = []
    for column in range(evaluations["timestep"].shape[1]):
        row = {k: np.asarray(v)[seed_index, column] for k, v in evaluations.items()}
        kept = {k: v.item() for k, v in row.items() if not np.isnan(v)}
        rows.append((int(row["timestep"]), kept))
    return rows


def worker_batch(run_id: str, rows: list[tuple[int, dict]]) -> list[tuple]:
    resumed = {"project": "pins", "name": f"run_{run_id}", "id": run_id}
    resumed.update(resume="must")
    steps = [("worker", "run.log", {"metrics": m, "step": t}) for t, m in rows]
    return [("worker", "init", resumed), *steps, ("worker", "run.finish", {})]


@pytest.mark.parametrize("agent_name", sorted(AGENTS))
def test_characterization_wandb_and_tensorboard_receive_each_evaluation(
    agent_name: str, tmp_path: Any, wandb_calls: str
) -> None:
    make, n_timesteps, every = AGENTS[agent_name]
    agent = make()
    logging = LoggingConfig(
        config=USER_CONFIG,
        project_name="pins",
        run_name="run",
        group_name="group",
        folder=str(tmp_path),
        log_frequency=every,
        use_wandb=True,
        use_tensorboard=True,
    )
    run_ids: list[str] = []
    _, evaluations = agent.train(
        seed=SEEDS,
        n_timesteps=n_timesteps,
        num_episode_test=2,
        logging_config=logging,
        on_ids_ready=run_ids.extend,
    )

    calls = fake_wandb.read_calls(wandb_calls)
    config = dict(USER_CONFIG)
    config.update(agent.config)
    assert sorted(os.listdir(tmp_path / "tensorboard")) == sorted(run_ids)
    for seed_index, (seed, run_id) in enumerate(zip(SEEDS, run_ids, strict=True)):
        rows = logged_rows(evaluations, seed_index)
        assert len(rows) == 12
        created = {"project": "pins", "name": f"run_{run_id}", "id": run_id}
        created.update(resume="never", config=fake_wandb.as_recorded(config))
        created.update(group="group", reinit="finish_previous")
        expected = [
            ("caller", "init", created),
            ("caller", "log", {"metrics": {"timestep": 0}, "step": None}),
            ("caller", "finish", {}),
            *worker_batch(run_id, rows[:10]),
            *worker_batch(run_id, rows[10:]),
        ]
        got = [
            (
                "caller" if c["pid"] == os.getpid() else "worker",
                c["call"],
                c["arguments"],
            )
            for c in calls
            if c["run_id"] == run_id
        ]
        assert got == expected

        # TensorBoard: each metric a scalar (float32) at its step, its tag
        # with spaces as underscores; config.json holds the merged config,
        # the seed and the run id, values that are not JSON as text.
        run_folder = tmp_path / "tensorboard" / run_id
        scalars = load_scalars_from_tfevents(run_folder)
        expected_scalars: dict[str, list] = {}
        for step, metrics in rows:
            for key, value in metrics.items():
                tag = key.replace(" ", "_")
                expected_scalars.setdefault(tag, []).append(
                    (step, float(np.float32(value)))
                )
        assert dict(scalars) == expected_scalars
        with open(run_folder / "config.json") as handle:
            written = json.load(handle)
        config_with_run = dict(config, seed=seed, run_id=run_id)
        assert written == json.loads(json.dumps(config_with_run, default=str))
