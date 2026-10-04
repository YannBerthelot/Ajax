"""TD-MPC2 multi-task validation pipeline on self-generated playground data (M9).

``docs/world_models/DESIGN.md`` section 7 ("Validation (M9)") and
``docs/world_models/VALIDATION.md``. The paper trains its multi-task models
offline on the pooled replay buffers of single-task agents (Sec. 4.1;
tdmpc2_spec 4.18-4.19); the official mt30 / mt80 datasets are out of scope,
so this pipeline makes its own, on the playground versions of the 19
original DMC tasks of mt30 (``5f6fade:tdmpc2/common/__init__.py:26-37``;
its 11 custom tasks have no playground version), in mt30 order:

1. ``sources``: single-task :class:`~ajax.TDMPC2` runs (the paper's
   single-task settings: ``model_size=5``, action repeat 2, 1000-step
   episodes) of :data:`PROTOCOL` ``source_budget`` agent steps, ``k``
   seeds each, evaluated every 50K steps like the paper-protocol runs. The
   replay keeps the run's full history: ``buffer_size`` (1M) is at least
   the run length, so nothing is evicted, and the export
   (:func:`~ajax.agents.TDMPC2.dataset.export_episodes`) raises if anything
   was. Each task's episodes (all seeds) are saved as a one-task dataset.
2. ``dataset``: the tasks pooled (:func:`~ajax.agents.TDMPC2.dataset.
   pool_tasks`, task id = mt30 position among the available tasks), saved
   (:func:`~ajax.agents.TDMPC2.dataset.save_dataset`), loaded back and
   checked equal (:func:`~ajax.agents.TDMPC2.dataset.load_dataset` and
   :meth:`~ajax.agents.TDMPC2.dataset.MultiTaskDataset.check`).
   ``dataset.json`` records its summary and the size and SHA-256 of
   ``dataset.npz`` and of every source export: a later stage refuses a
   dataset whose files changed since (a source run redone), and the
   offline run's specification holds it, so a resume or a report on
   another dataset is refused.
3. ``train``: :class:`~ajax.TDMPC2MultiTask` trained offline on it (batch
   1024, the paper's multi-task batch; ``model_size=19``, the paper's
   ablation size, spec 4.30), every task evaluated after each chunk (10
   episodes of the planner in ``eval_mode``, the offline trainer's
   protocol, spec 4.20).
4. ``report``: per task, the offline model's return against its source
   agents' final return, judged by ``wm_acceptance.judge_multitask``
   (fraction fixed before any run), as a validation of the multi-task
   mechanisms on Ajax's own data, not of the paper's multi-task numbers.

The budgets are ours (the paper's datasets came from much longer runs):
``1_000_000`` updates of batch 1024 sample each stored transition about
``1e6 * 1024 / 28.5e6 = 36`` times (19 tasks x 3 seeds x 500K steps), close
to the paper's mt30 run (``10e6 * 1024 / 345e6 = 30``, spec 4.18-4.19).
Playground tasks are MJX ports of dm_control: none of these returns is
comparable with the paper's.

Usage::

    python benchmarks/world_models/multitask_validation.py --out mt/  # all
    python benchmarks/world_models/multitask_validation.py --out mt/ \\
        --stage sources --tasks walker-stand walker-walk  # split over GPUs
    JAX_PLATFORMS=cpu python benchmarks/world_models/multitask_validation.py \\
        --smoke --out /tmp/mt-smoke  # 2 tasks, tiny: plumbing only

Every stage resumes: a finished source run or stage is skipped, an
interrupted run continues from its last saved state.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from typing import Any, Optional

import numpy as np
from paper_protocol import TDMPC2_KWARGS, TDMPC2_REFERENCE, TDMPC2_TINY
from paper_report import AJAX_COLOR, REFERENCE_COLOR, provenance_cells
from wm_acceptance import (
    MT_FRACTION,
    MT_SOURCE_FLOOR,
    TASK_SHARE,
    judge_multitask,
)
from wm_runs import (
    CURVE_FILE,
    RUN_FILE,
    SAVE_EVERY_S,
    STATE_FILE,
    Probe,
    RunSpec,
    jsonable,
    load_reference,
    read_records,
    run_chunked,
    run_single_task,
    spec_differences,
    write_atomic,
)

STAGES = ("sources", "dataset", "train", "report")
CAVEAT = (
    "This validates the multi-task mechanisms on data Ajax generated itself,"
    " on mujoco_playground's MJX ports of the DMC tasks (physics,"
    " observations and sometimes rewards differ from dm_control's): it is"
    " not a reproduction of the paper's mt30 / mt80 numbers."
)


@dataclasses.dataclass(frozen=True)
class MultiTaskProtocol:
    """The pipeline's fixed settings (module docstring).

    Attributes:
        tasks: TD-MPC2 task names, in mt30 order (those with a playground
            version).
        source_kwargs: the source agents' :class:`~ajax.TDMPC2` arguments.
        source_budget, source_chunk: agent steps per source run and per
            evaluation.
        source_seeds: the ``k`` source seeds of every task.
        eval_episodes: evaluation episodes per task (sources and offline).
        offline_kwargs: :class:`~ajax.TDMPC2MultiTask` arguments.
        offline_updates, offline_chunk: offline updates, and per evaluation.
        offline_seeds: the offline model's seeds.
        smoke: tiny plumbing settings (no verdict).
    """

    tasks: tuple[str, ...]
    source_kwargs: dict
    source_budget: int
    source_chunk: int
    source_seeds: tuple[int, ...]
    eval_episodes: int
    offline_kwargs: dict
    offline_updates: int
    offline_chunk: int
    offline_seeds: tuple[int, ...]
    smoke: bool = False

    def __post_init__(self) -> None:
        capacity = self.source_kwargs.get("buffer_size", 1_000_000)
        if capacity < self.source_budget:
            raise ValueError(
                f"the source runs' buffer_size {capacity} must hold their"
                f" {self.source_budget} steps: the dataset is their full history"
            )


def mt30_playground() -> dict[str, str]:
    """mt30's tasks with a playground version, in mt30 order -> env id."""
    return dict(load_reference(TDMPC2_REFERENCE)["mt30"]["playground"])


PROTOCOL = MultiTaskProtocol(
    tasks=tuple(mt30_playground()),
    source_kwargs=dict(TDMPC2_KWARGS),
    source_budget=500_000,  # 1M env steps
    source_chunk=50_000,
    source_seeds=(0, 1, 2),
    eval_episodes=10,
    offline_kwargs={
        "model_size": 19,
        "batch_size": 1024,
        "action_repeat": 2,
        "episode_length": 1000,
    },
    offline_updates=1_000_000,
    offline_chunk=100_000,
    offline_seeds=(0,),
)

SMOKE = MultiTaskProtocol(
    # Two cheap tasks of different observation and action dims, mt30 order.
    tasks=("reacher-easy", "pendulum-swingup"),
    source_kwargs=dict(TDMPC2_TINY),
    source_budget=60,  # 3 episodes of T = 20
    source_chunk=60,
    source_seeds=(0, 1),
    eval_episodes=2,
    offline_kwargs={
        "model_size": 1,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "task_dim": 8,
        "batch_size": 16,
        "num_samples": 32,
        "num_elites": 4,
        "num_pi_trajs": 4,
        "iterations": 2,
        "action_repeat": 2,
        "episode_length": 40,
    },
    offline_updates=20,
    offline_chunk=10,
    offline_seeds=(0,),
    smoke=True,
)


def source_spec(protocol: MultiTaskProtocol, task: str) -> RunSpec:
    env_id = mt30_playground()[task]
    return RunSpec(
        agent="TDMPC2",
        env_id=env_id,
        reference_task=task,
        description=f"multi-task source: TD-MPC2 on playground {env_id}",
        kwargs=dict(protocol.source_kwargs),
        budget=protocol.source_budget,
        chunk=protocol.source_chunk,
        seeds=protocol.source_seeds,
        num_eval_episodes=protocol.eval_episodes,
        smoke=protocol.smoke,
    )


def episodes_path(out: str, task: str) -> str:
    return os.path.join(out, "sources", task, "episodes.npz")


def run_sources(
    protocol: MultiTaskProtocol,
    out: str,
    tasks: Optional[list[str]] = None,
    save_every_s: float = SAVE_EVERY_S,
) -> None:
    """Stage 1: the source runs and their exported episodes."""
    from ajax.agents.TDMPC2.dataset import (
        concatenate_episodes,
        export_episodes,
        pool_tasks,
        save_dataset,
    )

    for task in tasks or protocol.tasks:
        if task not in protocol.tasks:
            raise SystemExit(f"{task!r} is not a task of the protocol {protocol.tasks}")
        path = episodes_path(out, task)
        if os.path.isfile(path):
            print(f"[sources] {task}: exported already", flush=True)
            continue
        run_dir = os.path.dirname(path)
        _, agent, state = run_single_task(
            source_spec(protocol, task),
            run_dir,
            save_every_s=save_every_s,
            keep_state=True,
            log=lambda message: print(f"[sources] {message}", flush=True),
        )
        if state is None:
            raise RuntimeError(
                f"{run_dir} finished without its state or episodes: rerun the"
                " task in a new directory"
            )
        # The full history: export_episodes raises if the ring evicted any.
        episodes = concatenate_episodes(
            [
                export_episodes(agent, state, seed_index=i, name=task)
                for i in range(len(protocol.source_seeds))
            ]
        )
        save_dataset(pool_tasks([episodes]), path)
        os.remove(os.path.join(run_dir, STATE_FILE))
        print(
            f"[sources] {task}: {episodes.num_episodes} episodes of"
            f" {episodes.rows} rows exported",
            flush=True,
        )


def file_digest(path: str) -> Optional[dict[str, Any]]:
    """``{"bytes", "sha256"}`` of a file (``None`` if there is none)."""
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return {"bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def dataset_paths(out: str) -> tuple[str, str]:
    """``dataset.npz`` and ``dataset.json`` (written last: the stage's
    "done" mark) of a pipeline directory."""
    return os.path.join(out, "dataset.npz"), os.path.join(out, "dataset.json")


def check_dataset(protocol: MultiTaskProtocol, out: str) -> dict[str, Any]:
    """``dataset.json``, after checking that ``dataset.npz`` and every
    source export are the files it was built from; raise otherwise."""
    path, meta_path = dataset_paths(out)
    with open(meta_path) as f:
        meta = json.load(f)
    changed = [
        task
        for task in protocol.tasks
        if meta["sources"].get(task) != file_digest(episodes_path(out, task))
    ]
    if changed or meta["dataset"] != file_digest(path):
        raise SystemExit(
            f"[dataset] {path} is stale: {changed or 'it'} changed since it"
            " was built. Delete dataset.npz, dataset.json and offline/ to"
            " rebuild them."
        )
    return meta


def build_dataset(protocol: MultiTaskProtocol, out: str) -> None:
    """Stage 2: pool, save, load back, check equal; record the files."""
    from ajax.agents.TDMPC2.dataset import load_dataset, pool_tasks, save_dataset

    path, meta_path = dataset_paths(out)
    if os.path.isfile(meta_path):
        check_dataset(protocol, out)
        print("[dataset] built already (its sources unchanged)", flush=True)
        return
    missing = [t for t in protocol.tasks if not os.path.isfile(episodes_path(out, t))]
    if missing:
        raise SystemExit(f"[dataset] the source runs of {missing} are not done")
    dataset = pool_tasks(
        [load_dataset(episodes_path(out, t)).task_episodes(0) for t in protocol.tasks]
    )
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    for field in ("obs", "action", "reward", "task"):
        if not np.array_equal(
            np.asarray(getattr(dataset, field)), getattr(loaded, field)
        ):
            raise AssertionError(f"the saved dataset's {field} does not round-trip")
    meta = ("obs_dims", "action_dims", "episode_lengths", "names")
    if any(getattr(dataset, m) != getattr(loaded, m) for m in meta):
        raise AssertionError("the saved dataset's metadata does not round-trip")
    summary = {
        "tasks": list(loaded.names),
        "episodes_per_task": loaded.episode_counts().tolist(),
        "rows": loaded.rows,
        "obs_dim": loaded.obs_dim,
        "action_dim": loaded.action_dim,
        "bytes": loaded.nbytes,
        "sources": {t: file_digest(episodes_path(out, t)) for t in protocol.tasks},
        "dataset": file_digest(path),
    }
    write_atomic(meta_path, json.dumps(summary, indent=1))
    print(f"[dataset] {summary}", flush=True)


class MultiTaskEvalProbe(Probe):
    """Every task's evaluation (:meth:`~ajax.TDMPC2MultiTask.evaluate`)."""

    def __init__(self, agent: Any, num_episodes: int) -> None:
        self.agent, self.num_episodes = agent, num_episodes

    def measure(self, state: Any, progress: int) -> dict[str, Any]:
        out = self.agent.evaluate(state, self.num_episodes)
        names = self.agent.tasks.names
        return {
            "returns": {
                name: out[f"Eval/{name}/episodic mean reward"].tolist()
                for name in names
            },
            "value": out["Eval/episodic mean reward"].tolist(),
            "normalized_score": out["Eval/normalized score"].tolist(),
        }


def train_offline(
    protocol: MultiTaskProtocol, out: str, save_every_s: float = SAVE_EVERY_S
) -> None:
    """Stage 3: offline multi-task training, chunked and resumable."""
    import ajax
    from ajax.agents.TDMPC2.dataset import load_dataset

    path, meta_path = dataset_paths(out)
    if not os.path.isfile(meta_path):
        raise SystemExit("[train] build the dataset first (--stage dataset)")
    dataset_record = check_dataset(protocol, out)
    dataset = load_dataset(path)
    playground = mt30_playground()
    agent = ajax.TDMPC2MultiTask(
        dataset,
        eval_envs=[playground[name] for name in dataset.names],
        **protocol.offline_kwargs,
    )
    del dataset
    seeds = list(protocol.offline_seeds)
    run_chunked(
        os.path.join(out, "offline"),
        spec={
            "protocol": dataclasses.asdict(protocol),
            "dataset": dataset_record,
            "resolved": jsonable(agent.config),
        },
        budget=protocol.offline_updates,
        chunk=protocol.offline_chunk,
        train=lambda state, n: agent.train(
            seed=seeds, n_timesteps=n, initial_state=state
        )[0],
        skeleton=lambda: agent.train(seed=seeds, n_timesteps=0)[0],
        progress_of=agent.resume_update_offset,
        probe=MultiTaskEvalProbe(agent, protocol.eval_episodes),
        save_every_s=save_every_s,
        log=lambda message: print(f"[train] {message}", flush=True),
    )


def pipeline_runs(protocol: MultiTaskProtocol, out: str) -> dict[str, dict]:
    """``{run: {"records", "off_protocol"}}`` for ``sources/<task>`` and
    ``offline``; ``off_protocol`` lists where a run's stored specification
    differs from ``protocol``'s (the offline run's also from the current
    ``dataset.json``), ``[]`` when it does not or there is no run."""
    meta_path = dataset_paths(out)[1]
    dataset = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            dataset = json.load(f)
    expected = {
        f"sources/{task}": {"run": dataclasses.asdict(source_spec(protocol, task))}
        for task in protocol.tasks
    }
    expected["offline"] = {
        "protocol": dataclasses.asdict(protocol),
        "dataset": dataset,
    }
    runs = {}
    for name, spec in expected.items():
        run_dir = os.path.join(out, name)
        reasons: list[str] = []
        if os.path.isfile(os.path.join(run_dir, RUN_FILE)):
            with open(os.path.join(run_dir, RUN_FILE)) as f:
                stored = json.load(f)["spec"]
            reasons = spec_differences(spec, {k: stored.get(k) for k in spec})
        runs[name] = {
            "records": read_records(os.path.join(run_dir, CURVE_FILE)),
            "off_protocol": reasons,
        }
    return runs


def collect(
    protocol: MultiTaskProtocol, runs: dict[str, dict]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per task, the source seeds' final returns and the offline model's
    final returns (finished runs on protocol only; empty otherwise), from
    :func:`pipeline_runs`."""

    def final(name: str, budget: int) -> Optional[dict]:
        run = runs[name]
        records = run["records"]
        if run["off_protocol"] or not records or records[-1]["progress"] < budget:
            return None
        return records[-1]

    source: dict[str, list[float]] = {}
    for task in protocol.tasks:
        record = final(f"sources/{task}", protocol.source_budget)
        source[task] = record["value"] if record else []
    record = final("offline", protocol.offline_updates)
    offline: dict[str, list[float]] = record["returns"] if record else {}
    return source, offline


def write_report(protocol: MultiTaskProtocol, out: str) -> str:
    """Stage 4: the markdown report (and a plot); returns the verdict."""
    runs = pipeline_runs(protocol, out)
    source, offline = collect(protocol, runs)
    results, verdict = judge_multitask(source, offline)
    judged = sum(r.status in ("pass", "fail") for r in results)
    playground = mt30_playground()
    lines = [
        f"# TD-MPC2 multi-task validation: {verdict}"
        + (" (smoke run: plumbing only)" if protocol.smoke else ""),
        "",
        CAVEAT,
        "",
        f"Criterion (`wm_acceptance.py`, fixed before any run): per task, the"
        f" offline model's mean return >= {MT_FRACTION:g} x the source agents'"
        f" final mean return; tasks whose sources stay below"
        f" {MT_SOURCE_FLOOR:g} are not judged; >= {TASK_SHARE:.0%} of the"
        f" tasks must be judged (else INCOMPLETE); PASS when >= {TASK_SHARE:.0%}"
        f" of the judged tasks pass. Judged: {judged} of {len(results)} tasks.",
        "",
        f"Sources: TD-MPC2 model_size {protocol.source_kwargs.get('model_size')},"
        f" {protocol.source_budget} agent steps, seeds"
        f" {list(protocol.source_seeds)}. Offline: TDMPC2MultiTask model_size"
        f" {protocol.offline_kwargs.get('model_size')},"
        f" {protocol.offline_updates} updates, seeds"
        f" {list(protocol.offline_seeds)}.",
        "",
        "| task | env | source final | offline | ratio | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.task} | {playground[r.task]} | {r.source:.0f} | {r.offline:.0f} |"
            f" {r.ratio:.2f} | {r.status} |"
        )
    records = runs["offline"]["records"]
    if records:
        lines += [
            "",
            "Offline normalised score (mean of return / 10) per evaluation: "
            + ", ".join(
                f"{r['progress']}: {np.mean(r['normalized_score']):.1f}"
                for r in records
            ),
        ]
    lines += [
        "",
        "Runs (off protocol: not used):",
        "",
        "| run | protocol | commits | tree changes | jax | devices |",
        "|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        reasons = run["off_protocol"]
        status = "**off protocol**: " + ", ".join(reasons) if reasons else "ok"
        if not run["records"] and not reasons:
            status = "no records"
        lines.append(f"| {name} | {status} |{provenance_cells(run['records'])}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "multitask_report.md"), "w") as f:
        f.write(text)
    if any(r.status != "missing" for r in results):
        _plot(results, os.path.join(out, "multitask.png"))
    print(text, flush=True)
    return verdict


def _plot(results: list, path: str) -> None:
    """Per task: source final and offline returns (bars)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [r.task for r in results]
    y = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(tasks) + 1.2))
    ax.barh(
        y - 0.2,
        [r.source for r in results],
        0.38,
        color=REFERENCE_COLOR,
        label="source agents (final)",
    )
    ax.barh(
        y + 0.2,
        [r.offline for r in results],
        0.38,
        color=AJAX_COLOR,
        label="offline multi-task",
    )
    ax.set_yticks(y, tasks, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("return (mean over seeds)")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        fontsize=8, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="pipeline directory")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument(
        "--tasks", nargs="*", default=None, help="source tasks to run (stage sources)"
    )
    parser.add_argument("--smoke", action="store_true", help="tiny runs (plumbing)")
    args = parser.parse_args(argv)
    protocol = SMOKE if args.smoke else PROTOCOL
    save_every_s = 0.0 if args.smoke else SAVE_EVERY_S
    stages = STAGES if args.stage == "all" else (args.stage,)
    if "sources" in stages:
        run_sources(protocol, args.out, args.tasks, save_every_s)
    if "dataset" in stages:
        build_dataset(protocol, args.out)
    if "train" in stages:
        train_offline(protocol, args.out, save_every_s)
    if "report" in stages:
        return 1 if write_report(protocol, args.out) == "FAIL" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
