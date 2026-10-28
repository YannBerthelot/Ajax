"""Where a run's logs go: backends behind one small protocol.

Every evaluation a run logs reaches each of its backends in the logging
worker, a process of its own, so that training never waits on a disk or a
network. The worker is spawned (CUDA is never forked), so a backend is built
in the caller's process and copied into the worker: it must be picklable,
and it learns about runs only through the calls below. One backend object
serves every run of a train call (one run per seed).
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import Any, Optional, Protocol, override, runtime_checkable


@dataclasses.dataclass(frozen=True)
class RunInfo:
    """What a backend is told when a run opens.

    run_id: 8 base-36 characters, unique to the run (W&B's id format).
    name: the LoggingConfig's run_name and the id, "<run_name>_<run_id>".
    config: the LoggingConfig's config, with the agent's run configuration
        merged in. In the worker, its values that are not JSON (an
        environment object, a function) are their text.
    seed: the run's seed, when the caller gives it.
    """

    run_id: str
    name: str
    config: Mapping[str, Any]
    seed: Optional[int] = None


@runtime_checkable
class LoggingBackend(Protocol):
    """A sink for the evaluations runs log: any picklable object with these
    three methods, all called in the logging worker.

    A backend may also have start(run), called in the caller's process for
    each run before training starts, for what must happen at once (W&B
    creates its run there, so a missing account fails before training and
    the run shows in the W&B UI at once). What a backend gathers in the
    worker stays there: to get it back, write it to a file.

    A backend of your own, here one printing each step:

        class PrintBackend:
            def open(self, run):
                print("run", run.name, "seed", run.seed)

            def log(self, run_id, step, metrics):
                print(run_id, step, {k: v.tolist() for k, v in metrics.items()})

            def close(self):
                pass

        LoggingConfig(backends=(PrintBackend(), TensorBoardBackend("runs")))

    It reaches the worker by pickling, so its class must be importable
    there: defined in a module or a script, not in a notebook cell.
    """

    def open(self, run: RunInfo) -> None:
        """A run starts; called before its first log."""

    def log(self, run_id: str, step: int, metrics: Mapping[str, Any]) -> None:
        """One evaluation of a run: its metrics at environment step step,
        each a numpy array. Metrics that are NaN (not computed at this step)
        are left out."""

    def close(self) -> None:
        """Training ended: write out anything buffered and release files."""


def _wandb() -> Any:
    """wandb, imported where a run logs to W&B and not before: the import
    takes about 0.4 s, and a broken install (e.g. a partial protobuf
    upgrade) must not break importing Ajax, nor a run that does not use
    W&B."""
    import wandb  # type: ignore[import-untyped]

    return wandb


@dataclasses.dataclass(frozen=True)
class TensorBoardBackend:
    """TensorBoard event files, a folder per run: folder/tensorboard/<run_id>/,
    beside the run's config.json (its configuration, seed and id, so runs
    can be found and paired by seed). Metrics that are not one number are
    left out."""

    folder: str = "."
    # The open writer of each run, filled in the worker.
    _writers: dict = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def open(self, run: RunInfo) -> None:
        # Imported in the worker, the one process that writes.
        from tensorboardX import SummaryWriter

        log_dir = os.path.join(self.folder, "tensorboard", run.run_id)
        writer = SummaryWriter(log_dir=log_dir)
        config_text = json.dumps(run.config, indent=2, default=str)
        writer.add_text("config", config_text, global_step=0)
        self._writers[run.run_id] = writer
        run_config = dict(run.config)
        if run.seed is not None:
            run_config["seed"] = run.seed
        run_config["run_id"] = run.run_id
        with open(os.path.join(log_dir, "config.json"), "w") as handle:
            json.dump(run_config, handle, indent=2, default=str, sort_keys=True)

    def log(self, run_id: str, step: int, metrics: Mapping[str, Any]) -> None:
        writer = self._writers[run_id]
        for key, value in metrics.items():
            # A metric that is not one number (an array) has no scalar.
            try:
                scalar = value.item() if hasattr(value, "item") else float(value)
                writer.add_scalar(key, scalar, step)
            except Exception:
                pass

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


@dataclasses.dataclass(frozen=True)
class WandbBackend:
    """Weights & Biases: the wandb extra (pip install ajax-rl[wandb]) and an
    account. wandb keeps one open run per process, so a run's steps are sent
    in batches of batch_size, each batch in one init, log, finish cycle; the
    run is created when training starts so it shows in the W&B UI at once."""

    project: Optional[str] = None
    group: Optional[str] = None
    # Larger batches cost fewer inits, but the steps reach the UI later.
    batch_size: int = 10
    # Per run, in the worker: its name and the steps not yet sent.
    _runs: dict = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def start(self, run: RunInfo) -> None:
        wandb = _wandb()
        wandb.init(
            project=self.project,
            name=run.name,
            id=run.run_id,
            resume="never",
            config=run.config,
            group=self.group,
            reinit="finish_previous",
        )
        wandb.log({"timestep": 0})
        wandb.finish()  # the worker resumes it, resume="must", for each batch
        print(f"Init wandb {run.name}, id={run.run_id}")

    def open(self, run: RunInfo) -> None:
        self._runs[run.run_id] = (run.name, [])

    def log(self, run_id: str, step: int, metrics: Mapping[str, Any]) -> None:
        pending = self._runs[run_id][1]
        pending.append((metrics, step))
        if len(pending) >= self.batch_size:
            self._send(run_id)

    def close(self) -> None:
        for run_id in self._runs:
            self._send(run_id)

    def _send(self, run_id: str) -> None:
        name, pending = self._runs[run_id]
        if not pending:
            return
        # A failed batch is reported and dropped: the other backends and the
        # run's later batches go on.
        try:
            run = _wandb().init(
                project=self.project, name=name, id=run_id, resume="must"
            )
            for metrics, step in pending:
                run.log(metrics, step=step)
            run.finish()
        except Exception as exc:
            print(f"[logging worker] wandb flush failed for {run_id}: {exc}")
        pending.clear()


@dataclasses.dataclass(frozen=True)
class _RunOnlyWandbBackend(WandbBackend):
    """use_wandb=True with sweep=True, as documented for the flags: the run is
    created, and the worker sends it no step."""

    @override
    def log(self, run_id: str, step: int, metrics: Mapping[str, Any]) -> None:
        pass
