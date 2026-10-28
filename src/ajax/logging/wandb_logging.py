"""LoggingConfig and the logging worker, which passes every run's
evaluations to its backends (ajax.logging.backends)."""

import dataclasses
import json
import multiprocessing as mp
import os
import pickle
import queue
import secrets
import string
import struct as pystruct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from flax import struct
from tensorboardX.proto import event_pb2  # tensorboardX includes this!
from tqdm import tqdm

from ajax.logging.backends import (
    LoggingBackend,
    RunInfo,
    TensorBoardBackend,
    WandbBackend,
)


@struct.dataclass
class LoggingConfig:
    """Evaluation and logging for a train call: every log_frequency
    environment steps the agent is evaluated, train returns the evaluations,
    and each run (one per seed) logs them to the backends.

    backends: the LoggingBackend objects every run logs to, such as
        TensorBoardBackend(folder) and WandbBackend(project, group), or one
        of your own (see LoggingBackend). Empty, use_wandb (on unless set
        False) and use_tensorboard (off unless set True) name them, as
        before; given, setting either flag as well is refused.
    config: stored with each run, with the agent's run configuration merged in.
    project_name, group_name: the W&B project and group of use_wandb.
    run_name: the prefix of each run's name, "<run_name>_<run_id>".
    log_frequency: environment steps between evaluations.
    folder: where use_tensorboard writes, folder/tensorboard/<run_id>/ (the
        working directory when None).
    sweep: for hyperparameter sweeps, evaluate only in the last fifth of
        training; with use_wandb, the W&B run gets no steps
        (WandbBackend(steps=False)).
    """

    config: dict = struct.field(default_factory=dict)
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    log_frequency: int = 1000
    group_name: Optional[str] = None
    folder: Optional[str] = None
    # None: not set, which tells the flags apart from backends.
    use_tensorboard: Optional[bool] = None
    use_wandb: Optional[bool] = None
    sweep: bool = False
    backends: tuple = struct.field(pytree_node=False, default=())

    def __post_init__(self) -> None:
        flag_set = self.use_wandb is not None or self.use_tensorboard is not None
        if self.backends and flag_set:
            raise ValueError(
                "LoggingConfig takes its backends either as backends=(...) or"
                " through use_wandb and use_tensorboard, not both: name every"
                " one in backends, such as backends=(TensorBoardBackend(folder),"
                " WandbBackend(project, group))."
            )
        for backend in self.backends:
            if not isinstance(backend, LoggingBackend):
                raise TypeError(
                    f"{backend!r} is not a logging backend: it needs open, log"
                    " and close methods (see LoggingBackend)."
                )

    @property
    def run_backends(self) -> tuple:
        """The backends every run logs to: backends, else those the flags
        name."""
        if self.backends:
            return tuple(self.backends)
        backends: list = []
        if self.use_wandb is None or self.use_wandb:
            backends.append(
                WandbBackend(
                    project=self.project_name,
                    group=self.group_name,
                    steps=not self.sweep,
                )
            )
        if self.use_tensorboard:
            backends.append(TensorBoardBackend(folder=self.folder or "."))
        return tuple(backends)

    @property
    def backend(self) -> bool:
        """Whether a run writes anywhere, so needs the logging worker;
        without a backend it still evaluates (train returns the
        evaluations)."""
        return bool(self.run_backends)


def new_run_id() -> str:
    """A fresh run id in W&B's format (``wandb.sdk.lib.runid.generate_id``:
    8 random base-36 digits), without importing wandb."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


# 'spawn' is required: we start the process AFTER JAX has initialised CUDA,
# so 'fork' would copy a live CUDA context into the child and cause crashes.
_mp_ctx = mp.get_context("spawn")

_log_queue: Optional[Any] = None  # _mp_ctx.JoinableQueue at runtime
_log_process: Optional[Any] = None  # _mp_ctx.Process at runtime
_pending_init_msgs: list = []  # messages buffered before worker starts
# The backends sent to the worker, each at the index the worker holds it at.
_sent_backends: list = []


def _call(backend: Any, method: str, *args: Any) -> None:
    """One backend call in the worker. A failure is reported, not raised:
    logging must never stop training, nor the other backends."""
    try:
        getattr(backend, method)(*args)
    except Exception as exc:
        print(f"[logging worker] {type(backend).__name__}.{method} failed: {exc}")


def _logging_worker(q: Any) -> None:
    """
    Single worker process: passes every run's messages to its backends.

    Message protocol:
        ("backend", index, backend)   — a backend, at its index
        ("open",    indices, run)     — a run (RunInfo) and its backends
        ("log",     run_id, metrics, step) — one evaluation of a run
        None                          — poison pill / shutdown
    """
    backends: dict = {}
    # run_id -> the backends its steps go to
    routes: dict = {}

    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as exc:
            # A message this process cannot unpickle, such as a backend whose
            # class it cannot import (one defined in a notebook): reported
            # and dropped, and counted done so that stopping does not hang.
            print(f"[logging worker] dropped a message it cannot read: {exc}")
            q.task_done()
            continue

        if item is None:  # poison pill — close every backend and shut down
            q.task_done()
            break

        kind = item[0]
        if kind == "backend":
            _, index, backend = item
            backends[index] = backend
        elif kind == "open":
            _, indices, run = item
            routes[run.run_id] = [backends[i] for i in indices if i in backends]
            for backend in routes[run.run_id]:
                _call(backend, "open", run)
        elif kind == "log":
            _, run_id, metrics, step = item
            for backend in routes.get(run_id, []):
                _call(backend, "log", run_id, step, metrics)
        q.task_done()

    for backend in backends.values():
        _call(backend, "close")


def start_async_logging() -> None:
    """Spawn the logging process (once; no-op if already running)."""
    global _log_queue, _log_process, _pending_init_msgs
    if _log_process is None or not _log_process.is_alive():
        _log_queue = _mp_ctx.JoinableQueue()
        # Force CPU in the child: importing this module imports ajax, and so
        # jax, which would otherwise pre-allocate GPU memory in the worker.
        _old = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"
        _log_process = _mp_ctx.Process(
            target=_logging_worker, args=(_log_queue,), daemon=True
        )
        _log_process.start()
        if _old is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = _old
        for msg in _pending_init_msgs:
            _log_queue.put(msg)
        _pending_init_msgs.clear()


def stop_async_logging() -> None:
    """Drain the queue, then stop the logging process."""
    global _log_queue, _log_process, _pending_init_msgs
    if _log_process is not None and _log_process.is_alive() and _log_queue is not None:
        _log_queue.put(None)  # poison pill
        try:
            _log_queue.join()  # wait until all task_done() calls balance put() calls
        except Exception:
            pass
        _log_process.join(timeout=300)
        if _log_process.is_alive():
            _log_process.terminate()
    _log_queue = None
    _log_process = None
    _pending_init_msgs.clear()
    _sent_backends.clear()


def _send(message: tuple) -> None:
    """A backend or a run to the worker, or held until it starts.

    Checked to pickle here, before training: the queue pickles in a thread
    of its own, where a failure is only printed, and stopping the worker
    then waits forever for the message.
    """
    try:
        pickle.dumps(message)
    except Exception as exc:
        raise TypeError(
            f"{message[-1]!r} cannot be copied into the logging worker, a"
            f" process of its own: it must be picklable ({exc})."
        ) from exc
    if _log_queue is not None:
        _log_queue.put(message)
    else:
        _pending_init_msgs.append(message)


def _worker_index(backend: Any) -> int:
    """backend's index in the worker, sent there on first use: one object
    serves every run of a train call, and equal backends are one sink."""
    for index, sent in enumerate(_sent_backends):
        if sent == backend:
            return index
    index = len(_sent_backends)
    _send(("backend", index, backend))
    _sent_backends.append(backend)
    return index


def init_logging(
    run_id: str, logging_config: LoggingConfig, run_seed: Optional[int] = None
) -> None:
    """
    Register a run with the logging config's backends: each backend's start,
    if it has one, runs here, before training (W&B creates the run, so it
    appears in the W&B UI immediately); the worker then opens the run.

    When ``run_seed`` is given (one call per seed), it is recorded alongside the
    config so every run is reproducible and can be paired across conditions by
    seed (see the per-run ``config.json`` written for TensorBoard runs).
    """
    run = RunInfo(
        run_id=run_id,
        name=f"{logging_config.run_name}_{run_id}",
        config=logging_config.config,
        seed=run_seed,
    )
    # Each backend once, by its index in the worker (equal ones are one).
    backends: dict = {}
    for backend in logging_config.run_backends:
        backends.setdefault(_worker_index(backend), backend)
    for backend in backends.values():
        start = getattr(backend, "start", None)
        if start is not None:
            start(run)
    if backends:
        # The worker gets a copy, so the config's values that are not JSON
        # (an environment, a function), which may not pickle, go as their text.
        plain = json.loads(json.dumps(run.config, default=str))
        _send(("open", list(backends), dataclasses.replace(run, config=plain)))


def vmap_log(
    log_metrics: Dict[str, Any],
    index: int,
    run_ids: Sequence[str],
) -> None:
    """Forward per-seed metrics to the logging process."""
    if _log_queue is None:
        return None

    run_id = run_ids[index]
    # The callback gets CPU arrays: numpy reads them without dispatching
    # two JAX ops per metric.
    metrics_np = {k: np.asarray(v) for k, v in log_metrics.items()}
    metrics_np = {k: v for k, v in metrics_np.items() if not np.isnan(v).any()}
    step = int(metrics_np["timestep"])

    _log_queue.put(("log", run_id, metrics_np, step))
    return None


def iter_tfrecord(path):
    """
    Iterate raw record data from a TFRecord-style TensorBoard event file.
    (Each record = ``[length][crc_len][data][crc_data]``)
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = pystruct.unpack("<Q", header)[0]
            f.read(4)  # skip CRC of length
            data = f.read(length)
            f.read(4)  # skip CRC of data
            if not data:
                break
            yield data


def load_scalars_from_tfevents(log_dir):
    """
    Reads all scalar summaries from TensorBoard event files
    using only tensorboardX (no TensorFlow, no tensorboard).
    """
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    all_events = defaultdict(list)

    for event_file in tqdm(event_files, desc="Reading events"):
        for record in iter_tfrecord(event_file):
            e = event_pb2.Event()
            e.ParseFromString(record)
            if not e.summary:
                continue
            for v in e.summary.value:
                if v.HasField("simple_value"):
                    all_events[v.tag].append((e.step, v.simple_value))

    for tag in all_events:
        all_events[tag].sort(key=lambda x: x[0])

    return all_events
