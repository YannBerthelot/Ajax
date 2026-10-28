"""LoggingConfig and the logging worker, which passes every run's
evaluations to its backends (ajax.logging.backends)."""

import dataclasses
import json
import multiprocessing as mp
import os
import secrets
import string
import struct as pystruct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from flax.struct import dataclass as flax_dataclass
from tensorboardX.proto import event_pb2  # tensorboardX includes this!
from tqdm import tqdm

from ajax.logging.backends import (
    RunInfo,
    TensorBoardBackend,
    WandbBackend,
    _RunOnlyWandbBackend,
)


@flax_dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    config: dict
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    log_frequency: int = 1000
    group_name: Optional[str] = None
    folder: Optional[str] = None
    use_tensorboard: bool = False
    use_wandb: bool = True
    sweep: bool = False

    @property
    def backend(self) -> bool:
        """Whether a run writes anywhere (W&B or TensorBoard), so needs the
        logging worker; without one it still evaluates (``train`` returns
        the evaluations)."""
        return self.use_wandb or self.use_tensorboard


def new_run_id() -> str:
    """A fresh run id in W&B's format (``wandb.sdk.lib.runid.generate_id``:
    8 random base-36 digits), without importing wandb."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _flag_backends(logging_config: LoggingConfig) -> tuple:
    """The backends use_wandb and use_tensorboard name."""
    backends: list = []
    if logging_config.use_wandb:
        wandb_class = _RunOnlyWandbBackend if logging_config.sweep else WandbBackend
        backends.append(
            wandb_class(
                project=logging_config.project_name, group=logging_config.group_name
            )
        )
    if logging_config.use_tensorboard:
        backends.append(TensorBoardBackend(folder=logging_config.folder or "."))
    return tuple(backends)


# 'spawn' is required: we start the process AFTER JAX has initialised CUDA,
# so 'fork' would copy a live CUDA context into the child and cause crashes.
_mp_ctx = mp.get_context("spawn")

_log_queue: Optional[Any] = None  # _mp_ctx.JoinableQueue at runtime
_log_process: Optional[Any] = None  # _mp_ctx.Process at runtime
_pending_init_msgs: list = []  # messages buffered before worker starts
# The backends sent to the worker, at the index the worker holds each at.
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
        ("backend", backend)          — a backend, at the next index
        ("open",    indices, run)     — a run (RunInfo) and its backends
        ("log",     run_id, metrics, step) — one evaluation of a run
        None                          — poison pill / shutdown
    """
    backends: list = []
    # run_id -> the backends its steps go to
    routes: dict = {}

    while True:
        try:
            item = q.get(timeout=1.0)
        except Exception:
            continue

        if item is None:  # poison pill — close every backend and shut down
            q.task_done()
            break

        kind = item[0]
        if kind == "backend":
            backends.append(item[1])
        elif kind == "open":
            _, indices, run = item
            routes[run.run_id] = [backends[index] for index in indices]
            for backend in routes[run.run_id]:
                _call(backend, "open", run)
        elif kind == "log":
            _, run_id, metrics, step = item
            for backend in routes.get(run_id, []):
                _call(backend, "log", run_id, step, metrics)
        q.task_done()

    for backend in backends:
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
    """To the worker, or held until it starts."""
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
    _sent_backends.append(backend)
    _send(("backend", backend))
    return len(_sent_backends) - 1


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
    indices = []
    for backend in _flag_backends(logging_config):
        start = getattr(backend, "start", None)
        if start is not None:
            start(run)
        indices.append(_worker_index(backend))
    if indices:
        # The worker gets a copy, so the config's values that are not JSON
        # (an environment, a function), which may not pickle, go as their text.
        plain = json.loads(json.dumps(run.config, default=str))
        _send(("open", indices, dataclasses.replace(run, config=plain)))


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
