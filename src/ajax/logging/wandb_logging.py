"""Helpers for Weights & Biases and TensorBoard logging"""

import functools
import json
import multiprocessing as mp
import os
import struct as pystruct
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import wandb
from flax.serialization import to_state_dict
from flax.struct import dataclass as flax_dataclass
from tensorboardX import SummaryWriter
from tensorboardX.proto import event_pb2  # tensorboardX includes this!
from tqdm import tqdm


@flax_dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    config: dict
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    log_frequency: int = 1000
    mode: str = "online"
    group_name: Optional[str] = None
    horizon: int = 10_000
    folder: Optional[str] = None
    use_tensorboard: bool = False
    use_wandb: bool = True
    sweep: bool = False


# ---------------------------------------------------------------------------
# Single-process sequential logging worker
# ---------------------------------------------------------------------------
# 'spawn' is required: we start the process AFTER JAX has initialised CUDA,
# so 'fork' would copy a live CUDA context into the child and cause crashes.
_mp_ctx = mp.get_context("spawn")

_log_queue: Optional[Any] = None   # _mp_ctx.JoinableQueue at runtime
_log_process: Optional[Any] = None  # _mp_ctx.Process at runtime
_pending_init_msgs: list = []       # messages buffered before worker starts

# Number of steps to buffer per run before calling wandb.init/finish.
# Higher = fewer inits (cheaper) but more latency before data appears in UI.
WANDB_LOG_BATCH_SIZE = 10


def _logging_worker(q: Any) -> None:
    """
    Single worker process: handles ALL run_ids sequentially.

    wandb only supports one active run per process, so we cannot keep
    multiple run objects alive concurrently.  Instead we buffer incoming
    log entries per run_id and flush them in a single init→log*N→finish
    cycle once WANDB_LOG_BATCH_SIZE steps have accumulated (or at shutdown).
    This amortises the wandb.init() cost over N steps.

    TensorBoard writers are kept persistently open (no init cost per step).

    Message protocol:
        ("init_wandb", run_id, project, name)      — register run metadata
        ("init_tb",    run_id, log_dir, config_str) — open TB writer
        ("log",        run_id, metrics, step)       — buffer one step
        None                                        — poison pill / shutdown
    """
    import wandb as _wandb  # re-import in fresh spawned process
    from tensorboardX import SummaryWriter as _SW
    from collections import defaultdict

    # run_id -> list of (metrics_dict, step)
    buffers: dict = defaultdict(list)
    # run_id -> (project, name)
    run_info: dict = {}
    # run_id -> bool (whether to log to wandb)
    use_wandb_map: dict = {}
    # run_id -> SummaryWriter
    tb_writers: dict = {}

    def flush_run(run_id: str) -> None:
        """Flush buffered steps for one run to wandb in a single open/close."""
        entries = buffers.get(run_id)
        if not entries:
            return
        if use_wandb_map.get(run_id):
            project, name = run_info[run_id]
            try:
                run = _wandb.init(
                    project=project,
                    name=name,
                    id=run_id,
                    resume="must",
                )
                for metrics, step in entries:
                    run.log(metrics, step=step)
                run.finish()
            except Exception as exc:
                print(f"[logging worker] wandb flush failed for {run_id}: {exc}")
        buffers[run_id].clear()

    while True:
        try:
            item = q.get(timeout=1.0)
        except Exception:
            continue

        if item is None:  # poison pill — flush everything and shut down
            q.task_done()
            break

        kind = item[0]

        if kind == "init_wandb":
            _, run_id, project, name, use_wb = item
            run_info[run_id] = (project, name)
            use_wandb_map[run_id] = use_wb
            q.task_done()

        elif kind == "init_tb":
            _, run_id, log_dir, config_str = item
            writer = _SW(log_dir=log_dir)
            writer.add_text("config", config_str, global_step=0)
            tb_writers[run_id] = writer
            q.task_done()

        elif kind == "log":
            _, run_id, metrics, step = item
            # Buffer the step for wandb (init is expensive; batch N steps)
            buffers[run_id].append((metrics, step))
            # Write to TensorBoard immediately (no open/close cost)
            tb_writer = tb_writers.get(run_id)
            if tb_writer is not None:
                for key, value in metrics.items():
                    try:
                        scalar = value.item() if hasattr(value, "item") else float(value)
                        tb_writer.add_scalar(key, scalar, step)
                    except Exception:
                        pass
            # Flush wandb once the batch is full
            if len(buffers[run_id]) >= WANDB_LOG_BATCH_SIZE:
                flush_run(run_id)
            q.task_done()

    # Shutdown: flush all remaining buffered data, then close TB writers
    for run_id in list(buffers.keys()):
        flush_run(run_id)
    for writer in tb_writers.values():
        try:
            writer.close()
        except Exception:
            pass


def start_async_logging() -> None:
    """Spawn the logging process (once; no-op if already running)."""
    global _log_queue, _log_process, _pending_init_msgs
    if _log_process is None or not _log_process.is_alive():
        _log_queue = _mp_ctx.JoinableQueue()
        _log_process = _mp_ctx.Process(
            target=_logging_worker, args=(_log_queue,), daemon=True
        )
        _log_process.start()
        # Flush any messages that arrived before the worker started
        for msg in _pending_init_msgs:
            _log_queue.put(msg)
        _pending_init_msgs.clear()


def stop_async_logging() -> None:
    """Drain the queue, then stop the logging process."""
    global _log_queue, _log_process, _pending_init_msgs
    if _log_process is not None and _log_process.is_alive() and _log_queue is not None:
        _log_queue.put(None)   # poison pill
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


def init_logging(run_id: str, logging_config: LoggingConfig) -> None:
    """
    Register a run: create it in the main process (so it appears in the W&B UI
    immediately), then send metadata to the worker so it can flush batches later.
    """
    if logging_config.use_wandb:
        wandb.init(
            project=logging_config.project_name,
            name=f"{logging_config.run_name}_{run_id}",
            id=run_id,
            resume="never",
            config=logging_config.config,
            group=logging_config.group_name,
            reinit="finish_previous",
        )
        wandb.log({"timestep": 0})
        wandb.finish()  # worker resumes with resume="must" on each flush
        jax.debug.print(
            "Init wandb {run}", run=f"{logging_config.run_name}_{run_id}, id={run_id}"
        )

    use_wb = logging_config.use_wandb and not logging_config.sweep

    init_msgs = []
    if logging_config.use_wandb:
        init_msgs.append((
            "init_wandb",
            run_id,
            logging_config.project_name,
            f"{logging_config.run_name}_{run_id}",
            use_wb,
        ))
    if logging_config.use_tensorboard:
        log_dir = os.path.join(logging_config.folder or ".", "tensorboard", run_id)
        config_str = json.dumps(logging_config.config, indent=2, default=str)
        init_msgs.append(("init_tb", run_id, log_dir, config_str))

    for msg in init_msgs:
        if _log_queue is not None:
            _log_queue.put(msg)
        else:
            _pending_init_msgs.append(msg)


def log_variables(variables_to_log: dict, commit: bool = True) -> None:
    """Log variables into the currently-active wandb run (main process)."""
    wandb.log(variables_to_log, commit=commit)


def finish_logging() -> None:
    """Terminate the active wandb run in the main process."""
    wandb.finish()


def flatten_dict(d: Dict) -> Dict:
    """Flatten nested dictionary keys with slashes"""
    result = {}
    for key, val in d.items():
        if isinstance(val, Dict):
            for subkey, subval in val.items():
                result[f"{key}/{subkey}"] = subval
        else:
            result[key] = val
    return result


def prepare_metrics(aux: Any) -> Dict[str, Any]:
    """Flatten and filter NaN metrics"""
    flat = flatten_dict(to_state_dict(aux))
    return {k: v for k, v in flat.items() if not jnp.isnan(v)}


def vmap_log(
    log_metrics: Dict[str, Any],
    index: int,
    run_ids: Tuple[int],
    logging_config: LoggingConfig,
) -> None:
    """Forward per-seed metrics to the logging process."""
    if _log_queue is None:
        return None

    run_id = run_ids[index]
    metrics_np = {
        k: jax.device_get(v)
        for k, v in log_metrics.items()
        if not jnp.any(jnp.isnan(v))
    }
    step = int(metrics_np["timestep"])

    _log_queue.put(("log", run_id, metrics_np, step))
    return None


def safe_get_env_var(var_name: str, default: str = "") -> str:
    """Safely retrieve an environment variable"""
    return os.environ.get(var_name, default)


def with_wandb_silent(func: Callable) -> Callable:
    """Temporarily set WANDB_SILENT during function execution"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        initial_wandb_silent = safe_get_env_var("WANDB_SILENT")
        try:
            os.environ["WANDB_SILENT"] = "true"
            return func(*args, **kwargs)
        finally:
            os.environ["WANDB_SILENT"] = (
                initial_wandb_silent if initial_wandb_silent != "" else "false"
            )

    return wrapper


def iter_tfrecord(path):
    """
    Iterate raw record data from a TFRecord-style TensorBoard event file.
    (Each record = [length][crc_len][data][crc_data])
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

    # Sort by step
    for tag in all_events:
        all_events[tag].sort(key=lambda x: x[0])

    return all_events


def merge_and_upload_tensorboard_to_wandb(log_dir: str):
    """
    Merge all TensorBoard event files in `log_dir` and upload to WandB
    with guaranteed step ordering and no duplicates.
    """
    all_events = load_scalars_from_tfevents(log_dir)
    # Sort events by step per tag and remove duplicates
    for tag, values in all_events.items():
        values.sort(key=lambda x: x[0])
        deduped = []
        last_step = None
        for step, val in values:
            if step != last_step:
                deduped.append((step, val))
            else:
                deduped[-1] = (step, val)  # overwrite duplicate step
            last_step = step
        all_events[tag] = deduped

    all_events_flat = []
    for tag, values in all_events.items():
        for step, val in values:
            all_events_flat.append((step, tag, val))

    all_events_flat.sort(key=lambda x: x[0])

    for step, tag, val in tqdm(all_events_flat, desc="Uploading to WandB"):
        wandb.log({tag: val}, step=step)

    wandb.finish()


def upload_tensorboard_to_wandb(
    run_ids: list[str],
    logging_config: LoggingConfig,
    base_folder: Optional[str] = None,
):
    """
    Upload existing TensorBoard log directories to W&B for the given run_ids.

    Args:
        run_ids: List of run IDs corresponding to TensorBoard runs.
        logging_config: LoggingConfig object.
        base_folder: Base folder where TensorBoard logs are stored. Defaults to logging_config.folder.
    """
    base_folder = base_folder or logging_config.folder or "."

    for run_id in tqdm(run_ids, desc="Run ids"):
        log_dir = os.path.join(base_folder, "tensorboard", run_id)
        if not os.path.exists(log_dir):
            print(f"TensorBoard log directory not found for run {run_id}: {log_dir}")
            continue

        try:
            print(f"Uploading TensorBoard logs for run {run_id} to W&B from {log_dir}")
            wandb.init(
                project=logging_config.project_name,
                name=f"{logging_config.run_name}_{run_id}",
                config=logging_config.config,
                resume="must",
                id=run_id,
            )
            merge_and_upload_tensorboard_to_wandb(log_dir)
        except Exception as e:
            print(f"Failed to upload TensorBoard logs for run {run_id} to W&B: {e}")
