"""Helpers for Weights & Biases and TensorBoard logging"""

import functools
import json
import os
import queue
import threading
from collections import defaultdict
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import tensorflow as tf
import wandb
from flax import struct
from flax.serialization import to_state_dict
from tensorboardX import SummaryWriter
from tqdm import tqdm


@struct.dataclass
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


# Global state for async logging
logging_queue = Queue()  # type: ignore[var-annotated]
logging_thread = None
stop_logging = threading.Event()

# Global map of tensorboard writers
tensorboard_writers: Dict[str, SummaryWriter] = {}


def init_logging(
    run_id: str,
    index: int,
    logging_config: LoggingConfig,
):
    """Init the wandb run and optionally TensorBoard"""
    if logging_config.use_wandb:
        wandb.init(
            project=logging_config.project_name,
            name=f"{logging_config.run_name}_{run_id}",
            id=run_id,
            resume="never",
            config=logging_config.config,
        )

        jax.debug.print(
            "Init wandb {run}", run=f"{logging_config.run_name}_{run_id}, id={run_id}"
        )
        wandb.log({"timestep": 0})
        wandb.finish()

    if logging_config.use_tensorboard:
        log_dir = os.path.join(logging_config.folder or ".", "tensorboard", run_id)

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            "config",
            json.dumps(logging_config.config, indent=2, default=str),
            global_step=0,
        )

        tensorboard_writers[run_id] = writer


def log_variables(variables_to_log: dict, commit: bool = True):
    """Log variables into wandb"""
    wandb.log(variables_to_log, commit=commit)


def finish_logging():
    """Terminate the wandb run"""
    wandb.finish()


def start_async_logging():
    """Start the async logging thread"""
    global logging_thread
    if logging_thread is None or not logging_thread.is_alive():
        stop_logging.clear()
        logging_thread = threading.Thread(target=_logging_worker)
        logging_thread.daemon = True
        logging_thread.start()


def stop_async_logging():
    """Stop the async logging thread and close TensorBoard writers"""
    global logging_thread
    if logging_thread is not None:
        # Wait until all queued logging tasks are marked done
        logging_queue.join()
        stop_logging.set()
        logging_thread.join()
        logging_thread = None

    for writer in tensorboard_writers.values():
        writer.close()
    # tensorboard_writers.clear()


def _logging_worker():
    """Worker thread that processes logging queue"""
    while not stop_logging.is_set():
        try:
            item = logging_queue.get(timeout=0.1)
            if item is None:
                continue

            run_id, metrics, step, project, name, sweep = item

            if project is not None and not sweep:
                while True:
                    try:
                        run = wandb.init(
                            project=project,
                            name=f"{name} {run_id}",
                            id=run_id,
                            resume="must",
                            reinit=True,
                        )
                        run.log(metrics, step=step)
                        break  # success → exit retry loop
                    except (wandb.errors.UsageError, OSError) as e:
                        print(f"W&B log failed, retrying: {e}")

            writer = tensorboard_writers.get(run_id)
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(key, value.item(), step)

            logging_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in logging worker: {e}")
            continue


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
):
    """Log metrics from a batched setup using vmap-style parallelism"""
    run_id = run_ids[index]

    metrics_np = {
        k: jax.device_get(v)
        for k, v in log_metrics.items()
        if not jnp.any(jnp.isnan(v))
    }

    step = log_metrics["timestep"]

    logging_queue.put(
        (
            run_id,
            metrics_np,
            step,
            logging_config.project_name,
            logging_config.run_name,
            logging_config.sweep,
        )
    )

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


def merge_and_upload_tensorboard_to_wandb(log_dir: str):
    """
    Merge all TensorBoard event files in `log_dir` and upload to WandB
    with guaranteed step ordering and no duplicates.
    """
    event_files = list(Path(log_dir).glob("**/events.out.tfevents*"))
    all_events = defaultdict(list)  # {tag_name: [(step, value), ...]}

    # Step 1: Read all events and collect by tag
    for event_file in tqdm(event_files, desc="Reading TensorBoard events"):
        for e in tf.compat.v1.train.summary_iterator(str(event_file)):
            if e.summary is None:
                continue
            for v in e.summary.value:
                # Only handle scalars
                if v.HasField("simple_value"):
                    all_events[v.tag].append((e.step, v.simple_value))

    # Step 2: Sort events by step per tag and remove duplicates
    for tag, values in all_events.items():
        # Sort by step
        values.sort(key=lambda x: x[0])
        # Deduplicate (keep last value for duplicate steps)
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

    # Sort by step
    all_events_flat.sort(key=lambda x: x[0])

    # Step 3: Upload to WandB
    for step, tag, val in tqdm(all_events_flat, desc="Uploading to WandB"):
        wandb.log({tag: val}, step=step)

    wandb.finish()


def upload_tensorboard_to_wandb(
    run_ids: list[str],
    logging_config: LoggingConfig,
    base_folder: Optional[str] = None,
    use_wandb: bool = False,
):
    """
    Upload existing TensorBoard log directories to W&B for the given run_ids,
    then optionally clean up.

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
            # # Find all event files
            # event_files = glob.glob(
            #     os.path.join(log_dir, "**", "events*"), recursive=True
            # )
            # # wandb.log({"empty": 1})

            # for event_file in tqdm(event_files):
            #     for e in tf.compat.v1.train.summary_iterator(event_file):
            #         if e.summary is not None:
            #             wandb.tensorboard._log(e.summary, step=e.step)
            # time.sleep(60)
            # wandb.finish()
        except Exception as e:
            print(f"Failed to upload TensorBoard logs for run {run_id} to W&B: {e}")
