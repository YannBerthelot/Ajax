"""Helpers for Weights & Biases and TensorBoard logging"""

import functools
import json
import os
import queue
import threading
from queue import Queue
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import wandb
import wandb.errors
from flax import struct
from flax.serialization import to_state_dict
from torch.utils.tensorboard import SummaryWriter


@struct.dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict
    log_frequency: int = 1000
    mode: str = "online"
    group_name: Optional[str] = None
    chunk_size: int = 1000
    horizon: int = 10_000
    folder: Optional[str] = None
    use_tensorboard: bool = False
    use_wandb: bool = True


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
            name=f"{logging_config.run_name}  {index}",
            id=run_id,
            resume="never",
            reinit=True,
            config=logging_config.config,
        )

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
        stop_logging.set()
        logging_thread.join()
        logging_thread = None

    for writer in tensorboard_writers.values():
        writer.close()
    tensorboard_writers.clear()


def _logging_worker():
    """Worker thread that processes logging queue"""
    while not stop_logging.is_set():
        try:
            item = logging_queue.get(timeout=0.1)
            if item is None:
                continue

            run_id, metrics, step, project, name = item
            try:
                run = wandb.init(
                    project=project,
                    name=f"{name} {run_id}",
                    id=run_id,
                    resume="must",
                    reinit=True,
                )
                run.log(metrics, step=step)
            except wandb.errors.UsageError:
                pass

            writer = tensorboard_writers.get(run_id)
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, step)

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
        k: jax.device_get(v) for k, v in log_metrics.items() if not jnp.isnan(v)
    }

    step = log_metrics["timestep"]
    logging_queue.put(
        (run_id, metrics_np, step, logging_config.project_name, logging_config.run_name)
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
            os.environ["WANDB_SILENT"] = initial_wandb_silent

    return wrapper
