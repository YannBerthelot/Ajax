import json
import re
import subprocess
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.logging import wandb_logging
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    load_scalars_from_tfevents,
    new_run_id,
    start_async_logging,
    stop_async_logging,
    vmap_log,
)

# These tests run the deprecated flags on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:LoggingConfig's use_wandb:FutureWarning"
)


def test_importing_ajax_does_not_import_wandb():
    """wandb (about 0.4 s to import) loads only when a run logs to W&B."""
    code = "import sys, ajax; from ajax import SAC, TDMPC2; assert 'wandb' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_run_ids_have_wandb_format():
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(re.fullmatch("[a-z0-9]{8}", run_id) for run_id in ids)


def test_vmap_log_queues_its_seeds_metrics_without_nan_as_numpy(monkeypatch):
    """One event per call: the seed's run id, the timestep, and each metric
    holding no NaN as a numpy array of its own dtype and shape."""
    sent: list = []
    monkeypatch.setattr(wandb_logging, "_log_queue", SimpleNamespace(put=sent.append))
    metrics = {
        "timestep": jnp.asarray(30, jnp.int32),
        "done": jnp.asarray(True),
        "loss": jnp.asarray([0.5, 1.5]),
        "partial": jnp.asarray([1.0, jnp.nan]),
    }
    vmap_log(metrics, 1, run_ids=["a", "b"])
    ((kind, run_id, got, step),) = sent
    assert (kind, run_id, step) == ("log", "b", 30)
    assert sorted(got) == ["done", "loss", "timestep"]
    for name, value in got.items():
        assert isinstance(value, np.ndarray) and value.dtype == metrics[name].dtype
        np.testing.assert_array_equal(value, metrics[name])


def test_async_tensorboard_logging_round_trip(tmp_path):
    """What the training loop sends reaches TensorBoard through the worker.

    Exercises the real chain end to end: ``init_logging`` before the worker
    exists (its messages are buffered), the spawned worker process, NaN
    filtering in ``vmap_log``, the shutdown drain in ``stop_async_logging``,
    and reading the event file back with ``load_scalars_from_tfevents``.
    """
    run_id = "run0"
    config = LoggingConfig(
        config={"lr": 0.1},
        run_name="test_run",
        folder=str(tmp_path),
        use_tensorboard=True,
        use_wandb=False,
    )
    init_logging(run_id, config, run_seed=3)
    start_async_logging()
    try:
        for step in (10, 20):
            metrics = {
                "timestep": jnp.asarray(step),
                "loss": jnp.asarray(step / 10.0),
                "not_ready": jnp.asarray(jnp.nan),
            }
            vmap_log(metrics, 0, run_ids=[run_id])
    finally:
        stop_async_logging()

    log_dir = tmp_path / "tensorboard" / run_id
    scalars = load_scalars_from_tfevents(log_dir)
    assert scalars["loss"] == [(10, 1.0), (20, 2.0)]
    assert scalars["timestep"] == [(10, 10.0), (20, 20.0)]
    assert "not_ready" not in scalars
    with open(log_dir / "config.json") as fh:
        assert json.load(fh) == {"lr": 0.1, "seed": 3, "run_id": run_id}
