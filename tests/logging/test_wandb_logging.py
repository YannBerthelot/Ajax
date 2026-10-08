import json
from unittest.mock import patch

import jax.numpy as jnp

from ajax.logging.wandb_logging import (
    LoggingConfig,
    finish_logging,
    init_logging,
    load_scalars_from_tfevents,
    log_variables,
    start_async_logging,
    stop_async_logging,
    vmap_log,
)


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
            vmap_log(metrics, 0, run_ids=[run_id], logging_config=config)
    finally:
        stop_async_logging()

    log_dir = tmp_path / "tensorboard" / run_id
    scalars = load_scalars_from_tfevents(log_dir)
    assert scalars["loss"] == [(10, 1.0), (20, 2.0)]
    assert scalars["timestep"] == [(10, 10.0), (20, 20.0)]
    assert "not_ready" not in scalars
    with open(log_dir / "config.json") as fh:
        assert json.load(fh) == {"lr": 0.1, "seed": 3, "run_id": run_id}


@patch("ajax.logging.wandb_logging.wandb.log")
def test_log_variables(mock_wandb_log):
    variables_to_log = {"metric1": 0.5, "metric2": 0.8}
    log_variables(variables_to_log)

    # Validate wandb logging
    mock_wandb_log.assert_called_once_with(variables_to_log, commit=True)


@patch("ajax.logging.wandb_logging.wandb.finish")
def test_finish_logging(mock_wandb_finish):
    finish_logging()

    # Validate wandb finish
    mock_wandb_finish.assert_called_once()
