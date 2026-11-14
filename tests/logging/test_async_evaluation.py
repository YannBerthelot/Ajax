"""LoggingConfig(evaluation_device=...): the evaluations scored in a thread on
a device of their own (ajax.async_evaluation).

Contract (the owner's brief for the asynchronous evaluator, 2026-11-12):
- each evaluation scores the policy as it was at its timestep, on the same
  keys, so train returns the evaluations inline evaluation returns, up to
  floating-point rounding (the reference here: the same run evaluated inline);
- each seed's run logs every evaluation, its scores included, under its own
  timestep, through the logging worker;
- an evaluation that fails fails the train call instead of leaving NaN rows;
- an agent that evaluates inside its training program (APG) is refused;
- on a device other than a GPU, a MuJoCo Warp task is evaluated on pure-JAX
  physics, as Warp runs on NVIDIA GPUs only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.async_evaluation import AsyncEvaluator
from ajax.evaluate import evaluation_physics
from ajax.logging import LoggingConfig
from tests import exact_pins
from tests.logging import fake_wandb
from tests.logging.test_logging_backends import RecordingBackend
from tests.logging.test_logging_pins import AGENTS, SEEDS, logged_rows

CPU = jax.devices("cpu")[0]
UDRL, UDRL_KW, UDRL_STEPS = exact_pins.CASES["UDRL"]
RUNS = {
    "SAC": AGENTS["SAC"],  # the replay loop
    "PPO": AGENTS["PPO"],  # the on-policy loop
    "UDRL": (lambda: UDRL(**UDRL_KW), UDRL_STEPS, 32),  # its own evaluation
}


def train(name: str, device: Any, backends: tuple = ()) -> tuple[dict, list[str]]:
    make, steps, every = RUNS[name]
    config = LoggingConfig(
        run_name="run", log_frequency=every, backends=backends, evaluation_device=device
    )
    run_ids: list[str] = []
    _, evaluations = make().train(
        seed=SEEDS,
        n_timesteps=steps,
        num_episode_test=2,
        logging_config=config,
        on_ids_ready=run_ids.extend,
    )
    return {k: np.asarray(v) for k, v in evaluations.items()}, run_ids


@pytest.mark.parametrize("name", sorted(RUNS))
def test_the_evaluator_returns_the_inline_evaluations(name: str) -> None:
    """Catches an evaluator scoring another state or key than the inline
    evaluation's (a later chunk's policy, a fresh key), or writing a seed's
    scores into another seed's rows."""
    inline, _ = train(name, None)
    later, _ = train(name, CPU)
    assert sorted(later) == sorted(inline)
    assert np.isfinite(later["Eval/episodic mean reward"]).all()
    for key, values in inline.items():
        # float32 episode returns: a separately compiled program may round
        # a few steps' sums differently.
        np.testing.assert_allclose(later[key], values, rtol=1e-5, err_msg=key)


def test_each_run_logs_its_evaluations_with_their_scores(tmp_path: Any) -> None:
    """Catches scores left out of the logs (the rows handed over before
    scoring, their NaN placeholders dropped) or logged to another seed's run."""
    path = str(tmp_path / "calls.jsonl")
    evaluations, run_ids = train("SAC", CPU, (RecordingBackend(path),))
    calls = fake_wandb.read_calls(path)
    for seed_index, run_id in enumerate(run_ids):
        logged = [
            (c["step"], c["metrics"])
            for c in calls
            if c["call"] == "log" and c["run_id"] == run_id
        ]
        expected = logged_rows(evaluations, seed_index)
        assert [step for step, _ in logged] == [step for step, _ in expected]
        for (_, got), (_, want) in zip(logged, expected, strict=True):
            assert "Eval/episodic mean reward" in got
            assert got == pytest.approx(want)


def test_a_failed_evaluation_fails_the_call_that_waits_for_it() -> None:
    """Catches an error in the evaluator's thread being swallowed, which
    would return NaN rows as if nothing had been evaluated."""

    def scores(state: Any, key: Any) -> dict:
        raise ValueError("the policy cannot be scored")

    evaluator = AsyncEvaluator(CPU, n_seeds=2, log_fn=lambda row, index: None)

    def evaluate(state: Any, key: Any, index: Any) -> None:
        metrics = {"timestep": jnp.int32(8)}
        evaluator.hand_over(True, scores, state, key, metrics, index)

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    jax.jit(jax.vmap(evaluate))(jnp.zeros((2, 3)), keys, jnp.arange(2))
    jax.effects_barrier()
    with pytest.raises(RuntimeError, match="asynchronous evaluation failed"):
        evaluator.wait()


def test_an_agent_evaluating_inside_its_iterations_is_refused() -> None:
    """Catches APG taking a device it cannot use: its iterations would hand
    over evaluations whose scores never reach the outputs it returns."""
    agent_class, kwargs, steps = exact_pins.CASES["APG"]
    config = LoggingConfig(log_frequency=40, backends=(), evaluation_device=CPU)
    with pytest.raises(ValueError, match="evaluation_device"):
        agent_class(**kwargs).train(seed=0, n_timesteps=steps, logging_config=config)


@pytest.mark.parametrize(
    ("physics", "device", "expected"),
    [("warp", CPU, "jax"), ("warp", None, "warp"), ("jax", CPU, "jax")],
)
def test_a_warp_task_is_evaluated_on_jax_physics_off_the_gpu(
    physics: str, device: Any, expected: str
) -> None:
    """Catches a CPU evaluator of a Warp run rebuilding its task on Warp,
    which has no CPU implementation, or a GPU evaluation leaving Warp."""
    env = SimpleNamespace(_ajax_physics=physics)
    with jax.default_device(device):
        assert evaluation_physics(env) == expected
