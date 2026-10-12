"""Every scenario of ``benchmarks/perf_bench.py`` builds against the
current SAC (two that passed removed SAC options crashed unnoticed until
run), and ``stress_sac_utd`` is ``stress_sac`` with ten critic updates per
step; one tiny trial runs on CPU. The timings themselves need a quiet
machine.

The script is not a package module; it is loaded from its path.
"""

import importlib.util
import os
import sys

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "benchmarks",
    "perf_bench.py",
)
_SPEC = importlib.util.spec_from_file_location("perf_bench", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
perf_bench = importlib.util.module_from_spec(_SPEC)
# Registered before execution: dataclasses resolves its classes' module there.
sys.modules["perf_bench"] = perf_bench
_SPEC.loader.exec_module(perf_bench)


def config(scenario: str) -> dict:
    """The constructor arguments the scenario's SAC received."""
    agent = perf_bench._build_agent(scenario, n_envs=4, learning_starts=50)
    return {k: v for k, v in agent.config.items() if k not in ("self", "__class__")}


@pytest.mark.parametrize("scenario", perf_bench.SCENARIOS)
def test_every_scenario_builds(scenario):
    if scenario.startswith("p3dcircle"):
        # Plane3DCircle comes from the optional "plane" group.
        pytest.importorskip("target_gym")
    assert config(scenario)["n_envs"] >= 4


def test_stress_sac_utd_is_stress_sac_with_ten_critic_updates():
    stress = config("stress_sac")
    assert config("stress_sac_utd") == {**stress, "num_critic_updates": 10}
    assert stress["num_critic_updates"] == 1  # SAC's default


def test_a_tiny_trial_runs_on_cpu():
    """A CPU device returns no memory stats: the trial records 0 bytes."""
    result = perf_bench._run_one_trial(
        tag="test",
        scenario="pure_sac",
        trial=0,
        timesteps=64,
        warmup_timesteps=32,
        n_envs=2,
        n_seeds=1,
        seed=0,
    )
    assert result.measured_s > 0 and result.steps_per_s > 0
    if result.jax_backend == "cpu":
        assert result.peak_bytes_in_use == 0
