"""TD3 against its original, sfujim/TD3 (vendored at 34770cc), one update at a time.

sfujim's own TD3.py trains on two gymnax tasks (main.py's loop) in its pinned PyTorch env and records
four states each, with the update's batch and target noise and the state after it in float32 and
float64 (benchmarks/reference/conformance_sfujim_td3.py, re-run on every test session). Ajax makes the
same update from the same state, batch and noise (td3_sfujim.py), and every leaf's change must agree
within 10 times the two sides' float32 rounding errors (submaps.rounding_null). MountainCarContinuous's
box is [-1, 1]; Pendulum's torque is [-2, 2], where Ajax's critics must read the env's units. A planted
fault (target noise 0.22 instead of 0.2) must fail the check.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tests.conformance import submaps

REPO = pathlib.Path(__file__).parents[2]
ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")
CPU = os.environ | {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture(scope="module")
def records(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("sfujim")
    for env_id in ENVS:
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "benchmarks/reference/envs/torch",
                "--locked",
                "python",
                "benchmarks/reference/conformance_sfujim_td3.py",
                env_id,
                str(out),
            ],
            cwd=REPO,
            env=CPU,
            check=True,
        )
    return out


def ajax_side(
    records: pathlib.Path, out: pathlib.Path, precision: str, fault: str = ""
) -> pathlib.Path:
    args = [
        sys.executable,
        "-m",
        "tests.conformance.td3_sfujim",
        str(records),
        str(out),
        precision,
    ]
    subprocess.run(args + ([fault] if fault else []), cwd=REPO, env=CPU, check=True)
    return out


@pytest.fixture(scope="module")
def ours(records, tmp_path_factory) -> dict:
    return {
        p: ajax_side(records, tmp_path_factory.mktemp(f"ajax_{p}"), p)
        for p in ("f32", "f64")
    }


def reports(
    records: pathlib.Path, ours: dict, env_id: str, update: pathlib.Path
) -> list:
    out = []
    for path in sorted(records.glob(f"{env_id}_*.npz")):
        ref = dict(np.load(path))
        a32, a64, new = (
            dict(np.load(d / path.name)) for d in (ours["f32"], ours["f64"], update)
        )
        old = {k: ref[f"in.{k}"] for k in a32}
        nulls = submaps.rounding_null(
            ({k: ref[f"f32.{k}"] for k in a32}, {k: ref[f"f64.{k}"] for k in a32}),
            (a32, a64),
            old=old,
        )
        out.append(
            submaps.compare(
                new,
                {k: ref[f"f32.{k}"] for k in a32},
                nulls,
                old=old,
                context=path.stem,
            )
        )
    assert out, f"no record for {env_id}"
    return out


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_sfujims(records, ours, env_id):
    for report in reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_a_planted_fault_fails_the_check(records, ours, tmp_path_factory):
    faulty = ajax_side(
        records, tmp_path_factory.mktemp("ajax_fault"), "f32", "target_noise"
    )
    assert any(report.failures for report in reports(records, ours, ENVS[0], faulty))
