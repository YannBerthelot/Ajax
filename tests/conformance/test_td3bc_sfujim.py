"""TD3+BC against its original, sfujim/TD3_BC (vendored at 8791ad7), one update at a time.

sfujim's TD3_BC.py trains offline by main.py's loop on a dataset collected from a gymnax task under
uniform random actions, its states normalised as main.py does, and records four states each
(benchmarks/reference/conformance_sfujim_td3_bc.py, originals.recorded). Ajax is TD3 with TD3+BC's term,
``ImitationLoss(None, target="replay", alpha=2.5)``: from the same state, batch (the reference's
normalised states) and noise it makes the same update (td3_sfujim.py), the paper's lambda-weighted loss
with the actions compared in the env's units (Pendulum's torque is in [-2, 2]). A planted fault (alpha
2.75 instead of 2.5) must fail the check.
"""

import pytest

from ajax.extensions.expert import ImitationLoss
from tests.conformance import originals, td3_sfujim

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


def td3bc(alpha: float = 2.5) -> tuple:
    return (ImitationLoss(None, target="replay", alpha=alpha),)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("td3bc")
    return originals.recorded(
        "sfujim_td3_bc", "conformance_sfujim_td3_bc.py", ENVS, scratch
    )


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: td3_sfujim.run(records, p, extensions=td3bc()) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_sfujims_td3_bc(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = td3_sfujim.run(records, "f32", extensions=td3bc(2.75))
    assert any(f.failures for f in originals.reports(records, ours, ENVS[0], faulty))
