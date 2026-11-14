"""Recurrent SAC and TD3 (memory=MemoryConfig("gru")) against their original, Ni et al. (2022)'s
twni2016/pomdp-baselines (vendored at e7c19c3), one update at a time.

The recorder (benchmarks/reference/conformance_ni_recurrent.py, in the torch env) builds Ni et al.'s
ModelFreeOffPolicy_Separate_RNN with their MuJoCo configs' settings for Pendulum-v1, runs a few
updates, and records the next one: 64 windows of 64 steps, every fourth ending its episode, with
the update's noises. From the same state, on the same windows, Ajax's update
(ni_recurrent_original.py) must move every leaf as ModelFreeOffPolicy_Separate_RNN.update does: the
actor, the critic's one memory and both Q-networks, the targets, both Adams, and SAC's temperature
with its Adam, within the float32-rounding tolerance. That covers the wiring no unit test sees
whole: the critics' summed errors training their shared memory, the actor reading the stepped
critic, SAC's temperature stepping alpha on the log-probabilities of the actor's own loss, TD3's
actor maximising min(Q1, Q2) at every update, the windows' first rows read as an episode's first.

torch's GRU has two biases for each of its r and z gates where flax's has one, so Ni et al.'s
update moves those biases twice as far as Ajax's, and its actor's step, which reads the stepped
critic, follows (ni_recurrent_original.py's docstring). The leaf-by-leaf check therefore runs
against the same update with torch's GRUs given one bias per gate, and the difference from Ni et
al.'s own is checked on its own. Planted faults (SAC's temperature stepping log alpha, TD3's
actor delayed to every second update) must fail the check.
"""

import pytest

from tests.conformance import ni_recurrent_original as ni
from tests.conformance import originals

ALGOS = ("sac", "td3")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("ni_recurrent")
    recorder = "conformance_ni_recurrent.py"
    return originals.recorded("ni_recurrent", recorder, ALGOS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {algo: ni.run(records, algo) for algo in ALGOS}


@pytest.mark.parametrize("algo", ALGOS)
def test_one_update_matches_nis_with_one_bias_per_gate(records, ours, algo):
    for report in ni.reports(records, ours[algo], algo, ours[algo]):
        assert not report.failures, str(report)


def upstream_of_the_gate_biases(name: str) -> bool:
    """The leaves one update's gate-bias step cannot reach: the critic's step and the
    temperature's are computed before it; the actor's reads the stepped critic."""
    networks = ("critic.", "critic_target.", "opt.critic.", "alpha.", "opt.alpha.")
    upstream = name == "log_alpha" or name.startswith(networks)
    return upstream and ni.GATE_BIAS.fullmatch(name) is None


@pytest.mark.parametrize("algo", ALGOS)
def test_nis_own_update_moves_its_gate_biases_twice_as_far(records, ours, algo):
    """Against Ni et al.'s own update: every leaf upstream of the GRU's gate biases matches, and
    the critic's pairs of r and z biases move twice as far as Ajax's one bias (each of the pair
    takes its step), their target accordingly. Would catch a gate bias mapped or stepped
    otherwise (one of the pair left out of the map, the pair stepped as one), or a change that
    makes the difference reach further back than the actor's step."""
    run = ours[algo]
    upstream = ni.reports(records, run, algo, run, upstream_of_the_gate_biases, "")
    for report in upstream + ni.gate_bias_reports(records, run, algo):
        assert not report.failures, str(report)


@pytest.mark.parametrize("algo", ALGOS)
def test_a_planted_fault_fails_the_check(records, ours, algo):
    faulty = ni.run(records, algo, precisions=("f32",), fault=True)
    reports = ni.reports(records, ours[algo], algo, faulty)
    assert all(report.failures for report in reports)
