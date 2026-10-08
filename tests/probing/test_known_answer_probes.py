"""Known-answer probes (see ``known_answers.py``): each agent learns the
exact answer of every probe that fits its action space, on 8 seeds."""

from __future__ import annotations

import pytest

import ajax

from .known_answers import AGENTS, CALIBRATION, NOT_PROBED, Probe, check

CASES = [
    pytest.param(name, probe, id=f"{name}-{probe.name}")
    for name, (_, probes) in AGENTS.items()
    for probe in probes
]


@pytest.mark.parametrize(("agent_name", "probe"), CASES)
def test_agent_learns_the_known_answer(agent_name: str, probe: Probe) -> None:
    budget, tolerances = CALIBRATION[agent_name, probe.name]
    verdict = check(agent_name, probe, budget, tolerances)
    assert verdict.passed, verdict.report


def test_every_probe_is_calibrated_within_its_wrong_answers() -> None:
    for name, (_, probes) in AGENTS.items():
        for probe in probes:
            _, tolerances = CALIBRATION[name, probe.name]
            assert set(tolerances) == {q.name for q in probe.queries}
            for q in probe.queries:
                if q.margin:
                    assert 0 < tolerances[q.name] < abs(q.truth)
                else:
                    assert tolerances[q.name] <= q.max_tolerance + 1e-9, (name, q.name)


def test_every_exported_agent_is_probed_or_says_why_not() -> None:
    assert set(AGENTS) | set(NOT_PROBED) == set(ajax.__all__)
    assert not set(AGENTS) & set(NOT_PROBED)
