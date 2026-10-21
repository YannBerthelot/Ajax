"""The slow tier's report (.github/scripts/slow_report.py): the failing tests
read off the junit reports, the issue it opens, comments on or closes, and
whether a bisect is cheap."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))
import slow_report

JUNIT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.probing.test_memory" name="test_learns[p0b-SAC-gru]" time="40.5"><failure/></testcase>
<testcase classname="tests.probing.test_memory" name="test_learns[p0b-PPO-gru]" time="12.0"/>
<testcase classname="tests.agents.test_option_smoke" name="test_a_row_of_options_trains[SAC-5]" time="9.0"><error/></testcase>
</testsuite></testsuites>"""


@pytest.fixture
def junit(tmp_path):
    (tmp_path / "junit-1").mkdir()
    (tmp_path / "junit-1" / "junit-1.xml").write_text(JUNIT)
    return str(tmp_path)


def test_the_failing_tests_are_read_off_the_reports(junit) -> None:
    assert slow_report.failures(junit) == {
        "tests/probing/test_memory.py::test_learns[p0b-SAC-gru]": 40.5,
        "tests/agents/test_option_smoke.py::test_a_row_of_options_trains[SAC-5]": 9.0,
    }


def test_a_bisect_is_cheap_for_few_short_failures_over_a_short_range() -> None:
    few = {"a": 100.0, "b": 200.0}
    assert slow_report.bisectable(few, 3)
    assert not slow_report.bisectable(few, 1)  # nothing to bisect
    assert not slow_report.bisectable(few, 40)  # over 4 steps
    assert not slow_report.bisectable({"a": 900.0}, 3)  # too long
    assert not slow_report.bisectable({str(i): 1.0 for i in range(6)}, 3)
    assert not slow_report.bisectable({}, 3)  # the run failed outside the tests


def test_a_red_run_opens_an_issue_and_a_green_one_closes_it(junit, capsys) -> None:
    common = [
        "--sha",
        "abc1234def",
        "--last-green",
        "",
        "--run-url",
        "URL",
        "--dry-run",
    ]
    slow_report.main([junit, "--result", "failure", *common])
    red = capsys.readouterr().out
    assert "gh issue create --label slow-tier-red" in red
    assert "test_learns[p0b-SAC-gru]" in red
    slow_report.main([junit, "--result", "success", *common])
    green = capsys.readouterr().out
    assert "issue close" not in green  # the dry run sees no open issue to close
