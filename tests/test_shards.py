"""Shards by recorded duration (tests/shards.py, ``pytest --shard I/N``)."""

import os
import subprocess
import sys
import textwrap

from tests import shards

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))
import durations as durations_script


def test_every_test_lands_in_one_shard_longest_first() -> None:
    """a (10) -> 0, b (6) -> 1, c (5) -> 1, d (1) -> 0, e (unknown: 1) -> 0
    (a tie: the lower shard). Loads 12 and 11."""
    durations = {"a": 10.0, "b": 6.0, "c": 5.0, "d": 1.0}
    assert shards.assign(["d", "c", "b", "a", "e"], durations, 2) == [0, 1, 1, 0, 0]


def test_the_split_is_balanced_and_the_same_in_every_process() -> None:
    durations = {f"t{i}": float(i % 7 + 1) for i in range(50)}
    ids = list(durations)
    picked = shards.assign(ids, durations, 3)
    assert picked == shards.assign(list(reversed(ids)), durations, 3)[::-1]
    loads = [
        sum(durations[i] for i, p in zip(ids, picked, strict=False) if p == s)
        for s in range(3)
    ]
    assert max(loads) - min(loads) <= max(durations.values())


def test_the_option_partitions_a_collection() -> None:
    def collected(*args: str) -> set[str]:
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *args]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return {line for line in out.splitlines() if "::" in line}

    whole = collected("tests/test_shards.py", "tests/test_tiers.py")
    parts = [
        collected("tests/test_shards.py", "tests/test_tiers.py", f"--shard={i}/3")
        for i in (1, 2, 3)
    ]
    assert set().union(*parts) == whole
    assert sum(len(p) for p in parts) == len(whole)


def test_a_files_tests_stay_together_and_a_file_without_a_time_gets_a_shard() -> None:
    """Split by file, a module's tests land in one shard (split, each shard
    would build the module's shared runs again), and a new file, without a
    recorded time, still runs: it counts FILE_DEFAULT (10 s). Longest
    first: a (10) -> 0, new (10) -> 1, b (6) -> 0 (a tie: the lower shard),
    c (5) -> 1. Loads 16 and 15."""
    seconds = {"a.py": 10.0, "b.py": 6.0, "c.py": 5.0}
    ids = ["c.py::t", "a.py::t1", "new.py::t", "b.py::T::t", "a.py::t2", "a.py::t3"]
    assert shards.assign_files(ids, seconds, 2) == [1, 0, 1, 0, 0, 0]


def test_the_option_keeps_each_file_in_one_shard_and_runs_every_test() -> None:
    """--shard-by file as CI's fast shards use it: the shards together hold
    every test once (a test in no shard would never run), and no file is in
    two."""

    def collected(*args: str) -> set[str]:
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *args]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return {line for line in out.splitlines() if "::" in line}

    paths = ["tests/test_shards.py", "tests/test_tiers.py", "tests/test_schedule.py"]
    whole = collected(*paths)
    parts = [collected(*paths, f"--shard={i}/2", "--shard-by=file") for i in (1, 2)]
    assert parts[0] | parts[1] == whole
    files = [{nodeid.split("::")[0] for nodeid in part} for part in parts]
    assert not files[0] & files[1]
    assert files[0] and files[1]


def test_file_times_are_summed_per_file_and_merged_into_the_known_ones(
    tmp_path,
) -> None:
    """`make file-durations`: a file's time is the sum of its tests' (0.4 +
    1.3 = 1.7, a class's test included: + 2.0 = 3.7), a file the reports
    do not name keeps its time (a partial refresh must not reset the other
    shards' files to the default), and a file that no longer exists goes."""
    report = tmp_path / "junit.xml"
    report.write_text(
        """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.test_shards" name="test_a" time="0.4"/>
<testcase classname="tests.test_shards" name="test_b[x]" time="1.3"/>
<testcase classname="tests.test_shards.TestC" name="test_c" time="2.0"/>
</testsuite></testsuites>"""
    )
    known = {
        "tests/test_shards.py": 99.0,
        "tests/test_tiers.py": 3.0,
        "tests/gone.py": 9,
    }
    assert durations_script.file_totals([str(report)], known) == {
        "tests/test_shards.py": 3.7,
        "tests/test_tiers.py": 3.0,
    }


def test_tests_sharing_a_module_fixture_build_it_on_one_worker(tmp_path) -> None:
    """Under --dist loadgroup, as CI's shards run, the tests reading one
    module fixture go to one xdist worker, which builds it once; spread over
    two workers it is built twice. Catches the groups no longer reaching
    xdist (a mark added after xdist reads it, or under another name)."""
    built = tmp_path / "built.txt"
    module = tmp_path / "test_shared.py"
    module.write_text(
        textwrap.dedent(
            f"""
            import pytest

            @pytest.fixture(scope="module")
            def run():
                with open({str(built)!r}, "a") as fh:
                    fh.write("built\\n")

            @pytest.mark.parametrize("i", range(8))
            def test_reads_the_run(run, i):
                pass
            """
        )
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += ["-p", "tests.conftest", "-n", "2", "--dist", "loadgroup", str(module)]
    subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    assert built.read_text().splitlines() == ["built"]


def test_a_grouped_tests_time_is_recorded_under_its_plain_node_id() -> None:
    """xdist names a grouped test ``nodeid@group``; its recorded time must
    keep the id --shard looks up, or every grouped test would count the
    default second and the slow tier's shards would drift apart."""
    name = "test_x[a]@tests/test_shards.py:a"
    got = durations_script.nodeid("tests.test_shards", name)
    assert got == "tests/test_shards.py::test_x[a]"
