"""The fast-tier catch measurement (tests/fault_catches.py) selects, per
planted fault, the tests that ran one of its lines."""

from tests import fault_catches


def test_a_faults_lines_are_those_its_old_text_spans() -> None:
    text = "a = 1\nb = 2\nc = 3\nd = 4\n"
    assert fault_catches.fault_lines(text, "b = 2\nc = 3\n") == [2, 3]
    assert fault_catches.fault_lines(text, "d = 4") == [4]


def test_the_tests_running_a_line_are_read_off_their_contexts() -> None:
    by_line = {
        2: ["tests/a.py::test_x[1]|run", "tests/a.py::test_x[1]|setup", ""],
        3: ["tests/b.py::test_y|run"],
        9: ["tests/c.py::test_z|run"],
    }
    assert fault_catches.covering(by_line, [2, 3]) == {
        "tests/a.py::test_x[1]",
        "tests/b.py::test_y",
    }


def test_every_catalogued_fault_has_a_family() -> None:
    from tests.probing import faults

    family = fault_catches.families()
    assert set(family) == set(faults.load())
    assert "memory" in set(family.values())
