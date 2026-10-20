"""The leaf-by-leaf comparison of two steps (tests/conformance/submaps.py)."""

import math

import numpy as np

from tests.conformance import submaps


def test_changes_not_values_are_compared():
    old = np.full(4, 1000.0)
    a, b = old + np.array([1.0, 1.0, 1.0, 1.001]), old + 1.0
    # As values the two agree to 1e-6; as changes they differ by 1e-3.
    assert submaps.change_diff(a, b) < 2e-6
    assert math.isclose(submaps.change_diff(a, b, old), 1e-3, rel_tol=1e-6)


def test_a_leaf_only_one_side_changes_differs_infinitely():
    old = np.zeros(3)
    assert submaps.change_diff(old, old, old) == 0.0
    assert submaps.change_diff(old + 1e-9, old, old) == math.inf


def test_discrete_leaves_must_be_equal_and_equal_nans_are_equal():
    assert submaps.change_diff(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.25
    nan = np.array([np.nan, 2.0])
    assert submaps.change_diff(nan, nan) == 0.0
    report = submaps.compare(
        {"action": np.array([1, 0])}, {"action": np.array([1, 1])}, nulls={}
    )
    assert [c.leaf for c in report.failures] == ["action"]


def test_the_tolerance_is_the_null_times_the_factor_with_a_floor():
    b = {"w": np.array([1.0, 2.0]), "v": np.array([3.0])}
    a = {"w": b["w"] * (1 + 5e-5), "v": b["v"] * (1 + 5e-7)}
    report = submaps.compare(a, b, nulls={"w": 1e-5, "v": 0.0})
    by_leaf = {c.leaf: c for c in report.checks}
    assert by_leaf["w"].tau == 1e-4 and by_leaf["w"].ok
    assert by_leaf["v"].tau == submaps.FLOOR and by_leaf["v"].ok
    report = submaps.compare(a, b, nulls={"w": 1e-6, "v": 0.0}, context="iteration 3")
    assert [c.leaf for c in report.failures] == ["w"]
    text = str(report)
    assert "1 of 2 leaves beyond tolerance (iteration 3)" in text and "w" in text


def test_the_null_is_the_largest_difference_within_each_side():
    old = {"w": np.zeros(2), "n": np.array([3])}
    jit_a, eager_a = (
        {"w": np.array([1.0, 1.0]), "n": np.array([3])},
        {"w": np.array([1.0, 1.01]), "n": np.array([3])},
    )
    jit_b, eager_b = (
        {"w": np.array([2.0, 2.0]), "n": np.array([3])},
        {"w": np.array([2.0, 2.0]), "n": np.array([3])},
    )
    nulls = submaps.null_of((eager_a, jit_a), (eager_b, jit_b), old=old)
    assert set(nulls) == {"w"} and math.isclose(nulls["w"], 0.01)
