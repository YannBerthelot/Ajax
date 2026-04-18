"""Tests for ajax.schedule — schedule_picker, Scheduled, resolve_schedulable."""

import jax.numpy as jnp
import pytest

from ajax.schedule import (
    Scheduled,
    _exponential_core,
    _linear_core,
    _polynomial_core,
    _window_schedule,
    make_scheduled,
    resolve_schedulable,
    schedule_picker,
)


# ---------------------------------------------------------------------------
# Core schedule shapes
# ---------------------------------------------------------------------------


def test_linear_core_monotonic_decrease():
    values = [_linear_core(jnp.asarray(u)) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)
    for a, b in zip(values, values[1:]):
        assert float(a) >= float(b)


def test_linear_core_min_value_floor():
    assert _linear_core(jnp.asarray(1.0), min_value=0.3) == pytest.approx(0.3)


def test_exponential_core_decays():
    v0 = float(_exponential_core(jnp.asarray(0.0), p=5.0))
    v1 = float(_exponential_core(jnp.asarray(1.0), p=5.0))
    assert v0 == pytest.approx(1.0)
    assert v1 < v0


def test_polynomial_core_boundaries():
    assert float(_polynomial_core(jnp.asarray(0.0), p=2)) == pytest.approx(1.0)
    assert float(_polynomial_core(jnp.asarray(1.0), p=2)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Window schedule: active only inside [start, end]
# ---------------------------------------------------------------------------


def test_window_schedule_before_start_is_one():
    core = lambda u: _linear_core(u, 0.0)  # noqa: E731
    sched = _window_schedule(core, start=0.3, end=0.8, min_value=0.0)
    assert float(sched(jnp.asarray(0.1))) == pytest.approx(1.0)


def test_window_schedule_after_end_is_min_value():
    core = lambda u: _linear_core(u, 0.0)  # noqa: E731
    sched = _window_schedule(core, start=0.0, end=0.5, min_value=0.25)
    assert float(sched(jnp.asarray(0.9))) == pytest.approx(0.25)


def test_window_schedule_interior_matches_core():
    core = lambda u: _linear_core(u, 0.0)  # noqa: E731
    sched = _window_schedule(core, start=0.0, end=1.0, min_value=0.0)
    # At u=0.5 the linear core returns 0.5.
    assert float(sched(jnp.asarray(0.5))) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# schedule_picker: name parsing and defaults
# ---------------------------------------------------------------------------


def test_schedule_picker_none_returns_constant_one():
    sched = schedule_picker(None)
    assert sched(0.123) == 1.0


def test_schedule_picker_linear_default_window():
    sched = schedule_picker("linear")
    # Default window 0 → 1, min_value=0.0.
    assert float(sched(jnp.asarray(0.0))) == pytest.approx(1.0)
    assert float(sched(jnp.asarray(1.0))) == pytest.approx(0.0)


def test_schedule_picker_linear_custom_window():
    sched = schedule_picker("linear_0.25_0.75")
    # Before start still 1, after end is 0, halfway through window ≈ 0.5.
    assert float(sched(jnp.asarray(0.1))) == pytest.approx(1.0)
    assert float(sched(jnp.asarray(0.9))) == pytest.approx(0.0)
    assert float(sched(jnp.asarray(0.5))) == pytest.approx(0.5)


def test_schedule_picker_linear_min_value_percent():
    # 4th segment is percent of the min-value floor.
    sched = schedule_picker("linear_0_1_40")
    assert float(sched(jnp.asarray(1.0))) == pytest.approx(0.4)


def test_schedule_picker_exponential_defaults():
    sched = schedule_picker("exponential")
    assert float(sched(jnp.asarray(0.0))) == pytest.approx(1.0)
    assert float(sched(jnp.asarray(1.0))) < 1.0


def test_schedule_picker_polynomial_defaults():
    sched = schedule_picker("polynomial")
    assert float(sched(jnp.asarray(0.0))) == pytest.approx(1.0)
    assert float(sched(jnp.asarray(1.0))) == pytest.approx(0.0)


def test_schedule_picker_polynomial_custom_p():
    # (1 - u)**p: higher p decays faster for u in (0, 1).
    sched_p3 = schedule_picker("polynomial_0_1_3")
    sched_p2 = schedule_picker("polynomial_0_1_2")
    assert float(sched_p3(jnp.asarray(0.5))) < float(sched_p2(jnp.asarray(0.5)))


@pytest.mark.parametrize(
    "bad_name",
    ["linear_0", "exponential_0", "polynomial_0", "unknown_0_1"],
)
def test_schedule_picker_rejects_bad_names(bad_name):
    with pytest.raises(ValueError):
        schedule_picker(bad_name)


# ---------------------------------------------------------------------------
# Scheduled dataclass + make_scheduled + resolve_schedulable
# ---------------------------------------------------------------------------


def test_scheduled_at_multiplies_value_by_schedule():
    s = make_scheduled(2.0, "linear")
    assert float(s.at(0.0)) == pytest.approx(2.0)
    assert float(s.at(1.0)) == pytest.approx(0.0)


def test_scheduled_mul_and_add():
    s = make_scheduled(4.0, None)  # schedule is constant 1 → s.at(t) == 4.0
    assert float(s.mul(0.5, 0.3)) == pytest.approx(2.0)
    assert float(s.add(1.0, 0.3)) == pytest.approx(5.0)


def test_scheduled_hash_ignores_value():
    # Hash only depends on (class, schedule_fn): two Scheduled built from
    # the same schedule_fn object compare equal for caching purposes.
    shared_fn = schedule_picker("linear")
    s1 = Scheduled(value=jnp.asarray(1.0), schedule_fn=shared_fn)
    s2 = Scheduled(value=jnp.asarray(5.0), schedule_fn=shared_fn)
    assert hash(s1) == hash(s2)


def test_resolve_schedulable_passes_through_scalars():
    assert resolve_schedulable(0.75, train_frac=0.5) == 0.75


def test_resolve_schedulable_evaluates_scheduled():
    s = make_scheduled(10.0, "linear")
    assert float(resolve_schedulable(s, train_frac=0.0)) == pytest.approx(10.0)
    assert float(resolve_schedulable(s, train_frac=1.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scheduled is a pytree — should flatten/unflatten through jax transforms
# ---------------------------------------------------------------------------


def test_scheduled_is_pytree_compatible():
    import jax

    s = make_scheduled(3.0, "linear")
    leaves, treedef = jax.tree_util.tree_flatten(s)
    # schedule_fn is marked pytree_node=False → only `value` is a leaf.
    assert len(leaves) == 1
    s_back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(s_back, Scheduled)
    assert float(s_back.at(0.0)) == pytest.approx(3.0)
