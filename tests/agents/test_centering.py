"""The centering rules (ajax.agents.centering), against their papers' equations.

Contract (Hisaki & Ono, ICML 2024, Eq. 18; Naik et al., RLC 2024, Eq. 10; RVI's reference function):
- RVIBatchMean: the target subtracts the stored offset; the next one is (1 - rate) offset + rate gain
  mean V'(s'). Iterated on constant next values v it reaches gain v geometrically.
- TDErrorCentering: the target subtracts the stored offset; the next one is offset + step_size mean delta.
- ReferencePoint: the target subtracts gain times the target critics' mean value at the reference row,
  read when the target is built; zeros stand for a reference given as None.
- Every rule commutes with a constant shift of the values (f(Q + k) = f(Q) + k), RVI's condition on f.
Expected values are worked by hand below.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.centering import (
    CenteringInputs,
    ReferencePoint,
    RVIBatchMean,
    TDErrorCentering,
)


def test_the_batch_mean_rule_moves_the_offset_towards_the_gained_mean():
    # mean V' = (2 + 4) / 2 = 3; 0.9 * 1 + 0.1 * 2 * 3 = 1.5
    rule = RVIBatchMean(rate=0.1, gain=2.0)
    inputs = CenteringInputs(jnp.array([[2.0], [4.0]]))
    assert float(rule.offset_for_target(jnp.asarray(1.0), inputs)) == 1.0
    np.testing.assert_allclose(
        rule.next_offset(jnp.asarray(1.0), inputs), 1.5, rtol=1e-6
    )


def test_the_batch_mean_rule_reaches_the_rate_of_constant_values():
    # offset_n = v + (1 - rate)^n (offset_0 - v): after 50 steps at rate 0.1, 7 + 0.9^50 (0 - 7)
    rule, offset = RVIBatchMean(rate=0.1), jnp.asarray(0.0)
    inputs = CenteringInputs(jnp.full((4, 1), 7.0))
    for _ in range(50):
        offset = rule.next_offset(offset, inputs)
    np.testing.assert_allclose(offset, 7.0 - 7.0 * 0.9**50, rtol=1e-5)


def test_the_td_error_rule_steps_by_the_mean_td_error():
    # mean delta = (1 - 3) / 2 = -1; 0.5 + 0.01 * -1 = 0.49
    rule = TDErrorCentering(step_size=0.01)
    inputs = CenteringInputs(jnp.zeros((2, 1)), td_errors=jnp.array([[1.0], [-3.0]]))
    assert float(rule.offset_for_target(jnp.asarray(0.5), inputs)) == 0.5
    np.testing.assert_allclose(
        rule.next_offset(jnp.asarray(0.5), inputs), 0.49, rtol=1e-6
    )


def test_the_reference_rule_reads_the_target_critics_at_its_row_when_the_target_is_built():
    seen = []

    def target_value_at(observation, action):
        seen.append((observation, action))
        return jnp.array([[3.0]])

    rule = ReferencePoint(observation=(1.0, 2.0), gain=0.5)
    inputs = CenteringInputs(jnp.zeros((2, 1)), target_value_at=target_value_at)
    offset = rule.offset_for_target(jnp.asarray(9.0), inputs)
    np.testing.assert_allclose(offset, 1.5)  # 0.5 * 3, whatever was stored
    assert seen == [((1.0, 2.0), None)]
    assert float(rule.next_offset(offset, inputs)) == 1.5


@pytest.mark.parametrize("shift", [-2.5, 0.0, 4.0])
def test_each_rule_commutes_with_a_constant_shift_of_the_values(shift):
    values = jnp.array([[1.0], [-2.0], [0.5]])
    shifted = CenteringInputs(
        values + shift, target_value_at=lambda o, a: jnp.array([[1.0 + shift]])
    )
    plain = CenteringInputs(values, target_value_at=lambda o, a: jnp.array([[1.0]]))
    batch_mean = RVIBatchMean(rate=0.2)
    np.testing.assert_allclose(
        batch_mean.next_offset(jnp.asarray(0.3 + shift), shifted),
        batch_mean.next_offset(jnp.asarray(0.3), plain) + shift,
        rtol=1e-6,
    )
    reference = ReferencePoint()
    np.testing.assert_allclose(
        reference.offset_for_target(jnp.asarray(0.0), shifted),
        reference.offset_for_target(jnp.asarray(0.0), plain) + shift,
        rtol=1e-6,
    )
