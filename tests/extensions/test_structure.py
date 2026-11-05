"""What RL changes around a controller it keeps (ajax.extensions.structure).

Contract, from the papers and not the code:

* Arroyo and Giraldo, arXiv 2008.13732: the inner loop is left as it is, and
  the outer loop only changes the reference it reads. So no change is the
  expert, and a change acts as the same change to the target would.
* Bloor et al., arXiv 2408.13566, eq. 10: Delta u = Kp Delta e + Ki dt e, at
  the step's gains, so a new gain alone moves no output.
* Hanus, Kinnaert and Henrotte 1987, as MERL TR2019-054 restates it: the state
  is fed the realisable error, e + (u_executed - u) / D. On a PI with state x =
  Ki I that is x+ = x + Ki dt / D (u_executed - x), with D = Kp + Ki dt here,
  since the law integrates before it outputs. Its fixed point is x = u_executed.

Beyond the papers, from the study's contract with TargetGym: the expert form is
the batched controller, channels come from the env's obs_value_index and
obs_target_index, the memory follows info["applied_action"], and a step the
expert chose itself stays its own.

The fake below is a PI on obs = [reading, target], limited to [-1, 1], which
holds its integral at the limit. Its numbers are worked by hand in each test.
"""

from typing import Any, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import environment, spaces

from ajax import SAC
from ajax.controller import Controller, TransformInputs, make_run_controller
from ajax.environments.interaction import ActingState, expert_collect_kwargs
from ajax.evaluate import evaluate
from ajax.extensions.expert import ExpertGains, JSRLCurriculum
from ajax.extensions.observe import ExpertStateInput
from ajax.extensions.structure import (
    BumplessTracking,
    ExpertLaw,
    GainScheduledLaw,
    OuterLoopReference,
)
from ajax.policy import make_act

KP, KI = 0.5, 0.1
D = KP + KI  # the feedthrough, at dt = 1
READING = (0,)


@struct.dataclass
class PIParams:
    Kp: float
    Ki: float


class PIMemory(NamedTuple):
    integral: Any


def fake_pi_step(params: PIParams, memory: PIMemory, obs: jax.Array, xp: Any = jnp):
    error = obs[1] - obs[0]
    integral = memory.integral + error
    wanted = params.Kp * error + params.Ki * integral
    action = xp.clip(wanted, -1.0, 1.0)
    held = xp.where(action == wanted, integral, memory.integral)
    return xp.array([action]), PIMemory(held)


def fake_pi(gain_names: tuple = ("Kp", "Ki")) -> ExpertLaw:
    return ExpertLaw(fake_pi_step, PIParams(KP, KI), PIMemory(0.0), gain_names)


def scheduled(law: Any, **options: Any) -> GainScheduledLaw:
    return GainScheduledLaw(law, obs_width=2, **options)


def observations(*errors: float) -> list[jax.Array]:
    """One env's observations with these errors, the reading at 0."""
    return [jnp.array([[0.0, error]]) for error in errors]


def assert_same_tree(left: Any, right: Any) -> None:
    jax.tree.map(np.testing.assert_array_equal, left, right)


def as_tuple(index: Any) -> tuple[int, ...]:
    return tuple(index) if isinstance(index, (tuple, list)) else (index,)


class RealTask(NamedTuple):
    controller: Any  # TargetGym's Controller
    expert: Any  # its batched form, the expert form the study calls
    rows: jax.Array  # (n_steps, n_envs, obs_width), along its closed loop
    memory: Any  # its batched memory at the last row
    readings: tuple[int, ...]
    targets: tuple[int, ...]


def real_task(task: str) -> RealTask:
    """task's scored PID, with observations along its closed loop."""
    target_gym = pytest.importorskip("target_gym")
    registry = pytest.importorskip("target_gym.registry")
    env, params = target_gym.make(task)
    controller = registry.REGISTRY[task].make_pid_controller()
    expert = controller.batched()
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, state = jax.vmap(env.reset, (0, None))(keys, params)
    memory = expert.init_state(3)
    rows, last_memory = [], memory
    for index in range(32):
        obs = obs.astype(jnp.float32)
        rows.append(obs)
        last_memory = memory
        action, memory = expert(memory, obs)
        keys = jax.random.split(jax.random.PRNGKey(index + 1), 3)
        obs, state, *_ = jax.vmap(env.step, (0, 0, 0, None))(
            keys, state, action, params
        )
    return RealTask(
        controller,
        expert,
        jnp.stack(rows),
        last_memory,
        as_tuple(env.obs_value_index),
        as_tuple(env.obs_target_index),
    )


#: Each real PID's gains, by the names its law gives them.
REAL_GAINS = {
    "four_tank": ("Kp_1", "Ti_1", "Kp_2", "Ti_2"),
    "cstr": ("Kp", "Ti", "Kp_T", "Ti_T"),
}


# -- outer-loop reference --------------------------------------------------
def reference(channels: tuple = (1,), bound: tuple = (0.25,)) -> OuterLoopReference:
    return OuterLoopReference(target_channels=channels, max_change=bound)


def test_a_zero_change_is_the_expert_to_the_bit() -> None:
    """Bug caught: the reference moved at rest, or the expert stepped twice."""
    law = fake_pi()
    obs = jnp.array([[0.1, 0.5]])
    memory = law.init_state(1)
    inputs = TransformInputs(law, memory, obs, None)
    action, after = reference().actuate(jnp.zeros((1, 1)), inputs)
    own_action, own_memory = law(memory, obs)
    assert_same_tree((action, after), (own_action, own_memory))


@pytest.mark.parametrize("task", sorted(REAL_GAINS))
def test_a_zero_change_is_the_batched_pid_to_the_bit(task: str) -> None:
    real = real_task(task)
    arm = reference(real.targets, tuple(0.01 for _ in real.targets))
    inputs = TransformInputs(real.expert, real.memory, real.rows[-1], None)
    actions = jnp.zeros((3, len(real.targets)))
    own = real.expert(real.memory, real.rows[-1])
    assert_same_tree(arm.actuate(actions, inputs), own)


def test_a_full_action_moves_the_target_by_its_bound() -> None:
    """A target 0.25 up on e = 0.4 is e = 0.65: u = 0.5 * 0.65 + 0.1 * 0.65."""
    law = fake_pi()
    inputs = TransformInputs(law, law.init_state(1), jnp.array([[0.1, 0.5]]), None)
    action, _ = reference().actuate(jnp.ones((1, 1)), inputs)
    np.testing.assert_allclose(action, [[0.39]], rtol=1e-6)


def test_the_policy_acts_in_the_width_of_the_targets() -> None:
    """The critics read the changes: an env-wide action would mix two spaces."""
    arm = reference((1, 3), (0.1, 0.2))
    assert arm.policy_action_dim(None, None) == 2
    assert Controller(transforms=(arm,)).actuates


def test_a_change_without_a_positive_bound_for_each_target_is_refused() -> None:
    with pytest.raises(ValueError, match="one max_change per channel"):
        OuterLoopReference(target_channels=(1, 2), max_change=(0.1,))
    with pytest.raises(ValueError, match="positive"):
        OuterLoopReference(target_channels=(1,), max_change=(0.0,))


# -- gain scheduling -------------------------------------------------------
def rollout(layer: GainScheduledLaw, rows: list, factors: list) -> list:
    state = layer.init_state(rows[0].shape[0])
    actions = []
    for obs, factor in zip(rows, factors, strict=True):
        action, state = layer.step_with_gains(state, obs, jnp.array([factor]))
        actions.append(float(action[0, 0]))
    return actions


def test_unit_multipliers_are_the_law_to_the_bit_through_its_limit() -> None:
    """Bug caught: the bias not 0 at rest, or the anti-windup lost. The errors
    drive the fake past its limit and back."""
    law = fake_pi()
    layer, memory = scheduled(law), law.init_state(1)
    state = layer.init_state(1)
    for obs in observations(0.2, 3.0, 3.0, 0.1, -0.2, -4.0, 0.3):
        action, state = layer.step_with_gains(state, obs, jnp.ones((1, 2)))
        own_action, memory = law(memory, obs)
        assert_same_tree(action, own_action)
    assert_same_tree(state.memory, memory)


def assert_layer_is_the_expert(layer: GainScheduledLaw, expert: Any, rows: Any) -> None:
    """Not to the bit: these laws divide by a gain, and XLA rewrites a division
    by one number as a product with its reciprocal, which a per-env gain
    escapes. One unit of rounding near 1 is 1.2e-7; 32 steps may add some."""
    state, memory = layer.init_state(rows.shape[1]), expert.init_state(rows.shape[1])
    multipliers = jnp.ones((rows.shape[1], len(layer.learnable_fields)))
    for obs in rows:
        action, state = layer.step_with_gains(state, obs, multipliers)
        own_action, memory = expert(memory, obs)
        np.testing.assert_allclose(action, own_action, atol=2e-6)


@pytest.mark.parametrize("task", sorted(REAL_GAINS))
def test_unit_multipliers_are_the_real_pid_to_float32_rounding(task: str) -> None:
    real = real_task(task)
    law = ExpertLaw.of(real.controller, REAL_GAINS[task])
    layer = GainScheduledLaw(law, obs_width=real.rows.shape[-1])
    assert_layer_is_the_expert(layer, real.expert, real.rows)


def test_the_layer_runs_on_the_batched_forms_own_step_with_gains() -> None:
    """TargetGym's FunctionalExpertPolicy has step_with_gains for the laws it
    registers, so a registered one needs no adapter."""
    real = real_task("plane3d_heading")
    layer = GainScheduledLaw(real.expert, obs_width=real.rows.shape[-1])
    assert layer.learnable_fields == real.expert.learnable_fields
    assert_layer_is_the_expert(layer, real.expert, real.rows)


def test_the_form_is_eq_10_of_the_paper() -> None:
    """e = 0.2 then 0.3, gains doubled at the second step. Step 1 at (0.5, 0.1)
    is u = 0.5 * 0.2 + 0.1 * 0.2 = 0.12. Eq. 10 at (1.0, 0.2) adds 1.0 * 0.1 +
    0.2 * 0.3 = 0.16, so u = 0.28."""
    layer = scheduled(fake_pi(), max_log_rate=1.0, bias_retention=1.0)
    actions = rollout(layer, observations(0.2, 0.3), [[1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_allclose(actions, [0.12, 0.28], rtol=1e-5)


def test_a_new_gain_alone_moves_no_output() -> None:
    """The bump of the position form: with an integral held at e = 0, doubling
    Ki doubles Ki I. Eq. 10 at e = 0 has no term to move."""
    layer = scheduled(fake_pi(), max_log_rate=1.0, bias_retention=1.0)
    rows = observations(0.5, 0.0, 0.0)
    actions = rollout(layer, rows, [[1.0, 1.0], [1.0, 1.0], [1.0, 2.0]])
    assert actions[2] == pytest.approx(actions[1], abs=1e-6)


def test_multipliers_move_no_faster_than_the_rate_limit() -> None:
    """From 1 towards 4, log 4 = 1.386 away, 0.5 a step: 3 steps to arrive."""
    layer = scheduled(fake_pi(), max_log_rate=0.5)
    state = layer.init_state(1)
    obs = jnp.array([[0.0, 0.1]])
    logs = []
    for _ in range(4):
        _, state = layer.step_with_gains(state, obs, jnp.array([[4.0, 1.0]]))
        logs.append(float(state.log_factors[0, 0]))
    np.testing.assert_allclose(logs, [0.5, 1.0, np.log(4.0), np.log(4.0)], rtol=1e-5)


def test_gain_mode_bounds_the_multipliers_by_its_factor() -> None:
    """Under ExpertGains(max_factor=2), a = +1 asks for 2 and a = -1 for 1 / 2."""
    layer = scheduled(fake_pi(), max_log_rate=10.0)
    obs = jnp.array([[0.0, 0.1]])
    inputs = TransformInputs(layer, layer.init_state(1), obs, None)
    _, state = ExpertGains(max_factor=2.0).actuate(jnp.array([[1.0, -1.0]]), inputs)
    np.testing.assert_allclose(jnp.exp(state.log_factors), [[2.0, 0.5]], rtol=1e-5)


class _NoFeatures:
    observers: tuple = ()


def test_the_policy_acts_in_the_width_of_the_gains() -> None:
    layer = scheduled(fake_pi())
    assert layer.learnable_fields == ("Kp", "Ki")
    np.testing.assert_array_equal(layer.anchor_gains, [1.0, 1.0])
    assert ExpertGains().policy_action_dim(_NoFeatures(), layer) == 2


def test_a_law_at_its_limit_is_not_held_off_it_by_an_old_bias() -> None:
    """A velocity form on a law that holds its integral at its limit could
    stick: the law flat at 1, its integral held, the bias keeping u below 1.
    Halving the bias per step ends it in the time the halving says."""
    layer = scheduled(fake_pi(), max_log_rate=1.0, bias_retention=0.5)
    rows = observations(0.5, 2.0, *[2.0] * 12)
    factors = [[1.0, 1.0], [0.5, 1.0], *[[0.5, 1.0]] * 12]
    assert rollout(layer, rows, factors)[-1] == pytest.approx(1.0, abs=1e-3)


def test_an_unknown_gain_name_is_refused() -> None:
    with pytest.raises(ValueError, match="no gains"):
        fake_pi(gain_names=("Kd",))


def test_a_law_keeps_its_other_params_as_they_are() -> None:
    """wind_turbine's law takes numpy to its params, which a tracer breaks, as
    TargetGym's form breaks it by making every param an array."""

    def step(params: PIParams, memory: PIMemory, obs: jax.Array) -> Any:
        assert isinstance(params.Ki, float)
        return fake_pi_step(params, memory, obs)

    law = ExpertLaw(step, PIParams(KP, KI), PIMemory(0.0), ("Kp",))
    obs = jnp.array([[0.0, 1.0]])
    action, _ = law.step_with_gains(law.init_state(1), obs, jnp.array([[0.2]]))
    np.testing.assert_allclose(action, [[0.3]], rtol=1e-6)


def test_a_nested_gain_is_found_by_its_dotted_name() -> None:
    """TargetGym's aircraft nest theirs (tecs.time_const)."""

    @struct.dataclass
    class Nested:
        inner: PIParams

    def step(params: Nested, memory: PIMemory, obs: jax.Array) -> Any:
        return fake_pi_step(params.inner, memory, obs)

    law = ExpertLaw(step, Nested(PIParams(KP, KI)), PIMemory(0.0), ("inner.Kp",))
    np.testing.assert_allclose(law.anchor_gains, [KP])
    layer = GainScheduledLaw(law, obs_width=2, max_log_rate=1.0)
    action, _ = layer.step_with_gains(
        layer.init_state(1), jnp.array([[0.0, 0.5]]), jnp.array([[2.0]])
    )
    # e = 0.5: 2 * 0.5 * 0.5 + 0.1 * 0.5 = 0.55.
    np.testing.assert_allclose(action, [[0.55]], rtol=1e-5)


# -- bumpless tracking -----------------------------------------------------
TRACKING = BumplessTracking(READING)


def track(law: Any, memory: Any, obs: jax.Array, command: Any, applied: Any) -> Any:
    return TRACKING.track_memory(law, memory, obs, command, applied)


def test_the_laws_own_action_leaves_its_step_as_it_is() -> None:
    """Bug caught: tracking changing a law that is acting. At the limit too:
    its own anti-windup stands (e = 5 saturates, the integral holds)."""
    law = fake_pi()
    for error in (0.2, 5.0):
        obs = jnp.array([[0.0, error]])
        memory = PIMemory(jnp.array([0.3]))
        action, own_memory = law(memory, obs)
        assert_same_tree(track(law, memory, obs, action, action), own_memory)


def test_a_step_the_law_chose_stays_its_own_whatever_the_actuator_did() -> None:
    """Under a rate limit the applied action is not the expert's own even when
    it acts alone. The study scores the expert as it ships, whose memory knows
    nothing of the actuator, so its own steps do not condition on it."""
    law = fake_pi()
    obs, memory = jnp.array([[0.1, 0.5]]), law.init_state(1)
    action, own_memory = law(memory, obs)
    tracked = track(law, memory, obs, action, jnp.array([[0.0]]))
    assert_same_tree(tracked, own_memory)


def test_the_memory_takes_the_realisable_error() -> None:
    """e = 0.4, I = 0: u = 0.24. u_executed = 0.09 gives e_r = 0.4 + (0.09 -
    0.24) / 0.6 = 0.15, so I+ = 0.15."""
    law = fake_pi()
    obs, executed = jnp.array([[0.1, 0.5]]), jnp.array([[0.09]])
    tracked = track(law, law.init_state(1), obs, executed, executed)
    np.testing.assert_allclose(tracked.integral, [0.15], rtol=1e-5)


def test_the_memory_follows_what_was_applied_not_what_was_sent() -> None:
    """As above with u_applied = 0.09, the command 0.3: PR 3's actuators rate
    limit and lag, so what reached the plant is what the memory must explain."""
    law = fake_pi()
    obs = jnp.array([[0.1, 0.5]])
    command, applied = jnp.array([[0.3]]), jnp.array([[0.09]])
    tracked = track(law, law.init_state(1), obs, command, applied)
    np.testing.assert_allclose(tracked.integral, [0.15], rtol=1e-5)


def takeover(law: ExpertLaw, error: float, executed: float, steps: int) -> tuple:
    """The integral of the tracked and of the plain law, while another
    controller holds the plant at this error and action."""
    obs, action = jnp.array([[0.0, error]]), jnp.array([[executed]])
    tracked = plain = law.init_state(1)
    for _ in range(steps):
        tracked = track(law, tracked, obs, action, action)
        _, plain = law(plain, obs)
    return tracked, plain


def test_the_integrator_settles_on_the_executed_action() -> None:
    """x+ = x + Ki / D (u - x) gives x_t = u (1 - (1 - 1 / 6)^t), with x = Ki I:
    the module docstring's fixed point at the rate the paper's form says."""
    tracked, _ = takeover(fake_pi(), 0.2, 0.3, 6)
    expected = 0.3 * (1.0 - (5.0 / 6.0) ** 6)
    np.testing.assert_allclose(KI * tracked.integral, [expected], rtol=1e-4)
    settled, _ = takeover(fake_pi(), 0.2, 0.3, 200)
    np.testing.assert_allclose(KI * settled.integral, [0.3], rtol=1e-4)


def test_a_takeover_does_not_wind_the_integral_up() -> None:
    """After 50 steps of e = 0.2 under another controller holding u = 0.3, the
    plain law's Ki I is 0.1 * 0.2 * 50 = 1 and it hands back u = 1, 0.7 above
    what was executed. The tracked one hands back Ki I + D e = 0.42."""
    tracked, plain = takeover(fake_pi(), 0.2, 0.3, 50)
    law, obs = fake_pi(), jnp.array([[0.0, 0.2]])
    np.testing.assert_allclose(law(plain, obs)[0], [[1.0]])
    np.testing.assert_allclose(law(tracked, obs)[0], [[0.42]], rtol=1e-2)


@pytest.mark.parametrize("task", sorted(REAL_GAINS))
def test_the_real_pid_takes_the_reading_that_gives_the_executed_action(
    task: str,
) -> None:
    """Below their limits these laws are affine in their readings. Finite
    differences, an oracle apart from conditioned_memory's derivative, give the
    reading at which the expert outputs the executed action; its memory there
    is the tracked one, and its output is the executed action."""
    real = real_task(task)
    obs, columns = real.rows[-1], jnp.asarray(real.readings)
    own, _ = real.expert(real.memory, obs)
    executed = jnp.clip(own + 0.02, -0.95, 0.95)
    step = 1e-3
    slopes = [
        (real.expert(real.memory, obs.at[:, column].add(step))[0] - own) / step
        for column in real.readings
    ]
    sensitivity = jnp.stack(slopes, axis=-1)
    shift = jax.vmap(jnp.linalg.solve)(sensitivity, executed - own)
    moved = obs.at[:, columns].add(shift)
    output, oracle = real.expert(real.memory, moved)
    np.testing.assert_allclose(output, executed, atol=2e-2)
    tracker = BumplessTracking(real.readings)
    tracked = tracker.track_memory(real.expert, real.memory, obs, executed, executed)
    jax.tree.map(
        lambda left, right: np.testing.assert_allclose(left, right, atol=2e-2),
        tracked,
        oracle,
    )


@pytest.mark.parametrize("task", sorted(REAL_GAINS))
def test_a_takeover_of_the_real_pid_stays_nearer_the_executed_action(
    task: str,
) -> None:
    """Bug caught: a law whose memory ignores the takeover. The same plant-free
    takeover from one state: readings frozen, an action held 0.2 away. The
    tracked law's hand-back is nearer it than the plain law's."""
    real = real_task(task)
    obs, tracker = real.rows[-1], BumplessTracking(real.readings)
    executed = jnp.clip(real.expert(real.memory, obs)[0] - 0.2, -0.9, 0.9)
    tracked = plain = real.memory
    for _ in range(64):
        tracked = tracker.track_memory(real.expert, tracked, obs, executed, executed)
        _, plain = real.expert(plain, obs)
    tracked_gap = jnp.abs(real.expert(tracked, obs)[0] - executed).mean()
    plain_gap = jnp.abs(real.expert(plain, obs)[0] - executed).mean()
    assert tracked_gap < plain_gap


def hand_back(tracker: Any, memory: Any, steps: int = 60) -> tuple[float, float]:
    """A first-order plant x+ = x + 0.1 (u - x), target 0.5, driven by the fake
    PI to the target, held at 0 by another controller for 60 steps, then handed
    back. The highest x after the hand-back, and the error it leaves at the end."""
    law, x = fake_pi(), 0.0

    def stepped(x: float, u: float) -> float:
        return x + 0.1 * (u - x)

    def obs_at(x: float) -> jax.Array:
        return jnp.array([[x, 0.5]])

    for _ in range(steps):
        action, memory = law(memory, obs_at(x))
        x = stepped(x, float(action[0, 0]))
    held = jnp.zeros((1, 1))
    for _ in range(steps):
        if tracker is None:
            _, memory = law(memory, obs_at(x))
        else:
            memory = tracker.track_memory(law, memory, obs_at(x), held, held)
        x = stepped(x, 0.0)
    peak = x
    for _ in range(2 * steps):
        action, memory = law(memory, obs_at(x))
        x = stepped(x, float(action[0, 0]))
        peak = max(peak, x)
    return peak, abs(x - 0.5)


def test_a_takeover_leaves_no_overshoot_on_a_real_loop() -> None:
    """What the technique is for, with a plant in the loop. Held at 0 for 60
    steps, the plain law integrates an error near 0.5 until its limit holds it,
    and overshoots the target by a third at the hand-back. The tracked one
    barely does. Both settle: tracking takes nothing from the loop."""
    fresh = fake_pi().init_state(1)
    plain_peak, plain_end = hand_back(None, fresh)
    tracked_peak, tracked_end = hand_back(TRACKING, fresh)
    assert plain_peak > 0.65
    assert tracked_peak < 0.55
    assert max(plain_end, tracked_end) < 0.05


class _FixedResult(NamedTuple):
    env_action: Any
    a_expert: Any = None
    new_expert_state: Any = None


def test_tracking_waits_for_the_env_step() -> None:
    """The pipeline's memory stays the expert's own, so nothing is conditioned
    on a command the actuator has not answered; track_expert does it after."""
    law = fake_pi()
    extensions = (TRACKING,)
    kwargs = expert_collect_kwargs(
        law,
        lambda *_: _FixedResult(jnp.array([[0.3]])),
        False,
        1,
        extensions=extensions,
    )
    acting = ActingState(
        obs=jnp.array([[0.1, 0.5]]),
        done=jnp.zeros(1, bool),
        timestep=jnp.zeros((), jnp.int32),
        expert_state=law.init_state(1),
    )
    result = kwargs["action_pipeline"](
        None, acting, acting.obs, None, False, None, None
    )
    assert_same_tree(result.new_expert_state, law(acting.expert_state, acting.obs)[1])
    command, applied = jnp.array([[0.3]]), jnp.array([[0.09]])
    tracked = kwargs["track_expert"](acting, acting.obs, command, applied)
    np.testing.assert_allclose(tracked.integral, [0.15], rtol=1e-5)


def test_the_ablation_and_an_actuated_expert_have_no_tracking() -> None:
    """Off is nothing traced; and an expert actuated by the policy acts itself."""
    law = fake_pi()
    assert "track_expert" not in expert_collect_kwargs(law, None, False, 1)
    arm = reference()
    kwargs = expert_collect_kwargs(law, None, False, 1, arm, extensions=(TRACKING,))
    assert "track_expert" not in kwargs


class _HeldPlant(environment.Environment):
    """An observation of [0, 1] forever, a constant error of 1, whose actuator
    applies 0 whatever it is sent, and says so when it reports. It pays the
    action sent, so a return reads what an actor sent."""

    def __init__(self, reports: bool) -> None:
        super().__init__()
        self.reports = reports

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams(max_steps_in_episode=4)

    def reset_env(self, key: Any, params: Any) -> tuple:
        return jnp.array([0.0, 1.0]), environment.EnvState(time=0)

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> tuple:
        state = state.replace(time=state.time + 1)
        info = {"applied_action": jnp.zeros_like(action)} if self.reports else {}
        done = state.time >= params.max_steps_in_episode
        return jnp.array([0.0, 1.0]), state, jnp.sum(action), done, info

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> jax.Array:
        return jnp.array([0.0, 1.0])

    def observation_space(self, params: Any) -> spaces.Box:
        return spaces.Box(-jnp.inf, jnp.inf, (2,))

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,))


@pytest.mark.parametrize("reports", [True, False])
def test_collection_follows_the_env_report_or_else_the_command(reports: bool) -> None:
    """A warm-up's uniform actions. u_applied = 0 at e = 1 from I = 0 gives e_r =
    1 + (0 - 0.6) / 0.6 = 0, so I stays 0, step after step. Told the command
    instead, the memory takes each uniform draw, and does not stay there."""
    env = _HeldPlant(reports)
    agent = SAC(
        env,
        env_params=env.default_params,
        n_envs=1,
        expert_policy=fake_pi(),
        extensions=(TRACKING,),
        learning_starts=64,
        batch_size=4,
        buffer_size=32,
    )
    state, _ = agent.train(seed=0, n_timesteps=3)
    memory = state.collector_state.expert_state
    assert memory is not None
    integral = np.asarray(memory.integral)
    if reports:
        np.testing.assert_allclose(integral, 0.0, atol=1e-5)
    else:
        assert np.abs(integral).max() > 1e-3


@struct.dataclass
class _Actor:
    """An actor sending 0.3 plus its observation's last column, the expert's
    integral under ExpertStateInput."""

    params: Any = None
    obs_norm_info: Any = None
    hidden_state: Any = None

    def apply(self, params: Any, obs: jax.Array) -> Any:
        mean = 0.3 + obs[..., -1:]
        return distrax.Normal(mean, jnp.full(mean.shape, 1e-3))


def evaluated_return(reports: bool, tracker: Any) -> float:
    env, law = _HeldPlant(reports), fake_pi()
    controller = Controller(observers=(ExpertStateInput(width=1),), tracker=tracker)
    returns = evaluate(
        env,
        _Actor(),
        1,
        jax.random.PRNGKey(0),
        env.default_params,
        expert_policy=law,
        agent_state=None,
        controller=controller,
    )[0]
    return float(jnp.mean(returns))


def test_evaluation_follows_the_env_report() -> None:
    """Applied 0 at e = 1 leaves I at 0, as in collection, so the actor sends
    0.3 on all 4 steps."""
    assert evaluated_return(True, TRACKING) == pytest.approx(1.2, rel=1e-3)


def test_a_deployed_policy_follows_the_command() -> None:
    """No env has stepped when a deployed policy acts: e = 0.4, u = 0.24,
    command 0.3 gives e_r = 0.5."""

    @struct.dataclass
    class Actor(_Actor):
        def apply(self, params: Any, obs: jax.Array) -> Any:
            mean = jnp.full((obs.shape[0], 1), 0.3)
            return distrax.Normal(mean, jnp.full(mean.shape, 1e-3))

    law = fake_pi()
    controller = Controller(transforms=(_Keeps(READING),), tracker=TRACKING)
    act = make_act(Actor(), False, True, law, None, controller)
    obs = jnp.array([[0.1, 0.5]])
    acted = act(obs, jnp.zeros(1), obs, law.init_state(1), None, None, None)
    np.testing.assert_allclose(acted.expert_state.integral, [0.5], rtol=1e-5)


class _Keeps(BumplessTracking):
    """A transform leaving the action as it is, so the expert is called."""

    def implemented_phases(self) -> frozenset:
        return frozenset({"transform_action"})


def test_the_run_controller_carries_the_option() -> None:
    class Args:
        store_normalized_obs = False
        observers: tuple = ()
        transforms: tuple = ()

    assert make_run_controller(Args(), None, (TRACKING,)).tracker is TRACKING
    assert make_run_controller(Args(), None, ()).tracker is None


def test_the_option_refuses_a_stateless_expert_and_no_channels() -> None:
    with pytest.raises(ValueError, match="stateful"):
        TRACKING.bind_to_agent(expert_policy=lambda obs: obs)
    with pytest.raises(ValueError, match="reading_channels"):
        BumplessTracking(())


# -- through an agent -------------------------------------------------------
TASK = "four_tank"
ARMS = ["reference", "gains", "tracking"]


def arm_of(name: str, real: RealTask) -> tuple:
    """The agent's expert and extensions in the arm of that name."""
    if name == "reference":
        changes = tuple(0.01 for _ in real.targets)
        return real.expert, (OuterLoopReference(real.targets, changes),)
    if name == "gains":
        law = ExpertLaw.of(real.controller, REAL_GAINS[TASK])
        layer = GainScheduledLaw(law, real.rows.shape[-1])
        return layer, (ExpertGains(max_factor=2.0),)
    arms = (JSRLCurriculum(schedule="time"), BumplessTracking(real.readings))
    return real.expert, arms


@pytest.mark.parametrize("name", ARMS)
def test_sac_trains_through_each_form(name: str) -> None:
    """Bug caught: a wiring gap the units above cannot see, between the agent,
    the collector and the evaluation."""
    target_gym = pytest.importorskip("target_gym")
    env, params = target_gym.make(TASK, max_steps_in_episode=12)
    expert, extensions = arm_of(name, real_task(TASK))
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        expert_policy=expert,
        extensions=extensions,
        learning_starts=4,
        batch_size=4,
        buffer_size=32,
    )
    state, _ = agent.train(seed=0, n_timesteps=16)
    replay = state.collector_state.buffer_state
    assert replay is not None
    assert np.isfinite(np.asarray(replay.experience["reward"])).all()
    dim = agent.env_args.policy_action_dim
    assert dim == {"reference": 2, "gains": 4, "tracking": None}[name]
