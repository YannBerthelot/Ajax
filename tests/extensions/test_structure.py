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

The fake below is a PI on obs = [reading, target], limited to [-1, 1], which
holds its integral at the limit. Its numbers are worked by hand in each test.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax import SAC
from ajax.controller import Controller, TransformInputs, make_run_controller
from ajax.environments.interaction import ActingState, expert_collect_kwargs
from ajax.environments.utils import ExpertInAgentUnits
from ajax.extensions.expert import ExpertGains, JSRLCurriculum
from ajax.extensions.structure import (
    BumplessTracking,
    ExpertLaw,
    GainScheduledLaw,
    OuterLoopReference,
    conditioned_memory,
)
from ajax.policy import make_act

KP, KI = 0.5, 0.1
D = KP + KI  # the feedthrough, at dt = 1


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


def fake_pi(**names: Any) -> ExpertLaw:
    names = {"reading_channels": (0,), "gain_names": ("Kp", "Ki"), **names}
    return ExpertLaw(fake_pi_step, PIParams(KP, KI), PIMemory(0.0), **names)


def scheduled(law: ExpertLaw, **options: Any) -> GainScheduledLaw:
    return GainScheduledLaw(law, obs_width=2, **options)


def observations(*errors: float) -> list[jax.Array]:
    """One env's observations with these errors, the reading at 0."""
    return [jnp.array([[0.0, error]]) for error in errors]


def assert_same_tree(left: Any, right: Any) -> None:
    jax.tree.map(np.testing.assert_array_equal, left, right)


def real_law(task: str, **names: Any) -> tuple[ExpertLaw, jax.Array, Any]:
    """task's scored PID as a law, with observations along its closed loop:
    (n_steps, n_envs, obs_width), at the task's own float32."""
    target_gym = pytest.importorskip("target_gym")
    registry = pytest.importorskip("target_gym.registry")
    env, params = target_gym.make(task)
    law = ExpertLaw.of(registry.REGISTRY[task].make_pid_controller(), **names)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, state = jax.vmap(env.reset, (0, None))(keys, params)
    memory = law.init_state(3)
    rows = []
    for index in range(32):
        obs = obs.astype(jnp.float32)
        rows.append(obs)
        action, memory = law(memory, obs)
        keys = jax.random.split(jax.random.PRNGKey(index + 1), 3)
        obs, state, *_ = jax.vmap(env.step, (0, 0, 0, None))(
            keys, state, action, params
        )
    return law, jnp.stack(rows), memory


#: Each real PID's channels, read off its law and observation layout.
REAL_LAWS = {
    "four_tank": {
        "reading_channels": (0, 1),
        "gain_names": ("Kp_1", "Ti_1", "Kp_2", "Ti_2"),
    },
    "cstr": {"reading_channels": (0,), "gain_names": ("Kp", "Ti", "Kp_T", "Ti_T")},
}
REAL_TARGETS = {"four_tank": (4, 5), "cstr": (2,)}


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


@pytest.mark.parametrize("task", sorted(REAL_LAWS))
def test_a_zero_change_is_the_real_pid_to_the_bit(task: str) -> None:
    law, rows, memory = real_law(task, **REAL_LAWS[task])
    channels = REAL_TARGETS[task]
    arm = reference(channels, tuple(0.01 for _ in channels))
    inputs = TransformInputs(law, memory, rows[-1], None)
    actions = jnp.zeros((3, len(channels)))
    assert_same_tree(arm.actuate(actions, inputs), law(memory, rows[-1]))


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
    rows = observations(0.2, 3.0, 3.0, 0.1, -0.2, -4.0, 0.3)
    layer, memory = scheduled(law), law.init_state(1)
    state = layer.init_state(1)
    for obs in rows:
        action, state = layer.step_with_gains(state, obs, jnp.ones((1, 2)))
        own_action, memory = law(memory, obs)
        assert_same_tree(action, own_action)
    assert_same_tree(state.memory, memory)


@pytest.mark.parametrize("task", sorted(REAL_LAWS))
def test_unit_multipliers_are_the_real_pid_to_float32_rounding(task: str) -> None:
    """Not to the bit: these laws divide by Ti, and XLA rewrites a division by
    one number as a product with its reciprocal, which a per-env gain escapes.
    One unit of rounding near 1 is 1.2e-7; 32 steps may add some."""
    law, rows, _ = real_law(task, **REAL_LAWS[task])
    layer = GainScheduledLaw(law, obs_width=rows.shape[-1])
    state, memory = layer.init_state(3), law.init_state(3)
    multipliers = jnp.ones((3, len(law.gain_names)))
    for obs in rows:
        action, state = layer.step_with_gains(state, obs, multipliers)
        own_action, memory = law(memory, obs)
        np.testing.assert_allclose(action, own_action, atol=2e-6)
    jax.tree.map(
        lambda left, right: np.testing.assert_allclose(left, right, rtol=1e-5),
        state.memory,
        memory,
    )


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
    position_form_jump = KI * 0.5
    assert position_form_jump > 0.04  # what the test would see without the bias


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


def test_the_policy_acts_in_the_width_of_the_gains() -> None:
    layer = scheduled(fake_pi())
    assert layer.learnable_fields == ("Kp", "Ki")
    np.testing.assert_array_equal(layer.anchor_gains, [1.0, 1.0])
    assert ExpertGains().policy_action_dim(_NoFeatures(), layer) == 2


class _NoFeatures:
    observers: tuple = ()


def test_a_law_at_its_limit_is_not_held_off_it_by_an_old_bias() -> None:
    """A velocity form on a law that holds its integral at its limit could
    stick: the law flat at 1, its integral held, the bias keeping u below 1.
    Halving the bias per step ends it in the time the halving says."""
    layer = scheduled(fake_pi(), max_log_rate=1.0, bias_retention=0.5)
    rows = observations(0.5, 2.0, *[2.0] * 12)
    factors = [[1.0, 1.0], [0.5, 1.0], *[[0.5, 1.0]] * 12]
    actions = rollout(layer, rows, factors)
    assert actions[-1] == pytest.approx(1.0, abs=1e-3)


def test_a_law_returning_a_number_acts_in_a_column() -> None:
    """TargetGym's battery returns one number, not a column, which an agent's
    (n_envs, n_actions) action refuses."""

    def scalar_step(params: PIParams, memory: PIMemory, obs: jax.Array) -> Any:
        action, memory = fake_pi_step(params, memory, obs)
        return action[0], memory

    law = ExpertLaw(
        scalar_step, PIParams(KP, KI), PIMemory(0.0), (0,), gain_names=("Kp", "Ki")
    )
    layer = scheduled(law)
    obs = jnp.array([[0.0, 0.2], [0.0, 0.4]])
    action, state = layer.step_with_gains(layer.init_state(2), obs, jnp.ones((2, 2)))
    assert action.shape == (2, 1)
    assert law(law.init_state(2), obs)[0].shape == (2, 1)
    np.testing.assert_allclose(action, [[0.12], [0.24]], rtol=1e-5)
    tracked = law.track(state.last_memory, obs, jnp.array([[0.12], [0.3]]))
    np.testing.assert_allclose(tracked.integral, [0.2, 0.5], rtol=1e-5)


def test_an_unknown_gain_name_is_refused() -> None:
    with pytest.raises(ValueError, match="no gains"):
        fake_pi(gain_names=("Kd",))


def test_a_nested_gain_is_found_by_its_dotted_name() -> None:
    """TargetGym's aircraft nest theirs (tecs.time_const)."""

    @struct.dataclass
    class Nested:
        inner: PIParams

    def step(params: Nested, memory: PIMemory, obs: jax.Array) -> Any:
        return fake_pi_step(params.inner, memory, obs)

    nested = Nested(PIParams(KP, KI))
    law = ExpertLaw(step, nested, PIMemory(0.0), gain_names=("inner.Kp",))
    layer = GainScheduledLaw(law, obs_width=2, max_log_rate=1.0)
    action, _ = layer.step_with_gains(
        layer.init_state(1), jnp.array([[0.0, 1.0]]), jnp.array([[2.0]])
    )
    # e = 1: 2 * 0.5 * 1 + 0.1 * 1 = 1.1, past the limit of 1.
    np.testing.assert_allclose(action, [[1.0]])


# -- bumpless tracking -----------------------------------------------------
def test_the_laws_own_action_leaves_its_step_as_it_is() -> None:
    """Bug caught: tracking changing a law that is acting. At the limit too:
    its own anti-windup stands (e = 5 saturates, the integral holds)."""
    law = fake_pi()
    for error in (0.2, 5.0):
        obs = jnp.array([[0.0, error]])
        memory = PIMemory(jnp.array([0.3]))
        action, own_memory = law(memory, obs)
        assert_same_tree(law.track(memory, obs, action), own_memory)


def test_the_memory_takes_the_realisable_error() -> None:
    """e = 0.4, I = 0: u = 0.24. u_executed = 0.09 gives e_r = 0.4 + (0.09 -
    0.24) / 0.6 = 0.15, so I+ = 0.15."""
    law = fake_pi()
    obs = jnp.array([[0.1, 0.5]])
    tracked = law.track(law.init_state(1), obs, jnp.array([[0.09]]))
    np.testing.assert_allclose(tracked.integral, [0.15], rtol=1e-5)


def takeover(law: ExpertLaw, error: float, executed: float, steps: int) -> tuple:
    """The integral of the tracked and of the plain law, while another
    controller holds the plant at this error and action."""
    obs = jnp.array([[0.0, error]])
    tracked = plain = law.init_state(1)
    for _ in range(steps):
        tracked = law.track(tracked, obs, jnp.array([[executed]]))
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


@pytest.mark.parametrize("task", sorted(REAL_LAWS))
def test_the_real_pid_steps_at_the_realisable_reading_to_the_executed_action(
    task: str,
) -> None:
    """Its law is affine in its readings below its limits, so one Newton step
    lands: the expert's output at the shifted reading is the executed action,
    within float32's reach on these gains."""
    law, rows, memory = real_law(task, **REAL_LAWS[task])
    obs = rows[-1]
    own_action, _ = law(memory, obs)
    executed = jnp.clip(own_action + 0.02, -0.95, 0.95)
    shifted = _realisable_output(law, memory, obs, executed)
    np.testing.assert_allclose(shifted, executed, atol=1e-3)


def _realisable_output(law: ExpertLaw, memory: Any, obs: Any, executed: Any) -> Any:
    """The law's output at the reading conditioned_memory's shift gives: it
    is found by asking for the memory with a recording step."""

    def one_env(state: Any, reading: Any, action: Any) -> Any:
        recorded = {}

        def recording(params: Any, inner: Any, shifted: Any) -> Any:
            out = law.step(params, inner, shifted)
            recorded["action"] = out[0]
            return out

        conditioned_memory(
            recording, law.params, state, reading, action, law.reading_channels
        )
        return recorded["action"]

    return jax.vmap(one_env)(memory, obs, executed)


@pytest.mark.parametrize("task", sorted(REAL_LAWS))
def test_a_takeover_of_the_real_pid_stays_nearer_the_executed_action(
    task: str,
) -> None:
    """Bug caught: a law whose memory ignores the takeover. The same plant-free
    takeover from one state: readings frozen, an action held 0.2 away. The
    tracked law's hand-back is nearer it than the plain law's."""
    law, rows, memory = real_law(task, **REAL_LAWS[task])
    obs = rows[-1]
    executed = jnp.clip(law(memory, obs)[0] - 0.2, -0.9, 0.9)
    tracked = plain = memory
    for _ in range(64):
        tracked = law.track(tracked, obs, executed)
        _, plain = law(plain, obs)
    tracked_gap = jnp.abs(law(tracked, obs)[0] - executed).mean()
    plain_gap = jnp.abs(law(plain, obs)[0] - executed).mean()
    assert tracked_gap < plain_gap


def test_tracking_reads_the_executed_action_back_in_env_units() -> None:
    """Bug caught: an agent-unit action fed to an env-unit law. Pendulum's box
    is [-2, 2], so agent 0.5 is env 1, and a law told 1 tracks that."""

    class Box:
        low, high, shape = jnp.full((1,), -2.0), jnp.full((1,), 2.0), (1,)

    class Env:
        def action_space(self, params: Any = None) -> Box:
            return Box()

    law = fake_pi()
    wrapped = ExpertInAgentUnits(law, Env(), None)
    obs, memory = jnp.array([[0.1, 0.5]]), law.init_state(1)
    in_agent_units = wrapped.track(memory, obs, jnp.array([[0.5]]))
    assert_same_tree(in_agent_units, law.track(memory, obs, jnp.array([[1.0]])))
    assert hasattr(wrapped, "track")
    assert not hasattr(ExpertInAgentUnits(lambda o: o, Env(), None), "track")


def collected_memory(extensions: tuple) -> tuple:
    """A pipeline step that executes 0.3 where the expert's own is 0.24."""
    law = fake_pi()

    class Result(NamedTuple):
        env_action: Any
        a_expert: Any = None
        new_expert_state: Any = None

    def fixed_pipeline(*args: Any) -> Result:
        del args
        return Result(jnp.array([[0.3]]))

    pipeline = expert_collect_kwargs(
        law, fixed_pipeline, False, 1, extensions=extensions
    )["action_pipeline"]
    acting = ActingState(
        obs=jnp.array([[0.1, 0.5]]),
        done=jnp.zeros(1, bool),
        timestep=jnp.zeros((), jnp.int32),
        expert_state=law.init_state(1),
    )
    result = pipeline(None, acting, acting.obs, None, False, None, None)
    return law, acting, result


def test_collection_tracks_the_executed_action_under_the_option() -> None:
    """e = 0.4, u = 0.24, u_executed = 0.3: e_r = 0.4 + 0.06 / 0.6 = 0.5."""
    _, _, result = collected_memory((BumplessTracking(),))
    np.testing.assert_allclose(result.new_expert_state.integral, [0.5], rtol=1e-5)


def test_collection_leaves_the_memory_as_it_was_without_the_option() -> None:
    """The ablation: memory advances on the error whoever acts."""
    law, acting, result = collected_memory(())
    assert_same_tree(result.new_expert_state, law(acting.expert_state, acting.obs)[1])


class _Actor:
    """An actor whose deterministic action is 0.3."""

    def apply(self, params: Any, obs: jax.Array) -> Any:
        del params
        return _Policy(obs.shape[0])

    def replace(self, **fields: Any) -> "_Actor":
        del fields
        return self

    params = None


class _Policy:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs

    def mean(self) -> jax.Array:
        return jnp.full((self.n_envs, 1), 0.3)

    def entropy(self) -> jax.Array:
        return jnp.zeros((self.n_envs,))


class _LeaveAlone(BumplessTracking):
    """A transform leaving the action as it is, so the expert is called."""

    def implemented_phases(self) -> frozenset:
        return frozenset({"transform_action"})


@pytest.mark.parametrize("tracks", [True, False])
def test_evaluation_tracks_the_executed_action_under_the_option(tracks: bool) -> None:
    """e = 0.4, u = 0.24, u_executed = 0.3: e_r = 0.5, as in collection; the
    plain law integrates e = 0.4."""
    law = fake_pi()
    controller = Controller(transforms=(_LeaveAlone(),), tracks_executed=tracks)
    act = make_act(_Actor(), False, True, law, None, controller)
    obs = jnp.array([[0.1, 0.5]])
    acted = act(obs, jnp.zeros(1), obs, law.init_state(1), None, None, None)
    expected = 0.5 if tracks else 0.4
    np.testing.assert_allclose(acted.expert_state.integral, [expected], rtol=1e-5)


def test_the_run_controller_carries_the_option() -> None:
    class Args:
        store_normalized_obs = False
        observers: tuple = ()
        transforms: tuple = ()

    assert make_run_controller(Args(), None, (BumplessTracking(),)).tracks_executed
    assert not make_run_controller(Args(), None, ()).tracks_executed


def test_the_option_refuses_an_expert_without_a_track() -> None:
    with pytest.raises(ValueError, match="BumplessTracking needs"):
        BumplessTracking().bind_to_agent(expert_policy=lambda state, obs: (obs, state))


# -- through an agent -------------------------------------------------------
TASK = "four_tank"
ARMS = ["reference", "gains", "tracking"]


def arm(name: str, law: ExpertLaw, width: int) -> tuple:
    """The agent's expert and extensions in the arm of that name."""
    if name == "reference":
        return law, (OuterLoopReference(REAL_TARGETS[TASK], (0.01, 0.01)),)
    if name == "gains":
        return GainScheduledLaw(law, width), (ExpertGains(max_factor=2.0),)
    return law, (JSRLCurriculum(schedule="time"), BumplessTracking())


@pytest.mark.parametrize("name", ARMS)
def test_sac_trains_through_each_form(name: str) -> None:
    """Bug caught: a wiring gap the units above cannot see, between the agent,
    the collector and the evaluation."""
    target_gym = pytest.importorskip("target_gym")
    registry = pytest.importorskip("target_gym.registry")
    env, params = target_gym.make(TASK, max_steps_in_episode=12)
    pid = registry.REGISTRY[TASK].make_pid_controller()
    law = ExpertLaw.of(pid, **REAL_LAWS[TASK])
    expert, extensions = arm(name, law, env.observation_space(params).shape[0])
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
