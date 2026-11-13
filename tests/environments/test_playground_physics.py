"""Which physics a Playground task runs on.

Contract (from the task: Ajax crashed on a GPU, and the CPU benchmark's
numbers all come from MJX's pure-JAX physics):

- every device gets pure-JAX physics unless the caller asks for Warp;
- ``physics="warp"`` reaches Playground;
- the evaluation rebuild keeps the training environment's physics;
- an unknown name is refused rather than passed on to Playground;
- Warp is refused, with its reason, where it cannot run: without a GPU (its
  kernels are CUDA only), and wherever gradients must cross the physics
  (``differentiable_reset``, APG), as MJX gives Warp no gradient rule;
- a Warp step that ran out of contact, convex-collision or constraint room
  stops the run rather than going on with dropped contacts;
- on Warp, a ``"mujoco:"`` task's physics call allocates convex-collision
  scratch for the task's geom pairs, not for the whole contact buffer (each
  CUDA graph Warp keeps reserves addresses for a call's scratch).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from ajax.environments.create import build_env_from_id
from ajax.evaluate import setup_environment

registry = pytest.importorskip("mujoco_playground").registry

TASK = "CartpoleBalance"


def physics_of(env) -> str:
    return env.unwrapped.mjx_model.impl.value


def test_a_gpu_does_not_fall_through_to_warp(monkeypatch):
    """Catches the crash on GPU: choosing pure JAX on a CPU only left a GPU on
    Playground's Warp default. No test machine has a GPU, so the backend query
    is stubbed to reach that path."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    env, _ = build_env_from_id(TASK)

    assert physics_of(env) == "jax"


@pytest.fixture
def physics_requested(monkeypatch):
    """Records the physics each build asks Playground for. Warp needs a GPU to
    build and no test machine has one, so the backend query is stubbed to pass
    the GPU check and the request is what can be observed: every build
    underneath stays pure JAX."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    requested = []
    load = registry.load

    def recording_load(env_id, config_overrides):
        requested.append(config_overrides["impl"])
        return load(env_id, config_overrides={"impl": "jax"})

    monkeypatch.setattr(registry, "load", recording_load)
    return requested


def test_physics_warp_reaches_playground(physics_requested):
    """Catches an option that is accepted and then ignored."""
    build_env_from_id(TASK, physics="warp")

    assert physics_requested == ["warp"]


def test_evaluation_keeps_the_training_physics(physics_requested):
    """Catches evaluating on other physics than the agent trained on: the
    rebuild starts again from the task's id."""
    env, env_params = build_env_from_id(TASK, physics="warp")

    setup_environment(env, env_params, num_episodes=2, norm_info=None, gamma=0.99)

    assert physics_requested == ["warp", "warp"]


def test_an_unknown_physics_is_refused():
    """Catches a misspelt name reaching Playground, whose error names neither
    the option nor the choices."""
    with pytest.raises(ValueError, match="physics"):
        build_env_from_id(TASK, physics="mujoco")


def test_warp_without_a_gpu_is_refused_with_the_reason(monkeypatch):
    """Catches Warp reaching MJX on a CPU, which fails deep in tracing with
    "Unknown backend cuda", naming neither the option nor a way out."""
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    with pytest.raises(ValueError, match="needs an NVIDIA GPU"):
        build_env_from_id(TASK, physics="warp")


def test_warp_with_gradients_through_the_reset_is_refused(physics_requested):
    """Catches a differentiable reset accepted on Warp, which has no gradient
    rule: the request fails before any build."""
    with pytest.raises(ValueError, match="cannot be differentiated through"):
        build_env_from_id(TASK, physics="warp", differentiable_reset=True)

    assert physics_requested == []


def test_apg_refuses_warp_physics_with_the_reason(physics_requested):
    """Catches APG, which back-propagates through the simulator, taking Warp
    physics without saying that Warp is why."""
    from ajax.agents.APG.APG import APG

    env, _ = build_env_from_id(TASK, n_envs=2, physics="warp")

    with pytest.raises(ValueError, match="cannot be differentiated through"):
        APG(env, n_envs=2)


@struct.dataclass
class _WarpCounters:
    """MuJoCo Warp's counters and capacities, as MJX keeps them in Data._impl."""

    nacon: jax.Array
    ncollision: jax.Array
    naconmax: jax.Array
    nefc: jax.Array
    njmax: jax.Array
    overflow: jax.Array
    naccdmax: jax.Array


@struct.dataclass
class _WarpData:
    _impl: _WarpCounters


@struct.dataclass
class _CountingState:
    data: _WarpData


class _CountingEnv:
    """A stand-in Warp env whose step reports the counts it is given: the
    batch's contacts and broadphase pairs, each world's constraint rows, and
    each world's overflow flags (MuJoCo Warp's OverflowType bits)."""

    def reset(self, rng):
        return self.step(None, (0, 0, 0, 0))

    def step(self, state, counts):
        contacts, pairs, rows, flags = counts
        warp = _WarpCounters(
            nacon=jnp.array([contacts]),
            ncollision=jnp.array([pairs]),
            naconmax=jnp.array(48),
            nefc=jnp.full(3, rows),
            njmax=jnp.array(64),
            overflow=jnp.array([0, flags, 0]),
            naccdmax=jnp.array(21),
        )
        return _CountingState(_WarpData(warp))


def _overflow_bits() -> tuple[int, int]:
    """MuJoCo Warp's flags for a convex (CCD) buffer that ran out, and for a
    solver that hit its iteration limit, which drops nothing."""
    from mujoco.mjx.third_party.mujoco_warp import OverflowType

    return int(OverflowType.CCD), int(OverflowType.ITERATIONS)


CONVEX, ITERATIONS = _overflow_bits()


@pytest.mark.parametrize(
    "counts, dropped",
    [
        ((49, 0, 0, 0), "contacts"),
        ((0, 49, 0, 0), "contacts"),
        ((0, 0, 65, 0), "constraints"),
        ((0, 0, 0, CONVEX), "convex collisions"),
    ],
)
def test_a_warp_step_that_ran_out_of_room_stops_the_run(counts, dropped):
    """Catches Warp dropping contacts, constraints or convex collisions with
    only a printed warning: with too many worlds for its contact buffer the
    physics went on wrong (47 Hopper worlds overflowed the default 48
    contacts). Convex collisions have no counter, only a per-world flag."""
    from ajax.wrappers import WarpCapacityWrapper

    env = WarpCapacityWrapper(_CountingEnv())
    step = jax.jit(env.step, static_argnums=1)
    jax.block_until_ready(step(None, (48, 48, 64, ITERATIONS)))  # full, not over

    with pytest.raises(jax.errors.JaxRuntimeError, match=f"Warp dropped {dropped}"):
        jax.block_until_ready(step(None, counts))


def test_only_a_warp_stack_checks_its_room(physics_requested):
    """Catches the check missing from the Warp stack, or costing pure JAX a
    host callback on every step."""
    from ajax.environments.utils import wrapper_chain
    from ajax.wrappers import WarpCapacityWrapper

    def checks(env) -> bool:
        return any(
            isinstance(layer, WarpCapacityWrapper) for layer in wrapper_chain(env)
        )

    assert checks(build_env_from_id(TASK, physics="warp")[0])
    assert not checks(build_env_from_id(TASK, physics="jax")[0])


def warp_scratch_megabytes(text: str) -> float:
    """The CPU peak of a ``wp.ScopedMemoryTracker`` report, in MB."""
    units = {"B": 1e-6, "KB": 1e-3, "MB": 1.0, "GB": 1e3}
    lines = text.splitlines()
    start = lines.index("  Peak usage:") + 1
    for line in lines[start:]:
        fields = line.split()
        if fields[0] == "cpu":
            return float(fields[1]) * units[fields[2]]
    raise ValueError(f"no CPU peak in the report:\n{text}")


@pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="on a GPU, Warp replays a captured CUDA graph: no allocation to count",
)
def test_a_pusher_physics_call_on_warp_allocates_scratch_for_its_pairs_only():
    """Catches the convex-collision scratch sized for the whole contact
    buffer: MuJoCo Warp sizes it for 49,152 collisions unless told, and
    every Pusher physics call allocated 235 MB (320 MB on a GPU). Warp keeps
    a CUDA graph per call and per new set of XLA buffer addresses, up to 32
    per Warp function, and each graph reserves its own addresses for the
    scratch: PPO's benchmark run added 5 graphs per evaluation, and on an
    11 GB GPU new ones failed after about 76 (21 GiB) with under 1.4 GB in
    use. Room for Pusher's 7 geom pairs in 1,024 worlds takes 47 MB at most;
    below 100 MB, the 96 graphs of three Warp functions fit in half of that
    21 GiB, and the old size fails. MJX runs Warp's kernels on a CPU when a
    task is made directly (no CUDA graphs), where Warp's tracker sees every
    allocation."""
    import warp as wp

    from ajax.environments.mujoco.tasks import make

    task = make("Pusher-v5", physics="warp")
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states = jax.jit(jax.vmap(task.reset))(keys)
    step = jax.jit(jax.vmap(task.step))
    actions = jnp.ones((2, task.action_size))
    jax.block_until_ready(step(states, actions))  # compiled outside the count

    reports: list[str] = []
    with wp.ScopedMemoryTracker("pusher", report_func=reports.append) as tracker:
        tracker.clear()
        jax.block_until_ready(step(states, actions))

    assert warp_scratch_megabytes(reports[0]) < 100


#: A Pusher state whose forearm and wrist touch the object (5 contacts, among
#: them capsule-on-cylinder ones that MuJoCo Warp's convex code handles),
#: found by sampling arm poses in C MuJoCo with the object under the tip.
PUSHER_ARM_ON_OBJECT = [0.633, 0.52, 1.492, -0.427, -1.492, -0.156, -1.399]
PUSHER_OBJECT_UNDER_TIP = [-0.24, 0.198]


def test_a_batch_beyond_the_convex_room_stops_the_run(monkeypatch):
    """Catches convex collisions dropped unnoticed, which no counter shows:
    with no room for them, Pusher's arm pressed on its object must stop
    the run, read from the overflow flag MuJoCo Warp itself sets."""
    from ajax.environments.mujoco import tasks
    from ajax.wrappers import WarpCapacityWrapper

    monkeypatch.setattr(tasks, "WARP_WORLDS", 0)
    task = tasks.make("Pusher-v5", physics="warp")
    env = WarpCapacityWrapper(task)
    qpos = jnp.array(PUSHER_ARM_ON_OBJECT + PUSHER_OBJECT_UNDER_TIP + [0.0, 0.0])

    def touching_step(key: jax.Array):
        state = task.reset(key)
        state = state.replace(data=task.set_state(qpos, task.init_qvel))
        return env.step(state, jnp.zeros(task.action_size))

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    with pytest.raises(
        jax.errors.JaxRuntimeError, match="Warp dropped convex collisions"
    ):
        jax.block_until_ready(jax.jit(jax.vmap(touching_step))(keys))
