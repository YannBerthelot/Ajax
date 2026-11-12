"""Which physics a Playground task runs on.

Contract (from the task: Ajax crashed on a GPU, and the CPU benchmark's
numbers all come from MJX's pure-JAX physics):

- every device gets pure-JAX physics unless the caller asks for Warp;
- ``physics="warp"`` reaches Playground;
- the evaluation rebuild keeps the training environment's physics;
- an unknown name is refused rather than passed on to Playground;
- Warp is refused, with its reason, where it cannot run: without a GPU (its
  kernels are CUDA only), and wherever gradients must cross the physics
  (``differentiable_reset``, APG), as MJX gives Warp no gradient rule.
"""

import jax
import pytest

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
