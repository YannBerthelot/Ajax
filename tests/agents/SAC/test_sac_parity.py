"""Plain-SAC behaviour-parity gate for the Extension-framework migration.

These golden values were captured from ``train_SAC.py`` (the pre-refactor
2712-line SAC) on a fixed tiny config/seed, *before* the migration to
``sac.py`` + the Extension framework. Plain SAC with an empty extension
stack must reproduce them to within ``1e-5`` — this is the
behaviour-preservation contract for Phase 2 of the agent-architecture
rework.

If this test fails after a change to SAC, the change altered the plain
SAC algorithm's numerics — investigate to root cause; do not re-bless
the goldens without understanding why they moved.
"""

import jax.numpy as jnp
import pytest
from optax.tree_utils import tree_norm

from ajax.agents.SAC.SAC import SAC

# Fixed tiny config — small enough to run on CPU in a few seconds, large
# enough that the policy / alpha updates actually fire (start thresholds
# lowered to 20 so the update path is exercised).
_PARITY_CONFIG = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "learning_starts": 20,
    "batch_size": 16,
    "buffer_size": 400,
    "policy_update_start": 20,
    "alpha_update_start": 20,
}
_PARITY_SEED = 0
_PARITY_TIMESTEPS = 200

# Golden values — re-captured on the post-rename baseline (commit 267f2c1,
# "SAC: rename train_SAC.py → sac.py, extract feature builders"). The
# original pre-refactor capture drifted by ~1e-5 (critic) / ~6e-5 (actor)
# under a later JAX/XLA environment; bisecting the four refactor commits
# showed the math is bit-identical across the migration, so the drift is
# environmental (fp32 SAC over 200 steps accumulates rounding at this
# scale). These goldens are stable across every commit on
# ``agent-architecture-rework`` and across two consecutive runs (the
# determinism test is the cross-check). Tolerance is set to 1e-4: tight
# enough that any genuine algorithmic divergence in the proven SAC loop
# would fail it, loose enough to absorb fp32 environmental noise.
#
# Re-blessed once, deliberately, when gradient clipping was removed from
# every non-PPO agent (SAC actor / critic / temperature all ran through
# ``clip_by_global_norm(0.5)`` before). The clip on the scalar
# ``log_alpha`` gradient zeroed out the entropy-target term whenever
# |H_pi - H_target| > 0.5, so ``target_entropy_per_dim`` had no effect
# until alpha had crawled to the target at lr per update; the actor /
# critic clip was a PPO convention with no counterpart in reference SAC.
# Previous goldens: critic 1046.9158935546875, actor 517.13427734375,
# target 1035.4898681640625, alpha 0.973069429397583.
#
# Re-blessed again when the shared encoder's output LayerNorm became opt-in
# (SAC's networks are plain MLPs, as in Haarnoja et al.): the trees lost
# the norm's scales, 256 per encoder (critic 1042.7 -> 565.3, actor 518.2 ->
# 267.3, target 1034.9 -> 524.0), alpha moved 0.04% (0.97270 -> 0.97225).
#
# Re-blessed once SAC's [-1, 1] action was mapped onto Pendulum's [-2, 2]
# torque box (the policy and the uniform warm-up act at twice the torque):
# critic -0.27%, actor -0.081%, target -0.0078%, alpha +0.0028%. Previous
# goldens: critic 565.2509155273438, actor 267.29229736328125, target
# 523.9547119140625, alpha 0.9722516536712646.
#
# Re-blessed once SAC's layers started as its reference code's (haarnoja/sac
# and softlearning: xavier_uniform kernels, zero biases, the mean and log-std
# heads included; were orthogonal(1.0), an orthogonal(0.01) mean and a
# zero-kernel log-std head at -1): critic -1.03%, actor +0.06%, target
# +2.29%, alpha +0.08%. Previous goldens: critic 563.753662109375, actor
# 267.07598876953125, target 523.9136352539062, alpha 0.9722799062728882.
#
# Re-blessed once SAC's Adam eps became TensorFlow's 1e-8, softlearning's
# (was 1e-5): critic +0.024%, actor +0.014%, target +0.0019%, alpha
# +0.0003%. Previous goldens: critic 557.9498901367188, actor
# 267.2409973144531, target 535.9058837890625, alpha 0.9731040596961975.
#
# Re-blessed once the flat replay drew rows alone instead of a row with its
# successor, so each env's newest row is sampled too: critic -0.0054%,
# actor +0.019%, target -0.0012%, alpha -0.0055%. Previous goldens: critic
# 558.0853881835938, actor 267.2794189453125, target 535.9161376953125,
# alpha 0.973106861114502.
#
# Re-blessed once SAC's actor step took its keys from the state (get_keys_and_new_state),
# its random stream moving: critic +0.21%, actor -0.018%, target +0.015%,
# alpha +0.0089%. Previous goldens: critic 558.0552978515625, actor
# 267.3292236328125, target 535.9094848632812, alpha 0.9730522036552429.
#
# Re-blessed once SAC stopped splitting a discarded third key off its replay
# key: critic +0.039%, actor +0.039%, target -0.015%, alpha +0.002%.
# Previous goldens: critic 559.20361328125, actor 267.2802734375, target
# 535.9905395507812, alpha 0.973141610622406.
_GOLDEN = {
    "critic": 559.4210205078125,
    "actor": 267.3836669921875,
    "target": 535.9125366210938,
    "alpha": 0.973161518573761,
}
_TOL = 1e-4


def _run_plain_sac():
    agent = SAC(**_PARITY_CONFIG)
    state, _metrics = agent.train(seed=_PARITY_SEED, n_timesteps=_PARITY_TIMESTEPS)
    return state


@pytest.fixture(scope="module")
def plain_sac_state():
    """One plain-SAC run, shared by the golden and determinism tests."""
    return _run_plain_sac()


def test_plain_sac_matches_golden_values(plain_sac_state):
    """Plain SAC (empty extension stack) reproduces the pre-refactor goldens."""
    state = plain_sac_state
    measured = {
        "critic": tree_norm(state.critic_state.params, squared=True),
        "actor": tree_norm(state.actor_state.params, squared=True),
        "target": tree_norm(state.critic_state.target_params, squared=True),
        "alpha": jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0],
    }
    for key, golden in _GOLDEN.items():
        got = float(measured[key])
        rel = abs(got - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, (
            f"plain-SAC parity broken for {key!r}: got {got!r}, "
            f"golden {golden!r} (relative error {rel:.2e} >= {_TOL:.0e})"
        )


def test_plain_sac_is_deterministic(plain_sac_state):
    """A second identical run of plain SAC reproduces the shared run."""
    s1 = plain_sac_state
    s2 = _run_plain_sac()
    for name, getter in (
        ("critic", lambda s: s.critic_state.params),
        ("actor", lambda s: s.actor_state.params),
        ("alpha", lambda s: s.alpha.params),
    ):
        v1 = tree_norm(getter(s1), squared=True)
        v2 = tree_norm(getter(s2), squared=True)
        assert jnp.allclose(v1, v2), f"plain SAC not reproducible for {name}"
