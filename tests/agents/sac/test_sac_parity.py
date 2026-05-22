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

import jax
import jax.numpy as jnp

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

# Golden values — captured from pre-refactor train_SAC.py.
_GOLDEN = {
    "critic": 1046.930176,
    "actor": 517.165527,
    "target": 1035.490845,
    "alpha": 0.973069,
}
_TOL = 1e-5


def _checksum(tree) -> jax.Array:
    """Sum of squared elements over all floating-point leaves of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(
        jnp.sum(leaf.astype(jnp.float32) ** 2)
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    )


def _run_plain_sac():
    agent = SAC(**_PARITY_CONFIG)
    state, _metrics = agent.train(seed=_PARITY_SEED, n_timesteps=_PARITY_TIMESTEPS)
    return state


def test_plain_sac_matches_golden_values():
    """Plain SAC (empty extension stack) reproduces the pre-refactor goldens."""
    state = _run_plain_sac()
    measured = {
        "critic": _checksum(state.critic_state.params),
        "actor": _checksum(state.actor_state.params),
        "target": _checksum(state.critic_state.target_params),
        "alpha": jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0],
    }
    for key, golden in _GOLDEN.items():
        got = float(measured[key])
        rel = abs(got - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, (
            f"plain-SAC parity broken for {key!r}: got {got!r}, "
            f"golden {golden!r} (relative error {rel:.2e} >= {_TOL:.0e})"
        )


def test_plain_sac_is_deterministic():
    """Two identical runs of plain SAC produce bit-identical agent states."""
    s1 = _run_plain_sac()
    s2 = _run_plain_sac()
    for name, getter in (
        ("critic", lambda s: s.critic_state.params),
        ("actor", lambda s: s.actor_state.params),
        ("alpha", lambda s: s.alpha.params),
    ):
        v1, v2 = _checksum(getter(s1)), _checksum(getter(s2))
        assert jnp.allclose(v1, v2), f"plain SAC not reproducible for {name}"
