"""Behaviour-equivalence tests for the SAC ``extensions=`` surface.

For each migrated extension, configuring SAC via the new
``extensions=[...]`` surface must produce byte-identical numerics to
configuring SAC via the equivalent legacy boolean-flag surface. This is
the behaviour-preservation contract for Phase 2 of the agent-architecture
rework: the resolver in :mod:`ajax.agents.SAC.sac._resolve_extension_stack`
translates each :class:`~ajax.extensions.base.Extension` config object
into the matching legacy ``make_train`` kwargs.

The tests deliberately use a tiny config so they run in a few seconds on
CPU; the checksum is a strong-enough fingerprint to catch any silent
divergence in the resolver mapping or in extension wiring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.extensions.expert import (
    ExpertGuidance,
    ExpertObsAugmentation,
    JSRLCurriculum,
    OnlineBC,
    ResidualPolicy,
)
from ajax.extensions.target_mods import IBRL

# --------------------------------------------------------------------------
# Test infrastructure
# --------------------------------------------------------------------------

_TINY = {
    "env_id": "Pendulum-v1",
    "n_envs": 1,
    "learning_starts": 20,
    "batch_size": 16,
    "buffer_size": 400,
    "policy_update_start": 20,
    "alpha_update_start": 20,
}
_TIMESTEPS = 80
_SEED = 0


def _checksum(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(
        sum(
            jnp.sum(leaf.astype(jnp.float32) ** 2)
            for leaf in leaves
            if jnp.issubdtype(leaf.dtype, jnp.floating)
        )
    )


def _agent_fingerprint(state) -> dict:
    return {
        "critic": _checksum(state.critic_state.params),
        "actor": _checksum(state.actor_state.params),
        "target": _checksum(state.critic_state.target_params),
        "alpha": float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0]),
    }


def _assert_same(legacy_state, new_state, label: str):
    legacy = _agent_fingerprint(legacy_state)
    new = _agent_fingerprint(new_state)
    for key, lo in legacy.items():
        rel = abs(new[key] - lo) / max(abs(lo), 1.0)
        assert rel < 1e-5, (
            f"{label}: extension surface diverged from legacy flags for {key!r}: "
            f"legacy={lo!r}, new={new[key]!r}, rel error {rel:.2e}"
        )


class _NoiseExpert:
    """A deterministic 'expert' for tests: ``tanh(sum(obs))`` action.

    Defined as a class so it supplies both the stateless call
    ``expert(obs)`` (used by the action pipeline, MC pretrain, IBRL
    target modifier) AND the stateful call ``expert(state, obs)`` (used
    by the eval loop), plus ``init_state(n_envs)``. The internal
    "state" is just a unit pytree carried unchanged so its presence
    does not perturb numerics.
    """

    def __init__(self, action_dim: int = 1):
        self.action_dim = action_dim

    def __call__(self, *args):
        if len(args) == 1:
            obs = args[0]
            return jnp.tanh(jnp.sum(obs, axis=-1, keepdims=True))
        state, obs = args
        return jnp.tanh(jnp.sum(obs, axis=-1, keepdims=True)), state

    def init_state(self, n_envs: int):
        return jnp.zeros((n_envs, 1), dtype=jnp.float32)


def _noise_expert(seed: int = 0):
    del seed
    return _NoiseExpert()


# --------------------------------------------------------------------------
# Plain SAC: empty stack equals no extensions equals legacy defaults
# --------------------------------------------------------------------------
def test_empty_stack_matches_legacy_plain_sac():
    legacy = SAC(**_TINY)
    new = SAC(**_TINY, extensions=())
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "empty_stack")


# --------------------------------------------------------------------------
# Expert-guidance base extension equals legacy ``expert_policy=``
# --------------------------------------------------------------------------
def test_expert_guidance_matches_legacy_expert_policy():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_expert_guidance=False,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
                use_expert_guidance=False,
            ),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "expert_guidance")


# --------------------------------------------------------------------------
# Online-BC equals legacy ``use_online_bc=True``
# --------------------------------------------------------------------------
def test_online_bc_matches_legacy_flag():
    """OnlineBC behaves like ``use_online_bc=True`` (the legacy default).

    BC is only active when ``expert_critic_params`` is populated (MC
    pre-training). Without MC pre-training the BC term is silently
    skipped — so this test exercises the resolver wiring rather than the
    BC term itself; the BC math is already covered by
    ``tests/modules/test_expert.py``.
    """
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_online_bc=True,
        bc_coef=0.5,
        critic_warmup_frac=0.5,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            OnlineBC(expert_policy=expert, bc_coef=0.5, critic_warmup_frac=0.5),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "online_bc")


# --------------------------------------------------------------------------
# Residual policy
# --------------------------------------------------------------------------
def test_residual_policy_matches_legacy_use_residual_rl():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        residual=True,
        residual_scale=0.5,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            ResidualPolicy(expert_policy=expert, scale=0.5),
        ),
        # ``residual`` on the SAC class still routes the residual-policy
        # flag onto network init; mirror it on both sides for an honest
        # equivalence check.
        residual=True,
        residual_scale=0.5,
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "residual_policy")


# --------------------------------------------------------------------------
# JSRL curriculum
# --------------------------------------------------------------------------
def test_jsrl_curriculum_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        jsrl_curriculum=True,
        jsrl_episode_length=50,
        jsrl_decay_frac=0.5,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            JSRLCurriculum(
                expert_policy=expert, episode_length=50, decay_frac=0.5
            ),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "jsrl_curriculum")


# --------------------------------------------------------------------------
# IBRL bootstrap
# --------------------------------------------------------------------------
def test_ibrl_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        ibrl_bootstrap=True,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            IBRL(expert_policy=expert),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "ibrl")


# --------------------------------------------------------------------------
# Expert-obs augmentation — sets augment_obs_with_expert_action=True
# which changes the network input dim. Just verify it constructs and
# trains (numerics covered by the legacy code path).
# --------------------------------------------------------------------------
def test_expert_obs_aug_constructs_and_trains():
    expert = _noise_expert()
    agent = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            ExpertObsAugmentation(expert_policy=expert, detach=False),
        ),
    )
    state, _ = agent.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    assert state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
