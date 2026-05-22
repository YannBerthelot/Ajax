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
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import (
    BellmanPretrain,
    MCPretrain,
    PhiRefresh,
)
from ajax.extensions.target_mods import (
    IBRL,
    CriticBlend,
    LCBGatedBootstrap,
    MCVarianceCorrection,
    ValueBox,
)

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
            JSRLCurriculum(expert_policy=expert, episode_length=50, decay_frac=0.5),
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


# --------------------------------------------------------------------------
# EDGE exploration — six gate variants. Each maps onto a different action-
# pipeline gate function (see ajax.modules.exploration), so all six are
# materially distinct code paths and worth pinning under the equivalence
# contract.
# --------------------------------------------------------------------------
def _edge_legacy_kwargs(gate: str) -> dict:
    """Map a gate name to its legacy-flag set (mirrors _resolve_extension_stack)."""
    return {
        "exploration_argmax": gate == "argmax",
        "exploration_boltzmann": gate == "boltzmann",
        "exploration_lcb": gate == "lcb",
        "exploration_argmax_lcb": gate == "argmax_lcb",
        "exploration_thompson": gate == "thompson",
    }


@pytest.mark.parametrize(
    "gate", ["fixed", "argmax", "boltzmann", "lcb", "argmax_lcb", "thompson"]
)
def test_edge_exploration_matches_legacy_flags(gate: str):
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_expert_guided_exploration=True,
        **_edge_legacy_kwargs(gate),
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            EDGEExploration(expert_policy=expert, gate=gate),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, f"edge_{gate}")


# --------------------------------------------------------------------------
# LCB-gated bootstrap — target modifier that soft-blends policy/expert
# next-actions by an LCB score (``lcb_gated_bootstrap=True``).
# --------------------------------------------------------------------------
def test_lcb_gated_bootstrap_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        lcb_gated_bootstrap=True,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            LCBGatedBootstrap(expert_policy=expert),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "lcb_gated_bootstrap")


# --------------------------------------------------------------------------
# MC critic pre-training (``use_mc_critic_pretrain=True``). Tiny config:
# small MC rollouts + few regression steps + few online-light steps so the
# test stays fast.
# --------------------------------------------------------------------------
_MC_KW = {
    "n_mc_steps": 200,
    "n_mc_episodes": 4,
    "n_steps": 20,
    "online_light_steps": 5,
}
_MC_LEGACY_KW = {
    "mc_pretrain_n_mc_steps": _MC_KW["n_mc_steps"],
    "mc_pretrain_n_mc_episodes": _MC_KW["n_mc_episodes"],
    "mc_pretrain_n_steps": _MC_KW["n_steps"],
    "online_critic_pretrain_steps": _MC_KW["online_light_steps"],
}


def test_mc_pretrain_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=True,
        **_MC_LEGACY_KW,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            MCPretrain(expert_policy=expert, **_MC_KW),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "mc_pretrain")


# --------------------------------------------------------------------------
# Bellman critic pre-training (``use_bellman_critic_pretrain=True``).
#
# Quirk: the legacy ``pretrain_critic_bellman`` path is currently broken
# under JIT — it calls a non-static ``update_target_fn`` inside a jitted
# init_fn, which raises ``TypeError: Error interpreting argument [...] as
# an abstract array``. Reproducible without any extension surface (just
# ``SAC(..., use_bellman_critic_pretrain=True)``). This is unrelated to
# the extension-resolver mapping under test here, so this test verifies
# the resolver wiring directly (mirroring the
# ``test_online_bc_matches_legacy_flag`` / ``test_expert_obs_aug_*``
# pattern) rather than running training end-to-end.
# --------------------------------------------------------------------------
def test_bellman_pretrain_resolver_wiring():
    from ajax.agents.SAC.sac import _resolve_extension_stack

    expert = _noise_expert()
    resolved = _resolve_extension_stack(
        (
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            BellmanPretrain(expert_policy=expert, n_steps=20),
        )
    )
    assert resolved.get("use_bellman_critic_pretrain") is True
    assert resolved.get("mc_pretrain_n_steps") == 20
    assert resolved.get("expert_policy") is expert


# --------------------------------------------------------------------------
# CriticBlend — warmup-decaying blend with the frozen expert critic value.
# Requires MC pre-training (which populates ``expert_critic_params``).
# --------------------------------------------------------------------------
def test_critic_blend_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=True,
        use_critic_blend=True,
        critic_warmup_frac=0.5,
        **_MC_LEGACY_KW,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            MCPretrain(expert_policy=expert, **_MC_KW),
            CriticBlend(expert_policy=expert, critic_warmup_frac=0.5),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "critic_blend")


# --------------------------------------------------------------------------
# MCVarianceCorrection — replace high-variance Bellman targets with the
# MC oracle when ensemble σ exceeds ``threshold``. Requires MC pre-training.
# Note: the threshold must be low enough that the correction actually fires
# at the tiny scale of this test — use 0.0 so it triggers on every batch.
# --------------------------------------------------------------------------
def test_mc_variance_correction_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=True,
        mc_variance_threshold=0.0,
        **_MC_LEGACY_KW,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            MCPretrain(expert_policy=expert, **_MC_KW),
            MCVarianceCorrection(threshold=0.0),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "mc_variance_correction")


# --------------------------------------------------------------------------
# ValueBox (``use_box=True``) — collection-time override of the policy
# action with the expert's whenever V_expert(s) exceeds a curriculum
# threshold. v_min/v_max come from MC pre-training, so MCPretrain must be
# present in the stack.
# --------------------------------------------------------------------------
def test_value_box_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=True,
        use_box=True,
        **_MC_LEGACY_KW,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            MCPretrain(expert_policy=expert, **_MC_KW),
            ValueBox(expert_policy=expert),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "value_box")


# --------------------------------------------------------------------------
# PhiRefresh (``use_phi_refresh=True``) — periodic self-consistent refresh
# of the frozen expert critic during training. Requires MC pre-training
# (which creates the refreshable φ*). Use a tiny interval so the refresh
# actually fires within _TIMESTEPS=80.
# --------------------------------------------------------------------------
def test_phi_refresh_matches_legacy_flag():
    expert = _noise_expert()
    legacy = SAC(
        **_TINY,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        expert_mix_fraction=0.0,
        use_mc_critic_pretrain=True,
        use_phi_refresh=True,
        phi_refresh_interval=30,
        phi_refresh_steps=2,
        **_MC_LEGACY_KW,
    )
    new = SAC(
        **_TINY,
        extensions=(
            ExpertGuidance(
                expert_policy=expert,
                expert_buffer_n_steps=0,
                expert_mix_fraction=0.0,
            ),
            MCPretrain(expert_policy=expert, **_MC_KW),
            PhiRefresh(expert_policy=expert, interval=30, steps=2),
        ),
    )
    s_legacy, _ = legacy.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    s_new, _ = new.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    _assert_same(s_legacy, s_new, "phi_refresh")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
