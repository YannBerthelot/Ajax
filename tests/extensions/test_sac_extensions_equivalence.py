"""Behaviour-equivalence tests for the SAC ``extensions=`` surface.

For each migrated extension, configuring SAC via the new
``extensions=[...]`` surface must produce the :func:`_agent_fingerprint`
the pre-Phase-5 legacy-flag surface produced.

Phase 5 stripped the legacy back-compat shim (``_resolve_extension_stack``,
``_auto_append_*`` helpers, the ``_locals.get(...)`` rebinding block,
and the matching ``SAC.__init__`` / ``make_train`` kwargs). The
pre-Phase-5 tests compared ``SAC(**legacy_flags)`` against
``SAC(extensions=[...])``. After Phase 5 the legacy-flag SAC is
unconstructible, so each configuration in ``CASES`` is now checked
against the fingerprint captured into ``_sac_equivalence_goldens.json``,
keyed by the name of the test that captured it.

The tests deliberately use a tiny config so they run in a few seconds on
CPU; the checksum is a strong-enough fingerprint to catch any silent
divergence in extension wiring.

Goldens were re-captured once, deliberately, when global-norm gradient
clipping was removed from every non-PPO agent (SAC's actor, critic and
temperature optimizers all clipped at 0.5 before). Every fingerprint
moved by 1e-3..2e-3 in the same direction — the plain-SAC parity goldens
in ``tests/agents/SAC/test_sac_parity.py`` moved identically — so the
shift is the optimizer change, not extension wiring.

Re-captured when SAC moved the extensions' ``post_update`` after its
update, on a key split from the state's as in ``TrainLoop.post_update``:
every non-empty stack moved by 6e-5..7.4e-4 (the new key stream), the
empty stack not at all. The five MCPretrain goldens were re-captured
again when SAC's init drew the extensions' keys as ``TrainLoop`` does
(MCPretrain trains phi* on a new key): moves of 9e-6..8.7e-4.

Re-captured when the shared encoder's output LayerNorm became opt-in (SAC's
networks are plain MLPs): every fingerprint lost the norm's scales, 256 per
encoder (critic about 1038 -> 525..552, actor 517 -> 267..271, target 1034
-> 522), alpha moved by at most 0.04%.

Re-captured when, on those plain MLPs, SAC drew its warm-up decision and
its uniform action from separate keys and CriticBlend kept reward, discount
and done mask in its target: the stacks that warm up through an expert moved
by 1.1e-3..4.5e-3, the empty stack, ValueBox and the residual policy not at
all (2e-7 at most).

The expert-policy golden (ExpertGuidance's until that inert extension
went) was re-captured when its stack became empty: SAC then draws no
post_update key, a move of 1.6e-5..2.9e-4.

Re-captured when every continuous action was mapped from [-1, 1] onto
Pendulum's [-2, 2] torque box: the policy and the uniform warm-up act at
twice the torque, and the expert (``tanh(sum(obs))``, read as env units)
is read at half its value in agent units, so every stack moved: by at
most 1.9e-3 (PhiRefresh's actor), 1.1e-3 (the expert policy's actor) and
1.0e-3 (EDGE thompson's critic), the others 1.3e-4..9.2e-4.

OnlineBC's golden was re-captured with MCPretrain in its stack, which every
reader of phi* now needs (it added nothing without it): critic 5.5e-3,
actor 4.8e-4, target 1.7e-4, alpha 3e-6.

Re-captured when SAC's layers started as its reference code's (haarnoja/sac
and softlearning: xavier_uniform kernels, zero biases, the mean and log-std
heads included): every stack moved alike, critic +1.5..2.2%, actor
-1.4..0.0%, target +2.4%, alpha +0.06..0.08%, as the plain-SAC parity
goldens did, so the shift is the initialisation, not extension wiring.

Re-captured when SAC's Adam eps became softlearning's, TensorFlow's 1e-8
(was 1e-5): every stack moved the same way, critic +2.3e-4..2.1e-3 (the
pretraining stacks most), actor at most 1.5e-4, target at most 6.5e-5,
alpha at most 1.2e-6; the plain-SAC parity goldens moved alike.

Re-captured when SAC's actor step took its keys from the state
(``get_keys_and_new_state``) and its extensions' key from its policy key
(``derive_extensions_key``), the random stream of every SAC run moving: up to
4.2e-3 (critic), 5.0e-4 (actor), 8.2e-5 (target), 1.2e-4 (alpha).

Re-captured when SAC stopped splitting a third, discarded key off its
replay key (left from a removed expert-batch mix): up to 2.0e-3 (critic),
3.7e-4 (actor), 7.6e-5 (target), 1.0e-4 (alpha).

BellmanPretrain's golden was re-captured when it came to require a seeded
replay: it had been recorded on the empty replay expert_buffer_n_steps=0
left, so it pre-trained on nothing; it now runs on ExpertPrefill(n_steps=40).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest
from optax.tree_utils import tree_norm

from ajax.agents.SAC.SAC import SAC
from ajax.extensions.expert import (
    ExpertWarmup,
    JSRLCurriculum,
    OnlineBC,
    ResidualPolicy,
)
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.observe import ExpertActionInput
from ajax.extensions.pretrain import (
    BellmanPretrain,
    ExpertPrefill,
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

# ``dict[str, Any]`` so that ``**_TINY`` unpacks into ``SAC.__init__``'s
# typed kwargs under mypy's incremental cache (it may infer
# ``dict[str, object]`` otherwise).
_TINY: dict[str, Any] = {
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

with (Path(__file__).parent / "_sac_equivalence_goldens.json").open() as _fh:
    _GOLDENS: dict[str, dict[str, float]] = json.load(_fh)


def _agent_fingerprint(state) -> dict:
    return {
        "critic": float(tree_norm(state.critic_state.params, squared=True)),
        "actor": float(tree_norm(state.actor_state.params, squared=True)),
        "target": float(tree_norm(state.critic_state.target_params, squared=True)),
        "alpha": float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0]),
    }


class _NoiseExpert:
    """A deterministic expert, ``tanh(sum(obs))``, callable both stateless,
    ``expert(obs)`` (the action pipeline, MC pretraining, the target
    modifiers), and stateful, ``expert(state, obs)`` (the evaluation), with
    ``init_state(n_envs)``. Its state is carried unchanged, so its presence
    does not perturb numerics."""

    def __call__(self, *args):
        action = jnp.tanh(jnp.sum(args[-1], axis=-1, keepdims=True))
        return action if len(args) == 1 else (action, args[0])

    def init_state(self, n_envs: int):
        return jnp.zeros((n_envs, 1), dtype=jnp.float32)


_EXPERT = _NoiseExpert()
_WITH_EXPERT: dict[str, Any] = {"expert_policy": _EXPERT}
#: The expert's share of the warm-up, which SAC took by default with an expert
#: (outside an action transform) when the goldens were recorded.
_MIX = ExpertWarmup(fraction=0.7)
# MC critic pretraining, small enough to stay fast.
_MC = MCPretrain(
    expert_policy=_EXPERT,
    n_mc_steps=200,
    n_mc_episodes=4,
    n_steps=20,
    online_light_steps=5,
)
_GATES = ("fixed", "argmax", "boltzmann", "lcb", "argmax_lcb", "thompson")


def _expert(*extensions: Any, **kwargs: Any) -> dict[str, Any]:
    return {**_WITH_EXPERT, **kwargs, "extensions": (*extensions, _MIX)}


# The SAC configurations by the golden captured for each, in the stacks'
# order. The comments say why a configuration is pinned.
CASES: dict[str, dict[str, Any]] = {
    # The empty stack equals pre-Phase-5 plain SAC.
    "test_empty_stack_matches_legacy_plain_sac": {"extensions": ()},
    # SAC's expert policy alone equals pre-Phase-5 ``expert_policy=``.
    "test_expert_policy_matches_legacy_expert_policy": _expert(),
    # OnlineBC reads the expert critic MC pretraining fits (the BC math:
    # tests/extensions/test_expert_math.py).
    "test_online_bc_matches_legacy_flag": _expert(
        _MC, OnlineBC(bc_coef=0.5, critic_warmup_frac=0.5)
    ),
    "test_residual_policy_matches_legacy_use_residual_rl": {
        "expert_policy": _EXPERT,
        "extensions": (ResidualPolicy(scale=0.5),),
    },
    # SAC counts the per-env ``step_in_episode`` the curriculum reads: the
    # golden was recorded with the retired ``jsrl_curriculum=True`` flag.
    "test_jsrl_curriculum_matches_legacy_flag": _expert(
        JSRLCurriculum(episode_length=50, decay_frac=0.5)
    ),
    "test_ibrl_matches_legacy_flag": _expert(IBRL(expert_policy=_EXPERT)),
    # Each EDGE gate is a different action-pipeline gate function.
    **{
        f"test_edge_exploration_matches_legacy_flags[{gate}]": _expert(
            EDGEExploration(gate=gate)
        )
        for gate in _GATES
    },
    # Soft-blends the policy's and the expert's next actions by an LCB score.
    "test_lcb_gated_bootstrap_matches_legacy_flag": _expert(
        LCBGatedBootstrap(expert_policy=_EXPERT)
    ),
    "test_mc_pretrain_matches_legacy_flag": _expert(_MC),
    # On the expert's prefill: recorded first on an empty replay, which
    # BellmanPretrain now refuses.
    "test_bellman_pretrain_matches_legacy_flag": _expert(
        ExpertPrefill(n_steps=40), BellmanPretrain(n_steps=20)
    ),
    # CriticBlend reads the expert critic MC pretraining fits. The legacy
    # default ``use_online_bc=True`` appended an OnlineBC whenever MC
    # pretraining ran; the golden was captured under it.
    "test_critic_blend_matches_legacy_flag": _expert(
        _MC,
        CriticBlend(expert_policy=_EXPERT, critic_warmup_frac=0.5),
        OnlineBC(bc_coef=1.0, critic_warmup_frac=0.5),
    ),
    "test_mc_variance_correction_matches_legacy_flag": _expert(
        _MC, MCVarianceCorrection(threshold=0.0)
    ),
    # ValueBox's v_min / v_max come from MC pretraining.
    "test_value_box_matches_legacy_flag": _expert(_MC, ValueBox()),
    # An interval short enough for the refresh to fire within 80 steps.
    "test_phi_refresh_matches_legacy_flag": _expert(
        _MC, PhiRefresh(expert_policy=_EXPERT, interval=30, steps=2)
    ),
}


def test_every_golden_has_a_configuration():
    assert set(CASES) == set(_GOLDENS)


@pytest.mark.parametrize("name", list(CASES), ids=lambda n: n.removeprefix("test_"))
def test_configuration_matches_its_golden(name: str):
    state, _ = SAC(**_TINY, **CASES[name]).train(seed=_SEED, n_timesteps=_TIMESTEPS)
    new = _agent_fingerprint(state)
    # Relative 1e-3: residual_policy reads 6.65e-4 on the CI runner, a
    # reduction-order drift; a wiring bug moves a fingerprint by O(1).
    for key, golden in _GOLDENS[name].items():
        rel = abs(new[key] - golden) / max(abs(golden), 1.0)
        assert rel < 1e-3, f"{name} {key}: golden {golden!r}, new {new[key]!r}"


def test_expert_obs_aug_constructs_and_trains():
    """Augmenting observations with the expert's action changes the
    networks' input width (init_SAC, collect_experience): it trains."""
    agent = SAC(**_TINY, **_WITH_EXPERT, extensions=(ExpertActionInput(),))
    state, _ = agent.train(seed=_SEED, n_timesteps=_TIMESTEPS)
    assert state is not None
