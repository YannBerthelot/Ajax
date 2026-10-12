"""Shared helpers of the TD-MPC2 parity tests: the reference's parameters in Ajax.

The parity fixtures (``fixtures/tdmpc2_update.npz``, ``fixtures/tdmpc2_plan.npz``,
``fixtures/tdmpc2_multitask_*.npz``, recorded by
``docs/world_models/parity/tdmpc2_*_fixtures.py``) store the ``state_dict`` of
the paper-era reference (``nicklashansen/tdmpc2@5f6fade``) under a key prefix.
:func:`torch_to_ajax` maps one snapshot onto Ajax's parameter trees (the
multi-task embedding table included); :class:`ErrorReport` asserts closeness
and keeps the worst errors for printing. The single- and multi-task parity
tests share the loading of a snapshot into an update state, the recorded
inputs, the checks of a planning decision and of the final parameters, and
their tolerances (justified in ``test_tdmpc2_parity.py``'s and
``test_tdmpc2_planner_parity.py``'s docstrings).
"""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.TDMPC2 import core, planner
from ajax.agents.TDMPC2.core import TASK_EMB
from ajax.agents.TDMPC2.state import TDMPC2Config

Fixture = dict[str, np.ndarray]

# Update parity, (rtol, atol).
SCALAR_TOL = (2e-5, 1e-5)
# Absolute only: one Adam step moves a parameter by ~lr (3e-4; 9e-5 for the
# encoder), so a missing or wrong step anywhere in the chain exceeds it.
PARAM_TOL = (0.0, 1.5e-5)
# Q values decoded from the post-step parameters: their float32 error,
# through the peaked softmax of the fixture's large heads and symexp.
POST_STEP_Q_TOL = (1e-4, 1e-4)
#: The logged values of an update compared with the reference's.
LOGGED = (
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "total_loss",
    "pi_loss",
    "grad_norm",
    "pi_grad_norm",
    "pi_scale",
)

# Plan parity, (rtol, atol): a few times the errors of the planner parity
# module's docstring, which may differ across platforms (macOS ARM / Linux
# x86). Values are O(1) to O(10); the fixture's tie margins,
# 2e-4 * max(1, |V|), stay 3x or more above this tolerance, so the elite
# sets can be compared exactly.
VALUE_TOL = (2e-5, 4e-5)
# Actions, means, stds in [-1, 1] / [0.05, 2].
ACTION_TOL = (0.0, 5e-6)
# Policy-prior samples: rounding amplified by |eps| exp(log_std) (docstring).
PI_ACTION_TOL = (0.0, 2e-5)
SCORE_TOL = (0.0, 1e-6)

# The reference's config keys that are TDMPC2Config fields.
CONFIG_FIELDS = (
    "latent_dim",
    "enc_dim",
    "mlp_dim",
    "num_enc_layers",
    "num_q",
    "simnorm_dim",
    "num_bins",
    "vmax",
    "dropout",
    "log_std_min",
    "log_std_max",
    "horizon",
    "rho",
    "consistency_coef",
    "reward_coef",
    "value_coef",
    "entropy_coef",
    "grad_clip_norm",
    "tau",
    "iterations",
    "num_samples",
    "num_elites",
    "num_pi_trajs",
    "min_std",
    "max_std",
    "temperature",
)


def load_fixture(path: Any) -> Fixture:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def reference_config(fx: Fixture) -> dict[str, Any]:
    """The reference configuration the fixture was recorded with."""
    return json.loads(str(fx["meta/config"]))


def ajax_config(ref: dict[str, Any]) -> TDMPC2Config:
    """The :class:`TDMPC2Config` of a reference configuration (its other
    fields at their defaults)."""
    assert ref["vmin"] == -ref["vmax"]
    return TDMPC2Config(**{name: ref[name] for name in CONFIG_FIELDS if name in ref})


def _dense(sd: Fixture, prefix: str) -> dict[str, np.ndarray]:
    """torch ``Linear`` (weight ``[out, in]``) -> flax ``Dense`` (kernel ``[in, out]``)."""
    return {"kernel": sd[f"{prefix}.weight"].T, "bias": sd[f"{prefix}.bias"]}


def _layer_norm(sd: Fixture, prefix: str) -> dict[str, np.ndarray]:
    return {"scale": sd[f"{prefix}.ln.weight"], "bias": sd[f"{prefix}.ln.bias"]}


def _trunk(sd: Fixture, prefix: str, n: int) -> dict[str, Any]:
    """NormedLinear layers ``prefix.0 .. prefix.{n-1}`` -> ``NormedMLP``."""
    out: dict[str, Any] = {}
    for i in range(n):
        out[f"Dense_{i}"] = _dense(sd, f"{prefix}.{i}")
        out[f"LayerNorm_{i}"] = _layer_norm(sd, f"{prefix}.{i}")
    return out


def _normed_linear(sd: Fixture, prefix: str) -> dict[str, Any]:
    """One NormedLinear ``prefix`` (the SimNorm heads) -> a one-layer ``NormedMLP``."""
    return {"Dense_0": _dense(sd, prefix), "LayerNorm_0": _layer_norm(sd, prefix)}


def _ensemble(sd: Fixture, prefix: str, names: list[str]) -> dict[str, Any]:
    """Stacked ``prefix.<i>`` tensors, in one member's parameter order ``names``
    (``[num_q, out, in]`` weights), -> the vmapped ``QFunction`` tree."""
    member = {name: sd[f"{prefix}.{i}"] for i, name in enumerate(names)}
    member = {
        name: v.transpose(0, 2, 1) if name.endswith("weight") and v.ndim == 3 else v
        for name, v in member.items()
    }
    trunk: dict[str, Any] = {}
    for i in range(2):
        trunk[f"Dense_{i}"] = {
            "kernel": member[f"{i}.weight"],
            "bias": member[f"{i}.bias"],
        }
        trunk[f"LayerNorm_{i}"] = {
            "scale": member[f"{i}.ln.weight"],
            "bias": member[f"{i}.ln.bias"],
        }
    out = {"kernel": member["2.weight"], "bias": member["2.bias"]}
    return {"members": {"trunk": trunk, "out": out}}


def torch_to_ajax(
    fx: Fixture, prefix: str, config: TDMPC2Config
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """World-model, target-Q and policy parameters of the snapshot ``prefix``."""
    sd = {k[len(prefix) :]: v for k, v in fx.items() if k.startswith(prefix)}
    names = [str(n) for n in fx["meta/q_param_names"]]
    hidden = max(config.num_enc_layers - 1, 1)
    wm: dict[str, Any] = {
        "encoder": {
            "trunk": _trunk(sd, "_encoder.state", hidden),
            "head": _normed_linear(sd, f"_encoder.state.{hidden}"),
        },
        "dynamics": {
            "trunk": _trunk(sd, "_dynamics", 2),
            "head": _normed_linear(sd, "_dynamics.2"),
        },
        "reward": {"trunk": _trunk(sd, "_reward", 2), "out": _dense(sd, "_reward.2")},
        "q": _ensemble(sd, "_Qs.params", names),
    }
    if "_task_emb.weight" in sd:  # multi-task: a world-model parameter
        wm[TASK_EMB] = sd["_task_emb.weight"]
    target_q = _ensemble(sd, "_target_Qs.params", names)
    pi = {"trunk": _trunk(sd, "_pi", 2), "out": _dense(sd, "_pi.2")}
    n_mapped = sum(len(jax.tree_util.tree_leaves(t)) for t in (wm, target_q, pi))
    assert n_mapped == len(sd), "every reference tensor is mapped exactly once"
    return wm, target_q, pi


def as_jnp(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


class ErrorReport:
    """Asserts closeness and keeps the worst absolute / relative error per name."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.worst: dict[str, tuple[float, float]] = {}

    def check(
        self, name: str, actual: Any, expected: Any, tol: tuple[float, float]
    ) -> None:
        rtol, atol = tol
        a_leaves = jax.tree_util.tree_leaves(actual)
        e_leaves = jax.tree_util.tree_leaves(expected)
        assert len(a_leaves) == len(e_leaves), name
        for a, e in zip(a_leaves, e_leaves, strict=False):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(e), rtol=rtol, atol=atol, err_msg=name
            )
        a = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in a_leaves])
        e = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in e_leaves])
        err = np.abs(a - e)
        big = np.abs(e) > 1e-3
        rel = float((err[big] / np.abs(e[big])).max(initial=0.0))
        old = self.worst.get(name, (0.0, 0.0))
        self.worst[name] = (max(old[0], float(err.max())), max(old[1], rel))

    def print(self) -> None:
        print(f"\n{self.title}: max |error|, max relative error (|reference| > 1e-3)")
        for name, (abs_err, rel_err) in self.worst.items():
            print(f"  {name:18s} {abs_err:.2e}  {rel_err:.2e}")


# ---------------------------------------------------------------------------
# Shared by the parity tests
# ---------------------------------------------------------------------------


def with_reference_params(
    state: Any, fx: Fixture, config: TDMPC2Config, prefix: str = "init/"
) -> Any:
    """The update ``state`` holding the snapshot ``prefix``'s parameters (its
    trees and shapes checked against Ajax's), with fresh optimizer moments."""
    wm, target_q, pi = (as_jnp(t) for t in torch_to_ajax(fx, prefix, config))
    for mapped, ajax in (
        (wm, state.world_model_state.params),
        (target_q, state.world_model_state.target_params),
        (pi, state.actor_state.params),
    ):
        # jaxlib does not type PyTreeDef's equality.
        assert jax.tree.structure(mapped) == jax.tree.structure(ajax)  # type: ignore[operator]
        assert jax.tree.map(jnp.shape, mapped) == jax.tree.map(jnp.shape, ajax)
    wm_state, pi_state = state.world_model_state, state.actor_state
    return state.replace(
        world_model_state=wm_state.replace(
            params=wm, target_params=target_q, opt_state=wm_state.tx.init(wm)
        ),
        actor_state=pi_state.replace(params=pi, opt_state=pi_state.tx.init(pi)),
    )


def update_inputs(
    fx: Fixture, k: int
) -> tuple[core.TDMPC2Batch, jax.Array | None, core.UpdateNoise]:
    """Update ``k``'s batch, task ids (``None`` single-task) and recorded
    draws (dropout is 0: the keys are unused)."""
    p = f"update{k}/"
    batch = core.TDMPC2Batch(
        obs=jnp.asarray(fx[p + "batch/obs"]),
        action=jnp.asarray(fx[p + "batch/action"]),
        reward=jnp.asarray(fx[p + "batch/reward"][..., 0]),
    )
    key = jax.random.PRNGKey(0)
    noise = core.UpdateNoise(
        td_eps=jnp.asarray(fx[p + "draws/td_eps"]),
        td_pair=jnp.asarray(fx[p + "draws/td_pair"]),
        td_dropout=key,
        value_dropout=key,
        pi_eps=jnp.asarray(fx[p + "draws/pi_eps"]),
        pi_pair=jnp.asarray(fx[p + "draws/pi_pair"]),
        pi_dropout=key,
    )
    task = fx.get(p + "batch/task")
    return batch, None if task is None else jnp.asarray(task, jnp.int32), noise


def check_final_params(
    report: ErrorReport, state: Any, fx: Fixture, config: TDMPC2Config
) -> None:
    """Every parameter after the last update against the snapshot ``final/``."""
    wm_ref, target_ref, pi_ref = torch_to_ajax(fx, "final/", config)
    report.check("wm_params", state.world_model_state.params, wm_ref, PARAM_TOL)
    report.check(
        "target_q_params", state.world_model_state.target_params, target_ref, PARAM_TOL
    )
    report.check("pi_params", state.actor_state.params, pi_ref, PARAM_TOL)


def recorded_plan_noise(fx: Fixture, d: int) -> planner.PlanNoise:
    """Decision ``d``'s draws (dropout is 0 in the fixtures: the keys are unused)."""
    p = f"decision{d}/draws/"
    iterations = fx[p + "q_pair"].shape[0]
    return planner.PlanNoise(
        pi_eps=jnp.asarray(fx[p + "pi_eps"]),
        candidate_eps=jnp.asarray(fx[p + "candidate_eps"]),
        terminal_eps=jnp.asarray(fx[p + "terminal_eps"]),
        q_pair=jnp.asarray(fx[p + "q_pair"]),
        q_dropout=jax.random.split(jax.random.PRNGKey(0), iterations),
        elite_uniform=jnp.asarray(fx[p + "elite_uniform"], jnp.float32),
        action_eps=jnp.asarray(fx[p + "action_eps"]),
    )


def check_decision(
    report: ErrorReport, fx: Fixture, d: int, action: Any, prev_mean: Any, info: Any
) -> None:
    """Decision ``d``'s planning statistics and outputs against the
    reference's: the elite sets exactly (not their order: module docstring
    of the planner parity test), the drawn elite exactly."""
    p = f"decision{d}/"
    report.check("init_mean", info.init_mean, fx[p + "init_mean"], ACTION_TOL)
    report.check("pi_actions", info.pi_actions, fx[p + "pi_actions"], PI_ACTION_TOL)
    report.check("value", info.value, fx[p + "value"], VALUE_TOL)
    np.testing.assert_array_equal(
        np.sort(info.elite_idx, axis=-1), np.sort(fx[p + "elite_idx"], axis=-1)
    )
    report.check("elite_value", info.elite_value, fx[p + "elite_value"], VALUE_TOL)
    report.check(
        "score",
        -np.sort(-np.asarray(info.score), axis=-1),
        -np.sort(-fx[p + "score"], axis=-1),
        SCORE_TOL,
    )
    report.check("mean", info.mean, fx[p + "mean"], ACTION_TOL)
    report.check("std", info.std, fx[p + "std"], ACTION_TOL)
    rank, ref_rank = int(info.elite_rank), int(fx[p + "elite_rank"])
    assert info.elite_idx[-1, rank] == fx[p + "elite_idx"][-1, ref_rank]
    report.check("action", action, fx[p + "action"], ACTION_TOL)
    report.check("prev_mean", prev_mean, fx[p + "prev_mean"], ACTION_TOL)
