"""Shared helpers of the TD-MPC2 parity tests: the reference's parameters in Ajax.

The parity fixtures (``fixtures/tdmpc2_update.npz``, ``fixtures/tdmpc2_plan.npz``,
``fixtures/tdmpc2_multitask_*.npz``, recorded by
``docs/world_models/parity/tdmpc2_*_fixtures.py``) store the ``state_dict`` of
the paper-era reference (``nicklashansen/tdmpc2@5f6fade``) under a key prefix.
:func:`torch_to_ajax` maps one snapshot onto Ajax's parameter trees (the
multi-task embedding table included); :class:`ErrorReport` asserts closeness
and keeps the worst errors for printing.
"""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.TDMPC2.core import TASK_EMB
from ajax.agents.TDMPC2.state import TDMPC2Config

Fixture = dict[str, np.ndarray]

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
    wm = {
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
        for a, e in zip(a_leaves, e_leaves):
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
