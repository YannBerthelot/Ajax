"""Shared tiny DreamerV3 world model and replay batch for the M5 tests."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.DreamerV3.networks import WorldModel, encode_action, init_world_model
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.world_model import ReplayContextBatch

#: The parity fixtures' sizes: 4 latents x 3 classes, deter 64 in 8 blocks,
#: RSSM width 12, MLP width 16. ``stoch != classes`` and ``hidden != units``,
#: so that swapping either pair changes a shape or a value.
TINY = DreamerV3Config(units=16, hidden=12, deter=64, stoch=4, classes=3, blocks=8)
OBS_DIM = 5
B, T = 2, 8


def perturbed(params: dict, seed: int, scale: float = 1.0) -> dict:
    """``params`` plus seeded Gaussian noise, so that no layer is degenerate.

    The zero reward head becomes non-zero and the KL crosses the free bits
    (kernels ``N(0, (0.6 scale / sqrt(fan_in))^2)``, biases and norm scales
    ``N(0, (0.3 scale)^2)``).
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    out = []
    for (path, leaf), key in zip(leaves, keys):
        name = jax.tree_util.keystr(path)
        if name.endswith("['kernel']"):
            std = 0.6 / np.sqrt(np.prod(leaf.shape[:-1]))
        else:
            std = 0.3
        out.append(leaf + scale * std * jax.random.normal(key, leaf.shape))
    return jax.tree_util.tree_unflatten(treedef, out)


def tiny_model(
    discrete: bool = False,
    seed: int = 0,
    config: DreamerV3Config = TINY,
    scale: float = 1.0,
) -> tuple[WorldModel, dict, int]:
    """The tiny world model, perturbed parameters (:func:`perturbed` with
    ``scale``) and the action width."""
    action_dim = 3 if discrete else 2
    params = init_world_model(jax.random.PRNGKey(seed), config, OBS_DIM, action_dim)
    return WorldModel(config, OBS_DIM), perturbed(params, seed + 100, scale), action_dim


def replay_batch(
    seed: int, discrete: bool = False, config: DreamerV3Config = TINY
) -> ReplayContextBatch:
    """A random ``[B, T + 1]`` replay batch with episode boundaries."""
    rng = np.random.default_rng(seed)
    length = T + 1
    is_first = np.zeros((B, length), bool)
    is_first[:, 0] = True
    is_first[0, 4] = True
    is_first[1, 1] = True
    is_first[1, 6] = True  # after a time-limit truncation at 5
    is_terminal = np.zeros((B, length), bool)
    is_terminal[0, 3] = True
    is_last = np.zeros((B, length), bool)
    is_last[:, :-1] |= is_first[:, 1:]
    reward = rng.normal(0.0, 2.0, (B, length)).astype(np.float32)
    reward[is_first] = 0.0
    if discrete:
        action = encode_action(jnp.asarray(rng.integers(0, 3, (B, length))), 3)
    else:
        action = jnp.asarray(rng.uniform(-1.5, 1.5, (B, length, 2)), jnp.float32)
    return ReplayContextBatch(
        obs=jnp.asarray(rng.normal(0.0, 3.0, (B, length, OBS_DIM)), jnp.float32),
        action=action,
        reward=jnp.asarray(reward),
        is_first=jnp.asarray(is_first),
        is_last=jnp.asarray(is_last),
        is_terminal=jnp.asarray(is_terminal),
        context_deter=jnp.asarray(rng.normal(0.0, 0.5, (B, config.deter)), jnp.float32),
        context_stoch=jnp.asarray(rng.integers(0, config.classes, (B, config.stoch))),
    )


# Reference parameter name -> Ajax path rules. ``{i}`` is the
# layer index of the reference's ``mlp{i}`` / ``h{i}`` / ``img{i}``.
_LAYER = re.compile(r"^(?P<prefix>.*?)(?P<index>\d+)$")
_DENSE = {"kernel": "kernel", "bias": "bias"}


def _hidden(module: tuple[str, ...], index: int, leaf: list[str]) -> tuple[str, ...]:
    """A reference hidden layer ``.../<layer>/{kernel,bias,norm/scale}``."""
    if leaf == ["norm", "scale"]:
        return (*module, f"RMSNorm_{index}", "scale")
    return (*module, f"Dense_{index}", _DENSE[leaf[0]])


def reference_to_ajax(name: str) -> tuple[str, ...]:
    """Ajax parameter path of the reference parameter ``name``.

    The layout differences of dreamerv3_spec 2.17 do not arise: Ajax keeps
    the reference's per-block ``[reset, cand, update]`` gate layout, its
    2411f7d decoder input order ``concat(deter, stoch)`` and a single
    observation key. Only the names differ (and the reward head's dropped
    256th logit, handled by the callers).
    """
    _, module, *rest = name.split("/")
    layer, leaf = rest[0], rest[1:]
    if module == "enc":
        return _hidden(("enc",), int(layer.removeprefix("mlp")), leaf)
    if module in ("dec", "rew", "con"):
        if layer in ("out_vector", "dist"):
            assert leaf[0] == "out", name
            return (module, "out", _DENSE[leaf[1]])
        index = int(_LAYER.match(layer)["index"])  # type: ignore[index]
        return _hidden((module, "mlp"), index, leaf)
    assert module == "dyn", name
    if layer in ("dyn0", "dyncore"):
        target = {"dyn0": "dynhid0", "dyncore": "dyngru"}[layer]
        if leaf == ["norm", "scale"]:
            return ("rssm", f"{target}_norm", "scale")
        return ("rssm", target, _DENSE[leaf[0]])
    if layer in ("obslogit", "imglogit"):
        target = {"obslogit": "obslogit", "imglogit": "priorlogit"}[layer]
        return ("rssm", target, _DENSE[leaf[0]])
    match = _LAYER.match(layer)
    assert match, name
    prefix, index = match["prefix"], int(match["index"])
    if prefix == "dynin":
        return _hidden(("rssm", layer), 0, leaf)
    module_name = {"obs": "obs", "img": "prior"}[prefix]
    return _hidden(("rssm", module_name), index, leaf)
