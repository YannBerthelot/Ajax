"""Shared by the DreamerV3 tests: the tiny world model, replay batch and
parameter groups; and the parity tests' plumbing, the reference's parameter
names and fixtures in Ajax's terms."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Callable, Mapping
from typing import Any

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
#: The action sizes of the training-step fixtures: a continuous action of
#: dimension 7, or 6 discrete actions -- distinct from every size above (not
#: the batch's 2 rows, not the latents' 3 classes), so that using one for
#: the other changes a shape or a value.
ACTION_DIM, NUM_ACTIONS = 7, 6


def perturbed(params: Mapping[str, Any], seed: int, scale: float = 1.0) -> dict:
    """``params`` plus seeded Gaussian noise, so that no layer is degenerate.

    The zero reward head becomes non-zero and the KL crosses the free bits
    (kernels ``N(0, (0.6 scale / sqrt(fan_in))^2)``, biases and norm scales
    ``N(0, (0.3 scale)^2)``).
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    out = []
    for (path, leaf), key in zip(leaves, keys, strict=False):
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
    ``scale``) and the action width (:data:`NUM_ACTIONS` or
    :data:`ACTION_DIM`)."""
    action_dim = NUM_ACTIONS if discrete else ACTION_DIM
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
        actions = rng.integers(0, NUM_ACTIONS, (B, length))
        action = encode_action(jnp.asarray(actions), NUM_ACTIONS)
    else:
        shape = (B, length, ACTION_DIM)
        action = jnp.asarray(rng.uniform(-1.5, 1.5, shape), jnp.float32)
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


#: The reference's actor output layers -> Ajax's (``nets.py:440-448``: the
#: ``action`` head's ``out`` is the mean, or the logits of a discrete
#: actor, and ``std`` the standard deviation).
_ACTOR_OUTPUTS = {
    ("out", False): "mean",
    ("out", True): "logits",
    ("std", False): "std",
}


def reference_to_ajax(name: str, discrete: bool = False) -> tuple[str, ...]:
    """Ajax parameter path of the reference parameter ``name``.

    The layout differences of dreamerv3_spec 2.17 do not arise: Ajax keeps
    the reference's per-block ``[reset, cand, update]`` gate layout, its
    2411f7d decoder input order ``concat(deter, stoch)`` and a single
    observation key. Only the names differ (and the two-hot heads' dropped
    256th logit, handled by the callers). World-model paths start with the
    module (``enc``, ``rssm``, ``dec``, ``rew``, ``con``); the actor's,
    critic's and slow critic's with ``actor``, ``critic`` and
    ``slowcritic``, followed by the path in that module's own tree.
    ``discrete`` names the actor's output layer.
    """
    _, module, *rest = name.split("/")
    layer, leaf = rest[0], rest[1:]
    if module == "actor":
        if layer == "action":
            output = _ACTOR_OUTPUTS[leaf[0], discrete]
            return ("actor", output, _DENSE[leaf[1]])
        return _hidden(("actor", "mlp"), int(layer.removeprefix("h")), leaf)
    if module in ("critic", "slowcritic"):
        if layer == "dist":
            assert leaf[0] == "out", name
            return (module, "out", _DENSE[leaf[1]])
        return _hidden((module, "mlp"), int(layer.removeprefix("h")), leaf)
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


# ------------------------------------------------------------ parity plumbing

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
#: The reference's loss keys for Ajax's terms.
REFERENCE_TERMS = {
    "rec": "vector",
    "rew": "reward",
    "con": "cont",
    "dyn": "dyn",
    "rep": "rep",
    "actor": "actor",
    "critic": "critic",
    "repval": "replay_critic",
}


def parity_error(ours: Any, theirs: Any) -> float:
    """``max |ours - theirs| / (1 + max |theirs|)``, in float64."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    return float(np.max(np.abs(ours - theirs)) / (1.0 + np.max(np.abs(theirs))))


def relative(ours: Any, theirs: Any) -> float:
    """``max |ours - theirs| / max |theirs|`` (absolute where theirs is 0)."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    scale = np.max(np.abs(theirs))
    return float(np.max(np.abs(ours - theirs)) / (scale if scale else 1.0))


def get_path(tree: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        tree = tree[key]
    return tree


def reference_group_path(name: str, discrete: bool) -> tuple[str, tuple[str, ...]]:
    """``(group, path)`` of a reference parameter in Ajax's learner state:
    group ``world_model``, ``actor``, ``critic`` or ``slowcritic``."""
    path = reference_to_ajax(name, discrete)
    if path[0] in ("actor", "critic", "slowcritic"):
        return path[0], path[1:]
    return "world_model", path


def cut(theirs: np.ndarray, width: int) -> np.ndarray:
    """``theirs`` cut to Ajax's last-axis ``width``: the reference's two-hot
    layers have a 256th logit that they drop (layout only: its gradients
    and updates are 0, and so are its parameters but in the perturbed
    world-model fixture); other tensors are unchanged."""
    assert not np.any(theirs[..., width:])
    return theirs[..., :width]


def reference_trees(
    flat: dict[str, np.ndarray], discrete: bool, bins: int, *, dropped_must_be_zero
) -> dict[str, dict]:
    """Reference ``{name: array}`` -> ``{group: nested Ajax params}``
    (:func:`reference_group_path`), each name on its own path, the two-hot
    layers' 256th logit dropped; asserted zero when ``dropped_must_be_zero``
    (the perturbed world-model fixture's reward head has a non-zero one)."""
    trees: dict = {}
    for name, value in flat.items():
        group, path = reference_group_path(name, discrete)
        node = trees.setdefault(group, {})
        for key in path[:-1]:
            node = node.setdefault(key, {})
        assert path[-1] not in node, name
        if path[-2] == "out" and value.shape[-1] == bins + 1:
            value = cut(value, bins) if dropped_must_be_zero else value[..., :bins]
        node[path[-1]] = jnp.asarray(value)
    return trees


def fixture_config(rssm: dict, units: int, **overrides: Any) -> DreamerV3Config:
    """Ajax's defaults at a fixture's widths (and ``overrides``): every
    other field keeps Ajax's default, which the parity modules pin against
    the reference's."""
    return dataclasses.replace(
        DreamerV3Config(),
        units=units,
        hidden=rssm["hidden"],
        deter=rssm["deter"],
        stoch=rssm["stoch"],
        classes=rssm["classes"],
        blocks=rssm["blocks"],
        **overrides,
    )


def fixture_batch(
    raw: dict[str, np.ndarray],
    num_actions: int | None,
    context_deter: np.ndarray,
    context_stoch: np.ndarray,
) -> ReplayContextBatch:
    """A fixture's replay batch (``num_actions``: one-hot actions)."""
    return ReplayContextBatch(
        obs=jnp.asarray(raw["vector"]),
        action=encode_action(jnp.asarray(raw["action"]), num_actions),
        reward=jnp.asarray(raw["reward"]),
        is_first=jnp.asarray(raw["is_first"]),
        is_last=jnp.asarray(raw["is_last"]),
        is_terminal=jnp.asarray(raw["is_terminal"]),
        context_deter=jnp.asarray(context_deter),
        context_stoch=jnp.asarray(context_stoch),
    )


def with_params(train_state: Any, params: Any, target: Any = None) -> Any:
    """``train_state`` holding ``params`` (and the slow critic ``target``)
    with fresh optimizer moments."""
    train_state = train_state.replace(
        params=params, opt_state=train_state.tx.init(params)
    )
    return train_state if target is None else train_state.replace(target_params=target)


# ------------------------------------------------------------ gradient routing

#: The world model's parameter groups, by their path in its tree.
WORLD_MODEL_GROUPS: dict[str, Callable[[tuple[str, ...]], bool]] = {
    "enc": lambda path: path[0] == "enc",
    "core": lambda path: path[0] == "rssm" and path[1].startswith("dyn"),
    "post": lambda path: path[0] == "rssm" and path[1] in ("obs", "obslogit"),
    "prior": lambda path: path[0] == "rssm" and path[1] in ("prior", "priorlogit"),
    "dec": lambda path: path[0] == "dec",
    "rew": lambda path: path[0] == "rew",
    "con": lambda path: path[0] == "con",
}


def reached(
    grads: Any, groups: dict[str, Callable[[tuple[str, ...]], bool]]
) -> set[str]:
    """The groups (name -> predicate on a parameter's key path; the first
    that matches) where ``grads`` is non-zero."""
    out = set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        keys = tuple(k.key for k in path)
        group = next(g for g, match in groups.items() if match(keys))
        if np.any(np.asarray(leaf)):
            out.add(group)
    return out
