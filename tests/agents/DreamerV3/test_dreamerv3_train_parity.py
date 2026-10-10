"""Parity of Ajax's DreamerV3 training step with the reference code itself.

The fixtures were produced by the real ``danijar/dreamerv3@29eb964`` (the
2411f7d algorithm plus the upstream replay-context fix) with
``docs/world_models/parity/dreamerv3_train_fixtures.py``: the tiny float32
model of the world-model fixtures (vector obs 5, deter 64, RSSM width 12, 4
latents x 3 classes, 8 blocks, MLP width 16) with a continuous action of
dimension 7 or 6 discrete actions (sizes distinct from all others, so that
Ajax cannot confuse them), every loss scale and every other hyperparameter at the
reference's default (imagination horizon 15, ...) except the learning-rate
warmup, 2 updates instead of 1000 so that the parameters move; three
consecutive ``Agent.train`` calls on ``[2, 9]`` replay batches with episode
boundaries, from perturbed parameters (the script's docstring says why).

Ajax runs with its own defaults at the fixture's widths and warmup
(:func:`test_default_hyperparameters_are_the_references` pins the
defaults). The reference's starting parameters are mapped onto Ajax's trees
(:func:`common.reference_to_ajax`) and the jitted :func:`train_step` is
chained over the three calls with the reference's random draws forced. Each
test compares one group of quantities call by call; the achieved errors are
printed (``pytest -s``) and asserted below :data:`TOLERANCES`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3 import learner
from ajax.agents.DreamerV3.networks import encode_action
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.world_model import ReplayContextBatch

from .common import B, T, reference_to_ajax

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
RUNS = ("continuous", "discrete")
OBS_DIM = 5
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
#: Bounds on the achieved errors (see each test for its error measure; the
#: achieved values on macOS ARM CPU are in the comments).
TOLERANCES = {
    "values": 1e-6,  # max |ajax - ref| / (1 + max |ref|); 3.7e-7
    "metrics": 1e-5,  # the same, for the scalar metrics; 8.2e-7
    "grads": 1e-5,  # per tensor, relative to its largest reference entry; 3.1e-6
    # Per tensor, max |ajax - ref| in learning rates; 5.5e-5. LaProp
    # normalises each entry by its own RMS, so an entry whose gradient is
    # small against its tensor's largest carries its relative gradient error
    # (float32 cancellation, about 1e-6 of the tensor's scale) into a step of
    # unit size: the bound leaves room for other platforms' summation orders.
    "updates": 1e-3,
    # Relative error of each tensor's moment norm, computed in float64 as the
    # generator does: an RMS moment ~1e-23 (gradients ~1e-10) squares to
    # below float32's range, so a float32 norm would be 0; 1.5e-6.
    "moments": 1e-5,
}


def _error(ours, theirs) -> float:
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    return float(np.max(np.abs(ours - theirs)) / (1.0 + np.max(np.abs(theirs))))


def _relative(ours, theirs) -> float:
    """``max |ours - theirs| / max |theirs|`` (absolute where theirs is 0)."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    scale = np.max(np.abs(theirs))
    return float(np.max(np.abs(ours - theirs)) / (scale if scale else 1.0))


def _report(run: str, what: str, errors: dict[str, float]) -> tuple[str, float]:
    worst = max(errors, key=errors.__getitem__)
    print(
        f"\n{run} {what}: {len(errors)} compared, max error {errors[worst]:.1e} ({worst})"
    )
    return worst, errors[worst]


# ------------------------------------------------------------------ fixtures


@functools.lru_cache(maxsize=None)
def load(run: str) -> dict:
    """The fixture of ``run``: ``meta``, ``init`` and one dict per call."""
    with np.load(FIXTURES / f"dreamerv3_train_{run}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    out: dict = {"meta": meta, "init": {}, "calls": [{} for _ in range(meta["calls"])]}
    for key, value in arrays.items():
        head, rest = key.split("/", 1)
        if head == "init":
            out["init"][rest] = value
        else:
            out["calls"][int(head[1:])][rest] = value
    return out


def _group(fixture_call: dict, prefix: str) -> dict[str, np.ndarray]:
    return {
        k[len(prefix) :]: v for k, v in fixture_call.items() if k.startswith(prefix)
    }


def _ajax_path(name: str, discrete: bool) -> tuple[str, tuple[str, ...]]:
    """``(group, path)`` of a reference parameter in Ajax's learner state:
    group ``world_model``, ``actor``, ``critic`` or ``slowcritic``."""
    path = reference_to_ajax(name, discrete)
    if path[0] in ("actor", "critic", "slowcritic"):
        return path[0], path[1:]
    return "world_model", path


def _get(tree: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        tree = tree[key]
    return tree


def _cut(theirs: np.ndarray, width: int) -> np.ndarray:
    """``theirs`` cut to Ajax's last-axis ``width``: the reference's two-hot
    layers have a 256th logit that they drop (layout only: its parameters,
    gradients and updates are all 0); other tensors are unchanged."""
    assert not np.any(theirs[..., width:])
    return theirs[..., :width]


def _trees(flat: dict[str, np.ndarray], discrete: bool, bins: int) -> dict:
    """Reference ``{name: array}`` -> ``{group: nested Ajax params}``."""
    trees: dict = {}
    for name, value in flat.items():
        group, path = _ajax_path(name, discrete)
        node = trees.setdefault(group, {})
        for key in path[:-1]:
            node = node.setdefault(key, {})
        if path[-2] == "out" and value.shape[-1] == bins + 1:
            value = _cut(value, bins)
        node[path[-1]] = jnp.asarray(value)
    return trees


def _config(meta: dict) -> DreamerV3Config:
    """Ajax's defaults at the fixture's widths and learning-rate warmup."""
    rssm = meta["rssm"]
    return dataclasses.replace(
        DreamerV3Config(),
        units=meta["overrides"]["enc.simple.units"],
        hidden=rssm["hidden"],
        deter=rssm["deter"],
        stoch=rssm["stoch"],
        classes=rssm["classes"],
        blocks=rssm["blocks"],
        warmup=meta["opt"]["warmup"],
    )


def _batch(call: dict, discrete: bool, action_dim: int) -> ReplayContextBatch:
    raw = _group(call, "batch/")
    num_actions = action_dim if discrete else None
    return ReplayContextBatch(
        obs=jnp.asarray(raw["vector"]),
        action=encode_action(jnp.asarray(raw["action"]), num_actions),
        reward=jnp.asarray(raw["reward"]),
        is_first=jnp.asarray(raw["is_first"]),
        is_last=jnp.asarray(raw["is_last"]),
        is_terminal=jnp.asarray(raw["is_terminal"]),
        context_deter=jnp.asarray(raw["context_deter"]),
        context_stoch=jnp.asarray(raw["context_stoch"]),
    )


def _noise(call: dict) -> learner.TrainNoise:
    return learner.TrainNoise(
        posterior=jnp.asarray(call["noise/post"]),
        prior=jnp.asarray(call["noise/prior"]),
        action=jnp.asarray(call["noise/action"]),
    )


class CallResult(NamedTuple):
    """Ajax's outputs for one reference call (numpy)."""

    total: Any
    aux: learner.LossAux
    grads: learner.Params
    updates: learner.Params
    before: learner.Params
    state: learner.LearnerState
    entries: Any
    metrics: dict


def _initial_state(fixture: dict, config: DreamerV3Config, discrete: bool):
    """:func:`learner.init_learner`'s state with the reference's starting
    parameters, fresh optimizers and the fixture's return normaliser."""
    action_dim = fixture["meta"]["action_dim"]
    state = learner.init_learner(
        jax.random.PRNGKey(0), config, OBS_DIM, action_dim, discrete
    )
    trees = _trees(fixture["init"], discrete, config.bins)

    def load_params(train_state, params, target=None):
        train_state = train_state.replace(
            params=params, opt_state=train_state.tx.init(params)
        )
        if target is None:
            return train_state
        return train_state.replace(target_params=target)

    lo, hi = fixture["meta"]["retnorm_init"]
    return state.replace(
        world_model_state=load_params(state.world_model_state, trees["world_model"]),
        actor_state=load_params(state.actor_state, trees["actor"]),
        critic_state=load_params(
            state.critic_state, trees["critic"], trees["slowcritic"]
        ),
        retnorm=state.retnorm.replace(lo=jnp.float32(lo), hi=jnp.float32(hi)),
    )


@functools.lru_cache(maxsize=None)
def ajax_run(run: str) -> tuple[dict, DreamerV3Config, bool, list[CallResult]]:
    """Ajax's :func:`train_step` chained over the fixture's calls.

    Next to the step itself, the same jitted function returns the joint
    gradient and each train state's optimizer update (``tx.update`` on that
    gradient), to compare with the reference's update tensors before they
    are added to the parameters.
    """
    fixture = load(run)
    discrete = run == "discrete"
    config = _config(fixture["meta"])
    state = _initial_state(fixture, config, discrete)

    @jax.jit
    def call(state, batch, noise):
        (total, aux), grads = learner.loss_and_grads(state, batch, noise, config=config)
        train_states = (state.world_model_state, state.actor_state, state.critic_state)
        updates = learner.Params(
            *(
                ts.tx.update(g, ts.opt_state, ts.params)[0]
                for ts, g in zip(train_states, grads)
            )
        )
        new_state, entries, metrics = learner.train_step(
            state, batch, noise, config=config
        )
        before = learner.learner_params(state)
        return total, aux, grads, updates, before, new_state, entries, metrics

    results = []
    action_dim = fixture["meta"]["action_dim"]
    with jax.default_matmul_precision("highest"):
        for fixture_call in fixture["calls"]:
            batch = _batch(fixture_call, discrete, action_dim)
            out = call(state, batch, _noise(fixture_call))
            state = out[5]
            results.append(CallResult(*jax.device_get(out)))
    return fixture, config, discrete, results


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("run", RUNS)
def test_default_hyperparameters_are_the_references(run):
    """Ajax's actor-critic and optimizer defaults equal the reference's
    (2411f7d ``configs.yaml``; the fixtures override widths and the
    warmup only), and the reference ran with the paper-era switches that
    Ajax hardcodes."""
    meta = load(run)["meta"]
    ours = DreamerV3Config()
    actor, critic, opt = meta["actor"], meta["critic"], meta["opt"]
    assert (ours.actor_layers, ours.critic_layers) == (
        actor["layers"],
        critic["layers"],
    )
    assert (
        (ours.minstd, ours.maxstd) == (actor["minstd"], actor["maxstd"]) == (0.1, 1.0)
    )
    assert ours.actor_unimix == actor["unimix"] == 0.01
    assert ours.bins == critic["bins"] == 255
    assert ours.imag_horizon == meta["imag_length"] == 15
    assert ours.lam == meta["return_lambda"] == 0.95
    assert ours.repval_lam == meta["return_lambda_replay"] == 0.95
    assert ours.return_horizon == meta["horizon"] == 333.0
    assert ours.actent == meta["actent"] == 3e-4
    assert ours.slowreg == meta["slowreg"] == 1.0
    assert ours.slow_rate == meta["slow_critic_fraction"] == 0.02
    assert meta["slow_critic_update"] == 1
    assert ours.retnorm_rate == meta["retnorm"]["rate"] == 0.01
    assert ours.retnorm_limit == meta["retnorm"]["limit"] == 1.0
    assert (meta["retnorm"]["perclo"], meta["retnorm"]["perchi"]) == (5.0, 95.0)
    scales = ours.loss_scales
    assert scales == {k: meta["scales"][v] for k, v in REFERENCE_TERMS.items()}
    assert ours.learning_rate == opt["lr"] == 4e-5
    assert (ours.agc, ours.agc_pmin) == (opt["agc"], opt["pmin"]) == (0.3, 1e-3)
    assert (ours.beta1, ours.beta2, ours.eps) == (
        opt["beta1"],
        opt["beta2"],
        opt["eps"],
    )
    assert ours.warmup == 1000 and opt["warmup"] == 2  # the fixtures' override
    assert (opt["scaler"], opt["momentum"], opt["schedule"]) == (
        "rms",
        True,
        "constant",
    )
    assert (opt["wd"], opt["globclip"]) == (0.0, 0.0)
    # The switches Ajax does not expose (DESIGN.md section 6.1).
    assert meta["contdisc"] is True and meta["slowtar"] is False
    assert meta["ac_grads"] == "none" and meta["reward_grad"] is True
    assert (meta["imag_start"], meta["imag_repeat"]) == ("all", 1)
    assert meta["replay_critic"] == [True, True, "imag"]
    assert meta["valnorm"]["impl"] == meta["advnorm"]["impl"] == "off"
    assert meta["actor_dist"] == ["onehot", "normal"]
    assert (actor["outscale"], critic["outscale"]) == (0.01, 0.0)


@pytest.mark.parametrize("run", RUNS)
def test_parameter_trees_match(run):
    """Every reference parameter of the world model, actor, critic and slow
    critic has an Ajax counterpart of the same shape, and Ajax has no other
    parameter. The fixture's action size is distinct from its other sizes
    (DESIGN.md section 10)."""
    fixture = load(run)
    discrete = run == "discrete"
    config = _config(fixture["meta"])
    action_dim = fixture["meta"]["action_dim"]
    sizes = (B, T, OBS_DIM, config.stoch, config.classes, config.blocks)
    assert action_dim not in sizes and action_dim != config.hidden
    fresh = learner.init_learner(
        jax.random.PRNGKey(0), config, OBS_DIM, action_dim, discrete
    )
    trees = _trees(fixture["init"], discrete, config.bins)
    ours = {
        "world_model": fresh.world_model_state.params,
        "actor": fresh.actor_state.params,
        "critic": fresh.critic_state.params,
        "slowcritic": fresh.critic_state.target_params,
    }
    shapes = jax.tree.map(lambda x: x.shape, ours)
    assert shapes == jax.tree.map(lambda x: x.shape, trees)


@pytest.mark.parametrize("run", RUNS)
def test_draws_and_write_back_entries_match(run):
    """The forced draws give the reference's samples: the posterior latents
    (returned for the write-back, with their ``deter``), the imagined prior
    latents and the actions at every imagined state (discrete: exactly)."""
    fixture, _, discrete, results = ajax_run(run)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        np.testing.assert_array_equal(ours.entries.stoch, theirs["out/stoch"])
        prior = np.asarray(ours.aux.prior_stoch).argmax(-1)
        np.testing.assert_array_equal(prior, theirs["draw/prior"])
        action = np.asarray(ours.aux.action)
        if discrete:
            np.testing.assert_array_equal(
                action.argmax(-1), theirs["tensor/act/action"]
            )
        else:
            errors[f"c{k}/action"] = _error(action, theirs["tensor/act/action"])
        errors[f"c{k}/deter"] = _error(ours.entries.deter, theirs["out/deter"])
    worst, error = _report(run, "draws and entries", errors)
    assert error < TOLERANCES["values"], (worst, error)


@pytest.mark.parametrize("run", RUNS)
def test_loss_terms_match(run):
    """Every per-element loss term, scaled as the reference returns them,
    and the total, call by call."""
    fixture, config, _, results = ajax_run(run)
    scales = config.loss_scales
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        for term, name in REFERENCE_TERMS.items():
            scaled = ours.aux.losses[term] * np.float32(scales[term])
            assert scaled.shape == theirs[f"loss/{name}"].shape, term
            errors[f"c{k}/{term}"] = _error(scaled, theirs[f"loss/{name}"])
        errors[f"c{k}/total"] = _error(ours.total, theirs["loss/total"])
    worst, error = _report(run, "losses", errors)
    assert error < TOLERANCES["values"], (worst, error)


@pytest.mark.parametrize("run", RUNS)
def test_imagination_and_return_normaliser_match(run):
    """The imagined rewards (index 0 from the data), cumulative weights,
    online values, lambda-returns, advantages, entropies, the replay
    returns, and the return normaliser's state and scale after each call."""
    fixture, _, _, results = ajax_run(run)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        imag = ours.aux.imagination
        lo, hi = ours.state.retnorm.lo, ours.state.retnorm.hi
        pairs = {
            "rew": (ours.aux.reward, "tensor/rew"),
            "weight": (imag.weight, "tensor/weight"),
            "val": (imag.value, "tensor/val"),
            "ret": (imag.ret, "tensor/ret"),
            "adv": (imag.adv, "tensor/adv"),
            "ent": (imag.entropy, "tensor/ent/action"),
            "ret_normed": ((imag.ret - lo) / imag.scale, "tensor/ret_normed"),
            "replay_ret": (ours.aux.replay_ret, "tensor/replay_ret"),
            "lo": (lo, "retnorm/low"),
            "hi": (hi, "retnorm/high"),
            "offset": (lo, "retnorm/offset"),
            "scale": (imag.scale, "retnorm/scale"),
        }
        for name, (value, key) in pairs.items():
            assert np.shape(value) == theirs[key].shape, name
            errors[f"c{k}/{name}"] = _error(value, theirs[key])
    worst, error = _report(run, "imagination", errors)
    assert error < TOLERANCES["values"], (worst, error)


def _per_tensor(fixture_call, prefix, ours_params, discrete, measure):
    errors = {}
    for name, theirs in _group(fixture_call, prefix).items():
        group, path = _ajax_path(name, discrete)
        ours = _get(getattr(ours_params, group), path)
        errors[name] = measure(ours, _cut(theirs, np.shape(ours)[-1]))
    return errors


@pytest.mark.parametrize("run", RUNS)
def test_gradients_match(run):
    """The joint gradient of the total loss, parameter by parameter, at each
    call (relative to the largest entry of the reference's tensor; the
    reference's dropped 256th two-hot logit gets no gradient)."""
    fixture, _, discrete, results = ajax_run(run)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        tensors = _per_tensor(theirs, "grad/", ours.grads, discrete, _relative)
        assert len(tensors) == len(jax.tree.leaves(ours.grads))
        errors.update({f"c{k}/{n}": e for n, e in tensors.items()})
    worst, error = _report(run, "gradients", errors)
    assert error < TOLERANCES["grads"], (worst, error)


@pytest.mark.parametrize("run", RUNS)
def test_optimizer_updates_match(run):
    """The optimizer's update of every parameter, in units of the
    learning rate of the call (``lr min(k / warmup, 1)``): exactly 0 at the
    first call, then the reference's update tensors. The new parameters
    are the old ones plus the update (to one float32 ulp: XLA may fuse the
    warmup factor and the addition into a multiply-add, as it does in the
    reference)."""
    fixture, config, discrete, results = ajax_run(run)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        after = learner.learner_params(ours.state)
        for leaf_before, leaf_update, leaf_after in zip(
            *(jax.tree.leaves(t) for t in (ours.before, ours.updates, after))
        ):
            ulp = np.spacing(np.maximum(np.abs(leaf_after), np.abs(leaf_update)))
            error = np.abs(leaf_after - (leaf_before + leaf_update).astype(np.float32))
            assert np.all(error <= ulp)
        if k == 0:
            assert not any(np.any(u) for u in jax.tree.leaves(ours.updates))
            assert not _group(theirs, "update/")
            continue
        lr = config.learning_rate * min(k / config.warmup, 1.0)

        def measure(ours_update, theirs_update, lr=lr):
            difference = np.asarray(ours_update, np.float64) - theirs_update
            return float(np.max(np.abs(difference)) / lr)

        tensors = _per_tensor(theirs, "update/", ours.updates, discrete, measure)
        assert len(tensors) == len(jax.tree.leaves(ours.updates))
        errors.update({f"c{k}/{n}": e for n, e in tensors.items()})
    worst, error = _report(run, "updates (in learning rates)", errors)
    assert error < TOLERANCES["updates"], (worst, error)


@pytest.mark.parametrize("run", RUNS)
def test_optimizer_state_matches(run):
    """The update counters of the three LaProp instances, equal to each
    other and to the reference's, and the norm of every parameter's RMS
    and momentum moments after each call."""
    fixture, _, discrete, results = ajax_run(run)
    keys = fixture["meta"]["opt_keys"]
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        train_states = (
            ours.state.world_model_state,
            ours.state.actor_state,
            ours.state.critic_state,
        )
        for train_state in train_states:
            _, rms, momentum, schedule = train_state.opt_state
            assert int(train_state.step) == k + 1 == int(theirs["opt/step"])
            assert int(rms.count) == int(theirs["opt/rms_count"])
            assert int(momentum.count) == int(theirs["opt/momentum_count"])
            assert int(schedule.count) == k + 1
        moments = {
            "nu": learner.Params(*(ts.opt_state[1].nu for ts in train_states)),
            "mu": learner.Params(*(ts.opt_state[2].ema for ts in train_states)),
        }
        for moment, tree in moments.items():
            for name, theirs_norm in zip(keys, theirs[f"opt/{moment}_norm"]):
                group, path = _ajax_path(name, discrete)
                ours = np.asarray(_get(getattr(tree, group), path), np.float64)
                ours_norm = np.linalg.norm(ours)
                scale = theirs_norm if theirs_norm else 1.0
                errors[f"c{k}/{moment}/{name}"] = abs(ours_norm - theirs_norm) / scale
    worst, error = _report(run, "optimizer moments", errors)
    assert error < TOLERANCES["moments"], (worst, error)


@pytest.mark.parametrize("run", RUNS)
def test_slow_critic_matches(run):
    """The slow critic after each call: a hard copy of the critic after the
    first update (bit-identical), then the EMA at rate 0.02 of the
    reference."""
    fixture, _, discrete, results = ajax_run(run)
    first = results[0].state.critic_state
    for ours, copy in zip(
        *(jax.tree.leaves(t) for t in (first.target_params, first.params))
    ):
        np.testing.assert_array_equal(ours, copy)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["calls"], results)):
        slow = {"slowcritic": ours.state.critic_state.target_params}
        for name, value in _group(theirs, "slow/").items():
            group, path = _ajax_path(name, discrete)
            mine = _get(slow[group], path)
            errors[f"c{k}/{name}"] = _error(mine, _cut(value, np.shape(mine)[-1]))
    worst, error = _report(run, "slow critic", errors)
    assert error < TOLERANCES["values"], (worst, error)


#: Ajax's metric names for the reference's, where they differ.
_METRIC_NAMES = {
    f"{ref}_loss{suffix}": f"{ours}_loss{suffix}"
    for ours, ref in REFERENCE_TERMS.items()
    for suffix in ("", "_std")
}
#: Reference metrics Ajax does not reproduce (``learner._loss_metrics``).
_NOT_REPRODUCED = ("rewstats/", "constats/", "opt_param_count")


@pytest.mark.parametrize("run", RUNS)
def test_metrics_match(run):
    """Every scalar metric of the reference's ``Agent.train`` that Ajax
    reproduces: the unscaled loss means and standard deviations, the
    imagination and replay statistics, the latent entropies and the
    optimizer's loss, norms and step count."""
    fixture, _, _, results = ajax_run(run)
    errors = {}
    for k, (theirs, ours) in enumerate(zip(fixture["meta"]["metrics"], results)):
        compared = 0
        for name, value in theirs.items():
            if name.startswith(_NOT_REPRODUCED):
                continue
            errors[f"c{k}/{name}"] = _error(
                ours.metrics[_METRIC_NAMES.get(name, name)], value
            )
            compared += 1
        assert compared == len(ours.metrics), sorted(set(ours.metrics))
    worst, error = _report(run, "metrics", errors)
    assert error < TOLERANCES["metrics"], (worst, error)
