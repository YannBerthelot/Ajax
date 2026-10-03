"""Parity fixtures for Ajax's DreamerV3 training step, made by the real reference code.

This script runs the paper-era DreamerV3 code itself -- ``danijar/dreamerv3``
at ``29eb964`` (the 2411f7d algorithm plus the upstream replay-context
``prevact`` fix; ``docs/world_models/deviations.md`` section 1) -- and saves
what Ajax's training step (``ajax.agents.DreamerV3.learner.train_step``) must
reproduce to ``tests/agents/DreamerV3/fixtures/dreamerv3_train_*.npz``. The
tests in ``tests/agents/DreamerV3/test_dreamerv3_train_parity.py`` load these
files; they do not need the reference code.

What it runs. The real ``dreamerv3.Agent`` at the tiny float32 sizes of the
world-model fixtures (``dreamerv3_world_model_fixtures.py``, whose builder,
batch and perturbation it reuses: vector observation 5, ``rssm`` deter 64 /
hidden 12 / stoch 4 / classes 3 / blocks 8, MLP units 16, batch 2 x (8
trained steps + 1 context step)), with a continuous action of dimension 7 or
6 discrete actions -- sizes distinct from every other one of the fixtures,
so that Ajax cannot confuse the number of actions with the latents' classes
(3) or the action dimension with the batch size (2), as the world-model
fixtures' 2 and 3 would allow -- **every loss scale at its default** and the
default imagination horizon 15.
The one other override is ``opt.warmup: 2`` (default 1000), so that the
parameters move within the recorded calls: the learning-rate factor is 0, 0.5
and 1 at updates 0, 1 and 2. Every other hyperparameter is the reference's
default and is recorded in ``meta`` so that the tests pin Ajax's defaults
against it. Three **consecutive** calls of the real ``Agent.train``
(``29eb964:dreamerv3/agent.py:166-223``: replay context, ``Optimizer`` over
``nj.grad(Agent.loss)``, ``SlowUpdater``) run on three fixed batches with
episode boundaries, each call starting from the state the previous one left
(parameters, optimizer moments and counters, return normaliser, slow
critic).

Starting parameters. The reference's initial parameters, perturbed (seeded
Gaussian noise as in the world-model fixtures, on the world model, actor,
critic and slow critic) so that no layer is degenerate, except the three
two-hot output layers (reward head, critic, slow critic), which get a bias
falling off linearly away from the middle bin, ``max(-0.5 |j - 127|, -27)``,
plus small noise. Three reasons. (1) The reference's zero-initialised
two-hot heads predict ``0.07`` to ``0.17``, not 0, under ``jit``: XLA fuses
its symmetric expectation into multiply-adds (deviation D22), and Ajax's
exactly-0 prediction would differ from the very first return. (2) With
non-uniform logits spread over all 255 bins, the expectation ``sum_j p_j
b_j`` sums terms of size ``|b_j| / 255 ~ 2e6`` to a result of order 1, so
float32 rounding alone moves it by about 0.1 in any implementation;
concentrating the probability on the middle bins makes it well conditioned
(``sum_j p_j |b_j| ~ 0.45``). (3) The floor keeps every bin's probability
above ``4e-13``: without it the outer bins' probabilities (``e^-63``) give
gradients below ``sqrt(1.18e-38 / (1 - beta2)) ~ 3.4e-18``, whose RMS
increment ``(1 - beta2) g^2`` falls below the smallest normal float32 and is
flushed to 0 by XLA, so that LaProp's ``eps = 1e-20`` turns ``g / (sqrt(nu)
+ eps)`` into steps of up to about 340 learning rates, whose size depends on
the platform's denormal handling (dreamerv3_spec 4.5, "float32 caveat").
The script asserts that every non-zero gradient is above that threshold
(it prints the margin) and that no optimizer step exceeds ``1.5 lr`` (the
bound of a normalised update at these step counts). The return normaliser
starts at ``lo = hi = 0`` (the
reference's init; its scale stays at the floor 1) in the continuous run and at
``lo = -1.5``, ``hi = 2.5`` in the discrete run, so that the advantages are
divided by a scale above 1.

Recording the random draws without changing them. Every random draw of a
training step is recorded next to the original draw, from the same seed:
the posterior and imagined-prior one-hot samples (tfp's JAX categorical:
``argmax(logits + gumbel(seed))``) and the actor's samples at the start state
and at every imagined state (continuous: ``normal(seed) * std + mean``,
``tfd.Normal._sample_n``; discrete: as the latents). Thin wrappers around
``jaxutils.OneHotDist.sample``, ``RSSM.observe``, ``RSSM.imagine``, the
module-level ``sample`` of ``agent.py``, ``jaxutils.scan`` (to carry the
noise drawn inside a scan body out of the scan), ``jaxutils.tensorstats``
and ``Moments.__call__`` (to read intermediate tensors), ``Agent.loss``,
``ninjax.grad`` (the gradient) and ``optax.apply_updates`` (the optimizer's
update) carry these values out of the traced function. None of them changes
a value the reference computes: each calls the original with the original
arguments, in the original order of ``nj.seed()`` calls. The script asserts,
for every draw, that the recorded noise reproduces the reference's sample
(exactly), and that the smallest margin between the two largest
``logits + noise`` of all categorical draws is far above float32 noise, so
that Ajax's forced draws cannot flip on a near-tie.

Saved per call ``k`` (prefix ``c{k}/``): the batch, the noise, every
(scaled, per-element) loss term and the total, the gradient of every
optimised parameter, the optimizer's update of every parameter (the update
of call 0 is exactly 0 and not saved), the slow critic after the call, the
return normaliser's state and the scale it used, the imagination tensors
the reference logs (returns, values, weights, advantages, entropies,
actions, ...), the posterior latents returned for the replay write-back, the
optimizer counters and the norm of each parameter's optimizer moments ``nu``
and ``mu``, and the classes of the imagined prior samples; once (prefix
``init/``) the starting parameters of every module including the slow
critic. The scalar metrics of each call are in ``meta["metrics"]``.

Running it (needs the reference code and its dependencies, so not part of the
Ajax test suite and not collected by pytest). From a checkout of
``https://github.com/danijar/dreamerv3`` at ``29eb964``, with the throwaway
venv of ``dreamerv3_world_model_fixtures.py`` (jax 0.4.26,
tensorflow-probability 0.24, optax 0.2.2, chex, einops, ruamel.yaml, numpy
1.26, pyzmq)::

    cd dreamerv3_29eb964
    JAX_PLATFORMS=cpu PYTHONPATH=. <venv>/bin/python \\
        <ajax>/docs/world_models/parity/dreamerv3_train_fixtures.py

``--out`` overrides the output directory (default: the Ajax checkout's
``tests/agents/DreamerV3/fixtures``).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.getcwd())  # the reference checkout (see the docstring)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dreamerv3_world_model_fixtures as wm  # noqa: E402
import embodied  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from dreamerv3 import agent as agt  # noqa: E402
from dreamerv3 import jaxutils, nets  # noqa: E402
from dreamerv3 import ninjax as nj  # noqa: E402
from tensorflow_probability.substrates.jax.internal import samplers  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

WARMUP = 2
NUM_CALLS = 3
#: The action sizes (module docstring), as ``wm._spaces`` / ``wm._batch`` take them.
ACTIONS = {"act_dim": 7, "num_actions": 6}
#: The world-model fixtures' widths and batch shape; their zeroed actor-critic
#: loss scales and shortened imagination are dropped (defaults instead).
TINY = {
    **{
        k: v
        for k, v in wm.TINY.items()
        if not k.startswith("loss_scales.") and k != "imag_length"
    },
    "opt.warmup": WARMUP,
}
TWOHOT_OUTPUTS = (
    "agent/rew/dist/out/",
    "agent/critic/dist/out/",
    "agent/slowcritic/dist/out/",
)
ACTOR_CRITIC = ("agent/actor/", "agent/critic/", "agent/slowcritic/")
OPTIMISED = (*wm.WORLD_MODEL, "agent/actor/", "agent/critic/")
RETNORM_INIT = {"continuous": (0.0, 0.0), "discrete": (-1.5, 2.5)}
#: The two-hot output biases: ``max(-0.5 |j - middle|, PROFILE_FLOOR)``.
PROFILE_SLOPE, PROFILE_FLOOR = 0.5, -27.0
#: Every draw's ``logits + noise`` must keep its top two this far apart.
MIN_MARGIN = 1e-4
#: The intermediate tensors read through ``jaxutils.tensorstats``.
TENSORS = (
    "adv",
    "rew",
    "weight",
    "val",
    "ret",
    "ret_normed",
    "replay_ret",
    "act/action",
    "ent/action",
    "rand/action",
    "prior_ent",
    "post_ent",
)


# ----------------------------------------------------------------- recording


class _Recorder:
    """Trace-time side channel for values the reference does not return.

    ``site`` names the latent draw being made (``post`` inside an observe
    step, ``prior`` inside an imagine step); ``pending`` collects the values
    recorded in the current scan body (or at the top level of the loss);
    ``scanned`` the per-scan stacks of those values.
    """

    def __init__(self) -> None:
        self.site: str | None = None
        self.pending: dict[str, Any] = {}
        self.scanned: list[dict[str, Any]] = []
        self.tensors: dict[str, Any] = {}
        self.retnorm: Any = None
        self.grad: Any = None
        self.updates: Any = None

    def put(self, key: str, value: Any) -> None:
        assert key not in self.pending, key
        self.pending[key] = value


REC = _Recorder()


def _gumbel(seed: Any, logits: Any) -> Any:
    """The Gumbel noise of tfp's JAX categorical sampler for ``seed``.

    ``random_generators._categorical_jax``: ``gumbel(seed, logits_2d.shape +
    (n,))`` and ``argmax`` over the classes; one sample per call (``n = 1``).
    """
    classes = logits.shape[-1]
    z = jax.random.gumbel(
        samplers.sanitize_seed(seed),
        (int(np.prod(logits.shape[:-1])), classes, 1),
        logits.dtype,
    )
    return z[..., 0].reshape(logits.shape)


def _install_recorders() -> None:
    """Wrap reference functions so they also report hidden values.

    Every wrapper calls the original function with the original arguments
    and returns its result unchanged; ``scan`` returns the body's original
    outputs and only adds the recorded values next to them.
    """
    orig_onehot_sample = jaxutils.OneHotDist.sample
    orig_observe = nets.RSSM.observe
    orig_imagine = nets.RSSM.imagine
    orig_scan = jaxutils.scan
    orig_tensorstats = jaxutils.tensorstats
    orig_moments = jaxutils.Moments.__call__
    orig_agent_loss = agt.Agent.inner.loss
    orig_grad = nj.grad
    orig_apply_updates = optax.apply_updates

    def onehot_sample(self, sample_shape=(), seed=None):
        out = orig_onehot_sample(self, sample_shape, seed)
        if REC.site is not None:
            logits = self._logits_parameter_no_checks()
            REC.put(f"{REC.site}_noise", _gumbel(seed, logits))
            REC.put(f"{REC.site}_logit", logits)
            # Independent asks its OneHotDist for a sample of shape (1,).
            REC.put(f"{REC.site}_sample", out.reshape(logits.shape))
        return out

    def observe(self, carry, action, embed, reset, bdims=2):
        if bdims != 1:
            return orig_observe(self, carry, action, embed, reset, bdims=bdims)
        REC.site = "post"
        try:
            return orig_observe(self, carry, action, embed, reset, bdims=1)
        finally:
            REC.site = None

    def imagine(self, carry, action, bdims=2):
        if bdims != 1:
            return orig_imagine(self, carry, action, bdims=bdims)
        REC.site = "prior"
        try:
            return orig_imagine(self, carry, action, bdims=1)
        finally:
            REC.site = None

    def sample(dist):
        # agent.py:20-21, ``{k: v.sample(seed=nj.seed())}``, the same seeds.
        out = {}
        for key, value in dist.items():
            seed = nj.seed()
            out[key] = value.sample(seed=seed)
            if isinstance(value, jaxutils.OneHotDist):
                logits = value._logits_parameter_no_checks()
                REC.put("act_noise", _gumbel(seed, logits))
                REC.put("act_logit", logits)
            else:  # tfd.Independent(tfd.Normal): Normal._sample_n with n = 1
                normal = value.distribution
                shape = (1, *normal.batch_shape)
                eps = jax.random.normal(samplers.sanitize_seed(seed), shape)
                REC.put("act_noise", eps[0])
                REC.put("act_mean", normal.loc)
                REC.put("act_std", normal.scale)
            REC.put("act_sample", out[key])
        return out

    def scan(fun, carry, xs, unroll=False, axis=0):
        def body(carry, x):
            outer, REC.pending = REC.pending, {}
            try:
                carry, ys = fun(carry, x)
                recorded = REC.pending
            finally:
                REC.pending = outer
            return carry, (ys, recorded)

        carry, (ys, recorded) = orig_scan(body, carry, xs, unroll, axis)
        REC.scanned.append(recorded)
        return carry, ys

    def tensorstats(tensor, prefix=None):
        if prefix in TENSORS:
            REC.tensors[prefix] = tensor
        return orig_tensorstats(tensor, prefix)

    def moments(self, x, update=True):
        offset, scale = orig_moments(self, x, update)
        if self.name == "retnorm":
            REC.retnorm = (offset, scale)
        return offset, scale

    def agent_loss(self, data, carry, update=True):
        REC.scanned, REC.pending, REC.tensors = [], {}, {}
        loss, (outs, carry, metrics) = orig_agent_loss(self, data, carry, update)
        observe_scan, imagine_scan = REC.scanned
        extra = {f"rec_{k}": v for k, v in observe_scan.items()}
        extra.update({f"rec_imag_{k}": v for k, v in imagine_scan.items()})
        extra.update({f"rec_start_{k}": v for k, v in REC.pending.items()})
        extra.update({f"rec_tensor/{k}": v for k, v in REC.tensors.items()})
        extra["rec_retnorm_offset"], extra["rec_retnorm_scale"] = REC.retnorm
        return loss, ({**outs, **extra}, carry, metrics)

    def grad(fun, keys, has_aux=False):
        inner = orig_grad(fun, keys, has_aux)

        def wrapper(*args, **kwargs):
            out = inner(*args, **kwargs)
            REC.grad = out
            return out

        return wrapper

    def apply_updates(params, updates):
        REC.updates = updates
        return orig_apply_updates(params, updates)

    jaxutils.OneHotDist.sample = onehot_sample
    nets.RSSM.observe = observe
    nets.RSSM.imagine = imagine
    agt.sample = sample
    jaxutils.scan = scan
    jaxutils.tensorstats = tensorstats
    jaxutils.Moments.__call__ = moments
    agt.Agent.inner.loss = agent_loss
    nj.grad = grad
    optax.apply_updates = apply_updates


# --------------------------------------------------------- starting state


def _starting_params(params: dict, discrete: bool) -> dict:
    """The reference's initial parameters, perturbed (module docstring)."""
    params = wm._perturbed(params, seed=4 + discrete)
    rng = np.random.default_rng(14 + discrete)
    out = dict(params)
    for name in sorted(params):
        if not name.startswith(ACTOR_CRITIC + TWOHOT_OUTPUTS):
            continue
        value = np.asarray(params[name])
        if name.startswith(TWOHOT_OUTPUTS):
            if name.endswith("/kernel"):
                std = 0.3 / np.sqrt(value.shape[0])
                new = rng.normal(0.0, std, value.shape).astype(np.float32)
                new[:, -1] = 0.0  # the dropped 256th logit
            else:
                middle = (value.shape[0] - 2) // 2  # 127 of 0..254
                distance = np.abs(np.arange(value.shape[0]) - middle)
                profile = np.maximum(-PROFILE_SLOPE * distance, PROFILE_FLOOR)
                new = (profile + rng.normal(0.0, 0.1, value.shape)).astype(np.float32)
                new[-1] = 0.0
            out[name] = jnp.asarray(new)
        else:
            if name.endswith("/kernel"):
                std = 0.6 / np.sqrt(np.prod(value.shape[:-1]))
            elif name.endswith("/bias"):
                std = 0.3
            else:
                std = 0.25
            noise = rng.normal(0.0, std, value.shape).astype(np.float32)
            out[name] = jnp.asarray(value + noise)
    kind = "discrete" if discrete else "continuous"
    low, high = RETNORM_INIT[kind]
    out["agent/retnorm/low/value"] = jnp.asarray(low, jnp.float32)
    out["agent/retnorm/high/value"] = jnp.asarray(high, jnp.float32)
    return out


def _batch(discrete: bool, call: int) -> dict[str, np.ndarray]:
    """The world-model fixtures' batch layout, new values for each call."""
    return wm._batch(discrete, seed=100 + 10 * call + int(discrete), **ACTIONS)


# ------------------------------------------------------------------ the run


def _train_call(agent: Any, params: dict, data: dict, carry: Any, seed: Any) -> dict:
    """One real ``Agent.train`` call; returns the recorded values."""

    def run(params, data, carry, seed):
        state, (train_outs, _, metrics) = nj.pure(agent.train)(
            params, data, carry, seed=seed
        )
        loss, _, grads, (outs, _, _) = REC.grad
        return {
            "state": state,
            "loss": loss,
            "grads": grads,
            "outs": outs,
            "train": train_outs,
            "metrics": metrics,
            "updates": REC.updates,
        }

    return jax.device_get(jax.jit(run)(params, data, carry, seed))


@jax.jit
def _scale_shift(noise: Any, std: Any, mean: Any) -> Any:
    return noise * std + mean


def _margin(logits: np.ndarray, noise: np.ndarray) -> float:
    scores = np.sort(np.asarray(logits) + np.asarray(noise), -1)
    return float(np.min(scores[..., -1] - scores[..., -2]))


def _check_draws(name: str, outs: dict, discrete: bool) -> dict[str, float]:
    """The recorded noise reproduces every sample; returns the tie margins."""
    margins = {}
    for site, prefix in (("post", "rec_post"), ("prior", "rec_imag_prior")):
        noise, logit = outs[f"{prefix}_noise"], outs[f"{prefix}_logit"]
        sample = np.asarray(outs[f"{prefix}_sample"])
        redrawn = (np.asarray(logit) + noise).argmax(-1)
        assert np.array_equal(sample.argmax(-1), redrawn), f"{name}: {site}"
        assert np.array_equal(sample.max(-1), np.ones(redrawn.shape)), name
        margins[site] = _margin(logit, noise)
    for prefix in ("rec_start_act", "rec_imag_act"):
        noise, sample = outs[f"{prefix}_noise"], np.asarray(outs[f"{prefix}_sample"])
        if discrete:
            logit = outs[f"{prefix}_logit"]
            redrawn = (np.asarray(logit) + noise).argmax(-1)
            assert np.array_equal(sample.argmax(-1), redrawn), f"{name}: {prefix}"
            margins["act"] = min(margins.get("act", np.inf), _margin(logit, noise))
        else:
            # Under jit, XLA contracts tfp's ``sampled * scale + loc`` into a
            # fused multiply-add; the same jitted expression reproduces it.
            mean, std = outs[f"{prefix}_mean"], outs[f"{prefix}_std"]
            redrawn = jax.device_get(_scale_shift(noise, std, mean))
            assert np.array_equal(sample, redrawn), f"{name}: {prefix}"
    return margins


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    """The scalar metrics as Python floats (exact for float32 and int32)."""
    out = {}
    for key, value in metrics.items():
        value = np.asarray(value)
        if value.shape == () and np.issubdtype(value.dtype, np.number):
            out[key] = float(value)
    return out


def _check_call(
    call: int, before: dict, result: dict, opt_keys: list[str], lr: float
) -> float:
    """Self-checks of one call; returns the smallest non-zero |gradient|.

    The update is the optimizer's: zero at call 0 (warmup), and the new
    parameters are the old ones plus it. XLA contracts the warmup factor and
    the addition into a fused multiply-add, so the stored parameters can
    differ from ``before + update`` by one ulp of the larger of the two. No
    step exceeds 1.5 learning rates (module docstring, reason 3). The slow
    critic is a hard copy of the critic after the first update, and the
    counters advance once per call.
    """
    state, grads, updates = result["state"], result["grads"], result["updates"]
    assert set(updates) == set(grads) == set(opt_keys)
    for key, update in updates.items():
        old, new = np.asarray(before[key]), np.asarray(state[key])
        error = np.abs(new - (old + update).astype(np.float32))
        assert np.all(error <= np.spacing(np.maximum(np.abs(new), np.abs(update)))), key
        assert np.max(np.abs(update)) <= 1.5 * lr, key
        if call == 0:
            assert not np.any(update) and np.array_equal(new, old), key
    if call == 0:
        for key in grads:
            if key.startswith("agent/critic/"):
                slow = key.replace("/critic/", "/slowcritic/")
                np.testing.assert_array_equal(state[slow], state[key])
    assert int(state["agent/opt/step/value"]) == call + 1
    assert int(state["agent/updater/updates/value"]) == call + 1
    return min(float(np.min(np.abs(g[g != 0]))) for g in grads.values() if np.any(g))


def _call_arrays(
    call: int, data: dict, result: dict, opt_keys: list[str], terms: list[str]
) -> dict[str, np.ndarray]:
    """What one call saves, under the prefix ``c{call}/``."""
    state, outs = result["state"], result["outs"]
    arrays: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if key in ("deter", "stoch"):  # only the context row is read
            arrays[f"batch/context_{key}"] = np.asarray(value[:, 0])
        elif key != "stepid":
            arrays[f"batch/{key}"] = np.asarray(value)
    arrays["noise/post"] = np.asarray(outs["rec_post_noise"])
    # [H, N, ...] scan stacks -> start-major [N, H, ...].
    arrays["noise/prior"] = np.swapaxes(outs["rec_imag_prior_noise"], 0, 1)
    imag_act = np.swapaxes(outs["rec_imag_act_noise"], 0, 1)
    start_act = np.asarray(outs["rec_start_act_noise"])[:, None]
    arrays["noise/action"] = np.concatenate([start_act, imag_act], 1)
    # The imagined prior samples' classes, start-major [N, H, S].
    prior_sample = np.asarray(outs["rec_imag_prior_sample"]).argmax(-1)
    arrays["draw/prior"] = np.swapaxes(prior_sample, 0, 1).astype(np.int32)
    for term in terms:
        arrays[f"loss/{term}"] = np.asarray(outs[f"{term}_loss"])
    arrays["loss/total"] = np.asarray(result["loss"], np.float32)
    for key, value in result["grads"].items():
        arrays[f"grad/{key}"] = np.asarray(value, np.float32)
    if call > 0:
        for key, value in result["updates"].items():
            arrays[f"update/{key}"] = np.asarray(value, np.float32)
    for key, value in state.items():
        if key.startswith("agent/slowcritic/"):
            arrays[f"slow/{key}"] = np.asarray(value, np.float32)
    for key in ("low", "high"):
        value = state[f"agent/retnorm/{key}/value"]
        arrays[f"retnorm/{key}"] = np.asarray(value, np.float32)
    arrays["retnorm/offset"] = np.asarray(outs["rec_retnorm_offset"])
    arrays["retnorm/scale"] = np.asarray(outs["rec_retnorm_scale"])
    for key in TENSORS:
        arrays[f"tensor/{key}"] = np.asarray(outs[f"rec_tensor/{key}"])
    replay = result["train"]["replay"]
    arrays["out/deter"] = np.asarray(replay["deter"], np.float32)
    arrays["out/stoch"] = np.asarray(replay["stoch"], np.int32)
    # The chain's state: (agc (), rms (count, nu), momentum (count, mu),
    # scale ()). The moments are stored as one Euclidean norm per parameter
    # (in the order of ``meta["opt_keys"]``): given the gradients and
    # updates, which are stored in full, they are redundant except after
    # call 0, whose update is 0.
    opt_state = state["agent/opt/state"]
    arrays["opt/rms_count"] = np.asarray(opt_state[1][0])
    arrays["opt/momentum_count"] = np.asarray(opt_state[2][0])
    arrays["opt/step"] = np.asarray(state["agent/opt/step/value"])
    for moment, values in (("nu", opt_state[1][1]), ("mu", opt_state[2][1])):
        arrays[f"opt/{moment}_norm"] = np.asarray(
            [np.linalg.norm(np.asarray(values[k], np.float64)) for k in opt_keys],
            np.float32,
        )
    return {f"c{call}/{key}": value for key, value in arrays.items()}


def _meta(agent: Any, kind: str) -> dict[str, Any]:
    """The reference's configuration, for the tests to pin Ajax's defaults."""
    cfg = agent.config
    return {
        "reference": "danijar/dreamerv3@29eb964",
        "run": kind,
        "calls": NUM_CALLS,
        "action_dim": ACTIONS["num_actions" if kind == "discrete" else "act_dim"],
        "overrides": {k: v for k, v in TINY.items() if not k.startswith("jax.")},
        "scales": {k: float(v) for k, v in agent.scales.items()},
        "opt": dict(cfg.opt),
        "rssm": dict(cfg.dyn.rssm),
        "actor": dict(cfg.actor),
        "critic": dict(cfg.critic),
        "rewhead": dict(cfg.rewhead),
        "imag_length": cfg.imag_length,
        "horizon": float(cfg.horizon),
        "return_lambda": cfg.return_lambda,
        "return_lambda_replay": cfg.return_lambda_replay,
        "actent": cfg.actent,
        "slowreg": cfg.slowreg,
        "slowtar": cfg.slowtar,
        "slow_critic_fraction": cfg.slow_critic_fraction,
        "slow_critic_update": cfg.slow_critic_update,
        "retnorm": dict(cfg.retnorm),
        "retnorm_init": RETNORM_INIT[kind],
        "valnorm": dict(cfg.valnorm),
        "advnorm": dict(cfg.advnorm),
        "contdisc": cfg.contdisc,
        "ac_grads": cfg.ac_grads,
        "imag_start": cfg.imag_start,
        "imag_repeat": cfg.imag_repeat,
        "replay_critic": [
            cfg.replay_critic_loss,
            cfg.replay_critic_grad,
            cfg.replay_critic_bootstrap,
        ],
        "reward_grad": cfg.reward_grad,
        "actor_dist": [cfg.actor_dist_disc, cfg.actor_dist_cont],
        "free": float(cfg.rssm_loss.free),
    }


def _record(discrete: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Three consecutive ``Agent.train`` calls; the arrays and the meta."""
    kind = "discrete" if discrete else "continuous"
    config = embodied.Config(agt.Agent.configs["defaults"]).update(TINY)
    agent, params, carry, dummy, _ = wm._build(config, discrete, **ACTIONS)
    params = nj.init(agent.train)(
        params, dummy, carry, seed=jnp.array([0, 0], jnp.uint32)
    )
    # The initialiser leaves every counter and statistic at its start.
    for key in ("agent/opt/step/value", "agent/updater/updates/value"):
        assert int(params[key]) == 0, key
    params = _starting_params(params, discrete)
    arrays: dict[str, np.ndarray] = {}
    for key, value in params.items():
        if key.startswith((*OPTIMISED, "agent/slowcritic/")):
            arrays[f"init/{key}"] = np.asarray(value, np.float32)
    margins: dict[str, float] = {}
    metrics: list[dict[str, float]] = []
    smallest_grad = np.inf
    opt_keys = sorted(k for k in params if k.startswith(OPTIMISED))
    for call in range(NUM_CALLS):
        data = _batch(discrete, call)
        seed = jnp.array([1, 2 + call], jnp.uint32)
        result = _train_call(agent, params, data, carry, seed)
        name = f"{kind} call {call}"
        outs = result["outs"]
        for site, margin in _check_draws(name, outs, discrete).items():
            margins[site] = min(margins.get(site, np.inf), margin)
        lr = agent.config.opt.lr * min(call / WARMUP, 1.0)
        smallest = _check_call(call, params, result, opt_keys, lr)
        smallest_grad = min(smallest_grad, smallest)
        arrays.update(_call_arrays(call, data, result, opt_keys, list(agent.scales)))
        metrics.append(_scalar_metrics(result["metrics"]))
        ret, scale = outs["rec_tensor/ret"], float(outs["rec_retnorm_scale"])
        print(
            f"{name}: loss {float(result['loss']):.5f}, "
            f"actor {float(np.mean(outs['actor_loss'])):.3e}, "
            f"critic {float(np.mean(outs['critic_loss'])):.4f}, "
            f"repval {float(np.mean(outs['replay_critic_loss'])):.4f}, "
            f"ret in [{np.min(ret):.3f}, {np.max(ret):.3f}], retnorm scale {scale:.4f}"
        )
        params = result["state"]

    meta = _meta(agent, kind)
    meta.update(
        min_margins=margins,
        smallest_nonzero_grad=smallest_grad,
        opt_keys=opt_keys,
        metrics=metrics,
    )
    # Below this, (1 - beta2) g^2 is not a normal float32 (module docstring).
    flushed = float(np.sqrt(np.finfo(np.float32).tiny / (1 - agent.config.opt.beta2)))
    print(f"{kind}: smallest top-2 margins of the categorical draws {margins}")
    print(
        f"{kind}: smallest non-zero |gradient| {smallest_grad:.2e}, "
        f"{smallest_grad / flushed:.1f} x the flush threshold {flushed:.2e}"
    )
    assert min(margins.values()) > MIN_MARGIN, margins
    assert smallest_grad > flushed, (smallest_grad, flushed)
    return arrays, meta


def main() -> None:
    default_out = (
        pathlib.Path(__file__).resolve().parents[3]
        / "tests"
        / "agents"
        / "DreamerV3"
        / "fixtures"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=default_out)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _install_recorders()
    for discrete in (False, True):
        arrays, meta = _record(discrete)
        arrays["meta"] = np.asarray(json.dumps(meta, sort_keys=True, default=str))
        path = args.out / f"dreamerv3_train_{meta['run']}.npz"
        np.savez_compressed(path, **arrays)
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
