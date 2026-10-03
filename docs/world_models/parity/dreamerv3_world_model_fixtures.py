"""Parity fixtures for Ajax's DreamerV3 world model, made by the real reference code.

This script runs the paper-era DreamerV3 code itself -- ``danijar/dreamerv3``
at ``29eb964`` (the 2411f7d algorithm plus the upstream replay-context
``prevact`` fix; ``docs/world_models/deviations.md`` section 1) -- on a tiny
float32 configuration and saves what Ajax's world model must reproduce to
``tests/agents/DreamerV3/fixtures/*.npz``. The tests in
``tests/agents/DreamerV3/test_dreamerv3_parity.py`` load these files; they do
not need the reference code.

What it runs. The real ``dreamerv3.Agent`` (the inner ninjax module; the
JAXAgent wrapper only adds devices, jit and threading) is built for a vector
observation ``vector`` of dimension 5 and either a continuous action of
dimension 2 or a discrete action with 3 classes, with ``rssm`` deter 64,
hidden 12, stoch 4, classes 3, blocks 8, encoder / decoder / reward /
continue units 16, ``jax.compute_dtype`` float32, batch 2 and
``batch_length`` 9 = 8 trained steps + ``replay_context`` 1. The sizes are
pairwise distinct where Ajax could confuse them (``stoch`` vs ``classes``,
the RSSM's ``hidden`` vs the MLPs' ``units``), so a mix-up changes shapes or
values instead of passing unnoticed. Only widths (and the batch shape) are
overridden: every other hyperparameter -- layer counts, unimix, free bits,
``horizon``, loss scales, bins -- is the reference's default, recorded in the
fixtures' ``meta`` so that the tests pin Ajax's defaults against it. Its parameters are initialised exactly as ``JAXAgent._init_params`` does
(``29eb964:dreamerv3/jaxagent.py:380-389``). One call of the real
``Agent.train`` (``dreamerv3/agent.py:166-223``: replay context, then
``Optimizer.__call__`` -> ``nj.grad(Agent.loss)``) then runs on a fixed
synthetic batch that contains episode boundaries: an ``is_first`` at the first
trained step (it zeroes the stored context latent), a mid-sequence
``is_first``, two ``is_terminal`` steps and a time-limit ``is_last``.

World-model terms only. ``loss_scales.actor``, ``.critic`` and
``.replay_critic`` are set to 0, so the gradient computed by the reference's
own ``nj.grad`` call is the gradient of the weighted world-model loss
``sum_k scale_k * mean(loss_k)`` over ``vector`` (rec), ``reward``, ``cont``,
``dyn`` and ``rep``. The rest of ``Agent.loss`` (imagination, actor, critic,
replay critic) still runs: it draws its random numbers only after the
posterior samples, so they are unaffected, and its terms enter the total as
``0 * term``. The script asserts that the actor and critic gradients are then
exactly zero, i.e. that nothing else reaches the recorded gradients.

Recording the random draws without changing them. The only random draws of
the world-model loss are the posterior one-hot samples
(``nets.py:70``, ``tfd.Independent(OneHotDist(logit)).sample(seed)``), which
tfp's JAX backend draws as ``argmax(logits + gumbel(seed))``. A wrapper around
``jaxutils.OneHotDist.sample`` calls the original method unchanged and, next to
it, evaluates ``jax.random.gumbel`` on the same seed and shape, i.e. the
noise the original draw used. Thin wrappers around ``RSSM.observe``,
``RSSM.loss``, ``Agent.loss`` and ``ninjax.grad`` carry that noise, the prior
logits, the KL before free bits and the gradients out of the traced
function; none of them changes a value the reference computes. The script
asserts ``argmax(posterior logits + noise) == recorded sample`` for every
latent, so the saved noise is the noise of the draw.

A third run perturbs the initial world-model parameters (seeded Gaussian
noise) and repeats the same call. At initialisation the reward head is zero
(no reward gradient reaches its trunk or the RSSM) and the KL can sit below
the free-bits threshold (no ``dyn`` / ``rep`` gradient); the perturbed run
exercises those paths. A last file records the parameter shapes of the real
``size12m`` world model (vector obs 24, continuous action 6) for the
parameter-count test.

Running it (needs the reference code and its dependencies, so not part of the
Ajax test suite and not collected by pytest). From a checkout of
``https://github.com/danijar/dreamerv3`` at ``29eb964``, with a throwaway venv
holding jax 0.4.26, tensorflow-probability 0.24, optax 0.2.2, chex, einops,
ruamel.yaml, numpy 1.26 and pyzmq::

    cd dreamerv3_29eb964
    JAX_PLATFORMS=cpu PYTHONPATH=. <venv>/bin/python \\
        <ajax>/docs/world_models/parity/dreamerv3_world_model_fixtures.py

``--out`` overrides the output directory (default: the Ajax checkout's
``tests/agents/DreamerV3/fixtures``). The files are small (kB) and committed.
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

import embodied  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from dreamerv3 import agent as agt  # noqa: E402
from dreamerv3 import jaxutils, nets  # noqa: E402
from dreamerv3 import ninjax as nj  # noqa: E402
from tensorflow_probability.substrates.jax.internal import samplers  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

WORLD_MODEL = ("agent/enc/", "agent/dyn/", "agent/dec/", "agent/rew/", "agent/con/")
OBS_DIM = 5
BATCH, LENGTH = 2, 9  # 8 trained steps + replay_context 1
TINY = {
    "jax.platform": "cpu",
    "jax.compute_dtype": "float32",
    "jax.param_dtype": "float32",
    "jax.transfer_guard": False,
    "batch_size": BATCH,
    "batch_length": LENGTH,
    "replay_context": 1,
    "dyn.rssm.deter": 64,
    "dyn.rssm.hidden": 12,
    "dyn.rssm.stoch": 4,
    "dyn.rssm.classes": 3,
    "dyn.rssm.blocks": 8,
    "enc.simple.units": 16,
    "dec.simple.units": 16,
    "rewhead.units": 16,
    "conhead.units": 16,
    "actor.units": 16,
    "critic.units": 16,
    "imag_length": 3,
    # World-model terms only (see the module docstring).
    "loss_scales.actor": 0.0,
    "loss_scales.critic": 0.0,
    "loss_scales.replay_critic": 0.0,
}
WM_TERMS = ("vector", "reward", "cont", "dyn", "rep")


# ----------------------------------------------------------------- recording


class _Recorder:
    """Trace-time side channel for values the reference does not return."""

    def __init__(self) -> None:
        self.in_observe_step = False
        self.step_noise: list[Any] = []
        self.observe_noise: Any = None
        self.rssm_loss: dict[str, Any] = {}
        self.grad: Any = None


REC = _Recorder()


def _install_recorders() -> None:
    """Wrap reference functions so they also report hidden values.

    Every wrapper calls the original function with the original arguments
    and returns its result unchanged (``RSSM.observe`` at ``bdims=1`` adds a
    ``gumbel`` entry to the step outputs, which its ``bdims=2`` wrapper
    removes again before anything downstream sees it).
    """
    orig_sample = jaxutils.OneHotDist.sample
    orig_observe = nets.RSSM.observe
    orig_rssm_loss = nets.RSSM.loss
    orig_agent_loss = agt.Agent.inner.loss
    orig_grad = nj.grad

    def sample(self, sample_shape=(), seed=None):
        out = orig_sample(self, sample_shape, seed)
        if REC.in_observe_step:
            # The draw of tfp's JAX backend, random_generators._categorical_jax:
            # gumbel(seed, logits_2d.shape + (n,)), argmax over the classes.
            # One sample per call (Independent passes n = 1).
            logits = self._logits_parameter_no_checks()
            classes = logits.shape[-1]
            z = jax.random.gumbel(
                samplers.sanitize_seed(seed),
                (int(np.prod(logits.shape[:-1])), classes, 1),
                logits.dtype,
            )
            REC.step_noise.append(z[..., 0].reshape(logits.shape))
        return out

    def observe(self, carry, action, embed, reset, bdims=2):
        if bdims == 1:
            REC.in_observe_step = True
            try:
                carry, outs = orig_observe(self, carry, action, embed, reset, bdims=1)
            finally:
                REC.in_observe_step = False
            (noise,) = REC.step_noise
            REC.step_noise.clear()
            return carry, {**outs, "gumbel": noise}
        carry, outs = orig_observe(self, carry, action, embed, reset, bdims=bdims)
        outs = dict(outs)
        REC.observe_noise = outs.pop("gumbel")
        return carry, outs

    def rssm_loss(self, outs, free=1.0):
        losses, metrics = orig_rssm_loss(self, outs, free)
        # Diagnostics only (not part of the loss): the prior logits and the
        # KL before free bits, recomputed with the reference's own methods.
        prior = self._prior(outs.get("feat", outs["deter"]))
        post = outs["logit"]
        kl = self._dist(jax.lax.stop_gradient(post)).kl_divergence(
            self._dist(jax.lax.stop_gradient(prior))
        )
        REC.rssm_loss = {"dyn": losses["dyn"], "rep": losses["rep"]}
        REC.rssm_loss.update(prior=prior, kl=kl)
        return losses, metrics

    def agent_loss(self, data, carry, update=True):
        loss, (outs, carry, metrics) = orig_agent_loss(self, data, carry, update)
        outs = {
            **outs,
            "rec_noise": REC.observe_noise,
            "rec_prior": REC.rssm_loss["prior"],
            "rec_kl": REC.rssm_loss["kl"],
            "rec_dyn": REC.rssm_loss["dyn"],
            "rec_rep": REC.rssm_loss["rep"],
        }
        return loss, (outs, carry, metrics)

    def grad(fun, keys, has_aux=False):
        inner = orig_grad(fun, keys, has_aux)

        def wrapper(*args, **kwargs):
            out = inner(*args, **kwargs)
            REC.grad = out
            return out

        return wrapper

    jaxutils.OneHotDist.sample = sample
    nets.RSSM.observe = observe
    nets.RSSM.loss = rssm_loss
    agt.Agent.inner.loss = agent_loss
    nj.grad = grad


# --------------------------------------------------------------- the agent


def _config(extra: dict[str, Any] | None = None) -> Any:
    config = embodied.Config(agt.Agent.configs["defaults"])
    return config.update({**TINY, **(extra or {})})


def _spaces(discrete: bool, obs_dim: int = OBS_DIM, act_dim: int = 2) -> tuple:
    obs_space = {
        "vector": embodied.Space(np.float32, (obs_dim,)),
        "reward": embodied.Space(np.float32),
        "is_first": embodied.Space(bool),
        "is_last": embodied.Space(bool),
        "is_terminal": embodied.Space(bool),
    }
    if discrete:
        action = embodied.Space(np.int32, (), 0, 3)
    else:
        action = embodied.Space(np.float32, (act_dim,), -1, 1)
    act_space = {"action": action, "reset": embodied.Space(bool)}
    return obs_space, act_space


def _build(config: Any, discrete: bool, **space_kwargs: Any) -> tuple:
    """The real agent and its initial parameters (``jaxagent.py:380-389``)."""
    obs_space, act_space = _spaces(discrete, **space_kwargs)
    agent = agt.Agent.inner(obs_space, act_space, config, name="agent")
    spaces = {**obs_space, **act_space, **agent.aux_spaces}
    keys = [k for k in spaces if k != "reset"]
    batch, length = config.batch_size, config.batch_length
    dummy = {
        k: np.zeros((batch, length, *spaces[k].shape), spaces[k].dtype) for k in keys
    }
    seed = jnp.array([config.seed, 0], jnp.uint32)
    params = nj.init(agent.init_train, static_argnums=[1])({}, batch, seed=seed)
    _, carry = jax.jit(nj.pure(agent.init_train), static_argnums=[1])(
        params, batch, seed=seed
    )
    return agent, params, carry, dummy, seed


# --------------------------------------------------------------- the batch


def _batch(discrete: bool, seed: int) -> dict[str, np.ndarray]:
    """A fixed replay batch ``[2, 9]`` with episode boundaries.

    Row 0: context, then an episode that terminates at index 3, a reset at 4
    and a terminal step at 8. Row 1: a reset at index 1 (the first trained
    step, so the stored context latent is zeroed), a time-limit truncation at
    5 (``is_last`` without ``is_terminal``) and a reset at 6. ``is_first[:, 0]``
    and ``is_last`` before every reset follow the replay annotation
    (``embodied/replay``, ``is_last |= next is_first``). Actions are zeroed at
    ``is_last`` as the driver does; continuous actions exceed ``|a| = 1`` so
    that the dynamics' ``a / max(1, |a|)`` is exercised.
    """
    rng = np.random.default_rng(seed)
    deter, stoch, classes = (
        TINY[f"dyn.rssm.{k}"] for k in ("deter", "stoch", "classes")
    )
    is_first = np.zeros((BATCH, LENGTH), bool)
    is_first[:, 0] = True
    is_first[0, 4] = True
    is_first[1, 1] = True
    is_first[1, 6] = True
    is_terminal = np.zeros((BATCH, LENGTH), bool)
    is_terminal[0, 3] = True
    is_terminal[0, 8] = True
    is_last = np.zeros((BATCH, LENGTH), bool)
    is_last[:, :-1] |= is_first[:, 1:]
    is_last |= is_terminal
    reward = rng.normal(0.0, 2.0, (BATCH, LENGTH)).astype(np.float32)
    reward[0, 2] = 37.5
    reward[1, 7] = -120.0
    reward[is_first] = 0.0
    vector = rng.normal(0.0, 3.0, (BATCH, LENGTH, OBS_DIM)).astype(np.float32)
    vector[0, 5, 1] = 250.0
    vector[1, 2, 3] = -80.0
    if discrete:
        action = rng.integers(0, 3, (BATCH, LENGTH)).astype(np.int32)
    else:
        action = rng.uniform(-1.6, 1.6, (BATCH, LENGTH, 2)).astype(np.float32)
    action[is_last] = 0
    return {
        "vector": vector,
        "reward": reward,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "action": action,
        "stepid": np.zeros((BATCH, LENGTH, 20), np.uint8),
        "deter": rng.normal(0.0, 0.5, (BATCH, LENGTH, deter)).astype(np.float32),
        "stoch": rng.integers(0, classes, (BATCH, LENGTH, stoch)).astype(np.int32),
    }


def _perturbed(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """World-model parameters plus seeded Gaussian noise (others unchanged).

    Kernels get ``N(0, (0.6 / sqrt(fan_in))^2)``, biases ``N(0, 0.3^2)``, norm
    scales ``N(0, 0.25^2)``: large enough that the zero reward head predicts
    non-zero values and that the KL crosses the free-bits threshold.
    """
    rng = np.random.default_rng(seed)
    out = dict(params)
    for name in sorted(params):
        if not name.startswith(WORLD_MODEL):
            continue
        value = np.asarray(params[name])
        if name.endswith("/kernel"):
            fan_in = int(np.prod(value.shape[:-1]))
            std = 0.6 / np.sqrt(fan_in)
        elif name.endswith("/bias"):
            std = 0.3
        else:
            std = 0.25
        noise = rng.normal(0.0, std, value.shape).astype(np.float32)
        out[name] = jnp.asarray(value + noise)
    return out


# ------------------------------------------------------------------ the run


def _train_call(agent: Any, params: dict, data: dict, carry: Any, seed: Any) -> dict:
    """One real ``Agent.train`` call; returns the recorded values."""

    def run(params, data, carry, seed):
        _, (train_outs, _, _) = nj.pure(agent.train)(params, data, carry, seed=seed)
        loss, _, grads, (outs, _, _) = REC.grad
        return {"loss": loss, "grads": grads, "outs": outs, "train": train_outs}

    return jax.device_get(jax.jit(run)(params, data, carry, seed))


def _record(name: str, agent: Any, params: dict, carry: Any, data: dict) -> dict:
    seed = jnp.array([1, 2], jnp.uint32)
    result = _train_call(agent, params, data, carry, seed)
    outs, grads = result["outs"], result["grads"]
    replay = outs["replay_outs"]
    scales = agent.scales

    # The recorded noise is the noise of the reference's draws.
    sample = np.asarray(replay["stoch"]).argmax(-1)
    redrawn = (np.asarray(replay["logit"]) + outs["rec_noise"]).argmax(-1)
    assert np.array_equal(sample, redrawn), f"{name}: noise does not match samples"
    assert np.array_equal(np.asarray(replay["stoch"]).max(-1), np.ones(sample.shape))
    # Only world-model terms reach the gradient.
    for key, value in grads.items():
        if not key.startswith(WORLD_MODEL):
            assert not np.any(np.asarray(value)), f"{name}: non-zero grad {key}"
    total = sum(np.mean(outs[f"{k}_loss"], dtype=np.float64) for k in WM_TERMS)
    np.testing.assert_allclose(result["loss"], total, rtol=1e-6)
    # The write-back entries are the posterior of the same forward pass.
    np.testing.assert_array_equal(result["train"]["replay"]["deter"], replay["deter"])
    np.testing.assert_array_equal(result["train"]["replay"]["stoch"], sample)
    # Unscaled terms: rec, reward, cont, dyn have scale 1 (scaled == unscaled).
    unscaled = {k: np.asarray(outs[f"{k}_loss"]) for k in ("vector", "reward", "cont")}
    unscaled["dyn"] = np.asarray(outs["rec_dyn"])
    unscaled["rep"] = np.asarray(outs["rec_rep"])
    for key in WM_TERMS:
        np.testing.assert_array_equal(
            np.asarray(outs[f"{key}_loss"]),
            (unscaled[key] * np.float32(scales[key])).astype(np.float32),
        )

    arrays: dict[str, np.ndarray] = {}
    for key, value in params.items():
        if key.startswith(WORLD_MODEL):
            arrays[f"param/{key}"] = np.asarray(value, np.float32)
    for key, value in grads.items():
        if key.startswith(WORLD_MODEL):
            arrays[f"grad/{key}"] = np.asarray(value, np.float32)
    for key, value in data.items():
        if key != "stepid":
            arrays[f"batch/{key}"] = np.asarray(value)
    for key in WM_TERMS:
        arrays[f"loss/{key}"] = unscaled[key]
    arrays["loss/weighted"] = np.asarray(result["loss"], np.float32)
    arrays["out/kl"] = np.asarray(outs["rec_kl"], np.float32)
    arrays["out/noise"] = np.asarray(outs["rec_noise"], np.float32)
    arrays["out/sample"] = sample.astype(np.int32)
    arrays["out/post_logit"] = np.asarray(replay["logit"], np.float32)
    arrays["out/prior_logit"] = np.asarray(outs["rec_prior"], np.float32)
    arrays["out/deter"] = np.asarray(replay["deter"], np.float32)
    arrays["out/embed"] = np.asarray(outs["embed"], np.float32)
    arrays["out/entry_stoch"] = np.asarray(result["train"]["replay"]["stoch"], np.int32)
    meta = {
        "reference": "danijar/dreamerv3@29eb964",
        "run": name,
        "scales": {k: float(scales[k]) for k in WM_TERMS},
        "horizon": float(agent.config.horizon),
        "free": float(agent.config.rssm_loss.free),
        "rssm": dict(agent.config.dyn.rssm),
        "layers": {
            "enc": agent.config.enc.simple.layers,
            "dec": agent.config.dec.simple.layers,
            "rew": agent.config.rewhead.layers,
            "con": agent.config.conhead.layers,
        },
        "bins": agent.config.rewhead.bins,
    }
    arrays["meta"] = np.asarray(json.dumps(meta, sort_keys=True))
    kl = arrays["out/kl"]
    print(
        f"{name}: loss {float(result['loss']):.6f}, "
        f"KL in [{kl.min():.3f}, {kl.max():.3f}], "
        f"{int((kl > 1.0).sum())}/{kl.size} above free bits, "
        f"reward loss {unscaled['reward'].mean():.4f}"
    )
    return arrays


def _shapes_12m() -> dict[str, np.ndarray]:
    """Parameter shapes of the real ``size12m`` world model (obs 24, action 6)."""
    config = embodied.Config(agt.Agent.configs["defaults"]).update(
        {
            **{k: v for k, v in TINY.items() if k.startswith("jax.")},
            **agt.Agent.configs["size12m"],
            "batch_size": 1,
            "batch_length": 3,
        }
    )
    agent, params, carry, dummy, seed = _build(
        config, discrete=False, obs_dim=24, act_dim=6
    )
    shapes = jax.eval_shape(nj.init(agent.train), params, dummy, carry, seed=seed)
    table = {
        key: list(value.shape)
        for key, value in sorted(shapes.items())
        if key.startswith(WORLD_MODEL)
    }
    meta: dict[str, Any] = {"reference": "danijar/dreamerv3@29eb964"}
    meta.update(preset="size12m", obs_dim=24, action_dim=6, shapes=table)
    return {"meta": np.asarray(json.dumps(meta, sort_keys=True))}


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

    runs = {}
    for discrete in (False, True):
        agent, params, carry, dummy, _ = _build(_config(), discrete)
        params = nj.init(agent.train)(
            params, dummy, carry, seed=jnp.array([0, 0], jnp.uint32)
        )
        data = _batch(discrete, seed=3 if discrete else 2)
        kind = "discrete" if discrete else "continuous"
        runs[f"{kind}_init"] = _record(f"{kind}_init", agent, params, carry, data)
        if not discrete:
            perturbed = _perturbed(params, seed=4)
            runs["continuous_perturbed"] = _record(
                "continuous_perturbed", agent, perturbed, carry, data
            )
    runs["shapes_12m"] = _shapes_12m()

    for name, arrays in runs.items():
        path = args.out / f"dreamerv3_wm_{name}.npz"
        np.savez_compressed(path, **arrays)
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
