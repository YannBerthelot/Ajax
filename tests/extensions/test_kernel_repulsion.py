"""KernelRepulsion: the SVGD-style repulsion between an ensemble's critics."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax.extensions.base import ExtensionContext
from ajax.extensions.ensemble import KernelRepulsion, q_kernel_repulsion

OBS = jnp.arange(8.0).reshape(4, 2)
ACTIONS = jnp.ones((4, 1))


def _apply(params: Any, x: jax.Array) -> jax.Array:
    """A linear ensemble: critic i predicts w_i * sum(obs, action)."""
    return params["w"][:, None, None] * x.sum(-1, keepdims=True)[None]


def _phase(params: Any, coef: float = 1.0) -> tuple[jax.Array, dict]:
    batch = {
        "observations": OBS,
        "actions": ACTIONS,
        "critic_params": params,
        "critic_state": SimpleNamespace(apply_fn=_apply),
    }
    ctx = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    return KernelRepulsion(coef=coef).critic_loss(None, (), batch, ctx)


def _term(params: Any, coef: float = 1.0) -> jax.Array:
    return _phase(params, coef)[0]


def test_the_term_is_coef_times_the_critics_mean_kernel() -> None:
    params = {"w": jnp.array([1.0, 1.5, 3.0, 4.0])}
    q_preds = _apply(params, jnp.concatenate((OBS, ACTIONS), axis=-1))
    np.testing.assert_allclose(
        _term(params, coef=0.5), 0.5 * q_kernel_repulsion(q_preds), rtol=1e-6
    )
    # Identical critics: every kernel entry is exp(0) = 1.
    assert float(q_kernel_repulsion(jnp.ones((4, 3, 1)))) == 1.0


def test_it_logs_the_term_the_kernel_and_the_bandwidth() -> None:
    """The bandwidth is the median heuristic over the critics' squared
    distances, ``median / log(n)``."""
    params = {"w": jnp.array([1.0, 1.5, 3.0, 4.0])}
    term, metrics = _phase(params, coef=0.5)
    feats = np.asarray(_apply(params, jnp.concatenate((OBS, ACTIONS), axis=-1)))
    feats = feats.reshape(4, -1)
    sq_dists = ((feats[:, None] - feats[None]) ** 2).sum(-1)
    np.testing.assert_allclose(
        metrics["kernel_repulsion/bandwidth"],
        np.median(sq_dists) / np.log(4),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        metrics["kernel_repulsion/kernel"], term / 0.5, rtol=1e-6
    )
    assert metrics["kernel_repulsion/term"] is term


def test_its_gradient_pushes_the_critics_apart() -> None:
    params = {"w": jnp.array([1.0, 1.1, 1.2, 1.3])}
    grads = jax.grad(_term)(params)
    stepped = params["w"] - 0.1 * grads["w"]
    assert float(jnp.std(stepped)) > float(jnp.std(params["w"]))
