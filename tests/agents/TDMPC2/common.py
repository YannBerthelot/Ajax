"""Shared by the TD-MPC2 tests: the tiny agent's sizes, random heads and the
TD target of an update state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from ajax.agents.TDMPC2 import core

#: A tiny agent: small networks, fewer candidates than the paper's 512 / 24
#: / 64 and 2 planning iterations.
TINY_AGENT: dict[str, Any] = {
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "num_q": 2,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 8,
    "num_pi_trajs": 4,
    "iterations": 2,
}


def random_kernel(
    params: Mapping[str, Any], path: tuple[str, ...], key: jax.Array, std: float
):
    """``params`` with the kernel under ``path`` drawn ``N(0, std^2)`` (a
    copy: ``params`` is unchanged)."""
    if not path:
        kernel = std * jax.random.normal(key, params["kernel"].shape)
        return {**params, "kernel": kernel}
    head, *rest = path
    return {**params, head: random_kernel(params[head], tuple(rest), key, std)}


def random_heads(
    wm_params: Mapping[str, Any], reward_key: jax.Array, q_key: jax.Array, std: float
) -> dict:
    """The reward and Q heads' final kernels drawn ``N(0, std^2)``.

    The paper zero-initialises them, so an untrained model's rewards and
    values are all 0: no gradient reaches the world model through them and
    the planner's candidate values all tie.
    """
    wm_params = random_kernel(wm_params, ("reward", "out"), reward_key, std)
    return random_kernel(wm_params, ("q", "members", "out"), q_key, std)


_NOISE = object()  # dropout_key default: the noise's own


def td_target_of(
    state: Any,
    batch: core.TDMPC2Batch,
    noise: core.UpdateNoise,
    config: Any,
    *,
    gamma: Any = 0.99,
    wm_params: Any = None,
    target_q: Any = None,
    pi_params: Any = None,
    dropout_key: Any = _NOISE,
    **kwargs: Any,
) -> tuple[jax.Array, jax.Array]:
    """:func:`core.td_target` of ``state``'s networks on ``batch`` with
    ``noise``'s TD draws: ``(y, next_z)``. The parameters default to the
    state's, the dropout key to ``noise.td_dropout``; ``kwargs`` (``task``,
    ``terminated``) pass through."""
    wm, pi = state.world_model_state, state.actor_state
    return core.td_target(
        wm_apply=wm.apply_fn,
        pi_apply=pi.apply_fn,
        wm_params=wm.params if wm_params is None else wm_params,
        target_q_params=wm.target_params if target_q is None else target_q,
        pi_params=pi.params if pi_params is None else pi_params,
        next_obs=batch.obs[1:],
        reward=batch.reward,
        gamma=gamma,
        eps=noise.td_eps,
        pair=noise.td_pair,
        dropout_key=noise.td_dropout if dropout_key is _NOISE else dropout_key,
        config=config,
        **kwargs,
    )
