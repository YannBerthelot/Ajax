"""TD-MPC2 update parity against the real paper-era reference code.

The fixture ``fixtures/tdmpc2_update.npz`` was recorded by running the
unmodified ``TDMPC2.update()`` of ``nicklashansen/tdmpc2@5f6fade`` for four
consecutive updates (``docs/world_models/parity/tdmpc2_update_fixtures.py``,
whose docstring explains the configuration and how to regenerate it). The
reference's initial parameters are mapped onto Ajax's parameter tree, Ajax's
jitted :func:`ajax.agents.TDMPC2.core.update` is chained over the same batches
with the recorded random draws (policy noise, Q-head pairs). After each update
every logged loss term, both gradient norms, the RunningScale and the policy
entropy are compared, together with the TD targets, the policy samples and the
policy loss' Q values (which read that update's post-step Q; the next update's
intermediates read all its post-update parameters). All parameters (world
model, policy prior, target Q) are compared after the last update, to an
absolute tolerance below one Adam step. The four updates cover the
gradient-clip cases, including the paper-era world-model clip norm that counts
the previous update's policy gradient (deviations.md §2).

Expected float32 differences: torch and XLA reduce in different orders, optax
rounds Adam's bias corrections in float32, and Ajax's two-hot decode, encode
and log-softmax differ from the reference's at float32 rounding level
(deviation T24: decode within ``eps (1 + 6 E|b|)`` in symlog space, which
``symexp`` turns into ~2e-5 relative on small values). Achieved on CPU with
``highest`` matmul precision (print them with ``-s``): losses, norms and the
scale within 1e-5 relative; TD targets within 2.1e-5 relative; the policy
loss' Q values (decoded from post-step parameters) within 1.7e-4 absolute;
parameters after four updates within 4.5e-6 absolute (the tolerance is
1.5e-5), the largest error being
one Q-head weight whose first gradient is ~5e-10, where Adam's
``g / (|g| + 1e-8)`` turns float32 rounding of a ~1e-9 softmax probability
into ~1e-2 of the 3e-4 step.

The fixture also records the numeric defaults of the reference's
``config.yaml``, against which the defaults of :class:`TDMPC2Config` and of
:func:`~ajax.agents.TDMPC2.core.create_update_state` are pinned.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2UpdateState
from ajax.state import BaseAgentConfig

from .reference_params import (
    ErrorReport,
    Fixture,
    ajax_config,
    as_jnp,
    load_fixture,
    reference_config,
    torch_to_ajax,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tdmpc2_update.npz"
N_UPDATES = 4
# (rtol, atol)
SCALAR_TOL = (2e-5, 1e-5)
# Absolute only: one Adam step moves a parameter by ~lr (3e-4; 9e-5 for the
# encoder), so a missing or wrong step anywhere in the chain exceeds it.
PARAM_TOL = (0.0, 1.5e-5)
# Q values decoded from the post-step parameters: their float32 error,
# through the peaked softmax of the fixture's large heads and symexp.
POST_STEP_Q_TOL = (1e-4, 1e-4)
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


@pytest.fixture(scope="module")
def fx() -> Fixture:
    return load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def ref_config(fx: Fixture) -> dict[str, Any]:
    return reference_config(fx)


@pytest.fixture(scope="module")
def config(ref_config: dict[str, Any]) -> TDMPC2Config:
    return ajax_config(ref_config)


def initial_state(
    fx: Fixture, ref_config: dict[str, Any], config: TDMPC2Config
) -> TDMPC2UpdateState:
    """Ajax's update state holding the reference's initial parameters."""
    state = core.create_update_state(
        jax.random.PRNGKey(0),
        config,
        ref_config["obs_dim"],
        ref_config["action_dim"],
        learning_rate=ref_config["lr"],
        enc_lr_scale=ref_config["enc_lr_scale"],
    )
    wm, target_q, pi = (as_jnp(t) for t in torch_to_ajax(fx, "init/", config))
    for mapped, ajax in (
        (wm, state.world_model_state.params),
        (target_q, state.world_model_state.target_params),
        (pi, state.actor_state.params),
    ):
        assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(
            ajax
        )
        assert jax.tree_util.tree_map(jnp.shape, mapped) == jax.tree_util.tree_map(
            jnp.shape, ajax
        )
    wm_state, pi_state = state.world_model_state, state.actor_state
    return state.replace(
        world_model_state=wm_state.replace(
            params=wm, target_params=target_q, opt_state=wm_state.tx.init(wm)
        ),
        actor_state=pi_state.replace(params=pi, opt_state=pi_state.tx.init(pi)),
    )


def batch_and_noise(fx: Fixture, k: int) -> tuple[core.TDMPC2Batch, core.UpdateNoise]:
    """Update ``k``'s batch and its recorded draws (dropout is 0: keys unused)."""
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
    return batch, noise


def test_fixture_records_the_paper_era_q_dropout_mode(fx):
    """The generator verified on the real reference that Q dropout ignores eval()."""
    assert bool(fx["meta/q_dropout_active_in_eval_mode"])
    assert str(fx["meta/reference_commit"]).startswith("5f6fade")


@pytest.fixture(scope="module")
def paper_config(fx: Fixture) -> dict[str, Any]:
    """The numeric entries of 5f6fade's ``config.yaml`` (the paper defaults)."""
    return json.loads(str(fx["meta/paper_config"]))


def test_config_defaults_are_the_paper_defaults(paper_config):
    """Every ``TDMPC2Config`` default is ``config.yaml``'s (the 5M model)."""
    default = TDMPC2Config()
    base = {f.name for f in dataclasses.fields(BaseAgentConfig)}
    names = [f.name for f in dataclasses.fields(default) if f.name not in base]
    assert {n: getattr(default, n) for n in names} == {
        n: paper_config[n] for n in names
    }
    assert paper_config["vmin"] == -default.vmax
    assert TDMPC2Config.from_model_size(5) == default


def test_create_update_state_defaults_are_the_paper_learning_rates(paper_config):
    """Adam's first step on unit gradients is ~lr: ``lr * enc_lr_scale`` for the
    encoder, ``lr`` for the rest of the world model and the policy
    (``tdmpc2.py:21-28``)."""
    config = TDMPC2Config(latent_dim=16, enc_dim=32, mlp_dim=32)
    state = core.create_update_state(jax.random.PRNGKey(0), config, 5, 2)
    lr = paper_config["lr"]

    def first_step(train_state):
        grads = jax.tree_util.tree_map(jnp.ones_like, train_state.params)
        return train_state.tx.update(grads, train_state.opt_state)[0]

    for part, updates in first_step(state.world_model_state).items():
        expected = lr * paper_config["enc_lr_scale"] if part == "encoder" else lr
        for leaf in jax.tree_util.tree_leaves(updates):
            np.testing.assert_allclose(leaf, -expected, rtol=1e-4, err_msg=part)
    for leaf in jax.tree_util.tree_leaves(first_step(state.actor_state)):
        np.testing.assert_allclose(leaf, -lr, rtol=1e-4)


def test_update_matches_reference_over_four_updates(fx, ref_config, config):
    gamma = float(fx["meta/discount"])
    state = initial_state(fx, ref_config, config)
    step = jax.jit(lambda s, b, n: core.update(s, b, n, config=config, gamma=gamma))
    report = ErrorReport("TD-MPC2 update parity")

    with jax.default_matmul_precision("highest"):
        for k in range(N_UPDATES):
            p = f"update{k}/"
            batch, noise = batch_and_noise(fx, k)
            wm, pi = state.world_model_state, state.actor_state

            # Intermediates on the pre-update state.
            td, next_z = core.td_target(
                wm_apply=wm.apply_fn,
                pi_apply=pi.apply_fn,
                wm_params=wm.params,
                target_q_params=wm.target_params,
                pi_params=pi.params,
                next_obs=batch.obs[1:],
                reward=batch.reward,
                gamma=gamma,
                eps=noise.td_eps,
                pair=noise.td_pair,
                dropout_key=None,
                config=config,
            )
            report.check("td_targets", td, fx[p + "td_targets"][..., 0], SCALAR_TOL)
            _, (_, zs) = core.world_model_loss(
                wm.params,
                wm_apply=wm.apply_fn,
                batch=batch,
                next_z=next_z,
                td_targets=td,
                dropout_key=None,
                config=config,
            )
            sample = core.policy_sample(
                pi.apply_fn, pi.params, zs, noise.pi_eps, config
            )
            report.check("pi_actions", sample.action, fx[p + "pi_actions"], SCALAR_TOL)
            report.check(
                "pi_log_pis", sample.log_pi, fx[p + "pi_log_pis"][..., 0], SCALAR_TOL
            )

            state, aux = step(state, batch, noise)

            # The policy loss' Q: post-step online heads at the pre-step latents.
            logits = core.q_logits(
                wm.apply_fn, state.world_model_state.params, zs, sample.action
            )
            q = core.reduce_q_pair(logits, noise.pi_pair, "avg", config.two_hot)
            report.check("pi_q", q, fx[p + "pi_q"][..., 0], POST_STEP_Q_TOL)
            for name in LOGGED:
                report.check(name, aux[name], fx[p + name], SCALAR_TOL)
            report.check("q_scale", state.q_scale.value, fx[p + "pi_scale"], SCALAR_TOL)
            report.check(
                "pi_entropy",
                aux["pi_entropy"],
                -np.mean(fx[p + "pi_log_pis"]),
                SCALAR_TOL,
            )

    wm_ref, target_ref, pi_ref = torch_to_ajax(fx, "final/", config)
    report.check("wm_params", state.world_model_state.params, wm_ref, PARAM_TOL)
    report.check(
        "target_q_params", state.world_model_state.target_params, target_ref, PARAM_TOL
    )
    report.check("pi_params", state.actor_state.params, pi_ref, PARAM_TOL)
    report.print()


def _max_deviation(actual: Any, expected: Any) -> float:
    return max(
        float(np.abs(np.asarray(a) - e).max())
        for a, e in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
        )
    )


def _chain(
    fx: Fixture, ref_config: dict[str, Any], config: TDMPC2Config, *, stale: bool
) -> list[tuple[TDMPC2UpdateState, dict[str, jax.Array]]]:
    """The state and logged values after each of the fixture's updates.

    ``stale=False`` zeroes the carried policy-gradient norm before every
    update, i.e. drops the paper-era clip-norm quirk.
    """
    gamma = float(fx["meta/discount"])
    step = jax.jit(lambda s, b, n: core.update(s, b, n, config=config, gamma=gamma))
    state = initial_state(fx, ref_config, config)
    out = []
    with jax.default_matmul_precision("highest"):
        for k in range(N_UPDATES):
            if not stale:
                state = state.replace(pi_gradnorm_sq=jnp.zeros(()))
            state, aux = step(state, *batch_and_noise(fx, k))
            out.append((state, aux))
    return out


def test_parity_depends_on_the_stale_policy_gradient_in_the_clip(
    fx, ref_config, config
):
    """Without the paper-era clip-norm quirk the updates no longer match.

    The carried scalar is the squared norm of the previous update's clipped
    policy gradient, which the reference's world-model clip counts. In
    update 1 that clip binds only because of it (norm 20 from update 0).
    Dropping the carry gives the world-model-only norm (also recorded by the
    generator) and a different clip coefficient. Adam is nearly invariant to a
    gradient scale, so the world-model parameters move little, but the policy
    loss, which reads the post-step Q, drifts: its update-1 value leaves the
    parity tolerance and the final policy parameters land about 100x further
    from the reference than the parity error.
    """
    with_quirk = _chain(fx, ref_config, config, stale=True)
    without = _chain(fx, ref_config, config, stale=False)
    for k in range(N_UPDATES - 1):
        np.testing.assert_allclose(
            with_quirk[k][0].pi_gradnorm_sq,
            fx[f"update{k + 1}/stale_pi_grad_sq_norm"],
            rtol=2 * SCALAR_TOL[0],  # a squared pi_grad_norm
        )
    own, logged = float(fx["update1/wm_own_grad_norm"]), float(fx["update1/grad_norm"])
    assert own < config.grad_clip_norm < logged
    np.testing.assert_allclose(with_quirk[1][1]["grad_norm"], logged, rtol=1e-5)
    np.testing.assert_allclose(without[1][1]["grad_norm"], own, rtol=1e-5)
    rtol, atol = SCALAR_TOL
    ref_pi_loss = float(fx["update1/pi_loss"])
    assert not np.isclose(without[1][1]["pi_loss"], ref_pi_loss, rtol=rtol, atol=atol)
    _, _, pi_ref = torch_to_ajax(fx, "final/", config)
    parity_error = _max_deviation(with_quirk[-1][0].actor_state.params, pi_ref)
    assert _max_deviation(without[-1][0].actor_state.params, pi_ref) > 10 * parity_error
