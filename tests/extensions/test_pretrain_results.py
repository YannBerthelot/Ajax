"""What the pretraining extensions keep of their fits, in their state, and
log at each evaluation: CloningPretrain's, MCPretrain's and BellmanPretrain's
first and last loss, and MCPretrain's mean expert value, each the value its
fit computed. NaN before the fit, and in a run resumed from a checkpoint
saved before they were kept."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.extensions import pretrain
from ajax.extensions.base import ExtensionContext
from ajax.extensions.pretrain import (
    BellmanPretrain,
    CloningPretrain,
    ExpertPrefill,
    MCPretrain,
)
from ajax.logging.wandb_logging import LoggingConfig
from tests.probing import readouts

KEY = jax.random.PRNGKey(0)
LOGGING = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
NAMES = (
    "cloning/first_loss",
    "cloning/last_loss",
    "mc_pretrain/first_loss",
    "mc_pretrain/last_loss",
    "mc_pretrain/q_expert_mean",
    "bellman_pretrain/first_loss",
    "bellman_pretrain/last_loss",
)


def expert(obs: jax.Array) -> jax.Array:
    return jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)


def _stack() -> tuple:
    return (
        CloningPretrain(expert_policy=expert, n_steps=64, epochs=3, batch_size=16),
        MCPretrain(expert, 200, 4, 20, online_light_steps=5),
        BellmanPretrain(n_steps=4),
        ExpertPrefill(n_steps=16),  # BellmanPretrain's replay
    )


def _agent() -> SAC:
    net = ("8", "relu")
    return SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        extensions=_stack(),
    )


def _logged(out: dict, name: str) -> np.ndarray:
    """``name``'s values at the run's evaluations."""
    evaluated = np.asarray(out["timestep"]).reshape(-1) >= 0
    assert evaluated.any()
    return np.asarray(out[name]).reshape(-1)[evaluated]


@pytest.fixture(scope="module")
def run() -> SimpleNamespace:
    """A logged SAC run with the three pretraining extensions, each fit's
    output recorded as it computed it."""
    fits: dict[str, Any] = {}

    def recording(name: str, fit: Callable, pick: Callable) -> Callable:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            out = fit(*args, **kwargs)
            jax.debug.callback(lambda v: fits.update({name: v}), pick(out))
            return out

        return recorded

    with pytest.MonkeyPatch.context() as patch:
        for name, pick in (
            ("pre_train", lambda out: out[1]["actor_loss"]),  # per epoch
            ("pretrain_critic_mc", lambda out: out[4]),  # MCPretrainAux
            ("pretrain_critic_bellman", lambda out: out[1]),  # per update
        ):
            patch.setattr(
                pretrain, name, recording(name, getattr(pretrain, name), pick)
            )
        agent = _agent()
        state, out = agent.train(
            seed=0, n_timesteps=64, num_episode_test=1, logging_config=LOGGING
        )
        jax.effects_barrier()
    return SimpleNamespace(agent=agent, state=jax.device_get(state), out=out, fits=fits)


def test_each_logged_result_is_what_its_fit_computed(run: SimpleNamespace) -> None:
    cloning, mc = run.fits["pre_train"], run.fits["pretrain_critic_mc"]
    bellman = run.fits["pretrain_critic_bellman"]
    assert (cloning.shape, bellman.shape) == ((3,), (4,))
    computed = {
        "cloning/first_loss": cloning[0],
        "cloning/last_loss": cloning[-1],
        "mc_pretrain/first_loss": mc.initial_loss,
        "mc_pretrain/last_loss": mc.final_loss,
        "mc_pretrain/q_expert_mean": mc.q_expert_mean,
        "bellman_pretrain/first_loss": bellman[0],
        "bellman_pretrain/last_loss": bellman[-1],
    }
    for name, value in computed.items():
        assert np.isfinite(value), name
        np.testing.assert_allclose(_logged(run.out, name), value, rtol=1e-6)


def test_the_state_has_one_structure_before_and_after_the_fit(
    run: SimpleNamespace,
) -> None:
    """``init_state`` builds what ``pretrain`` returns (here batched over
    the run's one seed), so a fresh skeleton holds a pretrained state."""
    for ext, kept in zip(_stack(), run.state.ext_state, strict=True):
        fresh = ext.init_state(None, KEY)
        assert type(kept) is type(fresh)
        for before, after in zip(
            jax.tree.leaves(fresh), jax.tree.leaves(kept), strict=True
        ):
            assert (np.shape(after), after.dtype) == ((1, *before.shape), before.dtype)


def test_an_extension_logs_nan_before_its_fit() -> None:
    """And a fit taking no step keeps NaN (BellmanPretrain with
    ``n_steps=0``, CloningPretrain with ``epochs=0``)."""
    context = ExtensionContext(step=jnp.asarray(0), rng=KEY, total_steps=100)
    bounds = SimpleNamespace(
        expert_v_min=jnp.asarray(0.0), expert_v_max=jnp.asarray(1.0)
    )
    logged: dict = {}
    for ext in _stack():
        logged.update(ext.eval_metrics(bounds, ext.init_state(None, KEY), KEY, context))
    assert {k for k in logged if k not in NAMES} == {
        "mc_pretrain/v_min",
        "mc_pretrain/v_max",
    }
    assert all(np.isnan(logged[name]) for name in NAMES)
    none = pretrain._first_and_last(jnp.zeros((0,)))
    assert np.isnan(none.first_loss) and np.isnan(none.last_loss)


def test_a_checkpoint_saved_before_the_results_were_kept_resumes(
    run: SimpleNamespace, tmp_path: Any
) -> None:
    """A state saved while the three extensions were stateless (``()`` each)
    restores with their results NaN and every other leaf as saved; its run
    resumes and logs them as NaN."""
    old = run.state.replace(ext_state=((), (), ()))
    path = str(tmp_path / "old.pkl")
    save_checkpoint(old, path)
    skeleton, _ = run.agent.train(seed=0, n_timesteps=0)
    restored = restore_into(skeleton, path)
    assert all(np.isnan(leaf).all() for leaf in jax.tree.leaves(restored.ext_state))
    assert not readouts.differing_leaves(
        old, jax.device_get(restored).replace(ext_state=((), (), ()))
    )
    _, out = run.agent.train(
        seed=0,
        n_timesteps=32,
        num_episode_test=1,
        initial_state=restored,
        logging_config=LOGGING,
    )
    for name in NAMES:
        assert np.isnan(_logged(out, name)).all(), name
