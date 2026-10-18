"""The ``critic_loss`` batch carries what a critic-loss extension needs: the
shared replay critic step hands :class:`BiasVorePenalty` (it reads the
observations, the targets and the critic) the targets, and on a batch
without them the penalty raises instead of adding 0. Every agent's batches
are test_phase_contract's."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from ajax.agents.loop import critic_step
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.extensions.instrumentation import BiasVorePenalty
from ajax.state import Transition

PENALTY = BiasVorePenalty(alpha=0.3)
CTX = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))


def test_critic_step_hands_the_extensions_the_targets() -> None:
    """The shared replay critic step (TD3, AVG, ASAC, REDQ) puts the target
    in the ``critic_loss`` batch: the penalty moves the critic."""
    obs = jnp.arange(4.0).reshape(4, 1)
    batch = Transition(
        obs=obs,
        action=jnp.zeros((4, 1)),
        reward=jnp.zeros((4, 1)),
        terminated=jnp.zeros((4, 1)),
        truncated=jnp.zeros((4, 1)),
        next_obs=obs,
    )
    critic = TrainState.create(
        apply_fn=lambda params, x: (x @ params["w"])[None],
        params={"w": jnp.ones((2, 1))},
        tx=optax.sgd(0.1),
    )
    state = SimpleNamespace(
        collector_state=SimpleNamespace(timestep=jnp.asarray(0)),
        critic_state=critic,
        ext_state=((),),
    )

    def value_loss(params: Any, target_q: jax.Array) -> tuple[jax.Array, Any]:
        q = critic.apply_fn(params, jnp.concatenate([obs, batch.action], -1))
        return jnp.mean((q - target_q) ** 2), ()

    def stepped(extension: BiasVorePenalty) -> jax.Array:
        critic_state, _, _ = critic_step(
            state,
            batch,
            jnp.ones((4, 1)),
            value_loss,
            ExtensionStack((extension,)),
            jax.random.PRNGKey(0),
            100,
            rewards=batch.reward,
            dones=batch.terminated,
            gamma=0.9,
            reward_scale=1.0,
        )
        return critic_state.params["w"]

    moved = stepped(PENALTY) - stepped(BiasVorePenalty(coeff_override=0.0))
    assert float(jnp.abs(moved).max()) > 0.0


@pytest.mark.parametrize(
    "batch, missing",
    [
        ({"observations": jnp.zeros((2, 1))}, "targets"),
        ({"observations": jnp.zeros((2, 1)), "targets": jnp.zeros((2, 1))}, "critic"),
    ],
    ids=["no-targets", "no-critic"],
)
def test_the_penalty_raises_on_a_batch_without_one_operand(
    batch: dict, missing: str
) -> None:
    """test_no_silent_noop's empty batch stops at the first check."""
    with pytest.raises(ValueError, match=f"BiasVorePenalty needs .*{missing}"):
        PENALTY.critic_loss(None, (), batch, CTX)
