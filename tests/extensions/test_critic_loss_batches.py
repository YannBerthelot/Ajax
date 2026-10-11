"""The ``critic_loss`` batch carries what a critic-loss extension needs: on
every agent family :class:`BiasVorePenalty` (it reads the observations, the
targets and the critic) has a nonzero gradient on the critic, and on a batch
without them it raises instead of adding 0."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from ajax import APO, ASAC, AVG, DQN, PPO, PQN, REDQ, SAC, TD3
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
        critic_state, _ = critic_step(
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
    "batch",
    [
        {"targets": jnp.zeros((2, 1))},
        {"observations": jnp.zeros((2, 1))},
        {"observations": jnp.zeros((2, 1)), "targets": jnp.zeros((2, 1))},
    ],
    ids=["no-observations", "no-targets", "no-critic"],
)
def test_the_penalty_raises_on_a_batch_without_its_operands(batch: dict) -> None:
    with pytest.raises(ValueError, match="BiasVorePenalty"):
        PENALTY.critic_loss(None, (), batch, CTX)


# --- One run per agent: the penalty's gradient on the critic the agent steps --
GRAD_NORMS: list[np.ndarray] = []


@dataclass(frozen=True)
class RecordedPenalty(BiasVorePenalty):
    """The penalty, recording its gradient's norm on the critic parameters
    (the Q-network's for DQN and PQN) of the batch it is given."""

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        def term(params: Any) -> jax.Array:
            if "q_state" in batch:
                with_params = {"q_state": batch["q_state"].replace(params=params)}
            else:
                with_params = {"critic_params": params}
            return super(RecordedPenalty, self).critic_loss(
                agent_state, ext_state, batch | with_params, ctx
            )

        params = (
            batch["q_state"].params if "q_state" in batch else batch["critic_params"]
        )
        norm = optax.global_norm(jax.grad(term)(params))
        jax.debug.callback(lambda n: GRAD_NORMS.append(np.asarray(n)), norm)
        return term(params)


_NET: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "actor_architecture": ("16", "relu"),
    "critic_architecture": ("16", "relu"),
}
_REPLAY: dict[str, Any] = {**_NET, "learning_starts": 20, "batch_size": 16}
_ROLLOUT: dict[str, Any] = {**_NET, "n_steps": 16, "batch_size": 16}
_Q: dict[str, Any] = {"env_id": "CartPole-v1", "architecture": ("16", "relu")}
AGENTS: dict[str, tuple[Any, dict[str, Any]]] = {
    "TD3": (TD3, _REPLAY),
    "AVG": (AVG, {**_NET, "learning_starts": 20}),
    "ASAC": (ASAC, _REPLAY),
    "REDQ": (REDQ, _REPLAY),
    "SAC": (SAC, {**_REPLAY, "policy_update_start": 20, "alpha_update_start": 20}),
    "PPO": (PPO, _ROLLOUT),
    "APO": (APO, _ROLLOUT),
    "DQN": (DQN, {**_Q, "n_envs": 2, "learning_starts": 20, "batch_size": 16}),
    "PQN": (PQN, {**_Q, "n_envs": 2, "n_steps": 16, "num_minibatches": 2}),
}


@pytest.mark.parametrize("agent", list(AGENTS))
def test_the_penalty_has_a_gradient_on_every_agent(agent: str) -> None:
    cls, kwargs = AGENTS[agent]
    GRAD_NORMS.clear()
    cls(**kwargs, extensions=(RecordedPenalty(alpha=0.3),)).train(
        seed=0, n_timesteps=64
    )
    norms = np.concatenate([np.ravel(n) for n in GRAD_NORMS])
    assert norms.size and (norms > 0).all(), norms
