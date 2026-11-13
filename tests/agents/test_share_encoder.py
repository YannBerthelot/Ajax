"""An actor and a critic sharing one encoder (``share_encoder``).

Contract (from the owner's request, 2026-11-13, and ppo2 @ea25b9e, whose
shared network is one torso under two heads trained by one step down the
summed loss, ppo2/model.py:75-107, common/policies.py:155-157):

- with ``share_encoder=True`` there is one encoder (and memory block): the
  critic's. The actor acts through it, so a change to it changes the
  policy; without sharing the critic's encoder never reaches the policy;
- the encoder's gradient is that of the summed loss: the value loss's plus
  the policy loss's, so the policy loss alone trains it;
- the actor's copy follows the critic's encoder after training and after
  cloning pretraining, which trains the encoder through the actor;
- ``None`` is the agent's reference: separate MLPs for PPO (ppo2's MuJoCo
  setting, defaults.py:12) and for APO (rlpyt's MujocoFfModel);
- the networks it cannot build are refused.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.cloning import pre_train
from ajax.agents.PPO.PPO import PPO
from ajax.agents.update_steps import compute_actor_critic_gradients
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.environments.interaction import get_pi
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import (
    mirror_shared_encoder,
    predict_value,
    take_actor_encoder,
    with_critic_encoder,
)
from ajax.state import Transition

N_STEPS, N_ENVS = 8, 2
SMALL: dict[str, Any] = {"n_steps": N_STEPS, "n_envs": N_ENVS, "n_epochs": 1}
SMALL.update({"num_minibatches": 2})
SMALL.update(
    {"actor_architecture": ("8", "tanh"), "critic_architecture": ("8", "tanh")}
)
OBS = jnp.linspace(-1.0, 1.0, 12).reshape(4, 3)  # Pendulum's 3 features


def _init(agent: Any) -> tuple[Any, Any]:
    _, context = agent.make_context(1_000)
    return agent.init(jax.random.PRNGKey(0), context), context


def _critic_encoder(state: Any) -> Any:
    return state.critic_state.params["params"]["encoder"]


def _shift_critic_encoder(state: Any, amount: float) -> Any:
    """state with every weight of the critic's encoder (shared or its own,
    inside its ensemble) moved by amount."""

    def shift(path: Any, weight: jax.Array) -> jax.Array:
        names = [getattr(key, "key", None) for key in path]
        return weight + amount if "encoder" in names else weight

    critic = state.critic_state
    params = jax.tree_util.tree_map_with_path(shift, critic.params)
    return state.replace(critic_state=critic.replace(params=params))


def _policy_mean(state: Any) -> np.ndarray:
    actor = state.actor_state
    return np.asarray(get_pi(actor, actor.params, OBS)[0].mean())


def _rollout() -> Transition:
    obs = jnp.tile(OBS[:2][None], (N_STEPS, 1, 1)) * jnp.arange(N_STEPS)[:, None, None]
    column = jnp.ones((N_STEPS, N_ENVS, 1))
    return Transition(
        obs=obs,
        action=0.3 * column,
        reward=column,
        terminated=0 * column,
        truncated=0 * column,
        next_obs=obs + 0.5,
        raw_obs=obs,
        log_prob=-column,
        raw_action=0.3 * column,
    )


@pytest.mark.parametrize("share_encoder", [True, False])
def test_the_policy_reads_the_critics_encoder_only_when_shared(share_encoder):
    agent = PPO(env_id="Pendulum-v1", share_encoder=share_encoder, **SMALL)
    state, _ = _init(agent)
    before = _policy_mean(state)

    moved = mirror_shared_encoder(_shift_critic_encoder(state, 0.5))

    changed = not np.allclose(_policy_mean(moved), before)
    assert changed == share_encoder


def test_a_shared_encoder_is_held_once_by_the_critic():
    state, _ = _init(PPO(env_id="Pendulum-v1", share_encoder=True, **SMALL))

    assert "encoder" not in state.actor_state.params["params"]
    chex_equal = jax.tree.map(
        np.array_equal,
        dict(state.actor_state.shared_params),
        {"encoder": _critic_encoder(state)},
    )
    assert all(jax.tree.leaves(chex_equal))


@pytest.mark.parametrize("share_encoder", [True, False])
def test_the_policy_loss_alone_trains_a_shared_encoder(share_encoder):
    """With vf_coef 0 the value loss has no gradient: the critic moves only
    through the policy loss, which reaches it only through a shared encoder."""
    agent = PPO(env_id="Pendulum-v1", share_encoder=share_encoder, vf_coef=0.0, **SMALL)
    state, context = _init(agent)

    updated, _ = agent.update(state, _rollout(), state, context)

    moved = jax.tree.map(
        lambda a, b: not np.array_equal(a, b),
        state.critic_state.params["params"],
        updated.critic_state.params["params"],
    )
    assert any(jax.tree.leaves(moved.get("encoder", {}))) == share_encoder
    assert not any(jax.tree.leaves(moved["ensemble"]))  # the value head: no gradient


def test_the_encoders_gradient_is_the_summed_losses():
    """Oracle: jax.grad of policy_loss + value_loss over both networks'
    parameters at once, ppo2's one network."""
    state, _ = _init(PPO(env_id="Pendulum-v1", share_encoder=True, **SMALL))
    actor, critic = state.actor_state, state.critic_state
    actions = jnp.full((4, 1), 0.3)

    def policy_loss(actor_params, critic_params):
        policy, _ = get_pi(with_critic_encoder(actor, critic_params), actor_params, OBS)
        loss = -policy.log_prob(actions).mean()
        return loss, loss

    def value_loss(critic_params):
        loss = jnp.mean(predict_value(critic, critic_params, OBS) ** 2)
        return loss, loss

    actor_grads, critic_grads, _, _ = compute_actor_critic_gradients(
        policy_loss, value_loss, actor, critic
    )

    def summed(actor_params, critic_params):
        return (
            policy_loss(actor_params, critic_params)[0] + value_loss(critic_params)[0]
        )

    expected_actor, expected_critic = jax.grad(summed, argnums=(0, 1))(
        actor.params, critic.params
    )
    # float32, a few dozen terms per entry: rounding only
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7),
        (actor_grads, critic_grads),
        (expected_actor, expected_critic),
    )


def test_the_actors_copy_follows_the_encoder_through_recurrent_training():
    agent = PPO(
        env_id="CartPole-v1",
        share_encoder=True,
        memory=MemoryConfig("gru", 8),
        **{**SMALL, "n_envs": 4},
    )

    state, _ = agent.train(seed=0, n_timesteps=2 * N_STEPS * 4)

    shared = state.actor_state.shared_params
    owned = state.critic_state.params["params"]
    assert set(shared) == {"encoder", "memory_cell"}
    same = jax.tree.map(np.array_equal, dict(shared), {k: owned[k] for k in shared})
    assert all(jax.tree.leaves(same))


def test_cloning_trains_a_shared_encoder_and_hands_it_to_the_critic():
    state, _ = _init(PPO(env_id="Pendulum-v1", share_encoder=True, **SMALL))
    dataset = Transition(
        obs=jnp.tile(OBS[:, None], (4, 1, 1)),
        action=jnp.full((16, 1, 1), 0.5),
        reward=jnp.zeros((16, 1, 1)),
        terminated=jnp.zeros((16, 1, 1)),
        truncated=jnp.zeros((16, 1, 1)),
        next_obs=jnp.tile(OBS[:, None], (4, 1, 1)),
        log_prob=None,
    )

    actor, _, _, _ = pre_train(
        jax.random.PRNGKey(1), state.actor_state, dataset, actor_epochs=2
    )
    cloned = take_actor_encoder(state.replace(actor_state=actor))

    moved = jax.tree.map(
        lambda a, b: not np.array_equal(a, b),
        _critic_encoder(state),
        _critic_encoder(cloned),
    )
    assert all(jax.tree.leaves(moved))
    same = jax.tree.map(
        np.array_equal, dict(actor.shared_params)["encoder"], _critic_encoder(cloned)
    )
    assert all(jax.tree.leaves(same))


def test_a_shared_encoder_state_restores_from_its_checkpoint(tmp_path):
    state, _ = _init(PPO(env_id="Pendulum-v1", share_encoder=True, **SMALL))
    path = str(tmp_path / "shared.pkl")
    save_checkpoint(state, path)

    restored = restore_into(state, path)

    same = jax.tree.map(
        np.array_equal,
        restored.actor_state.shared_params,
        state.actor_state.shared_params,
    )
    assert all(jax.tree.leaves(same))


@pytest.mark.parametrize("agent_class", [PPO, APO])
def test_the_reference_keeps_mlp_networks_apart(agent_class):
    assert not agent_class(env_id="Pendulum-v1", **SMALL).network_args.share_encoder
    shared = agent_class(env_id="Pendulum-v1", share_encoder=True, **SMALL)
    assert shared.network_args.share_encoder


def test_one_encoder_needs_one_architecture():
    agent = PPO(
        env_id="Pendulum-v1",
        share_encoder=True,
        **{**SMALL, "critic_architecture": ("16", "tanh")},
    )

    with pytest.raises(ValueError, match="actor_architecture and critic_architecture"):
        _init(agent)
