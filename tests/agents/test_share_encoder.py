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
- ``None`` is the agent's reference: for PPO one encoder with images
  (ppo2's Atari defaults), separate MLPs (ppo2's MuJoCo setting,
  defaults.py:12) and, by the owner's choice, separate memories; separate
  MLPs for APO (rlpyt's MujocoFfModel);
- the networks it cannot build are refused.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ajax
from ajax.agents.APO.APO import APO
from ajax.agents.cloning import pre_train
from ajax.agents.PPO.PPO import PPO
from ajax.agents.recurrent import sample_replay
from ajax.agents.TD3.TD3 import compute_td3_td_target
from ajax.agents.update_steps import compute_actor_critic_gradients
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.environments.interaction import get_pi
from ajax.extensions.observe import PreviousAction
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import (
    mirror_shared_encoder,
    predict_value,
    take_actor_encoder,
    with_critic_encoder,
)
from ajax.networks.utils import clip_by_member_norm
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


@pytest.mark.parametrize(
    "network, shared",
    [
        ({}, False),  # ppo2's MuJoCo copy (defaults.py:12), CleanRL's, PureJaxRL's
        # ppo2 and PureJaxRL share a memory, behind per-head layers Ajax lacks
        ({"memory": MemoryConfig("gru", 8)}, False),
        ({"env_id": "Breakout-MinAtar", "cnn_image_shape": (10, 10, 4)}, True),
    ],
)
def test_ppo_follows_its_reference_per_network(network, shared):
    """ppo2 shares its network with images (Atari defaults, defaults.py:15-21)
    and copies MLPs on MuJoCo; the owner chose the copy for every MLP, and
    separate networks with memory until Ajax has PureJaxRL's per-head layers
    after it (2026-11-13)."""
    agent = PPO(**{"env_id": "CartPole-v1", **SMALL, **network})
    assert agent.network_args.share_encoder == shared
    assert not PPO(
        **{"env_id": "CartPole-v1", **SMALL, **network, "share_encoder": False}
    ).network_args.share_encoder


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


# -- Q-critic agents (SAC-AE's and DrQ's rule) ------------------------------
# Contract (owner, 2026-11-13; SAC-AE, DrQ and SB3's share_features_extractor):
# the critic's loss trains the shared encoder and the actor reads it detached,
# as the critic step just left it; the action joins after the encoder; a target
# actor reads the target critic's encoder; every reference keeps its networks
# apart, so the default stays separate.

Q_AGENTS = ("SAC", "TD3", "REDQ", "TQC", "ASAC", "RVISAC", "AVG", "IQL")
Q_SMALL: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
}
REPLAY = {"buffer_size": 256, "batch_size": 8, "learning_starts": 16}


def _one_seed(state: Any) -> Any:
    """A trained state without its leading seed axis."""
    return jax.tree.map(lambda leaf: leaf[0], state)


def _q_agent(name: str, **options: Any) -> Any:
    kwargs = dict(Q_SMALL)
    if name != "AVG":
        kwargs.update(REPLAY)
    kwargs.update(options)
    if name == "ASAC":
        kwargs["max_grad_norm"] = 1.0  # its per-member clip, with a shared encoder
    if name == "REDQ":
        kwargs["num_critic_updates"] = 2  # a fast test's few updates, not 20 a step
    return getattr(ajax, name)(env_id="Pendulum-v1", **kwargs)


@pytest.mark.parametrize("name", Q_AGENTS)
def test_a_q_critic_agent_trains_on_one_encoder_its_actor_reads(name):
    assert not _q_agent(name).network_args.share_encoder  # the references' default
    agent = _q_agent(name, share_encoder=True)

    state = _one_seed(agent.train(seed=0, n_timesteps=48)[0])

    copy = dict(state.actor_state.shared_params)["encoder"]
    same = jax.tree.map(np.array_equal, copy, _critic_encoder(state))
    assert all(jax.tree.leaves(same))
    moved = mirror_shared_encoder(_shift_critic_encoder(state, 0.5))
    assert not np.allclose(_policy_mean(moved), _policy_mean(state))


@pytest.mark.parametrize(
    "critic_learning_rate, encoder_moves", [(0.0, False), (3e-4, True)]
)
def test_only_the_critics_loss_trains_the_encoder(critic_learning_rate, encoder_moves):
    """With the critic's learning rate at 0 an update leaves the encoder as it
    was, though the actor steps: the actor reads it detached. The actor's
    copy is the encoder the critic step left, read by the actor step after it."""
    agent = _q_agent(
        "SAC", share_encoder=True, critic_learning_rate=critic_learning_rate
    )
    state = _one_seed(agent.train(seed=0, n_timesteps=32)[0])
    _, context = agent.make_context(64)

    updated, _ = agent.update(state, context)

    moved = jax.tree.map(
        lambda a, b: not np.array_equal(a, b),
        _critic_encoder(state),
        _critic_encoder(updated),
    )
    assert all(jax.tree.leaves(moved)) == encoder_moves
    copy = dict(updated.actor_state.shared_params)["encoder"]
    assert all(
        jax.tree.leaves(jax.tree.map(np.array_equal, copy, _critic_encoder(updated)))
    )


@pytest.mark.parametrize("target, changes", [(True, True), (False, False)])
def test_td3s_target_actor_reads_the_target_critics_encoder(target, changes):
    state, _ = _init(_q_agent("TD3", share_encoder=True))
    critic = state.critic_state
    shifted = jax.tree_util.tree_map_with_path(
        lambda path, w: w + 0.5 if "encoder" in str(path) else w,
        critic.target_params if target else critic.params,
    )
    moved = critic.replace(**{"target_params" if target else "params": shifted})
    observations, column = OBS, jnp.zeros((4, 1))

    def next_actions(critic_state):
        _, _, actions, _ = compute_td3_td_target(
            state.actor_state,
            critic_state,
            jax.random.PRNGKey(0),
            observations,
            column,
            column,
            0.99,
            0.0,
            0.5,
            1.0,
        )
        return np.asarray(actions)

    assert (not np.allclose(next_actions(moved), next_actions(critic))) == changes


def test_each_member_and_a_shared_encoder_are_clipped_on_their_own():
    """Members (under "ensemble") clip by their own norms, the shared encoder
    by its own: norms 5 (3-4), 1 and 10 (6-8), clipped to 2."""
    updates = {
        "params": {
            "encoder": {"kernel": jnp.array([6.0, 8.0])},
            "ensemble": {"kernel": jnp.array([[3.0, 4.0], [0.6, 0.8]])},
        }
    }
    clipped, _ = clip_by_member_norm(2.0).update(updates, optax.EmptyState())

    np.testing.assert_allclose(
        clipped["params"]["encoder"]["kernel"], [1.2, 1.6], rtol=1e-6
    )
    np.testing.assert_allclose(
        clipped["params"]["ensemble"]["kernel"], [[1.2, 1.6], [0.6, 0.8]], rtol=1e-6
    )


def test_iqls_state_value_keeps_its_own_encoder():
    state, _ = _init(_q_agent("IQL", share_encoder=True))

    assert "encoder" in state.critic_state.params["params"]  # shared, the Q critics'
    assert "encoder" not in state.value_state.params["params"]  # V's, in its ensemble


# -- A memory shared with a Q-critic ------------------------------------------
# Contract (owner, 2026-11-13; Ni et al. 2022's shared RNN, twni2016/
# pomdp-baselines @e7c19c3 policies/models/policy_rnn_shared.py): one memory,
# the critic's, reads the observation and the previous action, which the
# PreviousAction feature puts in the observation for both networks; the
# queried action joins after it; the critic's loss alone trains it and the
# actor reads it detached; a target actor warms up through the target
# critic's; without PreviousAction it is refused.

RECURRENT_Q_AGENTS = ("SAC", "TD3", "REDQ", "TQC", "ASAC", "RVISAC")
SHARED_MEMORY: dict[str, Any] = {
    "share_encoder": True,
    "memory": MemoryConfig("gru", 8),
    "learning_starts": 64,
    "extensions": (PreviousAction(),),
}


def _shift_critic_module(state: Any, module: str, amount: float) -> Any:
    """state with every weight of the critic's module (its shared encoder or
    memory block), online and target, moved by amount."""

    def shift(params: Any) -> Any:
        return jax.tree_util.tree_map_with_path(
            lambda path, w: w + amount if module in str(path) else w, params
        )

    critic = state.critic_state
    moved = critic.replace(
        params=shift(critic.params), target_params=shift(critic.target_params)
    )
    return state.replace(critic_state=moved)


def _recurrent_policy_mean(state: Any) -> np.ndarray:
    """The policy's mean at the first observations of a fresh episode."""
    actor = state.actor_state
    n_envs = jax.tree.leaves(actor.hidden_state)[0].shape[0]
    observations = jnp.concatenate([OBS[:n_envs], jnp.zeros((n_envs, 1))], axis=-1)
    done = jnp.ones((n_envs,), bool)
    pi, _ = get_pi(actor, actor.params, observations, done, recurrent=True)
    return np.asarray(pi.mean())


def test_a_shared_memory_with_a_q_critic_needs_the_previous_action():
    agent = _q_agent("SAC", **{**SHARED_MEMORY, "extensions": ()})

    with pytest.raises(ValueError, match="PreviousAction"):
        _init(agent)


@pytest.mark.parametrize("name", RECURRENT_Q_AGENTS)
def test_a_q_critic_agent_trains_on_one_memory_its_actor_reads(name):
    agent = _q_agent(name, **SHARED_MEMORY)

    state = _one_seed(agent.train(seed=0, n_timesteps=96)[0])

    shared = dict(state.actor_state.shared_params)
    assert set(shared) == {"encoder", "memory_cell"}
    owned = state.critic_state.params["params"]
    same = jax.tree.map(np.array_equal, shared, {k: owned[k] for k in shared})
    assert all(jax.tree.leaves(same))
    moved = mirror_shared_encoder(_shift_critic_module(state, "memory_cell", 0.5))
    assert not np.allclose(_recurrent_policy_mean(moved), _recurrent_policy_mean(state))


@pytest.mark.parametrize(
    "critic_learning_rate, memory_moves", [(0.0, False), (3e-4, True)]
)
def test_only_the_critics_loss_trains_the_shared_memory(
    critic_learning_rate, memory_moves
):
    agent = _q_agent("SAC", **SHARED_MEMORY, critic_learning_rate=critic_learning_rate)
    state = _one_seed(agent.train(seed=0, n_timesteps=64)[0])
    _, context = agent.make_context(128)

    updated, _ = agent.update(state, context)

    def memory(agent_state: Any) -> Any:
        return agent_state.critic_state.params["params"]["memory_cell"]

    moved = jax.tree.map(
        lambda a, b: not np.array_equal(a, b), memory(state), memory(updated)
    )
    assert all(jax.tree.leaves(moved)) == memory_moves
    copy = dict(updated.actor_state.shared_params)["memory_cell"]
    assert all(jax.tree.leaves(jax.tree.map(np.array_equal, copy, memory(updated))))


@pytest.mark.parametrize("target, changes", [(True, True), (False, False)])
def test_td3s_target_actor_warms_up_through_the_target_critics_memory(target, changes):
    """The target actor's warmed-up memory moves with the target critic's
    memory block, not with the online critic's."""
    agent = _q_agent("TD3", **SHARED_MEMORY)
    state = _one_seed(agent.train(seed=0, n_timesteps=64)[0])
    _, context = agent.make_context(128)
    critic = state.critic_state
    which = "target_params" if target else "params"
    shifted = jax.tree_util.tree_map_with_path(
        lambda path, w: w + 0.5 if "memory_cell" in str(path) else w,
        getattr(critic, which),
    )
    # As the training loop leaves it: the actor's copy is the online critic's.
    moved = mirror_shared_encoder(
        state.replace(critic_state=critic.replace(**{which: shifted}))
    )
    key = jax.random.PRNGKey(0)

    def warmed_up(agent_state: Any) -> np.ndarray:
        _, carries = sample_replay(
            agent_state, context.buffer, key, True, agent.burn_in, False, True
        )
        assert carries is not None  # a window's, with memory
        return np.asarray(jax.tree.leaves(carries.target_actor_hidden)[0])

    assert (not np.allclose(warmed_up(moved), warmed_up(state))) == changes
