"""PPO code-tier tests: programmatic invariants of the active training path.

These are the "wire tests" -- they assert that config values are
honoured by the code path that actually runs, not that the agent learns
anything (that's the smoke / probing tests' job) and not that the
networks have a particular shape (that's the structure tests' job).

These tests catch bugs of the form:
* "config knob is defined but never read by the active path"
  (e.g. ``num_minibatches=32`` silently treated as 1)
* "active path is the wrong function"
  (e.g. ``do_update`` instead of ``update_agent``)
* "the loss function ignores a kwarg" (e.g. ``vf_coef`` defaults to 1.0
  because the caller doesn't pass it)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ajax import PPO
from ajax.agents.PPO.state import PPOConfig
from ajax.agents.PPO.train_PPO import Geometry, minibatch_geometry
from ajax.networks.memory import MemoryConfig

# Small CartPole config sized so one training iteration's rollout
# exactly equals n_envs * n_steps, with num_minibatches > 1 so the
# minibatch-iteration bug would be visible.
_CFG = {
    "env_id": "CartPole-v1",
    "n_envs": 4,
    "n_steps": 8,  # rollout per iter = 4 * 8 = 32 transitions
    "batch_size": 32,  # ignored when num_minibatches > 0
    "num_minibatches": 4,  # → mb_size = 32 / 4 = 8
    "n_epochs": 2,
    "actor_architecture": ("16", "tanh", "16", "tanh"),
    "critic_architecture": ("16", "tanh", "16", "tanh"),
}


@pytest.mark.parametrize(
    "extra, num_minibatches",
    [
        ({}, 4),
        ({"num_minibatches": 8, "batch_size": 64}, 8),
        ({"n_epochs": 4}, 4),
        # num_minibatches=0 (legacy): max(batch_size, n_steps) // min(...) = 1
        ({"num_minibatches": 0, "batch_size": 8}, 1),
    ],
    ids=["4-minibatches", "8-minibatches", "4-epochs", "legacy-1-minibatch"],
)
def test_apply_gradients_runs_once_per_epoch_and_minibatch(extra, num_minibatches):
    """Per training iteration PPO calls ``apply_gradients`` once per (epoch,
    minibatch) pair on BOTH the actor and the critic, so each Adam step
    counter reads ``n_epochs * num_minibatches * n_updates`` (whatever the
    number of iterations ``n_updates`` the loop ran).

    This is the **wire-test that would have caught the May 2026 bug**
    where ``do_update.body_fn`` passed the whole (num_minibatches,
    mb_size, feat) tensor to the loss as one batch, performing only
    ``n_epochs`` updates per iteration instead of
    ``n_epochs * num_minibatches``; the 8-minibatch and 4-epoch rows catch
    a knob honoured for the reshape but not for the scan length.
    """
    cfg = {**_CFG, **extra}
    rollout = cfg["n_envs"] * cfg["n_steps"]
    state, _ = PPO(**cfg).train(seed=[0], n_timesteps=rollout, num_episode_test=1)
    # train() seed-vmaps: every counter has a leading (1,) seed axis.
    n_updates = int(state.n_updates[0])
    expected = cfg["n_epochs"] * num_minibatches * n_updates
    for name in ("actor", "critic"):
        step = int(getattr(state, f"{name}_state").step[0])
        assert step == expected, (
            f"{name}.step={step}, expected n_epochs * num_minibatches * "
            f"n_updates = {cfg['n_epochs']} * {num_minibatches} * {n_updates} = "
            f"{expected}. Likely cause: the training body isn't iterating over "
            f"minibatches (would give {cfg['n_epochs'] * n_updates}, off by a "
            f"factor of {num_minibatches})."
        )


@pytest.mark.parametrize(
    "config, n_envs, recurrent, geometry",
    [
        # brax's fragments, opted into; unroll_length = n_steps: whole rollouts
        ({"num_minibatches": 4, "unroll_length": 8}, 2, False, ("time", 4, 8)),
        ({"num_minibatches": 2, "unroll_length": 32}, 4, False, ("time", 2, 32)),
        # flat by default, even where the env axis splits evenly (the paper)
        ({"num_minibatches": 2}, 4, False, ("flat", 2, 32)),
        ({"num_minibatches": 4}, 4, False, ("flat", 4, 32)),
        ({"num_minibatches": 4}, 2, False, ("flat", 4, 32)),
        ({"num_minibatches": 0, "batch_size": 32}, 4, False, ("flat", 1, 32)),
        # recurrent: sequences of bptt_length steps; one minibatch for one env
        ({"num_minibatches": 2, "bptt_length": 8}, 2, True, ("recurrent", 2, 8)),
        ({"num_minibatches": 2}, 1, True, ("recurrent", 1, 32)),
    ],
)
def test_the_rollout_geometry(config, n_envs, recurrent, geometry):
    agent_config = PPOConfig(n_steps=32, **config)
    got = minibatch_geometry(agent_config, 32, n_envs, recurrent)
    assert got == Geometry(*geometry)


@pytest.mark.parametrize(
    "config",
    [
        {"unroll_length": 5},  # does not divide n_steps
        {"unroll_length": 16, "n_envs": 1},  # 2 fragments, 4 minibatches
        {"unroll_length": 8, "num_minibatches": 1},  # one minibatch
        {"unroll_length": 8, "memory": MemoryConfig(kind="gru", hidden_size=4)},
    ],
    ids=["indivisible", "too-few-fragments", "one-minibatch", "recurrent"],
)
def test_an_unusable_unroll_length_is_refused(config):
    """The update would fall back to whole rollouts: refused at construction."""
    kw = {"env_id": "Pendulum-v1", "n_steps": 32, "n_envs": 2, "num_minibatches": 4}
    with pytest.raises(ValueError, match="unroll_length"):
        PPO(**kw | config)


def test_a_recurrent_rollout_too_small_to_split_warns():
    """One env's whole rollout cannot make 2 minibatches: one, and a warning."""
    with pytest.warns(UserWarning, match="trains on one"):
        PPO(
            "Pendulum-v1",
            n_envs=1,
            n_steps=32,
            num_minibatches=2,
            memory=MemoryConfig(kind="gru", hidden_size=4),
        )


def test_vf_coef_zero_leaves_the_critic_untrained():
    """``vf_coef`` scales the value loss the critic steps on: at 0 every
    critic gradient is exactly 0, so Adam's moments stay 0 while the critic
    steps. The actor's moments move, so the walk is not vacuous."""
    rollout = _CFG["n_envs"] * _CFG["n_steps"]
    state, _ = PPO(**_CFG, vf_coef=0.0).train(
        seed=[0], n_timesteps=rollout, num_episode_test=1
    )

    def moments(train_state):
        leaves = jax.tree.leaves(train_state.opt_state)
        return [x for x in leaves if jnp.issubdtype(x.dtype, jnp.floating)]

    assert int(state.critic_state.step[0]) > 0
    assert moments(state.critic_state)
    assert not any(jnp.any(m) for m in moments(state.critic_state))
    assert any(jnp.any(m) for m in moments(state.actor_state))
