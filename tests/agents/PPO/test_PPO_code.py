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

import itertools

import jax
import jax.numpy as jnp
import pytest

from ajax import PPO
from ajax.agents.minibatch import RolloutSplit, choose_rollout_split
from ajax.agents.PPO.PPO import PPOConfig
from ajax.networks.memory import MemoryConfig

# Small CartPole config sized so one training iteration's rollout
# exactly equals n_envs * n_steps, with num_minibatches > 1 so the
# minibatch-iteration bug would be visible.
_CFG = {
    "env_id": "CartPole-v1",
    "n_envs": 4,
    "n_steps": 8,  # rollout per iter = 4 * 8 = 32 transitions
    "batch_size": 32,  # ignored when num_minibatches > 0
    "num_minibatches": 4,  # → minibatch size = 32 / 4 = 8
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
        # one env: the default bptt_length halves its rollout into 2 sequences
        ({"n_envs": 1, "num_minibatches": 2, "memory": MemoryConfig("gru", 4)}, 2),
    ],
    ids=[
        "4-minibatches",
        "8-minibatches",
        "4-epochs",
        "legacy-1-minibatch",
        "recurrent-one-env",
    ],
)
def test_apply_gradients_runs_once_per_epoch_and_minibatch(extra, num_minibatches):
    """Per training iteration PPO calls ``apply_gradients`` once per (epoch,
    minibatch) pair on BOTH the actor and the critic, so each Adam step
    counter reads ``n_epochs * num_minibatches * n_updates`` (whatever the
    number of iterations ``n_updates`` the loop ran).

    This is the **wire-test that would have caught the May 2026 bug**
    where ``do_update.body_fn`` passed the whole (num_minibatches,
    minibatch_size, feat) tensor to the loss as one batch, performing only
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
    "config, n_envs, recurrent, split",
    [
        # brax's fragments, opted into; unroll_length = n_steps: whole rollouts
        ({"num_minibatches": 4, "unroll_length": 8}, 2, False, ("time", 4, 8)),
        ({"num_minibatches": 2, "unroll_length": 32}, 4, False, ("time", 2, 32)),
        # flat by default, even where the env axis splits evenly (the paper)
        ({"num_minibatches": 2}, 4, False, ("flat", 2, 32)),
        ({"num_minibatches": 4}, 4, False, ("flat", 4, 32)),
        ({"num_minibatches": 4}, 2, False, ("flat", 4, 32)),
        ({"num_minibatches": 0, "batch_size": 32}, 4, False, ("flat", 1, 32)),
        # recurrent: sequences of bptt_length steps; by default the longest
        # that split: whole rollouts when the envs do (Baselines' ppo2),
        # else each rollout in num_minibatches // gcd(n_envs, it) pieces
        ({"num_minibatches": 2, "bptt_length": 8}, 2, True, ("recurrent", 2, 8)),
        ({"num_minibatches": 2}, 4, True, ("recurrent", 2, 32)),
        ({"num_minibatches": 2}, 1, True, ("recurrent", 2, 16)),
        ({"num_minibatches": 4}, 6, True, ("recurrent", 4, 16)),
        ({"num_minibatches": 1}, 3, True, ("recurrent", 1, 32)),
    ],
)
def test_the_rollout_split(config, n_envs, recurrent, split):
    agent_config = PPOConfig(n_steps=32, **config)
    got = choose_rollout_split(agent_config, 32, n_envs, recurrent)
    assert got == RolloutSplit(*split)


def test_the_geometry_names_resolve_to_the_rollout_split() -> None:
    from ajax.agents import minibatch
    from ajax.agents.PPO import train_PPO

    with pytest.warns(FutureWarning, match="minibatch.Geometry moved to"):
        assert minibatch.Geometry is RolloutSplit
    with pytest.warns(FutureWarning, match="minibatch.minibatch_geometry moved"):
        assert minibatch.minibatch_geometry is choose_rollout_split
    with pytest.warns(FutureWarning, match="train_PPO.Geometry moved to"):
        assert train_PPO.Geometry is RolloutSplit
    with pytest.warns(FutureWarning, match="train_PPO.minibatch_geometry moved"):
        assert train_PPO.minibatch_geometry is choose_rollout_split


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


@pytest.mark.parametrize(
    "config, match",
    [
        ({"num_minibatches": 3}, "3 equal pieces"),  # 3 does not divide 32
        ({"num_minibatches": 2, "bptt_length": 32}, "must divide 16"),
    ],
    ids=["no-default-splits", "bptt_length-does-not-split"],
)
def test_a_recurrent_rollout_that_does_not_split_is_refused(config, match):
    """As Baselines' and CleanRL's recurrent PPO assert: never one minibatch
    where more were asked for."""
    memory = MemoryConfig(kind="gru", hidden_size=4)
    with pytest.raises(ValueError, match=match):
        PPO("Pendulum-v1", n_envs=1, n_steps=32, memory=memory, **config)


def test_bptt_length_is_the_longest_that_splits_and_explicit_ones_divide_it():
    """By brute force, the rule docs/guide/memory.md gives: None picks the
    longest length whose sequences split into the minibatches, an explicit
    length is accepted exactly when its sequences split, and those are the
    divisors of the default."""
    for n_steps, n_envs, k in itertools.product((6, 8, 12), range(1, 7), range(1, 9)):
        divisors = [n for n in range(1, n_steps + 1) if n_steps % n == 0]
        splits = [n for n in divisors if n_steps // n * n_envs % k == 0]
        if splits:
            assert splits == [n for n in divisors if max(splits) % n == 0]
        for length in (None, *divisors):
            config = PPOConfig(n_steps=n_steps, num_minibatches=k, bptt_length=length)
            want = max(splits, default=None) if length is None else length
            if want is None or want not in splits:
                with pytest.raises(ValueError):
                    choose_rollout_split(config, n_steps, n_envs, True)
            else:
                got = choose_rollout_split(config, n_steps, n_envs, True)
                assert got == RolloutSplit("recurrent", k, want), (n_steps, n_envs, k)


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
