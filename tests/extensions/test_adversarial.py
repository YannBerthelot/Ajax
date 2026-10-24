"""AdversarialReward (GAIL): the rollout's rewards from the discriminator,
its updates, and where it runs. Whether a policy imitates through it is the
probes' (tests/probing/test_imitation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, SAC
from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.adversarial import AdversarialReward
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.extensions.capabilities import MissingCapability, NotWired

KEY = jax.random.PRNGKey(0)
ROWS = 16


def demos() -> TransitionDataset:
    """Demonstrations acting 0.5 at observations near 1."""
    obs = np.ones((ROWS, 1))
    ends = np.ones(ROWS, bool)
    return TransitionDataset(
        obs, np.full((ROWS, 1), 0.5), np.zeros(ROWS), ends, ~ends, obs
    )


def rollout() -> dict:
    """A (4, 2) rollout acting -0.5 at the same observations."""
    return {
        "raw_observations": jnp.ones((4, 2, 1)),
        "actions": jnp.full((4, 2, 1), -0.5),
    }


def test_the_rewards_are_minus_log_one_minus_d_before_the_discriminator_steps() -> None:
    gail = AdversarialReward(demos(), n_updates=3)
    state = gail.init_state(None, KEY)
    batch, ctx = rollout(), ExtensionContext(step=jnp.asarray(0), rng=KEY)
    rewards, stepped, metrics = gail.reward(
        None, state, batch, jnp.zeros((4, 2, 1)), ctx
    )
    x = jnp.concatenate([batch["raw_observations"], batch["actions"]], -1)
    want = -jax.nn.log_sigmoid(-state.apply_fn(state.params, x))
    np.testing.assert_allclose(rewards, want, rtol=1e-6)
    assert int(stepped.step) == 3 and set(metrics) >= {"gail/disc_loss", "gail/reward"}


def test_the_discriminator_reads_the_executed_action() -> None:
    """A sample beyond [-1, 1] runs as the bound, and is scored as it; a
    demonstration's action too."""
    gail = AdversarialReward(demos())
    ctx = ExtensionContext(step=jnp.asarray(0), rng=KEY)
    state = gail.init_state(None, KEY)
    beyond, bound = (rollout() | {"actions": jnp.full((4, 2, 1), a)} for a in (3, 1))
    zeros = jnp.zeros((4, 2, 1))
    np.testing.assert_array_equal(
        gail.reward(None, state, beyond, zeros, ctx)[0],
        gail.reward(None, state, bound, zeros, ctx)[0],
    )
    wide = demos()
    wide.action[:] = -2.0
    assert float(AdversarialReward(wide).rows()[:, -1].max()) == -1.0


def test_the_discriminator_learns_to_tell_the_rollout_from_the_demonstrations() -> None:
    gail = AdversarialReward(demos(), n_updates=50, learning_rate=1e-2)
    state, batch = gail.init_state(None, KEY), rollout()
    losses = []
    for i in range(4):
        ctx = ExtensionContext(step=jnp.asarray(i), rng=jax.random.fold_in(KEY, i))
        _, state, metrics = gail.reward(None, state, batch, jnp.zeros((4, 2, 1)), ctx)
        losses.append(float(metrics["gail/disc_loss"]))
    assert losses[-1] < losses[0]
    assert float(metrics["gail/expert_acc"]) == float(metrics["gail/policy_acc"]) == 1.0


def test_the_fold_threads_the_discriminators_state() -> None:
    gail = AdversarialReward(demos())
    stack = ExtensionStack((gail,))

    class State:
        ext_state = (gail.init_state(None, KEY),)

        def replace(self, ext_state):
            new = State()
            new.ext_state = ext_state
            return new

    state, rewards, _ = stack.apply_reward(
        State(), rollout(), jnp.zeros((4, 2, 1)), 0, KEY, 8
    )
    assert int(state.ext_state[0].step) == gail.n_updates and rewards.shape == (4, 2, 1)


def test_ppo_alone_folds_the_reward_phase_with_continuous_actions() -> None:
    PPO("Pendulum-v1", extensions=(AdversarialReward(demos()),))
    with pytest.raises(NotWired, match="reward"):
        SAC("Pendulum-v1", extensions=(AdversarialReward(demos()),))
    with pytest.raises(MissingCapability, match="continuous_actions"):
        PPO("CartPole-v1", extensions=(AdversarialReward(demos()),))
