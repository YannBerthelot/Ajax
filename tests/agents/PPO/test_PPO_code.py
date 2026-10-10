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
import pytest

from ajax import PPO
from ajax.agents.PPO.state import PPOConfig
from ajax.agents.PPO.train_PPO import Geometry, minibatch_geometry

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


def _train(extra_kwargs=None) -> tuple[int, int, int]:
    """Train PPO for ~1 training iteration's worth of timesteps and
    return ``(actor_step, critic_step, n_updates)``.

    ``actor_step`` and ``critic_step`` are the Adam step counters on
    each network -- they increment once per ``apply_gradients`` call
    and are part of the pytree so the JIT trace updates them (no
    spying needed).

    ``n_updates`` is the number of training iterations Ajax actually
    ran. We don't assume this equals 1 -- Ajax's training loop may
    ceil-round or run an extra iter; the wire-test asserts step ==
    n_epochs * num_minibatches * n_updates which is invariant to the
    iter count.

    ``train()`` always seed-vmaps, so ``.step`` has shape ``(n_seeds,)``
    even for a single seed. All seeds run lockstep so we take ``[0]``.
    """
    cfg = {**_CFG, **(extra_kwargs or {})}
    agent = PPO(**cfg)
    rollout = cfg["n_envs"] * cfg["n_steps"]
    out = agent.train(seed=[0], n_timesteps=rollout, num_episode_test=1)
    if isinstance(out, tuple):
        state = out[0]
    else:
        state = out
    a_step = jax.numpy.asarray(state.actor_state.step).reshape(-1)[0]
    c_step = jax.numpy.asarray(state.critic_state.step).reshape(-1)[0]
    n_upd = jax.numpy.asarray(state.n_updates).reshape(-1)[0]
    return int(a_step), int(c_step), int(n_upd)


def test_apply_gradients_called_n_epochs_times_num_minibatches():
    """Per training iteration, PPO must call ``apply_gradients`` once
    per (epoch, minibatch) pair, on BOTH the actor and critic. The
    Adam step counter on each network must therefore be
    ``n_epochs * num_minibatches`` after one training iteration.

    This is the **wire-test that would have caught the May 2026 bug**
    where ``do_update.body_fn`` passed the whole (num_minibatches,
    mb_size, feat) tensor to the loss as one batch, performing only
    ``n_epochs`` updates per iteration instead of
    ``n_epochs * num_minibatches``.
    """
    actor_step, critic_step, n_upd = _train()
    per_iter = _CFG["n_epochs"] * _CFG["num_minibatches"]
    expected = per_iter * n_upd
    assert actor_step == expected, (
        f"actor.step={actor_step}, expected n_epochs * num_minibatches * "
        f"n_updates = {_CFG['n_epochs']} * {_CFG['num_minibatches']} * "
        f"{n_upd} = {expected}. Likely cause: training body isn't "
        "iterating over minibatches (would give actor.step == "
        f"{_CFG['n_epochs'] * n_upd}, off by a factor of "
        f"{_CFG['num_minibatches']})."
    )
    assert critic_step == expected, (
        f"critic.step={critic_step}, expected {expected}. " "Same diagnosis as actor."
    )


def test_num_minibatches_actually_changes_step_count():
    """If we double num_minibatches, the per-iteration step count
    must double. Catches "num_minibatches honoured for reshape but
    not for the scan length" -- the precise shape of the May 2026 bug.
    """
    a4, _, _ = _train()  # num_minibatches=4
    a8, _, _ = _train(extra_kwargs={"num_minibatches": 8, "batch_size": 64})
    # a4 should be n_epochs * 4 = 8, a8 should be n_epochs * 8 = 16
    assert a8 == 2 * a4, (
        f"Doubling num_minibatches should double the per-iter step count, "
        f"got {a4} vs {a8}. Likely cause: num_minibatches is read for "
        "the batch reshape but not for the gradient-update count."
    )


def test_n_epochs_actually_changes_step_count():
    """Doubling n_epochs must double the per-iteration step count.
    Symmetric to the num_minibatches test; ensures the outer scan
    length really is n_epochs.
    """
    a2, _, _ = _train()  # n_epochs=2
    a4, _, _ = _train(extra_kwargs={"n_epochs": 4})
    assert a4 == 2 * a2, (
        f"Doubling n_epochs should double the per-iter step count, "
        f"got {a2} vs {a4}."
    )


def test_legacy_path_still_works_without_num_minibatches():
    """When ``num_minibatches=0`` (legacy default), the formula
    ``max(batch_size, n_steps) // min(...)`` is used. Verify the path
    is exercised and the step count matches the implied minibatch
    count.

    For batch_size=8, n_steps=8 → max/min = 1 → num_minibatches=1.
    Per iter: n_epochs * 1 = n_epochs steps.
    """
    actor_step, critic_step, n_upd = _train(
        extra_kwargs={
            "num_minibatches": 0,
            "batch_size": 8,
        }
    )
    expected = _CFG["n_epochs"] * 1 * n_upd  # implied 1 minibatch
    assert actor_step == expected
    assert critic_step == expected


@pytest.mark.parametrize(
    "config, n_envs, recurrent, geometry",
    [
        # brax's fragments; without unroll_length, whole rollouts per env
        ({"num_minibatches": 4, "unroll_length": 8}, 2, False, ("time", 4, 8)),
        ({"num_minibatches": 2}, 4, False, ("time", 2, 32)),
        # the fragments do not split evenly, or one minibatch: flat
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
