"""Smoke tests for the Extension framework on PPO / DQN / PQN.

End-to-end checks that an :class:`~ajax.extensions.base.Extension`
instance passed via the agent's ``extensions=`` kwarg is wired through
the training loop: its :meth:`init_state` materialises in
``BaseAgentState.ext_state`` and its :meth:`post_update` actually fires
once per training iteration so the stateful counter grows.

These cover the minimal contract from Phase 3a of the agent-architecture
rework: the new surface is callable on each agent and the per-phase fold
runs. Numerical equivalence is out of scope (no SAC-style legacy mirror
exists for these three agents yet); the SAC suite holds that contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ajax.agents.DQN.DQN import DQN
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.extensions.base import Extension, ExtensionContext


@dataclass(frozen=True)
class CounterExt(Extension):
    """Stateful Extension: increments a counter on every ``post_update``.

    Hashable (frozen dataclass) so it composes as a JIT static arg.
    """

    name: str = "counter"

    def init_state(self, agent_state, rng):
        del agent_state, rng
        return jnp.asarray(0, dtype=jnp.int32)

    def post_update(self, agent_state, ext_state, ctx: ExtensionContext):
        del ctx
        return agent_state, ext_state + 1


# --------------------------------------------------------------------------
# Agent-specific smoke tests
# --------------------------------------------------------------------------
def _final_counter(agent_state) -> int:
    """Pull the CounterExt counter out of the (possibly vmapped) agent_state.

    ``agent.train`` returns a tuple ``(state, metrics)`` when there are
    extra outputs; here it returns just the state (no logging). The
    state is vmapped over seeds, so the counter has shape ``(n_seeds,)``.
    """
    if isinstance(agent_state, tuple):
        agent_state = agent_state[0]
    counter = agent_state.ext_state[0]
    # vmap over seeds -> take the first seed
    return int(jnp.asarray(counter).reshape(-1)[0])


def test_pqn_extension_counter_advances():
    """A CounterExt threaded into PQN's post_update increments per iteration."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        extensions=[CounterExt()],
    )
    # n_timesteps=256, n_envs=2, n_steps=8  ->  ~256/(2*8)=16 scan iters
    # (+1 from num_updates formula) ⇒ counter should grow > 1.
    state = agent.train(seed=42, n_timesteps=256)
    counter = _final_counter(state)
    assert counter > 1, (
        f"PQN CounterExt.post_update did not advance the ext_state counter: "
        f"got {counter}"
    )


def test_dqn_extension_counter_advances():
    """A CounterExt threaded into DQN's post_update fires after learning_starts."""
    agent = DQN(
        env_id="CartPole-v1",
        learning_starts=16,
        n_envs=1,
        architecture=("16", "relu"),
        batch_size=8,
        buffer_size=256,
        target_update_interval=10,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=200)
    counter = _final_counter(state)
    # post_update is gated on the do_update branch (timestep >=
    # learning_starts). With learning_starts=16 and n_timesteps=200 we
    # should see many ticks.
    assert counter > 1, (
        f"DQN CounterExt.post_update did not advance the ext_state counter: "
        f"got {counter}"
    )


def test_ppo_extension_counter_advances():
    """A CounterExt threaded into PPO's post_update fires every iteration."""
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=128)
    counter = _final_counter(state)
    assert counter > 1, (
        f"PPO CounterExt.post_update did not advance the ext_state counter: "
        f"got {counter}"
    )


# --------------------------------------------------------------------------
# eval_metrics fold smoke check (lightweight: just confirms train completes
# when an extension contributes an eval_metrics dict — exercises the fold
# path without needing to capture the log stream).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricExt(Extension):
    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        del agent_state, ext_state, rng, ctx
        return {"ext_smoke/x": jnp.asarray(1.0)}


def test_pqn_eval_metrics_fold_runs():
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=128)


def test_dqn_eval_metrics_fold_runs():
    agent = DQN(
        env_id="CartPole-v1",
        learning_starts=16,
        n_envs=1,
        architecture=("16", "relu"),
        batch_size=8,
        buffer_size=256,
        target_update_interval=10,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=128)


def test_ppo_eval_metrics_fold_runs():
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=128)


# --------------------------------------------------------------------------
# Empty extensions list — confirm zero-cost path still works on each agent.
# --------------------------------------------------------------------------
def test_pqn_empty_extensions_is_noop():
    """Empty extensions tuple ⇒ no ext_state, same as not passing the kwarg."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        extensions=(),
    )
    state = agent.train(seed=42, n_timesteps=128)
    if isinstance(state, tuple):
        state = state[0]
    assert state.ext_state == ()


def test_dqn_empty_extensions_is_noop():
    agent = DQN(
        env_id="CartPole-v1",
        learning_starts=16,
        n_envs=1,
        architecture=("16", "relu"),
        batch_size=8,
        buffer_size=256,
        target_update_interval=10,
        extensions=(),
    )
    state = agent.train(seed=42, n_timesteps=128)
    if isinstance(state, tuple):
        state = state[0]
    assert state.ext_state == ()


def test_ppo_empty_extensions_is_noop():
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        extensions=(),
    )
    state = agent.train(seed=42, n_timesteps=128)
    if isinstance(state, tuple):
        state = state[0]
    assert state.ext_state == ()


# --------------------------------------------------------------------------
# Composition: two extensions in the same stack both run.
# --------------------------------------------------------------------------
def test_pqn_two_extensions_compose():
    """Two CounterExts stacked yield two independent counters."""

    @dataclass(frozen=True)
    class CounterA(CounterExt):
        name: str = "counter_a"

    @dataclass(frozen=True)
    class CounterB(CounterExt):
        name: str = "counter_b"

    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        extensions=[CounterA(), CounterB()],
    )
    state = agent.train(seed=42, n_timesteps=128)
    if isinstance(state, tuple):
        state = state[0]
    assert len(state.ext_state) == 2
    a = int(jnp.asarray(state.ext_state[0]).reshape(-1)[0])
    b = int(jnp.asarray(state.ext_state[1]).reshape(-1)[0])
    assert (
        a > 0 and a == b
    ), f"CounterA={a}, CounterB={b}: post_update should fire on each per iter"
