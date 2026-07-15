"""Smoke tests for the Extension framework on Phase 3b agents.

End-to-end checks that an :class:`~ajax.extensions.base.Extension`
instance passed via the agent's ``extensions=`` kwarg is wired through
the training loop for every remaining agent migrated in Phase 3b:
TD3 / UDRL / REDQ / ASAC / SafeSAC / APO / AVG.

Cloning (``ajax.agents.cloning``) is a helper module — it provides
``CloningConfig`` and pre-train utilities used by other agents
(SAC / TD3 / REDQ / ASAC / APO), not a standalone agent class — so it
has no smoke test of its own. The migration plan called it out as
"train-time-only, no env interaction loop" precisely because there is
no per-step training loop to fold extension phases into.

SafeSAC subclasses SAC and re-uses ``super().__init__(*args, **kwargs)``
verbatim, so the Phase 2 SAC extension wiring is inherited end-to-end;
this file only adds a smoke test confirming that inheritance is alive.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SafeSAC.SafeSAC import SafeSAC
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.UDRL.UDRL import UDRL
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


@dataclass(frozen=True)
class MetricExt(Extension):
    """Stateless Extension contributing an extra eval-metric key."""

    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx: ExtensionContext):
        del agent_state, ext_state, rng, ctx
        return {"ext_smoke/x": jnp.asarray(1.0)}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _unwrap(state):
    """``agent.train()`` may return ``(state, metrics)``; strip the tuple."""
    if isinstance(state, tuple):
        state = state[0]
    return state


def _final_counter(agent_state) -> int:
    """Pull the CounterExt counter out of the (possibly vmapped) agent_state.

    The state is vmapped over seeds, so the counter has shape ``(n_seeds,)``.
    """
    agent_state = _unwrap(agent_state)
    counter = agent_state.ext_state[0]
    return int(jnp.asarray(counter).reshape(-1)[0])


# --------------------------------------------------------------------------
# TD3
# --------------------------------------------------------------------------
def test_td3_extension_counter_advances():
    """TD3 threads CounterExt through every gradient-step's post_update."""
    agent = TD3(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=64)
    assert _final_counter(state) > 1


def test_td3_empty_extensions_is_noop():
    agent = TD3(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=(),
    )
    state = _unwrap(agent.train(seed=42, n_timesteps=64))
    assert state.ext_state == ()


# --------------------------------------------------------------------------
# UDRL
# --------------------------------------------------------------------------
def test_udrl_extension_counter_advances():
    """UDRL is supervised / actor-only; post_update is folded after the
    sample-and-train inner scan."""
    agent = UDRL(
        env_id="CartPole-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        n_updates_per_iter=4,
        buffer_capacity=8,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=128)
    assert _final_counter(state) > 1


def test_udrl_empty_extensions_is_noop():
    agent = UDRL(
        env_id="CartPole-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        n_updates_per_iter=4,
        buffer_capacity=8,
        extensions=(),
    )
    state = _unwrap(agent.train(seed=42, n_timesteps=128))
    assert state.ext_state == ()


# --------------------------------------------------------------------------
# REDQ (SAC lineage — own train loop, separately wired)
# --------------------------------------------------------------------------
def test_redq_extension_counter_advances():
    agent = REDQ(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        num_critic_updates=2,
        num_critics=4,
        subset_size=2,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=48)
    assert _final_counter(state) > 1


def test_redq_eval_metrics_fold_runs():
    agent = REDQ(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        num_critic_updates=2,
        num_critics=4,
        subset_size=2,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=48)


# --------------------------------------------------------------------------
# ASAC (SAC lineage — average-reward variant, separately wired)
# --------------------------------------------------------------------------
def test_asac_extension_counter_advances():
    agent = ASAC(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=48)
    assert _final_counter(state) > 1


def test_asac_eval_metrics_fold_runs():
    agent = ASAC(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=48)


# --------------------------------------------------------------------------
# SafeSAC (subclass of SAC — inherits Phase 2 extension wiring verbatim)
# --------------------------------------------------------------------------
def test_safesac_extension_counter_advances():
    """SafeSAC.__init__ forwards ``extensions=`` via ``super().__init__``;
    the entire SAC extension stack (init / pretrain / on_target / critic_loss /
    actor_loss / action / eval_action / post_update / eval_metrics) is
    inherited — no SafeSAC-specific re-wiring required."""
    agent = SafeSAC(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=48)
    assert _final_counter(state) > 1


# --------------------------------------------------------------------------
# APO (on-policy, average-reward — no gamma)
# --------------------------------------------------------------------------
def test_apo_extension_counter_advances():
    agent = APO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=128)
    assert _final_counter(state) > 1


def test_apo_eval_metrics_fold_runs():
    agent = APO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        extensions=[MetricExt()],
    )
    agent.train(seed=42, n_timesteps=128)


# --------------------------------------------------------------------------
# AVG (own non-ActorCritic base — extension_stack built on-instance)
# --------------------------------------------------------------------------
def test_avg_extension_counter_advances():
    agent = AVG(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "leaky_relu"),
        critic_architecture=("16", "leaky_relu"),
        extensions=[CounterExt()],
    )
    state = agent.train(seed=42, n_timesteps=64)
    assert _final_counter(state) > 1


def test_avg_empty_extensions_is_noop():
    agent = AVG(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "leaky_relu"),
        critic_architecture=("16", "leaky_relu"),
        extensions=(),
    )
    state = _unwrap(agent.train(seed=42, n_timesteps=64))
    assert state.ext_state == ()


# --------------------------------------------------------------------------
# Composition: two extensions stacked on TD3 (covers the SAC-lineage path)
# --------------------------------------------------------------------------
def test_td3_two_extensions_compose():
    """Two CounterExts stacked yield two independent counters."""

    @dataclass(frozen=True)
    class CounterA(CounterExt):
        name: str = "counter_a"

    @dataclass(frozen=True)
    class CounterB(CounterExt):
        name: str = "counter_b"

    agent = TD3(
        env_id="Pendulum-v1",
        n_envs=1,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=[CounterA(), CounterB()],
    )
    state = _unwrap(agent.train(seed=42, n_timesteps=48))
    assert len(state.ext_state) == 2
    a = int(jnp.asarray(state.ext_state[0]).reshape(-1)[0])
    b = int(jnp.asarray(state.ext_state[1]).reshape(-1)[0])
    assert (
        a > 0 and a == b
    ), f"CounterA={a}, CounterB={b}: post_update should fire on each per iter"
