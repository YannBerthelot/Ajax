"""EDGE's acting: its fixed gate draws without a score and reads no critic,
where the other gates call one."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.extensions.base import ExtensionContext
from ajax.extensions.exploration import EDGEExploration

KEY = jax.random.PRNGKey(0)
CONTEXT = ExtensionContext(step=jnp.asarray(0), rng=KEY, total_steps=100)


def learner(critic_state: Any) -> SimpleNamespace:
    timestep = SimpleNamespace(timestep=jnp.asarray(0))
    return SimpleNamespace(critic_state=critic_state, collector_state=timestep)


def step(policy: list, executed: list, expert: list) -> dict:
    """The stage's dict for rows choosing policy, executing executed and
    offered the expert's expert, each an (n, 1) column."""
    return {
        "policy_action": jnp.array(policy)[:, None],
        "post_warmup_action": jnp.array(executed)[:, None],
        "expert_action": jnp.array(expert)[:, None],
        "observations": jnp.zeros((len(policy), 3)),
        "gate_rng": KEY,
    }


def test_the_fixed_gate_reads_no_critic(critic_stub: Any) -> None:
    """It runs beside a critic that scores no action (a V(s) network; here
    one failing on any call), which a scoring gate calls."""

    def scores_no_action(x: jax.Array) -> jax.Array:
        raise AssertionError("the gate called the critic")

    agent_state = learner(critic_stub(scores_no_action))
    edge = EDGEExploration(gate="fixed", fixed_prob=1.0, decay_frac=0.0)
    taken = step([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    np.testing.assert_allclose(edge.action(agent_state, (), taken, KEY, CONTEXT), 1.0)
    for gate in ("argmax", "boltzmann", "lcb", "argmax_lcb", "thompson"):
        with pytest.raises(AssertionError, match="called the critic"):
            EDGEExploration(gate=gate).action(
                agent_state, (), dict(taken), KEY, CONTEXT
            )
