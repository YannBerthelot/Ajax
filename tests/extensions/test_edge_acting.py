"""EDGE's acting: its scoring gates score the action the step would execute,
which every critic is trained on (under a residual policy, the residual
through the expert); its fixed gate draws without a score and reads no
critic."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.controller import ActingState
from ajax.environments.action_stage import apply_action_extensions
from ajax.environments.interaction import ActionPipelineResult
from ajax.extensions import exploration
from ajax.extensions.base import ExtensionContext
from ajax.extensions.exploration import EDGEExploration

KEY = jax.random.PRNGKey(0)
CONTEXT = ExtensionContext(step=jnp.asarray(0), rng=KEY, total_steps=100)
SCORING_GATES = ("argmax", "boltzmann", "lcb", "argmax_lcb", "thompson")
#: Two rows: the expert acts 0.5 and -0.5, the policy chooses the residuals
#: 0.2 and -0.2, so the step executes 0.7 and -0.7.
EXPERT, RESIDUAL = jnp.array([[0.5], [-0.5]]), jnp.array([[0.2], [-0.2]])


def learner(critic_state: Any) -> SimpleNamespace:
    timestep = SimpleNamespace(timestep=jnp.asarray(0))
    return SimpleNamespace(
        critic_state=critic_state, collector_state=timestep, ext_state=((),)
    )


def step(policy: list, executed: list, expert: list) -> dict:
    """The stage's dict for rows choosing policy, executing executed and
    offered the expert's expert, each an (n, 1) column."""
    return {
        "policy_action": jnp.array(policy)[:, None],
        "post_warmup_action": jnp.array(executed)[:, None],
        "expert_action": jnp.array(expert)[:, None],
        "observations": jnp.zeros((len(policy), 3)),
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
    for gate in SCORING_GATES:
        with pytest.raises(AssertionError, match="called the critic"):
            EDGEExploration(gate=gate).action(
                agent_state, (), dict(taken), KEY, CONTEXT
            )


def residual_step(agent_state, acting, raw_obs, key, uniform, mix_key, action_key):
    """A residual policy's pipeline: the step executes the residual through
    the expert, clip(a_E + a_pi)."""
    executed = jnp.clip(acting.expert_action + RESIDUAL, -1.0, 1.0)
    return ActionPipelineResult(env_action=executed, policy_action=RESIDUAL)


def run_stage(edge: EDGEExploration, critic_state: Any) -> ActionPipelineResult:
    """The residual step through the action stage, edge in its slot."""
    staged = apply_action_extensions(residual_step, (edge,), 2, 100)
    observations, done, timestep = jnp.zeros((2, 3)), jnp.zeros(2, bool), jnp.asarray(0)
    acting = ActingState(observations, done, timestep, expert_action=EXPERT)
    no_warmup = jnp.asarray(False)
    return staged(learner(critic_state), acting, None, KEY, no_warmup, KEY, KEY)


def test_under_a_residual_the_expert_is_compared_with_the_executed_action(
    critic_stub: Any,
) -> None:
    """On Q = a the expert's 0.5 loses to the executed 0.7 and its -0.5 beats
    the executed -0.7; against the residuals 0.2 and -0.2 both would flip."""
    edge = EDGEExploration(gate="argmax", decay_frac=0.0)
    result = run_stage(edge, critic_stub(lambda x: x[..., -1:]))
    np.testing.assert_allclose(result.env_action, [[0.7], [-0.5]], rtol=1e-6)
    np.testing.assert_array_equal(result.is_expert_flag, [[0.0], [1.0]])


@pytest.mark.parametrize("gate", SCORING_GATES)
def test_every_scoring_gate_scores_the_executed_action(
    gate: str, critic_stub: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each reads the critic at the executed 0.7 and -0.7."""
    scored = []

    def score(*inputs: Any) -> tuple:
        scored.append(inputs[1])  # (obs, scored action, expert action, ...)
        values = jnp.zeros((2, 2, 1))  # an ensemble of two
        return values, values

    monkeypatch.setattr(exploration, "score_action_pair", score)
    run_stage(EDGEExploration(gate=gate), critic_stub(lambda x: x))
    np.testing.assert_allclose(scored[0], [[0.7], [-0.7]], rtol=1e-6)
