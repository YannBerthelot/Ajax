"""IBRL's soft choice, its authors' "ibrl_soft" (hengyuan-hu/ibrl 3c457da,
rl/q_agent.py _act_ibrl_soft and update_critic): the expert's action is
taken with probability sigmoid(beta (Q_E - Q_pi)), their softmax over the
two candidates, sampled in collection, in evaluation and in the bootstrap.
Off by default, where the choice is the hard one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.controller import Controller
from ajax.evaluate import evaluate
from ajax.extensions import target_mods
from ajax.extensions.target_mods import IBRL
from ajax.random_keys import derive_key
from ajax.wrappers import InitialStateWrapper
from tests.environments.identity_env import IdentityEnv
from tests.extensions.test_ibrl_acting import CONTEXT, KEY, Learner, half, step
from tests.test_evaluate import _fix_c, _LinearActor, _pays_one

ROWS = 40_000  # the share's standard deviation at most 0.0025
GAPS = jnp.array([-0.2, -0.05, 0.05, 0.1, 0.3])


def choose_on_gaps(ibrl: IBRL, key: jax.Array) -> np.ndarray:
    """Per gap, the share of ROWS rows whose expert's value tops the
    policy's by it that choose the expert."""
    expert_values = jnp.broadcast_to(GAPS, (ROWS, len(GAPS)))
    chosen = ibrl.choose_expert(expert_values, jnp.zeros_like(expert_values), key)
    return np.asarray(chosen).mean(0)


def test_the_default_choice_is_the_hard_one_and_draws_nothing():
    """soft_choice is off by default (beta 10 once on): the expert's action
    where its value is higher, a tie keeping the policy's, with no key."""
    ibrl = IBRL(expert_policy=half)
    assert not ibrl.soft_choice and ibrl.soft_choice_beta == 10.0
    expert_values = jnp.array([[1.0], [0.5], [0.0]])
    chosen = ibrl.choose_expert(expert_values, jnp.full((3, 1), 0.5), None)
    np.testing.assert_array_equal(chosen, [[True], [False], [False]])


@pytest.mark.parametrize("beta", [10.0, 3.0])
def test_the_sampled_share_is_the_sigmoid_of_beta_times_the_gap(beta: float):
    """On fixed values the expert's share converges to sigmoid(beta x gap),
    a chance of choosing it below the policy's value too."""
    ibrl = IBRL(expert_policy=half, soft_choice=True, soft_choice_beta=beta)
    want = jax.nn.sigmoid(beta * GAPS)
    np.testing.assert_allclose(choose_on_gaps(ibrl, KEY), want, atol=0.01)


def test_a_large_beta_is_the_hard_choice():
    ibrl = IBRL(expert_policy=half, soft_choice=True, soft_choice_beta=1e4)
    np.testing.assert_array_equal(choose_on_gaps(ibrl, KEY), GAPS > 0)


def test_collection_samples_the_choice_on_the_steps_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    """The acting choice draws on a key derived from its own, and flags the
    rows the expert's action took."""
    q_expert, q_policy = jnp.full((2, ROWS, 1), 0.1), jnp.zeros((2, ROWS, 1))
    monkeypatch.setattr(
        target_mods, "score_action_pair", lambda *a: (q_expert, q_policy)
    )
    ibrl = IBRL(expert_policy=half, soft_choice=True)
    taken = step(jnp.zeros((ROWS, 1)))
    taken.update(observations=jnp.zeros((ROWS, 2)), raw_obs=jnp.zeros((ROWS, 2)))
    chosen = ibrl.action(Learner(), (), taken, KEY, CONTEXT)
    flagged = np.asarray(taken["expert_mask"])
    np.testing.assert_allclose(flagged.mean(), jax.nn.sigmoid(1.0), atol=0.01)
    np.testing.assert_array_equal(np.asarray(chosen) == 0.5, flagged)
    uniform = jax.random.uniform(derive_key(KEY, "soft_choice"), flagged.shape)
    np.testing.assert_array_equal(flagged, uniform < jax.nn.sigmoid(1.0))


def test_evaluation_samples_the_choice_too():
    """As the authors' evaluation: from the actor's mean t / 10 against the
    expert's 0.5 on the target critics (Q = a), the expert is taken on a
    share sigmoid(beta (0.5 - t / 10)) of each step t of the three, where
    the hard choice takes it at every step."""
    env = InitialStateWrapper(IdentityEnv(), lambda k, s, p: _fix_c(k, s, p, 0.5))
    soft = {"soft_choice": True, "soft_choice_beta": 2.0}
    ibrl = IBRL(expert_policy=_pays_one, reads_stored_expert=True, **soft)
    out = evaluate(
        env,
        _LinearActor(),
        4000,
        KEY,
        env.default_params,
        expert_policy=_pays_one,
        agent_state=Learner(),
        controller=Controller(choosers=((0, "ibrl#0", ibrl),)),
    )
    want = np.mean(jax.nn.sigmoid(2.0 * (0.5 - jnp.arange(3) / 10)))
    np.testing.assert_allclose(out[6], want, atol=0.01)


def test_the_bootstrap_samples_the_choice(monkeypatch: pytest.MonkeyPatch):
    """The target's choice is sampled too: an expert's next action worth
    0.05 below the policy's is chosen on a share sigmoid(-0.5), and those
    rows bootstrap on its lower value (the hard choice never takes it)."""
    q_expert, q_policy = jnp.full((2, ROWS, 1), -0.05), jnp.zeros((2, ROWS, 1))
    monkeypatch.setattr(
        target_mods, "score_next_actions", lambda *a: (q_expert, q_policy)
    )
    batch = {"dones": jnp.zeros((ROWS, 1)), "gamma": 0.5, "critic_subset": None}
    ibrl = IBRL(expert_policy=half, soft_choice=True)
    target, metrics = ibrl.on_target(None, None, batch, jnp.zeros((ROWS, 1)), CONTEXT)
    share = jax.nn.sigmoid(-0.5)
    np.testing.assert_allclose(metrics["ibrl/expert_wins"], share, atol=0.01)
    chosen = np.asarray(target) < 0
    np.testing.assert_allclose(np.asarray(target)[chosen], 0.5 * -0.05, rtol=1e-6)
    np.testing.assert_allclose(chosen.mean(), metrics["ibrl/expert_wins"])
