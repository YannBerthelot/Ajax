"""IBRL's acting as its authors' (hengyuan-hu/ibrl 3c457da, rl/q_agent.py
_act_ibrl): it executes the better of the agent's action and the expert's,
each scored by the target critics' minimum, over a fresh random subset where
the target takes one; in evaluation from the agent's evaluation action. The
replay flags the steps the expert's action took."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax import IQL
from ajax.controller import Controller
from ajax.evaluate import evaluate
from ajax.extensions import target_mods
from ajax.extensions.base import ExtensionContext
from ajax.extensions.capabilities import NotWired
from ajax.extensions.expert import ResidualPolicy
from ajax.extensions.target_mods import IBRL
from ajax.logging.wandb_logging import LoggingConfig
from ajax.wrappers import InitialStateWrapper
from tests.agents.test_agent_contract import CASES
from tests.environments.identity_env import IdentityEnv
from tests.extensions.test_ibrl_target import Q_EXPERT, Q_POLICY
from tests.test_evaluate import _fix_c, _LinearActor, _pays_one

KEY = jax.random.PRNGKey(0)
CONTEXT = ExtensionContext(step=jnp.asarray(0), rng=KEY, total_steps=100)


def half(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def torque_one(obs: jax.Array) -> jax.Array:
    return jnp.ones(obs.shape[:-1] + (1,))


@struct.dataclass
class SlopeCritic:
    """Q_k(s, a) = slope a for each of two critics: the target's slope +1,
    the live critic's -1, so the two rank any two actions oppositely."""

    params: Any = -1.0
    target_params: Any = 1.0
    apply_fn: Any = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        def apply(slope, x):
            return jnp.stack([slope * x[..., -1:]] * 2)

        object.__setattr__(self, "apply_fn", apply)


@struct.dataclass
class Learner:
    critic_state: Any = SlopeCritic()
    ext_state: Any = ((),)


def step(action: jax.Array, executed: str = "post_warmup_action") -> dict:
    return {
        executed: action,
        "expert_action": None,
        "observations": jnp.zeros((3, 2)),
        "raw_obs": jnp.zeros((3, 2)),
        "gate_rng": KEY,
    }


def test_collection_executes_the_expert_where_the_target_critics_rate_it_higher():
    """Rows acting -1, 0.5 and 1 against the expert's 0.5: on the target's
    critics (Q = a) the expert's wins the first alone (a tie keeps the
    agent's); the live critic would pick it on the last. Every critic is
    read, no key is drawn."""
    ibrl = IBRL(expert_policy=half, hybrid_acting=True)
    taken = step(jnp.array([[-1.0], [0.5], [1.0]]))
    chosen = ibrl.action(Learner(), (), taken, KEY, CONTEXT)
    np.testing.assert_allclose(chosen, [[0.5], [0.5], [1.0]])
    np.testing.assert_array_equal(taken["expert_mask"], [[True], [False], [False]])
    assert taken["gate_rng"] is KEY


def test_on_a_subset_it_chooses_on_a_fresh_draw_of_the_targets_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REDQ's subset size bound, it draws critics (2, 3 here) on the step's
    key and chooses on their minimum: the expert's wins rows 0 and 1, where
    every critic's minimum would pick it on row 1 alone."""
    monkeypatch.setattr(
        target_mods, "score_action_pair", lambda *a: (Q_EXPERT, Q_POLICY)
    )
    drawn = []

    def draw(size: int, num_critics: int, key: jax.Array) -> jax.Array:
        drawn.append((size, num_critics))
        return jnp.array([2, 3])

    monkeypatch.setattr(target_mods, "draw_selection_critics", draw)
    ibrl = IBRL(expert_policy=half, hybrid_acting=True, critic_subset_size=2)
    taken = step(jnp.zeros((3, 1)))
    ibrl.action(Learner(), (), taken, KEY, CONTEXT)
    np.testing.assert_array_equal(taken["expert_mask"][:, 0], [True, True, False])
    assert drawn == [(2, 4)] and taken["gate_rng"] is not KEY


def test_it_binds_the_targets_subset_size_and_acts_only_when_asked():
    """REDQ hands its subset size, the others none; without hybrid_acting
    IBRL has no acting phase (the stage and the evaluation leave it out)."""
    bound = IBRL(expert_policy=half).bind_to_agent(subset_size=2, expert_policy=None)
    assert bound.critic_subset_size == 2
    assert IBRL(expert_policy=half).bind_to_agent().critic_subset_size is None
    acting = {"action", "eval_action"}
    assert acting <= IBRL(expert_policy=half, hybrid_acting=True).implemented_phases()
    assert not acting.intersection(IBRL(expert_policy=half).implemented_phases())


def test_evaluation_executes_the_better_of_the_mean_and_the_experts_action():
    """The actor's mean t / 10 against the agent's expert's 0.5 (1 on the
    identity task's Box(-2, 2), paid 1 a step), read as IBRL's own
    (reads_stored_expert): the target critics (Q = a) pick the expert's at
    every step, return 3 and share 1; the live critic, or the plain policy,
    would earn the mean's 0.6."""
    env = InitialStateWrapper(IdentityEnv(), lambda k, s, p: _fix_c(k, s, p, 0.5))
    ibrl = IBRL(expert_policy=_pays_one, hybrid_acting=True, reads_stored_expert=True)
    out = evaluate(
        env,
        _LinearActor(),
        2,
        KEY,
        env.default_params,
        expert_policy=_pays_one,
        agent_state=Learner(),
        controller=Controller(choosers=((0, ibrl),)),
    )
    np.testing.assert_allclose(out[0], 3.0, rtol=1e-6)
    np.testing.assert_allclose(out[6], 1.0)


@pytest.mark.parametrize("name", ["SAC", "TD3", "REDQ", "TQC", "IQL"])
def test_the_replay_flags_the_rows_the_experts_action_took(name: str) -> None:
    """Without an agent expert, each agent acts with IBRL's: past the
    uniform warm-up (TQC has none) some rows store its action 0.5
    (Pendulum's torque 1), those the replay flags as the expert's where it
    keeps flags (IQL's keeps none, as for JSRL); evaluations log both
    shares."""
    case = CASES[name]
    ibrl = IBRL(expert_policy=torque_one, hybrid_acting=True)
    agent = case.cls(env_id="Pendulum-v1", n_envs=2, extensions=(ibrl,), **case.kwargs)
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    state, out = agent.train(
        seed=0, n_timesteps=case.n_timesteps, num_episode_test=1, logging_config=logging
    )
    evaluated = np.asarray(out["timestep"]) >= 0
    for share in ("Eval/expert share", "ibrl/expert_frac"):
        assert np.isfinite(np.asarray(out[share])[evaluated]).all(), share
    buffer_state = state.collector_state.buffer_state
    assert buffer_state is not None
    rows = buffer_state.experience
    stop = case.n_timesteps // 2
    took = np.isclose(np.asarray(rows["action"])[..., :stop, 0], 0.5, 0, 1e-6)
    assert took.any()
    if "is_expert" in rows:
        np.testing.assert_array_equal(rows["is_expert"][..., :stop, 0] > 0.5, took)
    warm = 0 if name == "TQC" else case.kwargs["learning_starts"] // 2
    assert not took[..., :warm].any()


def test_iql_under_a_residual_policy_refuses_hybrid_acting():
    """IQL's replay keeps the residual its actor chose: an executed expert's
    action leaves none to keep (as for JSRL and EDGE)."""
    residual = ResidualPolicy()
    IQL("Pendulum-v1", expert_policy=half, extensions=(IBRL(half), residual))
    with pytest.raises(NotWired, match="residual"):
        hybrid = IBRL(half, hybrid_acting=True)
        IQL("Pendulum-v1", expert_policy=half, extensions=(hybrid, residual))
