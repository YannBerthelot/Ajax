"""One budget of environment steps per run (ajax.ledger).

Contract (the expert-RL benchmark plan: "one step ledger that charges
pretraining and datasets", a 1M-step budget, expert queries logged):

- run_ledger(agent, n_timesteps) counts every environment step of a run,
  whoever acted: prefill, recorded rows, cloning and Monte-Carlo rollouts,
  which sit outside n_timesteps, then warm-up and online steps inside it.
- It counts the expert's queries: one per step it was called on.
- assert_budget raises unless the run spent exactly the budget; online_steps
  gives the n_timesteps that does. What it cannot count, it refuses.

Oracles: hand arithmetic from the extensions' documented n_steps ("per env"),
and on one real run the plant's own count of its step_env calls.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import SAC
from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.base import Extension
from ajax.extensions.observe import ExpertStateInput
from ajax.extensions.porl import porl
from ajax.extensions.pretrain import (
    CloningPretrain,
    ExpertPrefill,
    MCPretrain,
    PriorData,
)
from ajax.extensions.retained import RLPD, RetainedDemonstrations
from ajax.extensions.target_mods import IBRL
from ajax.ledger import BUDGET, online_steps, pretraining_charges, run_ledger


def expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


def agent_with(*extensions, n_envs=1, learning_starts=1_000, **kwargs) -> SAC:
    if any(isinstance(e, ExpertPrefill) for e in extensions):
        kwargs.setdefault("expert_policy", expert)
    return SAC(
        "Pendulum-v1",
        n_envs=n_envs,
        learning_starts=learning_starts,
        extensions=extensions,
        **kwargs,
    )


def dataset(rows: int, expert_rows: int) -> TransitionDataset:
    flag = np.zeros(rows, bool)
    flag[:expert_rows] = True
    ends = np.zeros(rows, bool)
    ends[-1] = True
    return TransitionDataset(
        obs=np.zeros((rows, 3)),
        action=np.zeros((rows, 1)),
        reward=np.zeros(rows),
        terminated=np.zeros(rows, bool),
        truncated=ends,
        next_obs=np.zeros((rows, 3)),
        is_expert=flag,
    )


def test_a_plain_run_spends_its_n_timesteps_with_no_expert_query():
    """The control arm is the budget's reference: it must not be charged
    twice for its warm-up, which n_timesteps already holds."""
    ledger = run_ledger(agent_with(), BUDGET)

    assert ledger.env_steps == BUDGET
    assert ledger.expert_queries == 0
    assert {c.source: c.env_steps for c in ledger.charges} == {
        "warm-up": 1_000,
        "online": BUDGET - 1_000,
    }


@pytest.mark.parametrize(
    "extension, outside, queries",
    [
        # n_steps per env, 2 envs: 2 x 2_000.
        (ExpertPrefill(n_steps=2_000), 4_000, 4_000),
        (CloningPretrain(expert, n_steps=100_000), 200_000, 200_000),
        # n_mc_steps x n_mc_episodes in all, whole ticks of 2 envs: 1_001 x 3.
        (MCPretrain(expert, n_mc_steps=1_001, n_mc_episodes=3), 3_002, 3_002),
        # Every row was a plant step; 60 of the 100 were the expert's.
        (PriorData(dataset(100, 60)), 100, 60),
    ],
    ids=["prefill", "cloning", "monte-carlo", "dataset"],
)
def test_pretraining_is_charged_to_the_budget(extension, outside, queries):
    """Today these sit outside n_timesteps: uncharged, a data arm gets more
    real experience than the control."""
    agent = agent_with(extension, n_envs=2)

    n_timesteps = online_steps(agent)
    ledger = run_ledger(agent, n_timesteps)

    assert n_timesteps == BUDGET - outside
    assert ledger.env_steps == BUDGET
    ledger.assert_budget()
    online_queries = n_timesteps if agent.expert is not None else 0
    assert ledger.expert_queries == queries + online_queries


def counting_expert_state(n_envs: int) -> jax.Array:
    return jnp.zeros((n_envs, 2))


class StatefulExpert:
    """An expert with a memory of two columns."""

    init_state = staticmethod(counting_expert_state)

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return expert(obs), state


@pytest.mark.parametrize(
    "keeps_the_experts_state", [False, True], ids=["no-expert-online", "expert-state"]
)
def test_porl_charges_its_clone_and_its_warm_up_once(keeps_the_experts_state):
    """PORL's clone rollouts sit outside n_timesteps, its pre-sample steps
    inside: 2 x 50_000 clone steps leave 900_000 to collect, T of them
    pre-sampled, which a second charge would count twice. The expert acts only
    in the rollouts, unless the observations hold its state, which needs a call
    on each step: 100_000 queries, or 1_000_000."""
    pid = StatefulExpert() if keeps_the_experts_state else expert
    features = (ExpertStateInput.for_expert(pid),) if keeps_the_experts_state else ()
    arm = porl(CloningPretrain(pid, n_steps=50_000), features=features)
    kwargs = {"expert_policy": pid} if keeps_the_experts_state else {}
    agent = SAC("Pendulum-v1", n_envs=2, **arm, **kwargs)

    n_timesteps = online_steps(agent)
    ledger = run_ledger(agent, n_timesteps)

    assert n_timesteps == BUDGET - 100_000
    ledger.assert_budget()
    assert ledger.expert_queries == (BUDGET if keeps_the_experts_state else 100_000)


def test_live_expert_is_queried_on_every_collected_step():
    """Live-query arms pay for the expert on each plant step, warm-up
    included: hybrid acting calls it whoever then acts."""
    agent = agent_with(IBRL(expert_policy=expert))

    assert run_ledger(agent, 5_000).expert_queries == 5_000


@pytest.mark.parametrize("spent", [BUDGET - 1, BUDGET + 1])
def test_a_run_off_the_budget_is_refused(spent):
    """A run short of the budget had less experience than its rivals, as one
    over it had more: neither is comparable."""
    with pytest.raises(ValueError, match="shared budget"):
        run_ledger(agent_with(), spent).assert_budget()


def test_a_harness_charges_a_dataset_it_brings_itself():
    """train_offline's dataset reaches no extension: the harness charges it."""
    ledger = run_ledger(agent_with(), 900_000).charge("recorded", 100_000, 100_000)

    ledger.assert_budget()
    assert ledger.expert_queries == 100_000


def test_pretraining_past_the_budget_leaves_no_run():
    """A negative n_timesteps would otherwise surface as a confusing failure,
    or none."""
    agent = agent_with(ExpertPrefill(n_steps=BUDGET + 1))

    with pytest.raises(ValueError, match="pretraining alone"):
        online_steps(agent)


def test_a_pretraining_it_does_not_know_is_refused():
    """A new arm's rollouts would otherwise go uncounted, silently."""

    class Unknown(Extension):
        name: str = "unknown"

        def pretrain(self, agent_state, ext_state, context):
            return agent_state, ext_state

    with pytest.raises(ValueError, match="Unknown pretrains"):
        online_steps(agent_with(Unknown()))


def test_the_ledger_is_the_plants_own_count():
    """The oracle is the plant: every step_env call counts, whoever made it,
    so a pretraining rolling out more than the ledger says fails here."""
    calls: list = []

    class CountingPendulum(Pendulum):
        def step_env(self, key, state, action, params):
            jax.debug.callback(calls.append, state.time)
            return super().step_env(key, state, action, params)

    env, params = CountingPendulum(), gymnax.make("Pendulum-v1")[1]
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=8,
        batch_size=4,
        expert_policy=expert,
        extensions=(
            CloningPretrain(expert, n_steps=6, epochs=1, batch_size=4),
            ExpertPrefill(n_steps=4),
            # 260 rows: its regression takes batches of 256.
            MCPretrain(
                expert,
                n_mc_steps=130,
                n_mc_episodes=2,
                n_steps=2,
                use_online_light=False,
            ),
        ),
    )
    state, _ = agent.train(seed=0, n_timesteps=16)
    jax.effects_barrier()

    # 2 envs: prefill 2 x 4, cloning 2 x 6, Monte Carlo 260, 16 online.
    assert len(calls) == 8 + 12 + 260 + 16
    timestep = int(state.collector_state.timestep[0])
    assert run_ledger(agent, timestep).env_steps == len(calls)


@pytest.mark.parametrize("module", [RLPD, RetainedDemonstrations])
def test_rows_kept_beside_the_replay_are_charged(module) -> None:
    """Their rows take no slot of the replay (replay_rows is 0), and they cost
    the plant all the same: charging replay_rows would give the arm free
    data. 500 expert steps in 2 envs are 1,000 steps and as many queries; a
    dataset of 300 rows, 120 of them the expert's, is 300 steps, 120 queries."""
    rolled = agent_with(module(n_steps=500), n_envs=2, expert_policy=expert)
    (charge,) = pretraining_charges(rolled)
    assert (charge.env_steps, charge.expert_queries) == (1_000, 1_000)
    logged = agent_with(module(n_steps=0, dataset=dataset(300, 120)))
    (charge,) = pretraining_charges(logged)
    assert (charge.source, charge.env_steps) == ("recorded dataset", 300)
    assert charge.expert_queries == 120
