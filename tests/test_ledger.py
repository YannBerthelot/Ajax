"""One budget of environment steps per run (ajax.ledger).

Contract (the expert-RL benchmark plan: "one step ledger that charges
pretraining and datasets", a 1M-step budget, expert queries logged):

- run_ledger(agent, n_timesteps) counts every environment step of a run,
  whoever acted: prefill, recorded rows, cloning and Monte-Carlo rollouts,
  which sit outside n_timesteps, then warm-up and online steps inside it.
- It counts the expert's queries: one per step it was called on.
- assert_budget raises unless the run spent exactly the budget; online_steps
  gives the n_timesteps that does. What it cannot count, it refuses.

- JSRLCurriculum's gate evaluates on the plant, so its episodes are charged,
  each at the task's length, with one expert query per step. A run's length is
  whole evaluations: where budget falls inside one, the run stops short of it
  by less than that evaluation, and no further.

Oracles: hand arithmetic from the extensions' documented n_steps ("per env"),
and on one real run the plant's own count of its step_env calls.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import PQN, SAC
from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.base import Extension
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy
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


def test_a_residual_policys_zero_start_costs_no_step():
    """It only zeroes the actor's head. A refusal here sends harnesses around
    the ledger, which then counts nothing of that arm."""
    agent = agent_with(ResidualPolicy(zero_start=True), expert_policy=expert)

    n_timesteps = online_steps(agent)
    ledger = run_ledger(agent, n_timesteps)

    assert n_timesteps == BUDGET
    ledger.assert_budget()
    assert ledger.expert_queries == BUDGET


def jsrl(length: int, gate: JSRLCurriculum, **kwargs) -> SAC:
    """SAC on Pendulum with episodes of length steps, the guide's horizon read
    off them, and gate."""
    params = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=length)
    kwargs.setdefault("learning_starts", 1_000)
    return SAC(
        Pendulum(),
        env_params=params,
        expert_policy=expert,
        extensions=(gate,),
        **kwargs,
    )


@pytest.mark.parametrize(
    "length, online, gate_steps",
    # n + length * n // 10_000 = 1M: one evaluation each 10,000 steps collected.
    [(200, 980_400, 19_600), (1_600, 862_400, 137_600)],
)
def test_the_gates_default_evaluations_fit_the_budget(length, online, gate_steps):
    """The owner's decision: charge the gate, and shrink it so JSRL fits 1M on
    both task lengths. At 10 episodes the same gate costs 200,000 and 1.6M
    steps, more than the whole budget on the long task. n + g length = 1M with
    g the evaluations by step n: 98 at 200 steps, 86 at 1,600."""
    agent = jsrl(length, JSRLCurriculum())

    n_timesteps = online_steps(agent)
    ledger = run_ledger(agent, n_timesteps)

    assert n_timesteps == online
    ledger.assert_budget()
    charged = {charge.source: charge for charge in ledger.charges}
    assert charged["jsrl_curriculum"].env_steps == gate_steps
    assert charged["jsrl_curriculum"].expert_queries == gate_steps
    assert ledger.expert_queries == BUDGET


def test_a_gate_evaluation_is_charged_every_episode_it_runs():
    """Three episodes of 200 steps per evaluation, at steps 10,000 to 30,000."""
    agent = jsrl(200, JSRLCurriculum(eval_episodes=3))

    ledger = run_ledger(agent, 30_000)

    assert ledger.env_steps == 30_000 + 3 * 3 * 200


def test_the_gate_waits_for_the_first_update():
    """post_update runs after an update, so an evaluation due before
    learning_starts never happens: of 10,000 to 40,000 only 30,000 and 40,000
    come after 25,000. Charging all four spends steps no run took."""
    agent = jsrl(200, JSRLCurriculum(), learning_starts=25_000)

    assert run_ledger(agent, 40_000).env_steps == 40_000 + 2 * 200


def test_the_gate_counts_the_steps_it_sees_on_many_envs():
    """On 2 envs the run reads steps 2, 4, ...; the first of each 5, by its rule
    step % 5 < 2: 6, 10, 16 and 20."""
    agent = jsrl(10, JSRLCurriculum(eval_every=5), n_envs=2, learning_starts=0)

    assert run_ledger(agent, 20).env_steps == 20 + 4 * 10


def test_a_budget_inside_an_evaluation_stops_short_of_it():
    """One evaluation of 1,600 comes at step 10,000. With 11,000 to spend, step
    10,000 would spend 11,600, so the run stops at 9,999, short by 1,001, which
    no run length can spend. The ledger takes that, not a whole evaluation."""
    agent = jsrl(1_600, JSRLCurriculum())

    n_timesteps = online_steps(agent, 11_000)

    assert n_timesteps == 9_999
    run_ledger(agent, n_timesteps).assert_budget(11_000)
    with pytest.raises(ValueError, match="shared budget"):
        run_ledger(agent, 9_400).assert_budget(11_000)
    with pytest.raises(ValueError, match="shared budget"):
        run_ledger(agent, 10_000).assert_budget(11_000)


@pytest.mark.parametrize("schedule", ["random", "time"])
def test_a_schedule_with_no_gate_spends_no_evaluation(schedule):
    agent = jsrl(200, JSRLCurriculum(schedule=schedule))

    assert online_steps(agent) == BUDGET
    run_ledger(agent, BUDGET).assert_budget()


def test_a_gate_on_a_run_with_no_ticks_of_steps_is_refused():
    """PQN updates once per rollout: the ledger cannot say when the gate fires,
    and refuses rather than leaving it uncounted."""
    agent = PQN(
        "CartPole-v1",
        n_envs=2,
        n_steps=8,
        n_epochs=1,
        num_minibatches=4,
        expert_policy=lambda obs: jnp.zeros((*obs.shape[:-1], 1), jnp.int32),
        extensions=(JSRLCurriculum(),),
    )

    with pytest.raises(ValueError, match="does not update once per tick"):
        online_steps(agent)


def test_the_gate_is_the_plants_own_count():
    """The oracle is the plant again. 2 envs, 24 steps: evaluations at 8, 16 and
    24, of 2 episodes of 6 steps, on envs of their own."""
    calls: list = []

    class CountingPendulum(Pendulum):
        def step_env(self, key, state, action, params):
            jax.debug.callback(calls.append, state.time)
            return super().step_env(key, state, action, params)

    params = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=6)
    agent = SAC(
        CountingPendulum(),
        env_params=params,
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=4,
        batch_size=4,
        expert_policy=expert,
        extensions=(JSRLCurriculum(eval_every=8, eval_episodes=2),),
    )
    state, _ = agent.train(seed=0, n_timesteps=24)
    jax.effects_barrier()

    assert len(calls) == 24 + 3 * 2 * 6
    timestep = int(state.collector_state.timestep[0])
    assert run_ledger(agent, timestep).env_steps == len(calls)


@pytest.mark.parametrize(
    "extension",
    [
        ExpertPrefill(n_steps=1_000, expert=expert),
        RetainedDemonstrations(n_steps=1_000, expert=expert),
        RLPD(n_steps=1_000, expert=expert, critic_layer_norm=False),
    ],
    ids=["prefill", "retained", "rlpd"],
)
def test_a_recorded_arm_queries_its_expert_before_training_alone(extension):
    """No expert acts online in these arms. Given their own expert, the agent
    tracks none, and the 1,000 steps of recording are all the queries it logs:
    999,000 more would class them with the live-query arms."""
    agent = SAC("Pendulum-v1", extensions=(extension,))

    ledger = run_ledger(agent, online_steps(agent))

    ledger.assert_budget()
    assert ledger.expert_queries == 1_000


def test_an_expert_only_the_prefill_holds_is_called_on_no_online_step():
    """The oracle is the expert: the ledger's zero is true only if collection
    does not call it. The agent's own expert_policy, in contrast, is called on
    each step, so its calls grow with the run."""
    calls: list = []

    def counted(obs: jax.Array) -> jax.Array:
        jax.debug.callback(lambda: calls.append(1))
        return expert(obs)

    def expert_calls(n_timesteps: int, **options) -> int:
        calls.clear()
        agent = SAC(
            "Pendulum-v1",
            learning_starts=4,
            batch_size=4,
            actor_architecture=("8", "relu"),
            critic_architecture=("8", "relu"),
            **options,
        )
        agent.train(seed=0, n_timesteps=n_timesteps)
        jax.effects_barrier()
        return len(calls)

    own = {"extensions": (ExpertPrefill(n_steps=4, expert=counted),)}
    assert expert_calls(16, **own) == expert_calls(8, **own) > 0
    agents = {"extensions": (ExpertPrefill(n_steps=4),), "expert_policy": counted}
    assert expert_calls(16, **agents) > expert_calls(8, **agents)


def test_offline_updates_cost_no_step():
    """They read the replay: a charge here would shorten IQL's online phase."""
    agent = agent_with(
        ExpertPrefill(n_steps=1_000, offline_updates=500), expert_policy=expert
    )

    assert online_steps(agent) == BUDGET - 1_000
