"""The agent families (ajax.agents.families) on toy declared agents that
never train: the constructor their fields generate, the run config, the
knobs reaching ActorCritic, and the literal dataclass decorator on every
family, mixin and declared agent (the API docs read the constructor from
it). Training is the declared agents' own tests'."""

import ast
import dataclasses
import inspect
import textwrap
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax import TD3
from ajax.agents import families
from ajax.agents.families import (
    DeclaredAgent,
    ExpertGuided,
    GradientClipping,
    ImageObservations,
    IncrementalAgent,
    ObservationNormalization,
    OnPolicyAgent,
    OwnIterationAgent,
    OwnReplayAgent,
    RecurrentReplayAgent,
    ReplayAgent,
    TrainContext,
)
from ajax.environments.row_collector import init_row_collector_state
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import Extension
from ajax.extensions.expert import ImitationLoss
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import action_value, state_value
from ajax.state import BaseAgentState, OptimizerConfig
from ajax.types import Architecture
from tests.agents.test_agent_surface import CONTRACT
from tests.test_exports import library_agents

NET: Architecture = ("8", "relu")


def push(obs):
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class Toy(GradientClipping, ObservationNormalization, ExpertGuided, ReplayAgent):
    actor_architecture: Architecture = NET
    critic_architecture: Architecture = NET
    gamma: float = 0.9
    reward_scale: float = 2.0
    learning_starts: int = 3
    buffer_size: int = 64
    batch_size: int = 8
    lam: float = 0.5

    # The hooks, none chaining super(): the family's part is its own.
    def check(self) -> None:
        if self.lam < 0:
            raise ValueError("Toy's lam is at least 0.")

    def actor_critic_args(self) -> dict[str, Any]:
        return {"tanh_mean": True}

    def optimizer_settings(self) -> dict[str, Any]:
        return {"beta_1": 0.5}

    def setup(self) -> None:
        self.doubled = 2 * self.batch_size

    @property
    def scaled(self) -> float:  # derived, as CrossQ's target entropy
        return 2 * self.lam

    name = "Toy"
    capabilities = frozenset({"discounted", "q_critic"})


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class Unsized(ReplayAgent):  # no buffer_size nor learning_starts, no discount
    batch_size: int = 8

    name = "Unsized"
    capabilities = frozenset({"discounted", "q_critic"})


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class Undiscounted(ReplayAgent):  # discounted, without gamma nor reward_scale
    buffer_size: int = 64
    batch_size: int = 8
    learning_starts: int = 0

    name = "Undiscounted"
    capabilities = frozenset({"discounted", "q_critic"})


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class RecurrentToy(RecurrentReplayAgent):
    learning_starts: int = 80
    buffer_size: int = 64
    batch_size: int = 4

    name = "RecurrentToy"


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class OnToy(ImageObservations, OnPolicyAgent):
    """Counts its updates; discounted without a reward scale."""

    actor_architecture: Architecture = NET
    critic_architecture: Architecture = NET
    gamma: float = 0.9
    n_steps: int = 4
    n_epochs: int = 1
    num_minibatches: int = 1

    def init(self, key: jax.Array, context: TrainContext) -> BaseAgentState:
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor, critic = context.make_actor_critic(
            init_key, continuous=False, action_value=False, num_critics=1
        )
        return BaseAgentState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor,
            critic_state=critic,
            collector_state=context.make_collector(collector_key),
        )

    def update(
        self, state: Any, rollout: Any, start: Any, context: TrainContext
    ) -> Any:
        assert rollout.obs.shape[:2] == (self.n_steps, self.n_envs)
        return state.replace(n_updates=state.n_updates + 1), {"r": rollout.reward}

    name = "OnToy"
    action_space = "discrete"
    capabilities = frozenset({"discounted", "v_critic"})


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class Undeclared(OnPolicyAgent):  # no n_steps, n_epochs nor num_minibatches
    name = "Undeclared"


def test_the_constructor_is_the_fields_keyword_only_after_env_id() -> None:
    params = inspect.signature(Toy).parameters
    assert next(iter(params)) == "env_id" and set(params) == Toy.field_names()
    assert params["env_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(p.kind is p.KEYWORD_ONLY for n, p in params.items() if n != "env_id")
    with pytest.raises(TypeError, match="positional"):
        Toy("Pendulum-v1", 2)  # type: ignore[misc]


def test_the_run_config_is_the_call_and_the_knobs_reach_the_actor_critic() -> None:
    agent = Toy(
        "Pendulum-v1",
        n_envs=2,
        expert_policy=push,
        max_grad_norm=1.0,
        weight_decay=0.1,
        normalize_observations=True,
    )
    assert set(agent.config) == Toy.field_names().union({"algo_name"})
    assert agent.config["algo_name"] == "Toy" and agent.config["n_envs"] == 2
    # The field and the run config keep the caller's expert; the agent reads
    # it in agent units.
    assert agent.config["expert_policy"] is push is agent.expert_policy
    assert agent.expert is not push
    assert agent.env_args.n_envs == 2 and agent.env_args.normalize_obs_in_agent
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        settings = (args.max_grad_norm, args.weight_decay, args.beta_1, args.eps)
        # optimizer_settings(), on OptimizerConfig's eps: the family sets none.
        assert settings == (1.0, 0.1, 0.5, OptimizerConfig(1.0).eps)
    assert agent.network_args.actor_architecture == NET  # a shared argument
    assert agent.network_args.tanh_mean  # actor_critic_args()
    assert agent.buffer is not None and agent.doubled == 16  # family's, then own


def test_actor_critic_args_only_fill_a_field_left_none() -> None:
    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class Filling(Toy):
        encoder_kernel_init: str | None = None

        def actor_critic_args(self) -> dict[str, Any]:
            return {"tanh_mean": True, "encoder_kernel_init": "lecun_uniform"}

    assert Filling("Pendulum-v1").network_args.encoder_kernel_init == "lecun_uniform"
    with pytest.raises(TypeError, match="fill a field left None.*encoder_kernel"):
        Filling("Pendulum-v1", encoder_kernel_init="he_normal")


def test_the_fields_are_fixed_once_built_and_replace_builds_another() -> None:
    """What trains is what the run config logs (owner, 2026-10-24)."""
    agent = Toy("Pendulum-v1", lam=0.25)
    with pytest.raises(AttributeError, match=r"lam is fixed.*dataclasses.replace"):
        agent.lam = 1.0
    agent.network_args = agent.network_args  # not a field: writable
    other = dataclasses.replace(agent, lam=1.0)
    assert (other.lam, other.config["lam"], other.scaled) == (1.0, 1.0, 2.0)
    assert agent.lam == 0.25


class HalfUnit:
    """A stateful expert (SAC's convention): half a unit of action."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return jnp.full(obs.shape[:-1] + (1,), 0.5), state


def test_replace_passes_the_callers_expert_to_an_agent_on_its_own_env() -> None:
    """dataclasses.replace reads the expert_policy field back: the caller's
    expert, which the new agent shares with its extensions and reads on its
    own env's bounds, as a fresh agent does."""
    expert = HalfUnit()
    agent = TD3(
        "Pendulum-v1",
        expert_policy=expert,
        extensions=(ImitationLoss(expert, coef=1.0),),
    )
    other = dataclasses.replace(agent, gamma=0.9)
    (imitation,) = other.extension_stack.extensions
    assert isinstance(imitation, ImitationLoss)
    assert other.expert_policy is expert and other.config["expert_policy"] is expert
    assert imitation.expert_policy is other.expert is not agent.expert
    obs = jnp.zeros((1, 2))
    moved = dataclasses.replace(
        TD3("Pendulum-v1", expert_policy=push), env_id="MountainCarContinuous-v0"
    )
    fresh = TD3("MountainCarContinuous-v0", expert_policy=push)
    # 0.5 on MountainCar's [-1, 1], not 0.25 as on Pendulum's [-2, 2].
    assert float(moved.expert(obs)[0, 0]) == float(fresh.expert(obs)[0, 0]) == 0.5


def test_checks_required_fields_and_action_spaces_refuse_at_construction() -> None:
    with pytest.raises(ValueError, match="lam"):
        Toy("Pendulum-v1", lam=-1.0)
    with pytest.raises(TypeError, match="buffer_size' and 'learning_starts'"):
        Unsized("Pendulum-v1")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Toy only supports continuous action"):
        Toy("CartPole-v1")


def test_the_value_estimate_reads_the_declared_discount_and_reward_scale() -> None:
    estimate = ValueEstimate(action_value, 0.9, 2.0)
    assert Toy("Pendulum-v1").value_estimate() == estimate
    with pytest.raises(TypeError, match="'discounted' capability.*declares gamma"):
        Undiscounted("Pendulum-v1")
    assert RecurrentToy("Pendulum-v1").value_estimate() is None  # not discounted
    # No reward_scale field: the rewards unscaled.
    assert OnToy("CartPole-v1").value_estimate() == ValueEstimate(state_value, 0.9)


def test_make_train_refuses_an_extension_context_it_cannot_give() -> None:
    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class Asking(Toy):
        extension_context = Toy.extension_context.union({"num_critics"})

    agent = Asking("Pendulum-v1")
    with pytest.raises(ValueError, match="extension_context names num_critics, whi"):
        agent.get_make_train()(
            agent.env_args,
            agent.actor_optimizer_args,
            agent.critic_optimizer_args,
            agent.network_args,
            total_timesteps=8,
            num_episode_test=1,
        )


@dataclasses.dataclass(frozen=True)
class Recorder(Extension):
    """Keeps the agent context it is bound to."""

    seen: tuple = ()
    name: str = "recorder"

    def bind_to_agent(self, **agent_context: Any) -> "Recorder":
        return dataclasses.replace(self, seen=tuple(sorted(agent_context.items())))


def test_the_extensions_get_the_declared_values_and_what_only_the_agent_has():
    """make_context gives the unbound extensions, the memory flag and the
    declared gamma and reward_scale; a discounted Q critic with targets and
    an update_critic gives BellmanPretrain its critic step and the targets'
    update (extension_values)."""

    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class Targeted(Toy):
        extension_context = Toy.extension_context.union(
            {"extensions", "gamma", "recurrent", "reward_scale", "update_value_fn"}
        )
        capabilities = Toy.capabilities.union({"target_critic"})

        def update_critic(self, agent_state, batch, carries, context):
            return agent_state, context, {}

    loop, _ = Targeted("Pendulum-v1").make_context(8, extensions=(Recorder(),))
    seen = dict(loop.extensions.extensions[0].seen)
    assert (seen["gamma"], seen["reward_scale"], seen["recurrent"]) == (0.9, 2.0, False)
    assert isinstance(seen["extensions"][0], Recorder)
    _, context, _ = seen["update_value_fn"]("state", "batch", False, 0.9, ())
    assert (context.extensions, context.executed, context.total_timesteps) == (
        (),
        None,
        1,
    )
    assert not ReplayAgent.extension_values(Toy("Pendulum-v1"), {})  # no targets


@struct.dataclass(kw_only=True)
class CountingState(BaseAgentState):
    seen: Any = 0  # the steps update_step_statistics saw


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class IncrementalToy(IncrementalAgent):
    """Counts the steps it saw and its updates; its critic's settings."""

    actor_architecture: Architecture = NET
    critic_architecture: Architecture = NET
    learning_starts: int = 4

    def network_settings(self) -> dict[str, Any]:
        return {"penultimate_normalization": True}

    def init(self, key: jax.Array, context: TrainContext) -> CountingState:
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor, critic = context.make_actor_critic(
            init_key, continuous=True, action_value=True, squash=True, num_critics=1
        )
        collector = context.make_collector(collector_key)
        return CountingState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor,
            critic_state=critic,
            collector_state=collector,
        )

    def update_step_statistics(self, agent_state, transition, context):
        return agent_state.replace(seen=agent_state.seen + 1)

    def update(self, agent_state, transition, context):
        assert transition.obs.shape[0] == self.n_envs  # the step alone
        return agent_state.replace(n_updates=agent_state.n_updates + 1), {}

    name = "IncrementalToy"


def test_an_incremental_agent_updates_on_each_step_after_learning_starts():
    """Six ticks of two envs: the statistics see every step, the update
    runs on each step collected from learning_starts (4) on, four times,
    never on one collected before it; the warm-up acts
    uniformly only for an agent claiming the warmup capability; the
    reference's network settings reach the networks; no offline training."""
    agent = IncrementalToy("Pendulum-v1", n_envs=2)
    assert agent.network_args.penultimate_normalization
    assert agent.uniform_steps() == 0 and Toy("Pendulum-v1").uniform_steps() == 0
    state, _ = agent.train(seed=0, n_timesteps=12)
    assert (np.asarray(state.seen).tolist(), np.asarray(state.n_updates).tolist()) == (
        [6],
        [4],
    )
    with pytest.raises(ValueError, match="cannot train offline"):
        agent.train_offline(dataset=None, n_updates=1)

    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class WarmingUp(IncrementalToy):
        capabilities = frozenset({"warmup"})

    assert WarmingUp("Pendulum-v1").uniform_steps() is None


def test_an_on_policy_agent_runs_its_update_on_each_rollout() -> None:
    params = inspect.signature(OnToy).parameters
    assert {"n_steps", "n_epochs", "num_minibatches", "cnn_spec"} <= set(params)
    assert params["expose_recent_rollout"].default is False
    with pytest.raises(TypeError, match="n_steps"):
        Undeclared("CartPole-v1")  # type: ignore[call-arg]
    agent = OnToy("CartPole-v1", cnn_extra_obs_dim=0)
    assert agent.network_args.cnn_image_shape is None  # ImageObservations'
    # The context runs init alone, as the run's first state.
    loop, context = agent.make_context(8)
    assert (
        context.extensions is loop.extensions
        and not context.recurrent
        and context.buffer is None
    )
    assert agent.init(jax.random.PRNGKey(0), context).collector_state.timestep == 0
    state, _ = agent.train(seed=0, n_timesteps=4)  # 4 // 4 + 1 rollouts
    assert np.asarray(state.n_updates).tolist() == [2]  # one seed
    with pytest.raises(ValueError, match="cannot train offline"):
        agent.train_offline(dataset=None, n_updates=1)


def test_a_recurrent_replay_agent_replays_windows_with_stored_carries() -> None:
    with pytest.raises(ValueError, match="stored_state=True requires a memory"):
        RecurrentToy("Pendulum-v1", stored_state=True)
    gru = MemoryConfig(kind="gru", hidden_size=4)
    agent = RecurrentToy("Pendulum-v1", memory=gru, stored_state=True, burn_in=2)
    assert agent.network_args.memory == gru and agent.stores_state()


@struct.dataclass(kw_only=True)
class ResumedState(BaseAgentState):
    resumes: Any = 0  # the times a resumed call transformed the state


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class OwnToy(OwnIterationAgent):
    """Counts its iterations of n_envs x 2 steps; a call given restart=True
    counts a resume it transforms."""

    actor_architecture: Architecture = NET
    critic_architecture: Architecture = NET

    def init(self, key: jax.Array, context: TrainContext) -> ResumedState:
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor, critic = context.make_actor_critic(
            init_key, continuous=True, action_value=False, num_critics=1
        )
        return ResumedState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor,
            critic_state=critic,
            collector_state=context.make_collector(collector_key),
        )

    def iterate(self, agent_state, context):
        collector = agent_state.collector_state
        timestep = collector.timestep + self.count_iteration_steps(context)[0]
        agent_state = agent_state.replace(
            n_updates=agent_state.n_updates + 1,
            collector_state=collector.replace(timestep=timestep),
        )
        return agent_state, {"timestep": timestep}

    def report(self, agent_state, aux):
        return {"timestep": aux["timestep"], "Train/updates": agent_state.n_updates}

    def evaluate(self, agent_state, key, context, num_episode_test):
        return {"Eval/episodes": jnp.asarray(num_episode_test, jnp.float32)}

    def count_iteration_steps(self, context):
        return 2 * context.env_args.n_envs, "n_envs x 2"

    def configure_call(self, context, n_iterations, restart: bool = False):
        def count_resume(agent_state):
            return agent_state.replace(resumes=agent_state.resumes + 1)

        return context, count_resume if restart else None

    name = "OwnToy"


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class StackedToy(OwnToy):
    returns_iterations = True

    def make_iteration_output(self, aux, evaluations):
        return {"aux": aux, "evaluations": evaluations}


def test_an_own_iteration_agent_runs_its_iterations_on_the_loop() -> None:
    """Three iterations of 2 envs x 2 steps, evaluated every 8 steps: the
    evaluations recorded, or every iteration's output returned with them; a
    call option reaches configure_call, whose transform runs on a resumed
    state alone; an option it does not take and a dataset are refused."""
    agent = OwnToy("Pendulum-v1", n_envs=2)
    logging = LoggingConfig(config={}, log_frequency=8, use_wandb=False)
    state, rows = agent.train(0, 12, num_episode_test=3, logging_config=logging)
    assert np.asarray(state.n_updates).tolist() == [3]
    assert np.asarray(rows["timestep"]).tolist() == [[8, -1]]  # ticks 1, (3)
    assert np.asarray(rows["Eval/episodes"])[0, 0] == 3.0
    stacked = StackedToy("Pendulum-v1", n_envs=2)
    _, out = stacked.train(0, 12, logging_config=logging)
    assert np.asarray(out["aux"]["timestep"]).tolist() == [[4, 8, 12]]
    assert np.asarray(out["evaluations"]["timestep"]).tolist() == [[-1, 8, -1]]
    _, out = stacked.train(0, 12)
    assert out["evaluations"] is None and out["aux"]["timestep"].shape == (1, 3)

    def counts(state: Any) -> list[int]:  # one seed's resumes and updates
        return [int(state.resumes[0]), int(state.n_updates[0])]

    assert counts(agent.train(0, 4, restart=True)[0]) == [0, 1]
    # A resume donates the state it is given: read each before the next call.
    state, _ = agent.train(0, 4, initial_state=state, restart=True)
    assert counts(state) == [1, 4]
    assert counts(agent.train(0, 4, initial_state=state)[0]) == [1, 5]
    with pytest.raises(TypeError, match="OwnToy.train takes no n_iterations"):
        agent.train(0, 4, n_iterations=1)
    with pytest.raises(ValueError, match="cannot train offline"):
        agent.train_offline(dataset=None, n_updates=1)


@struct.dataclass(kw_only=True)
class RowsState(BaseAgentState):
    stored: Any = 0  # the rows collect stored
    restarts: Any = 0  # the ticks after_log saw a log written


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class ReplayToy(OwnReplayAgent):
    """Stores n_envs rows a tick in a replay as long as the call's last
    tick, and updates after every odd tick."""

    actor_architecture: Architecture = NET
    critic_architecture: Architecture = NET

    def prepare_replay(self, start_tick, end_tick, agent_state):
        self.prepared = (start_tick, end_tick)
        return end_tick, agent_state

    def init(self, key, context):
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor, critic = context.make_actor_critic(
            init_key, continuous=True, action_value=False, num_critics=1
        )
        zero = jnp.zeros((), jnp.int32)
        return RowsState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor,
            critic_state=critic,
            collector_state=init_row_collector_state(collector_key, context.env_args),
            stored=zero,
            restarts=zero,
        )

    def collect(self, agent_state, tick, context):
        rows = context.env_args.n_envs
        collector = agent_state.collector_state
        collector = collector.replace(
            rows=collector.rows + rows, timestep=collector.timestep + rows
        )
        stored = agent_state.stored + rows
        return agent_state.replace(collector_state=collector, stored=stored), tick

    def count_updates(self, agent_state, tick, context):
        return tick % 2

    def update(self, agent_state, experience, context):
        return agent_state.replace(n_updates=agent_state.n_updates + 1), None

    def report(self, agent_state, aux):
        return {"timestep": agent_state.collector_state.timestep}

    def evaluate(self, agent_state, key, context, num_episode_test):
        return {"Eval/replay": jnp.asarray(context.buffer, jnp.float32)}

    def make_after_log(self):
        def restart(agent_state, logged):
            return agent_state.replace(restarts=agent_state.restarts + logged)

        return restart

    def check_run(self, agent_state, end_tick):
        self.checked = end_tick

    name = "ReplayToy"


def test_an_own_replay_agent_runs_its_ticks_on_a_replay_sized_per_call() -> None:
    """Six ticks of 2 envs, evaluated every 4: the replay of the call's ticks
    reaches the evaluation, after_log sees the log, check_run the last tick;
    a resume continues at its tick; a horizon ending before the call is
    refused before prepare_replay; a dataset is refused."""
    agent = ReplayToy("Pendulum-v1", n_envs=2)
    logging = LoggingConfig(config={}, log_frequency=8, use_wandb=False)
    state, rows = agent.train(0, 12, logging_config=logging)
    assert agent.prepared == (0, 6) and agent.checked == 6
    assert np.asarray(rows["timestep"]).tolist() == [[8, -1]]  # ticks 3, (7)
    assert np.asarray(rows["Eval/replay"])[0, 0] == 6.0
    counts = [state.stored, state.n_updates, state.restarts]
    assert [int(count[0]) for count in counts] == [12, 3, 1]  # odd ticks 1, 3, 5
    # A resume donates the state it is given: read each before the next call.
    state, _ = agent.train(0, 8, initial_state=state)
    assert agent.prepared == (6, 10) and agent.checked == 10
    assert [int(state.stored[0]), int(state.n_updates[0])] == [20, 5]  # 7, 9
    with pytest.raises(ValueError, match="ends before this call does"):
        agent.train(0, 4, initial_state=state, total_timesteps=22)
    assert agent.prepared == (6, 10)
    with pytest.warns(UserWarning, match="n_envs"):
        agent.train(0, 5)
    with pytest.raises(ValueError, match="cannot train offline"):
        agent.train_offline(dataset=None, n_updates=1)


def test_an_own_replay_agent_compares_no_value_estimate() -> None:
    """Its evaluation is its own: even a discounted Q-critic (a world
    model's, reading a latent state) gives no estimate to compare."""

    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class DiscountedReplayToy(ReplayToy):
        gamma: float = 0.99
        capabilities = frozenset({"discounted", "q_critic"})

    assert DiscountedReplayToy("Pendulum-v1", n_envs=2).value_estimate() is None


def test_an_agent_says_what_its_memory_cannot_take() -> None:
    """A run with memory refuses extensions (check_recurrent_options),
    unless the agent carries them through its memory, as APG's rollout."""

    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class Remembering(OwnToy):
        memory: Any = None

        supports_memory = True

    gru = MemoryConfig(kind="gru", hidden_size=4)
    with pytest.raises(
        NotImplementedError,
        match="OwnToy does not support these options yet: extensions",
    ):
        Remembering("Pendulum-v1", memory=gru).make_context(8, extensions=(Recorder(),))

    @dataclasses.dataclass(kw_only=True, eq=False, repr=False)
    class CarryingThrough(Remembering):
        def check_recurrent_options(self, extensions):
            del extensions

    agent = CarryingThrough("Pendulum-v1", memory=gru)
    loop, _ = agent.make_context(8, extensions=(Recorder(),))
    assert isinstance(loop.extensions.extensions[0], Recorder)


def declared_classes() -> list[type]:
    """The families, the mixins and the library's declared agents."""
    own = [c for c in vars(families).values() if inspect.isclass(c)]
    own = [c for c in own if c.__module__ == families.__name__]
    own.remove(families.TrainContext)
    return own + sorted(library_agents(DeclaredAgent), key=lambda c: c.__name__)


def decorator(cls: type) -> dict[str, Any] | None:
    """The keywords of the literal ``@dataclass(...)`` on ``cls``."""
    node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    assert isinstance(node, ast.ClassDef)
    for d in node.decorator_list:
        if isinstance(d, ast.Call) and ast.unparse(d.func) in (
            "dataclass",
            "dataclasses.dataclass",
        ):
            return {str(k.arg): ast.literal_eval(k.value) for k in d.keywords}
    return None


@pytest.mark.parametrize("cls", declared_classes(), ids=lambda c: c.__name__)
def test_each_is_a_literal_dataclass_whose_contract_is_no_field(cls: type) -> None:
    # DeclaredAgent's env_id stays positional (KW_ONLY after it).
    want = {"eq": False, "repr": False}
    if cls is not DeclaredAgent:
        want.update({"kw_only": True})
    assert decorator(cls) == want
    names = {f.name for f in dataclasses.fields(cls)}
    assert not names.intersection({*CONTRACT, "make_train"})
