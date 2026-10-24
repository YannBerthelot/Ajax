"""The agent families (ajax.agents.families) on toy declared agents that
never train: the constructor their fields generate, the run config, the
``agent_config`` view, the knobs reaching ActorCritic, and the literal
dataclass decorator on every family, mixin and declared agent (the API docs
read the constructor from it). Training is the declared agents' own tests'."""

import ast
import dataclasses
import inspect
import textwrap
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TD3
from ajax.agents import families
from ajax.agents.families import (
    DeclaredAgent,
    ExpertGuided,
    GradientClipping,
    ImageObservations,
    ObservationNormalization,
    OnPolicyAgent,
    RecurrentReplayAgent,
    ReplayAgent,
    TrainContext,
)
from ajax.evaluate import ValueEstimate
from ajax.extensions.expert import ImitationLoss
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

    def build(self) -> None:
        self.doubled = 2 * self.batch_size

    @property
    def scaled(self) -> float:  # derived, as CrossQ's target entropy
        return 2 * self.lam

    name = "Toy"
    capabilities = frozenset({"discounted", "q_critic"})


ToyConfig = Toy.declare_config("gamma lam scaled")


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

    def init(self, key: jax.Array, ctx: TrainContext) -> BaseAgentState:
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor, critic = ctx.actor_critic(
            init_key, continuous=False, action_value=False, num_critics=1
        )
        return BaseAgentState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor,
            critic_state=critic,
            collector_state=ctx.collector(collector_key),
        )

    def update(self, state: Any, rollout: Any, start: Any, ctx: TrainContext) -> Any:
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
    assert set(agent.config) == Toy.field_names() | {"algo_name"}
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


def test_agent_config_is_a_view_of_the_fields_declared_once() -> None:
    view = Toy("Pendulum-v1", lam=0.25).agent_config
    assert (view.gamma, view.lam, view.scaled) == (0.9, 0.25, 0.5)
    assert view == Toy("Pendulum-v1", lam=0.25).agent_config
    assert view.replace(gamma=0.5) == ToyConfig(gamma=0.5, lam=0.25, scaled=0.5)
    assert ToyConfig(scaled=1.0).lam == 0.5  # the fields' defaults
    with pytest.raises(TypeError, match="scaled"):
        ToyConfig()  # a derived entry has no default
    assert (ToyConfig.__qualname__, ToyConfig.__module__) == ("ToyConfig", __name__)
    assert type(view) is ToyConfig


def test_the_fields_are_fixed_once_built_and_replace_builds_another() -> None:
    """What trains is what the run config logs (owner, 2026-10-24)."""
    agent = Toy("Pendulum-v1", lam=0.25)
    with pytest.raises(AttributeError, match=r"lam is fixed.*dataclasses.replace"):
        agent.lam = 1.0
    with pytest.raises(AttributeError, match=r"views its fields.*replace\(agent, g"):
        agent.agent_config = agent.agent_config.replace(lam=1.0)
    agent.network_args = agent.network_args  # not a field: writable
    other = dataclasses.replace(agent, lam=1.0)
    assert (other.lam, other.config["lam"], other.agent_config.scaled) == (
        1.0,
        1.0,
        2.0,
    )
    assert agent.lam == 0.25
    with pytest.raises(ValueError, match="trains on its fields"):
        agent.make_train(
            agent.env_args,
            agent.actor_optimizer_args,
            agent.critic_optimizer_args,
            agent.network_args,
            agent.agent_config.replace(lam=1.0),
            total_timesteps=8,
            num_episode_test=1,
        )


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
        extension_context = Toy.extension_context | {"gamma"}

    agent = Asking("Pendulum-v1")
    with pytest.raises(ValueError, match="extension_context names gamma, which"):
        agent.get_make_train()(
            agent.env_args,
            agent.actor_optimizer_args,
            agent.critic_optimizer_args,
            agent.network_args,
            agent.agent_config,
            total_timesteps=8,
            num_episode_test=1,
        )


def test_an_on_policy_agent_runs_its_update_on_each_rollout() -> None:
    params = inspect.signature(OnToy).parameters
    assert {"n_steps", "n_epochs", "num_minibatches", "cnn_spec"} <= set(params)
    assert params["expose_recent_rollout"].default is False
    with pytest.raises(TypeError, match="n_steps"):
        Undeclared("CartPole-v1")  # type: ignore[call-arg]
    agent = OnToy("CartPole-v1", cnn_extra_obs_dim=0)
    assert agent.network_args.cnn_image_shape is None  # ImageObservations'
    # The context runs init alone, as the run's first state.
    loop, ctx = agent.context(8)
    assert ctx.extensions is loop.stack and not ctx.recurrent and ctx.buffer is None
    assert agent.init(jax.random.PRNGKey(0), ctx).collector_state.timestep == 0
    state, _ = agent.train(seed=0, n_timesteps=4)  # 4 // 4 + 1 rollouts
    assert np.asarray(state.n_updates).tolist() == [2]  # one seed
    with pytest.raises(ValueError, match="trains on its fields"):
        agent.make_train(
            agent.env_args,
            agent.actor_optimizer_args,
            agent.critic_optimizer_args,
            agent.network_args,
            agent.agent_config.replace(),
            total_timesteps=8,
            num_episode_test=1,
        )
    with pytest.raises(ValueError, match="cannot train offline"):
        agent.train_offline(dataset=None, n_updates=1)


def test_a_recurrent_replay_agent_replays_windows_with_stored_carries() -> None:
    with pytest.raises(ValueError, match="stored_state=True requires a memory"):
        RecurrentToy("Pendulum-v1", stored_state=True)
    gru = MemoryConfig(kind="gru", hidden_size=4)
    agent = RecurrentToy("Pendulum-v1", memory=gru, stored_state=True, burn_in=2)
    assert agent.network_args.memory == gru and agent.stores_state()


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
    assert decorator(cls) == want | ({} if cls is DeclaredAgent else {"kw_only": True})
    names = {f.name for f in dataclasses.fields(cls)}
    assert not names & {*CONTRACT, "make_train"}
