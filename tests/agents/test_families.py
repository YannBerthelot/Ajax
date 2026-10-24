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
import pytest

from ajax import TD3
from ajax.agents import families
from ajax.agents.families import (
    DeclaredAgent,
    ExpertGuided,
    GradientClipping,
    ObservationNormalization,
    RecurrentReplayAgent,
    ReplayAgent,
)
from ajax.evaluate import ValueEstimate
from ajax.extensions.expert import ImitationLoss
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import action_value
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
    learning_starts: int = 3
    buffer_size: int = 64
    batch_size: int = 8
    lam: float = 0.5

    def check(self) -> None:
        if self.lam < 0:
            raise ValueError("Toy's lam is at least 0.")

    def networks(self) -> dict[str, Any]:
        return {"tanh_mean": True}

    @property
    def scaled(self) -> float:  # derived, as CrossQ's target entropy
        return 2 * self.lam

    name = "Toy"
    capabilities = frozenset({"discounted", "q_critic"})


ToyConfig = Toy.declare_config("gamma lam scaled")


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class Unsized(ReplayAgent):  # no buffer_size, and no discount
    batch_size: int = 8

    name = "Unsized"
    capabilities = frozenset({"discounted", "q_critic"})


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class RecurrentToy(RecurrentReplayAgent):
    learning_starts: int = 80
    buffer_size: int = 64
    batch_size: int = 4

    name = "RecurrentToy"


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
        assert (args.max_grad_norm, args.weight_decay, args.eps) == (1.0, 0.1, 1e-8)
    assert agent.network_args.actor_architecture == NET  # a shared argument
    assert agent.network_args.tanh_mean  # networks()
    assert agent.buffer is not None  # the family's build


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
    with pytest.raises(AttributeError, match="agent_config is fixed"):
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
    with pytest.raises(TypeError, match="buffer_size"):
        Unsized("Pendulum-v1")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Toy only supports continuous action"):
        Toy("CartPole-v1")


def test_the_value_estimate_reads_a_declared_discount() -> None:
    assert Toy("Pendulum-v1").value_estimate() == ValueEstimate(action_value, 0.9)
    assert Unsized("Pendulum-v1", buffer_size=64).value_estimate() is None


def test_a_recurrent_replay_agent_replays_windows_with_stored_carries() -> None:
    with pytest.raises(ValueError, match="stored_state=True requires a memory"):
        RecurrentToy("Pendulum-v1", stored_state=True)
    gru = MemoryConfig(kind="gru", hidden_size=4)
    agent = RecurrentToy("Pendulum-v1", memory=gru, stored_state=True, burn_in=2)
    assert agent.network_args.memory == gru and agent.keeps_carries()


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
