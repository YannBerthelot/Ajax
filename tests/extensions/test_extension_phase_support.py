"""Agents declare what their algorithm has and the extension phases they
fold; an extension needing anything else is refused, naming it."""

import dataclasses
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp
import pytest

from ajax import PPO, SAC, TD3
from ajax.agents.base import ActorCritic
from ajax.agents.loop import LOOP_PHASES
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionStack,
    check_extensions,
)
from ajax.extensions.capabilities import (
    ALGORITHM_CAPABILITIES,
    MissingCapability,
    NotWired,
    run_capabilities,
)
from ajax.extensions.ensemble import KernelRepulsion
from ajax.extensions.expert import (
    ExpertGains,
    JSRLCurriculum,
    OnlineBC,
    ResidualPolicy,
)
from ajax.extensions.instrumentation import BiasVorePenalty
from ajax.extensions.pretrain import MCPretrain, PhiRefresh
from ajax.extensions.target_mods import CriticBlend, MCVarianceCorrection, ValueBox

AGENTS = "APG APO ASAC AVG DQN DreamerV3 PPO PQN REDQ SAC TD3 TDMPC2 UDRL".split()
#: The agents a residual policy runs on: their critics score its executed
#: actions, or their update reads the policy's own (PPO, APO, IQL's actor).
RESIDUAL = {"SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC", "AVG"}
RESIDUAL.update({"PPO", "APO", "IQL", "APG", "UDRL"})
#: The agents whose collection runs the warm-up's chooser (ExpertWarmup).
WARMUP = {"SAC", "TD3", "REDQ", "ASAC", "AVG", "DQN"}
#: The agents whose replay the replay family's collector writes, which runs
#: the stored action's chooser (StorePolicyAction).
REPLAY_FAMILY = {"ASAC", "REDQ", "SAC", "TD3", "TQC", "DQN"}
#: The collection phases an agent collecting on its own folds itself.
OWN_FOLDS = {("APG", "observe")}


@dataclass(frozen=True)
class TargetTweak(Extension):
    name: str = "target-tweak"

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {}


@dataclass(frozen=True)
class ActorNudge(Extension):
    name: str = "actor-nudge"

    def actor_loss(self, agent_state, ext_state, batch, context):
        return 0.0


@dataclass(frozen=True)
class Instrument(Extension):
    name: str = "instrument"

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {}

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state


@dataclass(frozen=True)
class EveryPhase(Extension):
    """Overrides every phase (with the base behaviour)."""

    name: str = "every-phase"

    def pretrain(self, agent_state, ext_state, context):
        return agent_state, ext_state

    def reward(self, agent_state, ext_state, batch, rewards, context):
        return rewards, ext_state, {}

    def observe(self, inputs, state):
        return jnp.zeros((*inputs.raw_obs.shape[:-1], 0)), state

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target

    def critic_loss(self, agent_state, ext_state, batch, context):
        return 0.0

    def actor_loss(self, agent_state, ext_state, batch, context):
        return 0.0

    def action(self, agent_state, ext_state, obs, rng, context):
        return None

    def eval_action(self, agent_state, ext_state, obs, rng, context):
        return None

    def transform_action(self, actions, inputs):
        return actions

    def actuate(self, actions, inputs):
        return actions, inputs.expert_state

    def choose_warmup_action(self, uniform_actions, inputs):
        return uniform_actions, jnp.zeros((), dtype=bool)

    def choose_replay_action(self, executed_actions, policy_actions):
        return executed_actions

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {}


class LatentAgent(ActorCritic):
    capabilities = frozenset({"q_critic"})
    supported_extension_phases = frozenset({"pretrain", "post_update", "eval_metrics"})

    def __init__(self, extensions=()):
        self.config = {}
        super().__init__(env_id="CartPole-v1", n_envs=1, extensions=extensions)


def test_unsupported_phases_are_rejected_at_construction():
    """An agent with the capability a phase needs (a critic for on_target)
    that does not fold the phase has not wired it."""
    with pytest.raises(NotWired) as error:
        LatentAgent(extensions=[Instrument(), TargetTweak()])
    message = str(error.value)
    assert "LatentAgent" in message
    assert "'target-tweak'" in message and "on_target" in message
    assert "instrument" not in message  # its phases are supported
    LatentAgent(extensions=[Instrument()])  # supported phases are accepted


def test_a_phase_needing_a_capability_the_algorithm_lacks_is_missing():
    """An actor-loss term on an agent without an actor (DQN) lacks a
    capability, which the refusal names with its meaning."""
    with pytest.raises(MissingCapability, match=r"actor \(a policy network"):
        check_extensions("DQN", [ActorNudge()], {"q_critic", "critic"}, frozenset())


def test_the_check_names_every_missing_capability_and_gap():
    """One refusal lists each extension with all it lacks; the missing
    capabilities come before what is not wired."""
    has = {"actor", "q_critic", "critic"}
    with pytest.raises(MissingCapability) as error:
        check_extensions("X", [KernelRepulsion(), TargetTweak()], has, frozenset())
    assert "critic_ensemble" in str(error.value)
    assert "target-tweak" not in str(error.value)  # wiring, reported after
    with pytest.raises(NotWired, match="takes no expert_policy"):
        check_extensions("X", [JSRLCurriculum()], {"off_policy"}, frozenset(PHASES))


def test_a_module_provides_capabilities_to_its_stack():
    """MCPretrain provides phi*, which its readers need, whatever the order."""
    has = run_capabilities(
        {"actor", "q_critic", "critic_ensemble", "discounted"},
        continuous=True,
        has_expert=True,
    )
    reader = MCVarianceCorrection(threshold=1.0)
    with pytest.raises(MissingCapability, match="phi_star"):
        check_extensions("SAC", [reader], has, frozenset(PHASES))
    check_extensions(
        "SAC",
        [reader, MCPretrain(expert_policy=_expert)],
        has,
        frozenset(PHASES),
        context=MCPretrain.context,
        batch_keys=SAC.phase_batch_keys,
    )


def test_run_capabilities_follow_the_environment_expert_and_critics():
    caps = run_capabilities(
        {"q_critic", "critic_ensemble"}, continuous=False, has_expert=False
    )
    assert caps == {
        "q_critic",
        "critic_ensemble",
        "critic",
        "scalar_target",
        "discrete_actions",
    }
    one = run_capabilities(
        {"v_critic", "critic_ensemble"}, continuous=True, has_expert=True, num_critics=1
    )
    assert one == {
        "v_critic",
        "critic",
        "scalar_target",
        "continuous_actions",
        "agent_expert",
    }


def test_a_distributional_critic_refuses_what_reads_one_target_per_row():
    """A critic regressing on quantiles (TQC's) lacks ``scalar_target``, so
    the extensions that blend its target toward one value, replace it with
    one, or compare one value with it refuse the agent."""
    algorithm = {"actor", "q_critic", "critic_ensemble", "discounted"}
    one = run_capabilities(algorithm, continuous=True, has_expert=True)
    many = run_capabilities(
        algorithm.union({"distributional_critic"}), continuous=True, has_expert=True
    )
    assert "scalar_target" in one and "scalar_target" not in many
    for module in (
        CriticBlend(expert_policy=_expert),
        MCVarianceCorrection(threshold=1.0),
        BiasVorePenalty(alpha=0.25),
    ):
        with pytest.raises(MissingCapability, match="scalar_target"):
            check_extensions("X", [module], many, frozenset(PHASES))


def test_agents_declare_only_known_capabilities():
    for name in AGENTS:
        agent = getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)
        assert agent.capabilities <= set(ALGORITHM_CAPABILITIES), name
        assert agent.capabilities, name


@dataclass(frozen=True)
class GapReader(Extension):
    """Reads ``gap``, which no agent of Ajax passes, and the common ``rewards``."""

    name: str = "gap-reader"
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset({"gap", "rewards"})
    }

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target + batch["gap"]


class GapAgent(LatentAgent):
    supported_extension_phases = LatentAgent.supported_extension_phases.union(
        {"on_target"}
    )
    phase_batch_keys: Mapping[str, frozenset] = {"on_target": frozenset({"gap"})}


def test_an_agent_declares_the_batch_keys_it_passes():
    """An agent written outside Ajax hands its extensions a key of its own
    by declaring it; the keys every agent passes need no declaration."""
    GapAgent(extensions=[GapReader()])
    undeclared = type("Undeclared", (GapAgent,), {"phase_batch_keys": {}})
    with pytest.raises(NotWired, match=r"puts no \['gap'\] in its on_target batch"):
        undeclared(extensions=[GapReader()])


def test_agents_accept_only_the_phases_they_fold():
    """The base accepts none; every agent the loop's, observe, action (the
    action stage's slots) and eval_action (the shared evaluation's
    choosers) but where it collects and evaluates on its own (APG's
    rollout observes the features itself),
    transform_action (a residual policy) the agents whose
    critics score executed actions, and choose_warmup_action those whose
    collection has a uniform warm-up, choose_replay_action those whose
    replay the replay family writes."""
    assert ActorCritic.supported_extension_phases == frozenset()
    assert EveryPhase().implemented_phases() == frozenset(PHASES)
    # PPO, on policy, lacks what replacing the executed action needs.
    with pytest.raises(MissingCapability, match="off_policy"):
        PPO("CartPole-v1", n_envs=1, extensions=[EveryPhase()])
    for name in AGENTS:
        agent = getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)
        phases = agent.supported_extension_phases
        own = {"observe", "action", "eval_action"}
        assert LOOP_PHASES.difference(own) <= phases, name
        own_collector = name in ("APG", "UDRL", "DreamerV3", "TDMPC2")
        for phase in own:
            folded = (name, phase) in OWN_FOLDS or not own_collector
            assert (phase in phases) == folded, (name, phase)
        assert ("transform_action" in phases) == (name in RESIDUAL), name
        # The warm-up chooser reaches the agents with a uniform warm-up, but
        # TD-MPC2, whose own collector does not run it yet.
        assert ("choose_warmup_action" in phases) == (name in WARMUP), name
        assert ("choose_replay_action" in phases) == (name in REPLAY_FAMILY), name


@dataclass(frozen=True)
class Unslotted(Extension):
    name: str = "unslotted"

    def action(self, agent_state, ext_state, obs, rng, context):
        return None


def _expert(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


@pytest.mark.parametrize("agent", [SAC, TD3])
def test_the_action_stage_runs_action_only_in_its_slots_with_an_expert(agent):
    """The action stage runs action only for an extension of a slot and
    only with an expert policy: any other is rejected at construction."""
    slotted = JSRLCurriculum()
    with pytest.raises(ValueError, match="'unslotted'"):
        agent("Pendulum-v1", expert_policy=_expert, extensions=[Unslotted()])
    with pytest.raises(ValueError, match="expert_policy"):
        agent("Pendulum-v1", extensions=[slotted])
    agent("Pendulum-v1", expert_policy=_expert, extensions=[slotted])


def test_sac_applies_a_residual_policy_only_through_its_pipeline():
    """ResidualPolicy needs the pipeline SAC runs with an expert policy and
    without the gain policy; otherwise it is rejected at construction."""
    residual = ResidualPolicy(scale=0.5)
    assert residual.implemented_phases() == {"transform_action"}
    gain_mode = {"expert_policy": _expert, "extensions": [residual, ExpertGains()]}
    for kwargs in ({"extensions": [residual]}, gain_mode):
        with pytest.raises(ValueError, match="'residual_policy'"):
            SAC("Pendulum-v1", **kwargs)
    SAC("Pendulum-v1", expert_policy=_expert, extensions=[residual])


@pytest.mark.parametrize(
    "name", sorted(set(AGENTS).difference(RESIDUAL, {"DreamerV3", "TDMPC2"}))
)
def test_an_agent_without_a_residual_transform_refuses_a_residual_policy(name):
    """An agent whose update does not score executed actions would train a
    plain policy and report it as residual, so it refuses one, as lacking
    continuous actions (discrete-action agents), an expert, or not wired."""
    module = importlib.import_module(f"ajax.agents.{name}.{name}")
    agent = getattr(module, name)
    env = "CartPole-v1" if name in ("DQN", "PQN", "UDRL") else "Pendulum-v1"
    with pytest.raises(ValueError, match=r"'residual_policy'"):
        agent(env, extensions=[ResidualPolicy()])


def test_online_bc_needs_the_agents_expert_policy():
    """OnlineBC clones the actions of the agent's own expert_policy: without
    one it would add nothing all run, so SAC refuses it at construction,
    whatever expert MCPretrain holds."""
    stack = [MCPretrain(expert_policy=_expert), OnlineBC()]
    with pytest.raises(MissingCapability, match="agent_expert"):
        SAC("Pendulum-v1", extensions=stack)
    SAC("Pendulum-v1", expert_policy=_expert, extensions=stack)


def test_the_phase_check_applies_to_any_stack():
    """The check ``ActorCritic`` runs at construction, also used by agents
    that are not ``ActorCritic`` (the offline multi-task TD-MPC2)."""
    supported = frozenset({"post_update", "eval_metrics"})
    has = {"q_critic", "critic"}
    check_extensions("Offline", ExtensionStack([Instrument()]), has, supported)
    check_extensions("Offline", [], has, supported)
    with pytest.raises(NotWired) as error:
        check_extensions("Offline", [Instrument(), TargetTweak()], has, supported)
    assert "Offline has what" in str(error.value)
    assert "'target-tweak'" in str(error.value) and "on_target" in str(error.value)


#: The extensions reading the frozen expert critic phi*, which MCPretrain fits.
PHI_STAR_READERS = {
    "OnlineBC": OnlineBC(),
    "CriticBlend": CriticBlend(expert_policy=_expert),
    "MCVarianceCorrection": MCVarianceCorrection(threshold=1.0),
    "ValueBox": ValueBox(),
    "PhiRefresh": PhiRefresh(expert_policy=_expert),
}


@pytest.mark.parametrize("name", list(PHI_STAR_READERS))
def test_a_reader_of_phi_star_needs_mc_pretrain(name: str) -> None:
    """Refused in a stack without MCPretrain (it would add nothing all run,
    or fail mid-trace), accepted with one."""
    reader = PHI_STAR_READERS[name]
    kwargs: dict[str, Any] = {
        "context": SAC.extension_context,
        "buffer_fields": SAC.supported_buffer_fields,
        "batch_keys": SAC.phase_batch_keys,
        "agent_expert": _expert,
        "takes_expert_policy": True,
    }
    has = run_capabilities(SAC.capabilities, continuous=True, has_expert=True)
    with pytest.raises(MissingCapability, match=f"^SAC cannot run {name}.*phi_star"):
        check_extensions("SAC", [reader], has, frozenset(PHASES), **kwargs)
    stack = [MCPretrain(expert_policy=_expert), reader]
    check_extensions("SAC", stack, has, frozenset(PHASES), **kwargs)


def test_an_agent_refuses_a_reader_of_phi_star_without_mc_pretrain() -> None:
    with pytest.raises(MissingCapability, match="add MCPretrain"):
        SAC("Pendulum-v1", expert_policy=_expert, extensions=[OnlineBC()])


class _Reached(Exception):
    """Raised by :class:`ContextSpy` with the agent context it was bound to
    and the context fields its pretraining got."""


@dataclass(frozen=True)
class ContextSpy(Extension):
    """Stops a run at its pretraining, reporting both contexts."""

    bound: frozenset = frozenset()
    name: str = "context-spy"

    def bind_to_agent(self, **agent_context):
        return dataclasses.replace(self, bound=frozenset(agent_context))

    def pretrain(self, agent_state, ext_state, context):
        fields = ("env_args", "update", "buffer")
        raise _Reached(
            self.bound, {f for f in fields if getattr(context, f) is not None}
        )

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {}


@pytest.mark.parametrize("name", AGENTS)
def test_each_agent_declares_the_context_it_gives(name: str) -> None:
    """``extension_context`` is the agent context make_train binds its
    extensions to, ``pretrain_context`` the fields its loop fills for their
    pretraining: a module needing another is refused."""
    agent_cls = getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)
    env = "CartPole-v1" if name in ("DQN", "PQN", "UDRL") else "Pendulum-v1"
    agent = agent_cls(env, extensions=[ContextSpy()])
    with pytest.raises(_Reached) as reached:
        agent.train(seed=[0], n_timesteps=8192)
    bound, filled = reached.value.args
    assert bound == agent_cls.extension_context
    assert filled == agent_cls.pretrain_context
