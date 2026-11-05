"""The action stage (ajax.environments.action_stage): an action
extension's slot is checked, the stage leaves a pipeline without one as it
is (its collection traces the same program), and on a pipeline's result it
substitutes the action outside the warm-up, flags where the expert took
the step and stores the substituted action in the pipeline's own layout."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.loop import TrainLoop
from ajax.controller import ActingState
from ajax.environments import interaction
from ajax.environments.action_stage import (
    apply_action_extensions,
    as_env_column,
    check_action_slots,
)
from ajax.environments.interaction import ActionPipelineResult, collect_experience
from ajax.extensions.base import Extension
from ajax.extensions.cheq import CHEQ
from ajax.extensions.expert import ExpertWarmup, ResidualPolicy
from tests.agents.test_agent_contract import CASES

N_ENVS = 2


@dataclass(frozen=True)
class HandOver(Extension):
    """Hands the expert the step in env 0, before the warm-up's choice."""

    name: str = "hand_over"
    action_slot: ClassVar[str] = "pre_warmup"

    def action(self, agent_state, ext_state, obs, rng, context):
        mask = jnp.arange(N_ENVS) == 0
        obs["expert_mask"] = mask
        column = mask.reshape((-1,) + (1,) * (obs["expert_action"].ndim - 1))
        return jnp.where(column, obs["expert_action"], obs["post_warmup_action"])


@dataclass(frozen=True)
class Unslotted(Extension):
    name: str = "unslotted"

    def action(self, agent_state, ext_state, obs, rng, context):
        return None


def _discrete_pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, key):
    """A discrete pipeline storing its own (n_envs, 1) column and no flags."""
    chosen = jnp.full((N_ENVS, 1), 2, jnp.int32)
    return ActionPipelineResult(
        env_action=chosen[:, 0],
        policy_action=chosen,
        buffer_action=chosen,
    )


def _stage_result(uniform: bool) -> ActionPipelineResult:
    staged = apply_action_extensions(_discrete_pipeline, (HandOver(),), N_ENVS, 10)
    acting = ActingState(
        obs=jnp.zeros((N_ENVS, 3)),
        done=jnp.zeros(N_ENVS, bool),
        timestep=jnp.asarray(0),
        expert_action=jnp.full((N_ENVS, 1), 1, jnp.int32),
    )
    agent_state = SimpleNamespace(ext_state=((),))
    key = jax.random.PRNGKey(0)
    return staged(agent_state, acting, None, key, jnp.asarray(uniform), key, key)


@dataclass(frozen=True)
class OwnExpertHandOver(Extension):
    """Hands env 0 its own expert's action, index 3, read off the raw obs."""

    name: str = "own_expert_hand_over"
    action_slot: ClassVar[str] = "pre_warmup"

    def action(self, agent_state, ext_state, obs, rng, context):
        assert obs["expert_action"] is None  # no agent expert
        mask = jnp.arange(N_ENVS) == 0
        obs["expert_mask"] = mask
        own = obs["raw_obs"][:, 0].astype(jnp.int32)
        return jnp.where(mask, own, obs["post_warmup_action"])


def test_without_the_agents_expert_the_stage_runs_an_extensions_own():
    """No agent expert: the stage still runs an extension acting with an
    expert of its own (IBRL); with none in a slot the pipeline stays None."""
    stage = interaction.expert_collect_kwargs(
        None, None, False, N_ENVS, extensions=(ExpertWarmup(fraction=0.5),)
    )
    assert stage == {"action_pipeline": None}
    staged = interaction.expert_collect_kwargs(
        None, _discrete_pipeline, False, N_ENVS, extensions=(OwnExpertHandOver(),)
    )["action_pipeline"]
    acting = ActingState(jnp.zeros((N_ENVS, 3)), jnp.zeros(N_ENVS, bool), 0)
    key, raw_obs = jax.random.PRNGKey(0), jnp.full((N_ENVS, 1), 3.0)
    agent_state = SimpleNamespace(ext_state=((),))
    result = staged(agent_state, acting, raw_obs, key, jnp.asarray(False), key, key)
    np.testing.assert_array_equal(result.buffer_action, [[3], [2]])
    np.testing.assert_array_equal(result.is_expert_flag, [[1.0], [0.0]])


def test_an_action_extension_without_a_slot_is_refused():
    check_action_slots((HandOver(), ExpertWarmup(fraction=0.5)))
    with pytest.raises(ValueError, match="unslotted"):
        check_action_slots((Unslotted(),))


def test_without_a_slotted_extension_the_pipeline_is_left_as_it_is():
    stack = (ExpertWarmup(fraction=0.5), ResidualPolicy())
    assert apply_action_extensions(_discrete_pipeline, stack, N_ENVS, 10) is (
        _discrete_pipeline
    )


def test_the_stage_substitutes_flags_and_stores_in_the_pipelines_layout():
    """Outside the warm-up env 0 executes the expert's index; the replay
    stores it in the pipeline's (n_envs, 1) int column, and flags it (the
    pipeline flagged nothing: zeros). In the warm-up the pipeline's action
    and flags stand."""
    result = _stage_result(uniform=False)
    np.testing.assert_array_equal(result.env_action, [1, 2])
    np.testing.assert_array_equal(result.buffer_action, [[1], [2]])
    assert result.buffer_action.dtype == jnp.int32
    np.testing.assert_array_equal(result.is_expert_flag, [[1.0], [0.0]])
    warm = _stage_result(uniform=True)
    np.testing.assert_array_equal(warm.env_action, [2, 2])
    np.testing.assert_array_equal(warm.is_expert_flag, [[0.0], [0.0]])


def _residual_pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, key):
    """A pipeline storing its choice (0.7) apart from the executed action
    (0.5), and the choice that executes the expert's action (0)."""
    return ActionPipelineResult(
        env_action=jnp.full((N_ENVS, 1), 0.5),
        policy_action=jnp.full((N_ENVS, 1), 0.7),
        policy_action_for_expert=jnp.zeros((N_ENVS, 1)),
    )


def test_the_replay_keeps_the_choice_executing_the_expert_where_it_was_handed():
    """Outside the warm-up the replay's choice column keeps 0, the choice
    executing the expert's action, in env 0, which the stage handed the
    expert, and the choice 0.7 in env 1; the choice itself, which
    StorePolicyAction and the transition read, stays 0.7. In the warm-up
    the choice stands. A pipeline giving no such choice stores none."""
    staged = apply_action_extensions(_residual_pipeline, (HandOver(),), N_ENVS, 10)
    acting = ActingState(
        obs=jnp.zeros((N_ENVS, 3)),
        done=jnp.zeros(N_ENVS, bool),
        timestep=jnp.asarray(0),
        expert_action=jnp.full((N_ENVS, 1), -0.2),
    )
    agent_state, key = SimpleNamespace(ext_state=((),)), jax.random.PRNGKey(0)
    for uniform, kept in ((False, [[0.0], [0.7]]), (True, [[0.7], [0.7]])):
        result = staged(agent_state, acting, None, key, jnp.asarray(uniform), key, key)
        np.testing.assert_allclose(result.stored_policy_action, kept)
        np.testing.assert_allclose(result.policy_action, 0.7)
    assert _stage_result(uniform=False).stored_policy_action is None


def _stage_rows(extension: Extension, uniform: bool) -> ActionPipelineResult:
    """The stage's result for the chosen action 0.5 (a warm-up's uniform draw
    where uniform), the policy's sample -0.5, the expert's action 0.1 and the
    observation's last column 0.25."""

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, key):
        return ActionPipelineResult(
            env_action=jnp.full((N_ENVS, 1), 0.5),
            policy_action=jnp.full((N_ENVS, 1), -0.5),
        )

    staged = apply_action_extensions(pipeline, (extension,), N_ENVS, 10)
    acting = ActingState(
        obs=jnp.full((N_ENVS, 3), 0.25),
        done=jnp.zeros(N_ENVS, bool),
        timestep=jnp.asarray(0),
        expert_action=jnp.full((N_ENVS, 1), 0.1),
    )
    key = jax.random.PRNGKey(0)
    state = SimpleNamespace(ext_state=((),))
    return staged(state, acting, None, key, jnp.asarray(uniform), key, key)


@pytest.mark.parametrize("uniform", [True, False])
def test_a_cheq_row_stores_the_chosen_action_and_executes_the_blend(uniform) -> None:
    """Algorithm 2 executes (1 - lambda) a_prior + lambda a_RL and stores a_RL,
    which appendix A.1's warm-up draws uniformly. By hand at lambda 0.25:
    0.75 * 0.1 + 0.25 * 0.5 = 0.2, stored 0.5, not the policy's -0.5."""
    result = _stage_rows(CHEQ(weight_column=2), uniform)
    np.testing.assert_allclose(result.env_action, 0.2, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result.buffer_action), 0.5, rtol=1e-6)


@dataclass(frozen=True)
class Override(Extension):
    """Overrides the action after the warm-up, writing no replay_action, as
    ValueBox does."""

    name: str = "override"
    action_slot: ClassVar[str] = "post_warmup"

    def action(self, agent_state, ext_state, obs, rng, context):
        return obs["expert_action"]


def test_an_override_writing_no_replay_action_stores_as_it_did() -> None:
    """Only a chooser writing replay_action changes the row: a stage without
    one leaves buffer_action to collection, which stores the executed action,
    the expert's here. A default of the policy's action would store -0.5."""
    result = _stage_rows(Override(), True)
    np.testing.assert_allclose(result.env_action, 0.1, rtol=1e-6)
    assert result.buffer_action is None


def test_a_mask_for_every_env_becomes_one_column():
    """EDGE's fixed gate draws one decision for every env."""
    np.testing.assert_array_equal(
        as_env_column(jnp.asarray(True), N_ENVS), [[1.0], [1.0]]
    )
    np.testing.assert_array_equal(
        as_env_column(jnp.asarray([True, False]), N_ENVS), [[1.0], [0.0]]
    )


def _push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


#: The agents on the shared off-policy collector that take an expert, each
#: running its own pipeline kind (the policy's, an exploration's, DAgger's,
#: IQL's residual after the choice).
OFF_POLICY = ("SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC", "AVG", "IQL", "DAgger")


class _Built(Exception):
    pass


def _collection_program(name: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The jaxpr text of one collection step of the agent with an expert
    (and a residual policy, but on DAgger, which would learn to cancel it),
    as its family's loop collects."""
    case = CASES[name]
    kwargs = dict(case.kwargs)
    kwargs.update({"expert_policy": _push})
    if name != "DAgger":
        kwargs.update({"extensions": (ResidualPolicy(),)})
    agent = case.cls(env_id="Pendulum-v1", n_envs=N_ENVS, **kwargs)
    built = []

    def spy(loop: Any, init: Any, *args: Any, **options: Any) -> Any:
        recurrent = options.get("recurrent", False)
        collect = loop.collect_kwargs(recurrent, **options["collect_kwargs"])
        built.append((init, collect))
        raise _Built

    monkeypatch.setattr(TrainLoop, "off_policy", spy)
    with pytest.raises(_Built):
        agent.train(seed=0, n_timesteps=8)
    init, collect = built[0]
    state = init(jax.random.PRNGKey(0))

    def step(state: Any) -> Any:
        return collect_experience(state, None, uniform=jnp.asarray(False), **collect)

    return str(jax.make_jaxpr(step)(state))


@pytest.mark.parametrize("name", OFF_POLICY)
def test_collection_traces_the_same_program_with_and_without_the_stage(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner 10-17: wiring a stage no extension acts in leaves each agent's
    program as it was, to the jaxpr text."""
    with_stage = _collection_program(name, monkeypatch)
    monkeypatch.setattr(
        interaction, "apply_action_extensions", lambda pipeline, *_: pipeline
    )
    assert _collection_program(name, monkeypatch) == with_stage
