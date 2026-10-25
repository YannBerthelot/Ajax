"""The expert prefill writes its rows as the live collector does
(replay_row): every field the replay holds, from the expert's steps, and
under an action transform what collection stores when the expert acts."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL, SAC
from ajax.buffers.utils import get_buffer, init_buffer
from ajax.environments.interaction import replay_row
from ajax.extensions.expert import ExpertGains, ResidualPolicy
from ajax.extensions.pretrain import collect_and_store_expert_transitions
from tests.agents.test_gain_mode import OneGainTwoActions, TwoGains
from tests.environments import paid_action
from tests.environments.identity_env import LIMIT, IdentityEnv


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def prefill(agent: Any = None, n_steps: int = 4, **fields: Any) -> dict:
    """The first ``n_steps`` rows of each env of the agent's replay (by
    default 2-env SAC on Pendulum with ``push``), holding the expert fields
    and ``fields``, after the expert's prefill."""
    agent = agent or SAC("Pendulum-v1", n_envs=2, expert_policy=push)
    buffer = get_buffer(64, 8, n_envs=2)
    state = init_buffer(buffer, agent.env_args, include_expert_fields=True, **fields)
    key = jax.random.PRNGKey(0)
    state = collect_and_store_expert_transitions(
        agent.expert_policy, agent.env_args, buffer, state, key, n_steps
    )
    return {
        name: np.asarray(value)[:, :n_steps] for name, value in state.experience.items()
    }


def test_the_prefill_fills_the_policy_action_and_the_next_action():
    """The expert chose its action, and the next one is unknown."""
    rows = prefill(include_policy_action=True, include_next_action=True)
    np.testing.assert_array_equal(rows["policy_action"], rows["action"])
    assert np.isnan(rows["next_action"]).all()


@pytest.mark.parametrize("expert", [OneGainTwoActions(), TwoGains()])
def test_gain_mode_prefill_rows_store_the_anchor_gains(expert):
    """Collection stores the gains chosen; at 0, the anchor gains, the expert
    acts as itself, its env action in a_expert."""
    env_width = 2 if isinstance(expert, OneGainTwoActions) else 1
    env, params = paid_action.make(length=8, dim=env_width)
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        expert_policy=expert,
        extensions=(ExpertGains(),),
    )
    rows = prefill(agent)
    assert rows["action"].shape[-1] == len(expert.learnable_fields)
    assert (rows["action"] == 0).all()
    anchor_action, _ = expert(expert.init_state(1), paid_action.OBS[None])
    np.testing.assert_allclose(
        rows["a_expert"], np.broadcast_to(anchor_action / 2, rows["a_expert"].shape)
    )


def test_a_residual_prefill_stores_a_zero_residual_and_the_experts_action():
    """IQL fits its stored residuals: the expert's rows are residual 0, and
    the executed action is the expert's (0.25 in agent units)."""
    agent = IQL(
        "Pendulum-v1", n_envs=2, expert_policy=push, extensions=(ResidualPolicy(),)
    )
    rows = prefill(agent, include_policy_action=True)
    assert (rows["policy_action"] == 0).all()
    np.testing.assert_allclose(rows["action"], 0.25)


def test_a_terminated_prefill_row_keeps_the_next_observation_collection_keeps():
    """After a termination the replay keeps the reset observation, the next
    row's (ASAC's target continues from it), and after a time limit the
    final one, as the live collector does (bootstrap_obs)."""
    agent = SAC(IdentityEnv(), n_envs=2, expert_policy=lambda obs: obs[..., 1:])
    rows = prefill(agent, n_steps=20)
    terminated, truncated = (
        rows["terminated"][..., 0] > 0,
        rows["truncated"][..., 0] > 0,
    )
    assert terminated.any() and truncated.any()
    continuing = ~truncated[:, :-1]
    np.testing.assert_array_equal(
        rows["next_obs"][:, :-1][continuing], rows["obs"][:, 1:][continuing]
    )
    assert (rows["next_obs"][truncated][:, 0] == LIMIT).all()


def test_the_prefill_refuses_a_recurrent_actors_stored_carry():
    with pytest.raises(ValueError, match="actor_carry"):
        prefill(actor_carry_dim=4)


def test_a_row_names_the_field_it_cannot_fill():
    zeros = jnp.zeros((2, 1))
    fields = "obs action reward terminated truncated next_obs raw_obs is_expert"
    fields += " a_expert next_a_expert raw_next_obs policy_action"
    with pytest.raises(ValueError, match="'episode'"):
        replay_row({"episode": zeros}, **dict.fromkeys(fields.split(), zeros))


def test_the_old_path_of_the_prefill_warns_and_resolves():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from ajax.agents.SAC.expert import collect_and_store_expert_transitions as old
    assert old is collect_and_store_expert_transitions
    assert any("ajax.extensions.pretrain" in str(w.message) for w in caught)
