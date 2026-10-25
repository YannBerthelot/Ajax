"""The expert prefill writes its rows as the live collector does
(replay_row): every field the replay holds, from the expert's steps."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.buffers.utils import get_buffer, init_buffer
from ajax.environments.interaction import replay_row
from ajax.extensions.pretrain import collect_and_store_expert_transitions


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def prefill(n_steps: int = 4, **fields: Any) -> dict:
    """A 2-env Pendulum replay holding the expert fields and ``fields``,
    after ``n_steps`` of the expert's prefill."""
    env_args = SAC("Pendulum-v1", n_envs=2).env_args
    buffer = get_buffer(64, 8, n_envs=2)
    state = init_buffer(buffer, env_args, include_expert_fields=True, **fields)
    key = jax.random.PRNGKey(0)
    state = collect_and_store_expert_transitions(
        push, env_args, buffer, state, key, n_steps
    )
    return state.experience


def test_the_prefill_fills_the_policy_action_and_the_next_action():
    """The expert chose its action, and the next one is unknown."""
    rows = prefill(include_policy_action=True, include_next_action=True)
    written = {name: np.asarray(value)[:, :4] for name, value in rows.items()}
    np.testing.assert_array_equal(written["policy_action"], written["action"])
    assert np.isnan(written["next_action"]).all()


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
