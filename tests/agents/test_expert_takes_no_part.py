"""An expert given to an agent acts only where an extension asks it to.

Contract (ExpertGuided): ``expert_policy`` feeds the features, a residual
policy and the extensions reading it, and each evaluation logs its return;
``eval_expert_policy`` "takes no other part in the run". The realistic bug, an
old form of SAC's: an expert given for logging silently adds a prefill or
warm-up steps, so a control arm receives expert data and measures another
method.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC

KWARGS: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "learning_starts": 16,
    "batch_size": 8,
    "buffer_size": 256,
    "policy_update_start": 0,
    "alpha_update_start": 0,
}
STEPS = 64


def _expert(obs: jax.Array) -> jax.Array:
    return jnp.ones(obs.shape[:-1] + (1,))


def _train(**kwargs: Any) -> Any:
    agent = SAC("Pendulum-v1", n_envs=2, **KWARGS, **kwargs)
    return jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=STEPS)[0])


@pytest.mark.parametrize("given", ["expert_policy", "eval_expert_policy"])
def test_an_expert_leaves_training_as_it_was(given: str) -> None:
    """The same seed trains the same networks on the same rows."""
    plain, guided = _train(), _train(**{given: _expert})
    for tree in ("actor_state", "critic_state"):
        jax.tree.map(
            np.testing.assert_array_equal,
            getattr(guided, tree).params,
            getattr(plain, tree).params,
        )
    plain_replay = plain.collector_state.buffer_state
    guided_replay = guided.collector_state.buffer_state
    assert int(guided_replay.current_index) == STEPS // 2  # no prefill rows
    np.testing.assert_array_equal(
        guided_replay.experience["action"], plain_replay.experience["action"]
    )


def test_no_row_of_a_run_without_an_expert_extension_is_the_experts() -> None:
    rows = _train(expert_policy=_expert).collector_state.buffer_state.experience
    assert np.asarray(rows["is_expert"]).max() == 0.0
