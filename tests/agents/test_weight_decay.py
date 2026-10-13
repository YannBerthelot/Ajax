"""Every agent's ``weight_decay`` is optax.adamw's decoupled decay on its
networks, multiplied by the learning rate: on a zero gradient, one step of a
network's optimizer multiplies each of its parameters by ``1 - lr *
weight_decay`` (downstream projects read ``1 / (lr * weight_decay)`` as the
decay's time constant). A scalar dual variable (the temperature) keeps
plain Adam. The default, 0, is plain Adam: a zero gradient moves nothing
(the agent pins show its runs unchanged)."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from tests.agents.test_agent_contract import CASES

LR, WD = 1e-2, 0.5
#: Network train states an agent may hold.
NETWORKS = ("actor_state", "critic_state")


def initial_state(agent: Any) -> Any:
    """The fresh state of seed 0, without its seed axis."""
    state, _ = agent.train(seed=0, n_timesteps=0)
    return jax.tree.map(lambda x: x[0], state)


def zero_gradient_step(train_state: TrainState) -> Any:
    """The parameters after one step of the state's own optimizer, from
    fresh moments (a zero-step run may already have updated, as an on-policy
    agent's first rollout does), on a zero gradient."""
    fresh = train_state.replace(opt_state=train_state.tx.init(train_state.params))
    zeros = jax.tree.map(jnp.zeros_like, train_state.params)
    return fresh.apply_gradients(grads=zeros).params


def assert_scaled(before: Any, after: Any, factor: float) -> None:
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_allclose(new, old * factor, rtol=1e-6, atol=1e-12)


def build(name: str, **overrides: Any) -> Any:
    """The contract table's tiny agent, every learning rate at ``LR`` and,
    where there is one, the temperature away from 1 (``log_alpha = 0`` would
    hide a decay)."""
    case = CASES[name]
    takes = inspect.signature(case.cls).parameters
    rates = "learning_rate actor_learning_rate critic_learning_rate alpha_learning_rate"
    kwargs: dict[str, Any] = {k: LR for k in rates.split() if k in takes}
    if "alpha_init" in takes:
        kwargs["alpha_init"] = 0.5
    return case.cls(env_id=case.envs[0], **case.kwargs, **kwargs, **overrides)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("weight_decay", [None, WD])  # None: the default
def test_a_step_shrinks_the_networks_by_one_minus_lr_times_wd(
    name: str, weight_decay: float | None
) -> None:
    given = {} if weight_decay is None else {"weight_decay": weight_decay}
    state = initial_state(build(name, **given))
    networks = [getattr(state, k) for k in NETWORKS]
    networks = [n for n in networks if isinstance(n, TrainState)]
    assert networks, name
    for network in networks:
        after = zero_gradient_step(network)
        assert_scaled(network.params, after, 1 - LR * (weight_decay or 0.0))
    if CASES[name].alpha:
        np.testing.assert_array_equal(
            zero_gradient_step(state.alpha)["log_alpha"],
            state.alpha.params["log_alpha"],
        )
