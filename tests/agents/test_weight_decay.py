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

from ajax.agents.APG.APG import APG
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY
from tests.agents.TDMPC2.test_TDMPC2MultiTask import make_dataset
from tests.agents.test_agent_contract import CASES

LR, WD = 1e-2, 0.5
#: Network train states an agent may hold.
NETWORKS = ("actor_state", "critic_state")
#: The keywords a test passes and the decay they set: the default, then WD.
DECAYS = pytest.mark.parametrize(
    "given, wd", [({}, 0.0), ({"weight_decay": WD}, WD)], ids=["default", "wd"]
)


def initial_state(agent: Any) -> Any:
    """The fresh state of seed 0, without its seed axis."""
    state, _ = agent.train(seed=0, n_timesteps=0)
    return jax.tree.map(lambda x: x[0], state)


def zero_gradient_step(train_state: TrainState, steps: int = 1) -> Any:
    """The parameters after ``steps`` steps of the state's own optimizer,
    from fresh moments (a zero-step run may already have updated, as an
    on-policy agent's first rollout does), on a zero gradient."""
    state = train_state.replace(opt_state=train_state.tx.init(train_state.params))
    zeros = jax.tree.map(jnp.zeros_like, train_state.params)
    for _ in range(steps):
        state = state.apply_gradients(grads=zeros)
    return state.params


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
@DECAYS
def test_a_step_shrinks_the_networks_by_one_minus_lr_times_wd(
    name: str, given: dict[str, Any], wd: float
) -> None:
    state = initial_state(build(name, **given))
    networks = [getattr(state, k) for k in NETWORKS]
    networks = [n for n in networks if isinstance(n, TrainState)]
    assert networks, name
    for network in networks:
        after = zero_gradient_step(network)
        assert_scaled(network.params, after, 1 - LR * wd)
    if CASES[name].alpha:
        np.testing.assert_array_equal(
            zero_gradient_step(state.alpha)["log_alpha"],
            state.alpha.params["log_alpha"],
        )


@pytest.mark.parametrize("multitask", [False, True])
@DECAYS
def test_tdmpc2_decays_each_group_at_its_own_learning_rate(
    given: dict[str, Any], wd: float, multitask: bool
) -> None:
    """TD-MPC2's world-model encoder learns at ``lr * enc_lr_scale``, so its
    decay is multiplied by that rate; the rest of the world model (with the
    multi-task embedding table) and the policy decay at ``lr``."""
    enc_lr_scale = 0.5
    kwargs = {"learning_rate": LR, "enc_lr_scale": enc_lr_scale, **given}
    agent = (
        TDMPC2MultiTask(make_dataset(), **TDMPC2_TINY, task_dim=4, **kwargs)
        if multitask
        else TDMPC2("Pendulum-v1", **TDMPC2_TINY, **kwargs)
    )
    state = initial_state(agent)
    world_model = state.world_model_state
    after = zero_gradient_step(world_model)
    assert "encoder" in world_model.params
    for group, params in world_model.params.items():
        rate = LR * enc_lr_scale if group == "encoder" else LR
        assert_scaled(params, after[group], 1 - rate * wd)
    policy = state.actor_state
    assert_scaled(policy.params, zero_gradient_step(policy), 1 - LR * wd)


@pytest.mark.parametrize("warmup", [0, 2])
@DECAYS
def test_dreamerv3_decays_at_the_warmed_up_learning_rate(
    given: dict[str, Any], wd: float, warmup: int
) -> None:
    """DreamerV3's LaProp adds the decay before its warmed-up learning rate,
    as AdamW does before Adam's: update ``k`` decays the world model, the
    actor and the critic at ``lr * min(k / warmup, 1)`` (``lr`` with the
    warmup off), so a warmup's first update decays nothing."""
    tiny = DREAMER_TINY | {"learning_rate": LR, "warmup": warmup} | given
    state = initial_state(DreamerV3("CartPole-v1", n_envs=2, **tiny))
    rates = [LR * min(k / warmup, 1) if warmup else LR for k in range(2)]
    for name in ("world_model_state", "actor_state", "critic_state"):
        network, factor = getattr(state, name), 1.0
        for steps, rate in enumerate(rates, start=1):
            factor *= 1 - rate * wd
            after = zero_gradient_step(network, steps)
            assert_scaled(network.params, after, factor)


@DECAYS
def test_apg_decays_its_controller(given: dict[str, Any], wd: float) -> None:
    """APG, which builds its own optimizer config, decays its controller
    (its one network) the same way."""
    agent = APG("Pendulum-v1", n_envs=2, horizon=4, learning_rate=LR, **given)
    controller = initial_state(agent).actor_state
    after = zero_gradient_step(controller)
    assert_scaled(controller.params, after, 1 - LR * wd)
