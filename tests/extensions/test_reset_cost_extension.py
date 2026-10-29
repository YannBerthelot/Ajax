"""ResetCost (ajax.extensions.reset_cost): Hisaki & Ono's automatic reset cost (ICML 2024, Section
3.3) for any discounted one-step agent with a replay, as their code gives SAC and ARO-DDPG.

Contract (the extension's docstring and the paper's Eqs. 16-24):
- on_target charges a terminated row reward_scale * c and no other, then steps the reset critic,
  its rate, the cost (down when the rate is below eps) and the reset critic's target.
- With it, the agent's target goes on through terminations: the on_target batch's dones are all 0
  while its terminated flags are not.
- It refuses the agents without a replay or a next-action target, and the average-reward agents,
  which price terminations themselves; it runs on SAC, TD3, REDQ, TQC, CrossQ and DQN.
- It refuses a stack reading what belongs to the episode a termination ended (an expert's next
  observation and action, an episode's return to go).
- A discrete agent's actions reach its reset critic one-hot.
Expected values are worked by hand below.
"""

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APO, ASAC, AVG, DQN, IQL, PPO, REDQ, RVISAC, SAC, TD3, TQC, CrossQ
from ajax.agents.reset_cost import create_reset_cost_state
from ajax.extensions.base import Extension, ExtensionStack
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.reset_cost import ResetCost
from tests.probing import envs

NET = ("16", "relu", "16", "relu")
SEEN: list[dict] = []


def chain(state, action, key):
    """Reward 1, a termination every third step."""
    del action, key
    return state.replace(s=(state.s + 1) % 3), 1.0, state.s == 2


ENDS = envs.Spec("ends", chain, obs_box=(0.0, 2.0), box=(-1.0, 1.0, 1))


@dataclass(frozen=True)
class CutSpy(Extension):
    name: str = "cut_spy"

    def on_target(self, agent_state, ext_state, batch, target, context):
        seen = {"dones": batch["dones"], "terminated": batch["terminated"]}
        jax.debug.callback(lambda s: SEEN.append(jax.tree.map(np.asarray, s)), seen)
        return target


@dataclass(frozen=True)
class ReadsTheEndedEpisode(Extension):
    name: str = "reads_the_ended_episode"
    batch_keys: ClassVar = {"on_target": frozenset({"raw_next_observations"})}

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target


def test_on_target_charges_terminated_rows_then_steps_the_cost_down_below_eps():
    # target [1, 1] - 2 * 0.5 * [1, 0] = [0, 1]; the reset critic reads 0 everywhere, so the rate
    # stays 0 < eps and Adam's first step moves the cost down by about its learning rate, 3e-4.
    extension = dataclasses.replace(
        ResetCost(), env_args=SimpleNamespace(continuous=True)
    )
    state = create_reset_cost_state(jax.random.PRNGKey(0), 3, ("8", "relu"))
    zeros = jax.tree.map(jnp.zeros_like, state.critic.params)
    cost = state.cost.replace(params=state.cost.params.copy({"cost": jnp.asarray(0.5)}))
    state = state.replace(
        critic=state.critic.replace(params=zeros), target_params=zeros, cost=cost
    )
    batch = {
        "observations": jnp.zeros((2, 2)),
        "actions": jnp.zeros((2, 1)),
        "next_observations": jnp.zeros((2, 2)),
        "next_actions": jnp.zeros((2, 1)),
        "terminated": jnp.array([[1.0], [0.0]]),
        "reward_scale": 2.0,
    }
    holder = SimpleNamespace(ext_state=(state,))
    target, metrics, (after,) = ExtensionStack((extension,)).apply_on_target(
        holder, batch, jnp.ones((2, 1)), 0, jax.random.PRNGKey(1), 10
    )
    np.testing.assert_allclose(target, [[0.0], [1.0]])
    assert float(metrics["reset_cost/cost"]) == 0.5 and float(after.rate) == 0.0
    np.testing.assert_allclose(after.cost.params["cost"], 0.5 - 3e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "stack", [(CutSpy(),), (ResetCost(), CutSpy())], ids=["alone", "with"]
)
def test_with_it_the_target_goes_on_through_terminations(stack):
    SEEN.clear()
    env, params = ENDS.make()
    agent = SAC(
        env,
        env_params=params,
        actor_architecture=NET,
        critic_architecture=NET,
        batch_size=16,
        learning_starts=32,
        buffer_size=256,
        extensions=stack,
    )
    agent.train(seed=0, n_timesteps=40)  # 8 updates
    jax.effects_barrier()
    terminated = np.concatenate([s["terminated"] for s in SEEN])
    dones = np.concatenate([s["dones"] for s in SEEN])
    assert terminated.any()
    if len(stack) == 1:
        np.testing.assert_array_equal(dones, terminated)
    else:
        assert not dones.any()


@pytest.mark.parametrize(
    "build",
    [
        lambda e: ASAC("Pendulum-v1", extensions=e),
        lambda e: RVISAC("Pendulum-v1", extensions=e),
        lambda e: PPO("Pendulum-v1", extensions=e),
        lambda e: APO("Pendulum-v1", extensions=e),
        lambda e: AVG("Pendulum-v1", extensions=e),
        lambda e: IQL("Pendulum-v1", extensions=e),
    ],
    ids=["ASAC", "RVISAC", "PPO", "APO", "AVG", "IQL"],
)
def test_it_refuses_agents_without_a_replay_a_next_action_or_a_discount(build):
    with pytest.raises(MissingCapability):
        build((ResetCost(),))


def test_it_runs_on_the_discounted_replay_agents():
    for cls in (SAC, TD3, REDQ, TQC, CrossQ):
        cls("Pendulum-v1", extensions=(ResetCost(),)).make_context(16)
    DQN("CartPole-v1", extensions=(ResetCost(),)).make_context(16)


def test_it_refuses_a_stack_reading_the_episode_a_termination_ended():
    with pytest.raises(ValueError, match="episode that ended"):
        SAC(
            "Pendulum-v1", extensions=(ResetCost(), ReadsTheEndedEpisode())
        ).make_context(16)


def test_a_discrete_agent_learns_its_reset_cost_on_one_hot_actions():
    agent = DQN(
        "CartPole-v1",
        architecture=NET,
        learning_starts=32,
        buffer_size=256,
        batch_size=16,
        extensions=(ResetCost(architecture=("8", "relu")),),
    )
    state, _ = agent.train(seed=0, n_timesteps=96)  # 64 updates
    reset = state.ext_state[0]
    # CartPole's 4 observation columns and its 2 actions one-hot.
    assert reset.critic.params["params"]["Dense_0"]["kernel"].shape[-2:] == (6, 8)
    assert np.isfinite(np.asarray(reset.rate)).all()
