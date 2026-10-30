"""RVI-SAC (Hisaki & Ono, ICML 2024), against the paper and the authors' settings.

Contract (the paper's Eqs. 17-24, Table 1, and yhisaki/average-reward-drl's rvi_sac.yaml):
- Its defaults are theirs: one learning rate 3e-4, batch 256, replay 1e6, 256-256 networks,
  tau 5e-3, kappa 5e-3, a 64-64 reset critic, eps 1e-3, 1e4 uniform warm-up steps, no discount,
  no gradient clip, the temperature learned from 1 towards -dim A.
- Its target is r - c 1(terminated) - xi + V'(s'), with no cut at a termination (the replay's
  next observation there is the reset one); the offset xi then follows the batch mean of V'(s') at
  kappa; the reset cost c charges terminated rows only.
- Its evaluation action is the mean of tanh at mu +- sigma.
- Every layer starts orthogonal with gain sqrt(1/3) and a zero bias, the reset critic's too.
- Without a reset cost it charges nothing and keeps no reset critic.
The exact update is checked against the authors' code in tests/conformance/test_rvisac_original.py;
expected values here are worked by hand.
"""

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import RVISAC
from ajax.agents.centering import RVIBatchMean
from ajax.agents.reset_cost import ResetCostSettings
from ajax.agents.RVISAC.RVISAC import compute_rvi_td_target
from ajax.extensions.base import Extension
from tests.conformance import kit
from tests.probing import envs

NET = ("16", "relu", "16", "relu")
SEEN: list[dict] = []


def chain(state, action, key):
    """Three steps of reward 1, the third terminating."""
    del action, key
    return state.replace(s=(state.s + 1) % 3), 1.0, state.s == 2


ENDS = envs.Spec("ends", chain, obs_box=(0.0, 2.0), box=(-1.0, 1.0, 1))


@dataclass(frozen=True)
class TargetSpy(Extension):
    """Keeps, for each update, the target's inputs and the state's offset and cost."""

    name: str = "target_spy"

    def on_target(self, agent_state, ext_state, batch, target, context):
        seen = {
            "target": target,
            "rewards": batch["rewards"],
            "next_values": batch["next_values"],
            "dones": batch["dones"],
            "terminated": batch["terminated"],
            "offset": agent_state.offset,
        }
        jax.debug.callback(lambda s: SEEN.append(jax.tree.map(np.asarray, s)), seen)
        return target


def spied_run(**kwargs) -> list[dict]:
    SEEN.clear()
    env, params = ENDS.make()
    agent = RVISAC(
        env,
        env_params=params,
        actor_architecture=NET,
        critic_architecture=NET,
        batch_size=16,
        learning_starts=32,
        buffer_size=256,
        extensions=(TargetSpy(),),
        **kwargs,
    )
    agent.train(seed=0, n_timesteps=48)  # 16 updates
    jax.effects_barrier()
    return list(SEEN)


def test_its_defaults_are_the_authors():
    agent = RVISAC("Pendulum-v1")
    assert (agent.actor_learning_rate, agent.critic_learning_rate) == (3e-4, 3e-4)
    assert agent.alpha_learning_rate == 3e-4 and agent.alpha_init == 1.0
    assert agent.target_entropy_per_dim == -1.0 and agent.max_grad_norm is None
    assert (agent.batch_size, agent.buffer_size, agent.learning_starts) == (
        256,
        10**6,
        10**4,
    )
    assert (
        agent.actor_architecture
        == agent.critic_architecture
        == ("256", "relu", "256", "relu")
    )
    assert agent.tau == 0.005 and agent.centering == RVIBatchMean(rate=0.005, gain=1.0)
    assert agent.reset_cost == ResetCostSettings(
        1e-3, 0.005, 1.0, ("64", "relu", "64", "relu")
    )
    assert "gamma" not in agent.field_names()


def test_the_target_is_the_charged_reward_minus_the_offset_plus_the_next_value():
    # 1 - 0.5 - 0.25 + 2 = 2.25; 3 - 0 - 0.25 + (-1) = 1.75
    rewards = jnp.array([[1.0], [3.0]]) - 0.5 * jnp.array([[1.0], [0.0]])
    target = compute_rvi_td_target(rewards, 0.25, jnp.array([[2.0], [-1.0]]))
    np.testing.assert_allclose(target, [[2.25], [1.75]])


def test_a_termination_does_not_cut_the_target_and_only_terminations_are_charged():
    seen = spied_run()
    terminated = np.concatenate([s["terminated"] for s in seen])
    assert terminated.any() and not terminated.all()
    for s in seen:
        assert not s["dones"].any()  # no cut anywhere
        np.testing.assert_allclose(
            s["target"],
            s["rewards"] - s["offset"] + s["next_values"],
            rtol=1e-5,
            atol=1e-5,
        )
        # Every reward is 1: a terminated row's is charged c, the others are not.
        np.testing.assert_array_equal(s["rewards"][s["terminated"] == 0], 1.0)


def test_the_offset_follows_the_batch_mean_of_the_next_values_at_kappa():
    seen = spied_run(centering=RVIBatchMean(rate=0.1))
    assert float(seen[0]["offset"]) == 0.0
    for before, after in itertools.pairwise(seen):
        moved = 0.9 * before["offset"] + 0.1 * before["next_values"].mean()
        np.testing.assert_allclose(after["offset"], moved, rtol=1e-5, atol=1e-6)


def test_without_a_reset_cost_nothing_is_charged():
    seen = spied_run(reset_cost=None)
    for s in seen:
        np.testing.assert_array_equal(s["rewards"], 1.0)


def fresh_state(**kwargs):
    env, params = ENDS.make()
    agent = RVISAC(
        env,
        env_params=params,
        actor_architecture=NET,
        critic_architecture=NET,
        **kwargs,
    )
    return agent, kit.capture_step(agent, 64).fresh(jax.random.PRNGKey(0))


def test_its_evaluation_action_is_the_mean_of_tanh_at_mu_plus_and_minus_sigma():
    _, state = fresh_state()
    actor = state.actor_state
    observations = jnp.linspace(0.0, 2.0, 5)[:, None]
    policy = actor.apply_fn(actor.params, observations)
    mu, sigma = policy.distribution.mean(), policy.distribution.stddev()
    expected = (jnp.tanh(mu + sigma) + jnp.tanh(mu - sigma)) / 2
    np.testing.assert_allclose(policy.mean(), expected, rtol=1e-6)
    assert not np.allclose(policy.mean(), jnp.tanh(mu))


@pytest.mark.parametrize(
    "path",
    [
        lambda s: s.actor_state.params["params"]["encoder"]["network"]["layers_2"],
        lambda s: jax.tree.map(
            lambda x: x[0],
            s.critic_state.params["params"]["ensemble"]["encoder"]["network"][
                "layers_2"
            ],
        ),
        lambda s: s.reset_cost.critic.params["params"]["Dense_1"],
    ],
    ids=["actor", "critic", "reset_critic"],
)
def test_every_layer_starts_orthogonal_with_gain_root_one_third_and_a_zero_bias(path):
    # A square orthogonal kernel W with gain g has W^T W = g^2 I = I / 3.
    _, state = fresh_state(reset_cost=ResetCostSettings(architecture=NET))
    layer = path(state)
    kernel = np.asarray(layer["kernel"], np.float64)
    np.testing.assert_allclose(kernel.T @ kernel, np.eye(16) / 3, atol=1e-5)
    np.testing.assert_array_equal(layer["bias"], 0.0)
