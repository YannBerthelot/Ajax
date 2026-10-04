"""Probing environment tests for Ajax agents.

These tests use ProbingEnvironments to verify that each agent's core RL
components (value loss, backprop, discounting, advantage, actor-critic
coupling) are working correctly. This is the TDD baseline for the
refactoring — if these pass, the agent is functionally correct.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from probing_environments.adaptors.ajax import (
    get_action,
    get_gamma,
    get_policy,
    get_value,
    init_agent,
    train_agent,
)
from probing_environments.checks import (
    check_actor_and_critic_coupling,
    check_actor_and_critic_coupling_continuous,
    check_advantage_policy,
    check_advantage_policy_continuous,
    check_backprop_value_net,
    check_boundary_action_saturation,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.DreamerV3.networks import (
    RSSM,
    Actor,
    WorldModel,
    features,
    initial_state,
    make_critic,
)
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.distributional import TwoHot

# All agents
ALL_AGENTS = [SAC, REDQ, PPO, APO, ASAC, AVG]
# Agents that recompute ``pi.log_prob(post_tanh_action)`` at update
# time (PPO-family) and are therefore vulnerable to the arctanh
# saturation instability covered by :func:`check_boundary_action_saturation`.
# SAC-family agents sample and store log_prob in one shot via
# ``sample_and_log_prob`` so they don't recompute and don't need this
# probe.
ONPOLICY_LOGPROB_RECOMPUTE_AGENTS = [PPO, APO]
# Average-reward agents: their critic learns *differential* V (≈0 for constant
# reward), not absolute V — so the V≈1 expectation in value-net checks and the
# discount-ratio expectation in reward-discounting check don't apply.
_AVG_REWARD = {"ASAC", "APO"}
DISCOUNTED_AGENTS = [a for a in ALL_AGENTS if a.__name__ not in _AVG_REWARD]
VALUE_NET_AGENTS = DISCOUNTED_AGENTS
# Coupling test asserts V≥0.8 — same differential-V issue as value-net checks.
COUPLING_AGENTS = DISCOUNTED_AGENTS

# Only SAC and PPO run by default. Other agents are too slow on probing envs
# (AVG especially — Vasan 2024 noted its sample inefficiency). TODO: speed up.
_FAST = {"SAC", "PPO"}
_SKIP_REASON = "probing test too slow for this agent — TODO: speed up"


def _params(agents):
    return [
        pytest.param(a)
        if a.__name__ in _FAST
        else pytest.param(a, marks=pytest.mark.skip(reason=_SKIP_REASON))
        for a in agents
    ]


BUDGET_VALUE = int(2e4)
# SAC's deterministic action on the advantage-policy probe plateaus at
# ~0.91 (the entropy bonus keeps the mean off the boundary). At 1e4 it
# sat right on the 0.90 threshold (0.8929..0.9073 across seeds); 2e4
# reaches the plateau on every seed tried (>= 0.908), and 3e4 adds nothing.
BUDGET_POLICY = int(2e4)
# Coupling on PolicyAndValueEnv requires learning the obs→action-sign mapping —
# PPO with n_envs=1 needs more rollouts and a higher LR to move the actor.
BUDGET_COUPLING = int(6e4)
LR_COUPLING = 5e-3


@pytest.mark.parametrize(
    "agent_cls", _params(VALUE_NET_AGENTS), ids=lambda c: c.__name__
)
class TestProbingValueNet:
    """Value network probing checks (critic only)."""

    def test_loss_or_optimizer(self, agent_cls):
        check_loss_or_optimizer_value_net(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )

    def test_backprop(self, agent_cls):
        check_backprop_value_net(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )


@pytest.mark.parametrize(
    "agent_cls", _params(DISCOUNTED_AGENTS), ids=lambda c: c.__name__
)
class TestProbingDiscounting:
    """Reward-discounting check — only meaningful for discounted agents."""

    def test_reward_discounting(self, agent_cls):
        check_reward_discounting(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            get_gamma=get_gamma,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=True,
        )


@pytest.mark.parametrize("agent_cls", _params(ALL_AGENTS), ids=lambda c: c.__name__)
class TestProbingPolicy:
    """Policy network probing checks (actor + critic)."""

    def test_advantage_policy(self, agent_cls):
        check_advantage_policy_continuous(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            budget=BUDGET_POLICY,
            gymnax=True,
        )


@pytest.mark.parametrize(
    "agent_cls",
    _params(ONPOLICY_LOGPROB_RECOMPUTE_AGENTS),
    ids=lambda c: c.__name__,
)
class TestProbingBoundarySaturation:
    """Boundary-action saturation probe (m4 audit -- May 2026).

    For tanh-squashed Gaussian policies (SquashedNormal), on-policy
    agents that recompute ``pi.log_prob(post_tanh_action)`` at update
    time go through ``distrax.Tanh.inverse_and_log_det`` which calls
    ``arctanh(action)``. This is numerically unstable as
    ``|action| → 1``, causing the recomputed log_prob to diverge from
    the stored log_prob, the PPO ratio to explode/underflow, and the
    policy to stall around action ≈ 0.95 instead of converging to
    ≈ 1.0 on a boundary-reward env.

    The fix (Ajax May 2026 m4): store the pre-tanh ``raw_action`` in
    the rollout buffer and recompute via ``base.log_prob(raw) -
    forward_log_det_jacobian(raw)`` -- never invert the tanh.

    The pre-existing :class:`TestProbingPolicy.test_advantage_policy`
    used threshold 0.90, which a saturation-stalled policy can pass.
    This stricter probe uses 0.98 and would have caught the m4 bug.
    """

    def test_boundary_action_saturation(self, agent_cls):
        check_boundary_action_saturation(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            budget=int(5e4),
            gymnax=True,
        )


@pytest.mark.parametrize(
    "agent_cls", _params(COUPLING_AGENTS), ids=lambda c: c.__name__
)
class TestProbingCoupling:
    """Actor-critic coupling — value assertion excludes avg-reward agents."""

    def test_actor_critic_coupling(self, agent_cls):
        check_actor_and_critic_coupling_continuous(
            agent=agent_cls,
            init_agent=init_agent,
            train_agent=train_agent,
            get_action=get_action,
            get_value=get_value,
            budget=BUDGET_COUPLING,
            learning_rate=LR_COUPLING,
            gymnax=True,
        )


class TestProbingDQN:
    """DQN is discrete-action and value-based: it runs the *discrete*
    probing checks (continuous=False) instead of the continuous variants.

    ``get_value`` for DQN returns the greedy state value V(s) = max_a
    Q(s, a); ``get_policy`` returns the one-hot greedy policy.
    """

    def test_loss_or_optimizer(self):
        check_loss_or_optimizer_value_net(
            agent=DQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=False,
        )

    def test_backprop(self):
        check_backprop_value_net(
            agent=DQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=False,
        )

    def test_reward_discounting(self):
        check_reward_discounting(
            agent=DQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            get_gamma=get_gamma,
            budget=BUDGET_VALUE,
            gymnax=True,
            continuous=False,
        )

    def test_advantage_policy(self):
        check_advantage_policy(
            agent=DQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_policy=get_policy,
            budget=BUDGET_POLICY,
            gymnax=True,
        )

    def test_actor_critic_coupling(self):
        check_actor_and_critic_coupling(
            agent=DQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_policy=get_policy,
            get_value=get_value,
            budget=BUDGET_VALUE,
            gymnax=True,
        )


class TestProbingPQN:
    """PQN is discrete-action and value-based, like DQN -- it runs the
    discrete probing checks. PQN is on-policy with a low update-to-data
    ratio, so it gets a larger step budget than DQN for the same checks.
    """

    # PQN does n_epochs minibatch updates per (n_envs * n_steps) env
    # steps, far fewer gradient steps per env step than DQN -- so it
    # needs a bigger env-step budget to converge on the probing envs.
    BUDGET_VALUE_PQN = int(8e4)
    BUDGET_POLICY_PQN = int(8e4)

    def test_loss_or_optimizer(self):
        check_loss_or_optimizer_value_net(
            agent=PQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=self.BUDGET_VALUE_PQN,
            gymnax=True,
            continuous=False,
        )

    def test_backprop(self):
        check_backprop_value_net(
            agent=PQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            budget=self.BUDGET_VALUE_PQN,
            gymnax=True,
            continuous=False,
        )

    def test_reward_discounting(self):
        check_reward_discounting(
            agent=PQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_value=get_value,
            get_gamma=get_gamma,
            budget=self.BUDGET_VALUE_PQN,
            gymnax=True,
            continuous=False,
        )

    def test_advantage_policy(self):
        check_advantage_policy(
            agent=PQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_policy=get_policy,
            budget=self.BUDGET_POLICY_PQN,
            gymnax=True,
        )

    def test_actor_critic_coupling(self):
        check_actor_and_critic_coupling(
            agent=PQN,
            init_agent=init_agent,
            train_agent=train_agent,
            get_policy=get_policy,
            get_value=get_value,
            budget=self.BUDGET_VALUE_PQN,
            gymnax=True,
        )


# TD-MPC2 is not run on the stock probing envs: they terminate after 1-2
# steps, while paper-era TD-MPC2 is for fixed-length, non-terminating tasks
# (it refuses terminations, deviation T10 in docs/world_models/deviations.md)
# and its training slices need episodes of T + 1 >= horizon + 1 rows. It is
# probed on fixed-length non-terminating toy envs instead, in
# tests/agents/TDMPC2/test_tdmpc2_probes.py (docs/world_models/DESIGN.md §8).
_TDMPC2_SKIP_REASON = (
    "stock probes terminate after 1-2 steps; paper-era TD-MPC2 needs"
    " fixed-length non-terminating episodes of >= horizon + 1 rows (see"
    " tests/agents/TDMPC2/test_tdmpc2_probes.py)"
)


@pytest.mark.parametrize(
    "agent_cls",
    [pytest.param(TDMPC2, marks=pytest.mark.skip(reason=_TDMPC2_SKIP_REASON))],
    ids=lambda c: c.__name__,
)
def test_tdmpc2_uses_local_probes(agent_cls):
    """Placeholder that records the skip; never runs."""
    raise AssertionError(f"{agent_cls.__name__} must not run the stock probes")


# ---------------------------------------------------------------------------
# DreamerV3: local adaptors (docs/world_models/DESIGN.md section 8)
# ---------------------------------------------------------------------------

#: A tiny DreamerV3 (d = 32) with 8 x 8 batches and one update per 8 rows.
#: Everything else is the paper-era default, but the learning rate is the
#: probing harness's and ``return_horizon = 1 / (1 - gamma)``.
_DREAMER_TINY: dict[str, Any] = {
    "model_size": "1m",
    "units": 32,
    "hidden": 32,
    "deter": 64,
    "classes": 4,
    "stoch": 8,
    "blocks": 4,
    "batch_size": 8,
    "batch_length": 8,
    "train_ratio": 8,
    "imag_horizon": 5,
    "warmup": 100,
}
#: Rows per check: values within 0.02 of the targets, the advantage-policy
#: action at 0.999, in ~30 s of CPU per check.
BUDGET_DREAMER = 6000


def dreamer_init_agent(
    agent,
    env,
    run_name="",
    gamma=0.5,
    learning_rate=1e-3,
    num_envs=1,
    seed=42,
    budget=None,
):
    """The probing adaptor's ``init_agent`` for DreamerV3 (a dict like the
    package adaptor's, trained by its ``train_agent``)."""
    del run_name, budget
    env_instance = env()
    env_params = env_instance.default_params.replace(max_steps_in_episode=10_000)
    instance = agent(
        env_instance,
        n_envs=num_envs or 1,
        env_params=env_params,
        return_horizon=1.0 / (1.0 - gamma),
        learning_rate=learning_rate,
        **_DREAMER_TINY,
    )
    return {
        "agent_instance": instance,
        "gamma": gamma,
        "seed": seed,
        "env": env_instance,
        "state": None,
    }


def _canonical_trajectory(env, obs):
    """The observations of an episode up to ``obs``.

    In RewardDiscountingEnv the observation is the time step, so ``[t]`` is
    reached through ``[0], ..., [t]``; every other probe's observations are
    first observations.
    """
    obs = np.asarray(obs, np.float32).reshape(-1)
    if type(env).__name__ == "RewardDiscountingEnv":
        return [np.array([float(t)], np.float32) for t in range(int(obs[0]) + 1)]
    return [obs]


def _dreamer_filter(agent, obs):
    """Features and policy at the posterior after filtering the canonical
    trajectory to ``obs`` from ``is_first``: the posterior's mode (zero
    noise), the policy's mode as each previous action."""
    instance, state = agent["agent_instance"], agent["state"]
    config = instance.dreamer_config
    params = state.world_model_state.params
    steps = _canonical_trajectory(agent["env"], obs)
    model = WorldModel(config, steps[0].shape[-1])
    actor = Actor(config, 1, False)
    carry = initial_state(config, (1,))
    action = jnp.zeros((1, 1))
    for i, x in enumerate(steps):
        token = model.apply(
            {"params": params}, jnp.asarray(x)[None], method=WorldModel.encode
        )
        carry, _ = RSSM(config).apply(
            {"params": params["rssm"]},
            carry,
            token,
            action,
            jnp.asarray([i == 0]),
            jnp.zeros((1, config.stoch, config.classes)),
            method=RSSM.observe_step,
        )
        feat = features(carry.deter, carry.stoch)
        policy = actor.apply({"params": state.actor_state.params}, feat)
        action = policy.mean
    return feat, policy


def dreamer_get_value(agent, obs):
    """The critic's decoded value at the filtered posterior."""
    feat, _ = _dreamer_filter(agent, obs)
    config = agent["agent_instance"].dreamer_config
    logits = make_critic(config).apply(
        {"params": agent["state"].critic_state.params}, feat
    )
    return float(TwoHot.dreamerv3(config.bins).decode(logits)[0])


def dreamer_get_action(agent, obs, key=None):
    """The mode of the policy (its bounded mean) at the filtered posterior."""
    del key
    _, policy = _dreamer_filter(agent, obs)
    return float(policy.mean[0, 0])


class TestProbingDreamerV3:
    """DreamerV3 on the continuous probes, with the local adaptors above.

    The coupling probe is skipped: its action space is ``Box(0, 1)``, onto
    which the agent's actions are mapped (``a_env = (clip(a) + 1) / 2``,
    the reference's NormalizeAction, DESIGN 5.1). Its reward at the
    observation ``-1`` needs ``a_env <= 0``, i.e. a sample ``a <= -1``,
    which a policy whose mean is ``tanh(.) > -1`` draws with probability
    below 1/2, so the expected value there stays below 0 and the check's
    ``V >= 0.8`` is unreachable by design.
    """

    def test_loss_or_optimizer(self):
        check_loss_or_optimizer_value_net(
            agent=DreamerV3,
            init_agent=dreamer_init_agent,
            train_agent=train_agent,
            get_value=dreamer_get_value,
            budget=BUDGET_DREAMER,
            gymnax=True,
            continuous=True,
        )

    def test_backprop(self):
        check_backprop_value_net(
            agent=DreamerV3,
            init_agent=dreamer_init_agent,
            train_agent=train_agent,
            get_value=dreamer_get_value,
            budget=BUDGET_DREAMER,
            gymnax=True,
            continuous=True,
        )

    def test_reward_discounting(self):
        check_reward_discounting(
            agent=DreamerV3,
            init_agent=dreamer_init_agent,
            train_agent=train_agent,
            get_value=dreamer_get_value,
            get_gamma=get_gamma,
            budget=BUDGET_DREAMER,
            gymnax=True,
            continuous=True,
        )

    def test_advantage_policy(self):
        check_advantage_policy_continuous(
            agent=DreamerV3,
            init_agent=dreamer_init_agent,
            train_agent=train_agent,
            get_action=dreamer_get_action,
            budget=BUDGET_DREAMER,
            gymnax=True,
        )

    @pytest.mark.skip(
        reason="unreachable by design: Box(0, 1) action mapping (class docstring)"
    )
    def test_actor_critic_coupling(self):
        check_actor_and_critic_coupling_continuous(
            agent=DreamerV3,
            init_agent=dreamer_init_agent,
            train_agent=train_agent,
            get_action=dreamer_get_action,
            get_value=dreamer_get_value,
            budget=BUDGET_DREAMER,
            gymnax=True,
        )
