"""The on-policy agents' targets read their own rollout, on two envs of the
identity task (tests/environments/identity_env.py), whose observation
[t, c] says which step of which episode a row is: PPO's GAE in the time
geometry (each minibatch's, with that step's critic) and in the flat one,
and PQN's Q(lambda) targets. Each step's target uses that step's flags,
reward and next state, and the agent's own gamma and lambda (none at its
default here), recomputed with the agent's functions from the rows it hands
its ``on_target`` extensions."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, PQN
from ajax.agents.PPO.PPO import _compute_gae
from ajax.agents.PQN.PQN import compute_q_lambda_targets
from ajax.extensions.base import Extension
from ajax.networks.networks import predict_value
from ajax.networks.q_networks import predict_q
from tests.environments.identity_env import IdentityEnv, ends

SEEN: list[dict] = []


def _keep(**arrays: Any) -> None:
    jax.debug.callback(
        lambda kw: SEEN.append({k: np.asarray(v) for k, v in kw.items()}), arrays
    )


@dataclass(frozen=True)
class _PPOTargets(Extension):
    """Keeps each minibatch's rows, its target and the GAE recomputed from
    the rows with the critic of that step."""

    gae_lambda: float = 0.8
    name: str = "ppo-targets"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        critic = agent_state.critic_state
        value = lambda x: predict_value(critic, critic.params, x).squeeze(0)  # noqa: E731
        values, next_values = (
            value(batch["observations"]),
            value(batch["next_observations"]),
        )
        _, want = _compute_gae(
            batch["rewards"],
            values,
            next_values,
            batch["terminated"],
            batch["truncated"],
            batch["gamma"],
            self.gae_lambda,
        )
        _keep(obs=batch["observations"], terminated=batch["terminated"],
              truncated=batch["truncated"], next_values=batch["next_values"],
              critic_next_values=next_values, target=target, want=want)  # fmt: skip
        return target


@dataclass(frozen=True)
class _PQNTargets(Extension):
    """Keeps each rollout's rows, its targets and the Q(lambda) targets
    recomputed from the rows with the Q-network of that step."""

    q_lambda: float = 0.5
    name: str = "pqn-targets"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        actor = agent_state.actor_state
        next_q = predict_q(actor, actor.params, batch["next_observations"])
        want = compute_q_lambda_targets(
            batch["rewards"] * batch["reward_scale"],
            jnp.max(next_q, axis=-1, keepdims=True),
            batch["terminated"],
            batch["truncated"],
            batch["gamma"],
            self.q_lambda,
        )
        _keep(obs=batch["observations"], terminated=batch["terminated"],
              truncated=batch["truncated"], target=target, want=want)  # fmt: skip
        return target


def _check_flags(rows: list[dict]) -> None:
    obs = np.concatenate([r["obs"].reshape(-1, 2) for r in rows])
    terminated, truncated = (np.asarray(x) for x in ends(jnp.asarray(obs)))
    assert terminated.any() and truncated.any(), "no episode end was learned from"
    for key, want in (("terminated", terminated), ("truncated", truncated)):
        got = np.concatenate([r[key].reshape(-1) for r in rows]) > 0
        np.testing.assert_array_equal(got, want, err_msg=key)


@pytest.mark.parametrize("unroll", [8, None], ids=["time", "flat"])
def test_ppo_gae_reads_each_steps_own_flags_and_next_state(unroll) -> None:
    """GAE with the agent's lambda on its rows' own termination flags and on V
    of their next observations: per minibatch in the time geometry
    (unroll_length), over the rollout in the flat one."""
    SEEN.clear()
    env = IdentityEnv(keep_raw=True)
    kw: dict[str, Any] = {
        "n_steps": 16,
        "unroll_length": unroll,
        "num_minibatches": 2,
        "n_epochs": 1,
    }
    kw |= {"gae_lambda": 0.8}
    agent = PPO(
        env, env_params=env.default_params, n_envs=2, extensions=[_PPOTargets()], **kw
    )
    agent.train(seed=0, n_timesteps=64)
    jax.effects_barrier()
    assert SEEN
    _check_flags(SEEN)
    for rows in SEEN:
        np.testing.assert_allclose(
            rows["next_values"], rows["critic_next_values"], rtol=1e-6
        )
        np.testing.assert_allclose(rows["target"], rows["want"], rtol=1e-5, atol=1e-6)


def test_pqn_targets_read_each_steps_own_reward_flags_and_next_state() -> None:
    """Q(lambda) on the scaled rewards (reward_scale 2), each step's own
    flags and the max over the next observation's Q-values, with the agent's
    gamma 0.9 and lambda 0.5."""
    SEEN.clear()
    env = IdentityEnv(discrete=True, keep_raw=True)
    kw: dict[str, Any] = {"n_steps": 16, "num_minibatches": 2, "n_epochs": 1}
    kw |= {"reward_scale": 2.0}
    kw |= {"gamma": 0.9, "q_lambda": 0.5}
    agent = PQN(
        env, env_params=env.default_params, n_envs=2, extensions=[_PQNTargets()], **kw
    )
    agent.train(seed=0, n_timesteps=64)
    jax.effects_barrier()
    assert SEEN
    _check_flags(SEEN)
    for rows in SEEN:
        np.testing.assert_allclose(rows["target"], rows["want"], rtol=1e-5, atol=1e-6)
