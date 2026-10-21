"""The on-policy agents' targets read their own rollout, on two envs of the
identity task (tests/environments/identity_env.py), whose observation
[t, c] says which step of which episode a row is: PPO's GAE in the time
geometry (each minibatch's, with that step's critic) and PQN's Q(lambda)
targets. Each step's target uses that step's flags, reward and next state,
recomputed here with the agent's own functions from the rows it hands its
``on_target`` extensions."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax import PPO, PQN
from ajax.agents.DQN.networks import predict_q
from ajax.agents.PPO.utils import _compute_gae
from ajax.agents.PQN.utils import compute_q_lambda_targets
from ajax.extensions.base import Extension
from ajax.networks.networks import predict_value
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

    gae_lambda: float = 0.95
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

    q_lambda: float = 0.65
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


def test_ppo_minibatch_gae_reads_each_steps_own_flags_and_next_state() -> None:
    """In the time geometry (unroll_length) each minibatch computes its GAE:
    on its rows' own termination flags and on V of their next observations."""
    SEEN.clear()
    env = IdentityEnv(keep_raw=True)
    kw = {"n_steps": 16, "unroll_length": 8, "num_minibatches": 2, "n_epochs": 1}
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
    flags and the max over the next observation's Q-values."""
    SEEN.clear()
    env = IdentityEnv(discrete=True, keep_raw=True)
    kw = {"n_steps": 16, "num_minibatches": 2, "n_epochs": 1, "reward_scale": 2.0}
    agent = PQN(
        env, env_params=env.default_params, n_envs=2, extensions=[_PQNTargets()], **kw
    )
    agent.train(seed=0, n_timesteps=64)
    jax.effects_barrier()
    assert SEEN
    _check_flags(SEEN)
    for rows in SEEN:
        np.testing.assert_allclose(rows["target"], rows["want"], rtol=1e-5, atol=1e-6)
