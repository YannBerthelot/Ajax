"""evaluate_and_log: the agents on the shared loop evaluate with a logging
config (with or without a backend) and do not even trace the evaluation
without one."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import LoggingConfig


def test_without_logging_nothing_is_evaluated_nor_traced() -> None:
    """No environment (``None``) and no log function to call: neither is
    reached, and there are no metrics."""
    state = {"n_logs": jnp.int32(3)}
    args = (state, None, 0, "gymnax", None, 1, False, False, None, None, 100)
    out, metrics = evaluate_and_log(*args)  # type: ignore[arg-type]
    assert out == state and metrics is None


_CONTINUOUS = {"env_id": "Pendulum-v1", "n_envs": 2}
_NETS = {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")}
_REPLAY = {"learning_starts": 8, "batch_size": 8, "buffer_size": 64}
_ROLLOUT = {"n_steps": 8, "batch_size": 8}
_DISCRETE = {"env_id": "CartPole-v1", "n_envs": 2, "architecture": ("8", "relu")}
AGENTS: dict[str, tuple[type, dict[str, Any]]] = {
    "SAC": (SAC, {**_CONTINUOUS, **_NETS, **_REPLAY}),
    "ASAC": (ASAC, {**_CONTINUOUS, **_NETS, **_REPLAY}),
    "REDQ": (REDQ, {**_CONTINUOUS, **_NETS, **_REPLAY}),
    "TD3": (TD3, {**_CONTINUOUS, **_NETS, **_REPLAY}),
    "AVG": (AVG, {**_CONTINUOUS, **_NETS, "learning_starts": 8}),
    "APO": (APO, {**_CONTINUOUS, **_NETS, **_ROLLOUT}),
    "PPO": (PPO, {**_CONTINUOUS, **_NETS, **_ROLLOUT}),
    "DQN": (DQN, {**_DISCRETE, **_REPLAY}),
    "PQN": (PQN, {**_DISCRETE, "n_steps": 8, "num_minibatches": 2}),
}
EVALUATE = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
SEEDS, BUDGET = 2, 64


@pytest.mark.parametrize("name", AGENTS)
def test_a_logged_run_traces_the_evaluation(name: str) -> None:
    """With a logging config and no backend, ``train`` evaluates every
    ``log_frequency`` steps and returns the evaluations: traced only
    (``jax.eval_shape``), the rows ``ajax.agents.base.ActorCritic.train``
    would return."""
    cls, kwargs = AGENTS[name]
    agent = cls(**kwargs)
    train = agent.get_make_train()(
        env_args=agent.env_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        network_args=agent.network_args,
        agent_config=agent.agent_config,
        total_timesteps=BUDGET,
        num_episode_test=1,
        run_ids=[],
        logging_config=EVALUATE,
    )
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(SEEDS))
    _, rows = jax.eval_shape(jax.vmap(train), keys, jnp.arange(SEEDS))
    assert rows["Eval/episodic mean reward"].shape == (SEEDS, BUDGET // 32)
    assert rows["timestep"].dtype == jnp.int32
