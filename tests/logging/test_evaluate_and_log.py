"""evaluate_and_log: the agents on the shared loop evaluate with a logging
config (with or without a backend) and do not even trace the evaluation
without one."""

from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import controller as controller_module
from ajax import evaluate as evaluate_module
from ajax import log
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
    "REDQ": (REDQ, {**_CONTINUOUS, **_NETS, **_REPLAY, "num_critic_updates": 2}),
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
        total_timesteps=BUDGET,
        num_episode_test=1,
        run_ids=[],
        logging_config=EVALUATE,
    )
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(SEEDS))
    _, rows = jax.eval_shape(jax.vmap(train), keys, jnp.arange(SEEDS))
    assert rows["Eval/episodic mean reward"].shape == (SEEDS, BUDGET // 32)
    assert rows["timestep"].dtype == jnp.int32


#: The agents whose critics estimate an average-reward (differential) value,
#: not a discounted return: they have no value bias.
AVERAGE_REWARD = {"APO", "ASAC"}
#: The agents whose evaluations also measure the reward per step
#: (``avg_reward_mode``).
MEASURES_REWARD_RATE = {"APO"}


@pytest.mark.parametrize("name", AGENTS)
def test_an_agent_logs_the_value_bias_of_its_critic(name: str) -> None:
    """``Eval/mean bias``, the critic's estimate minus the realised
    discounted return (:func:`ajax.evaluate.evaluate`), is measured for an
    agent with a discounted critic and NaN for an average-reward one; APO's
    evaluations also measure its objective, the reward per step."""
    cls, kwargs = AGENTS[name]
    _, rows = cls(**kwargs).train(
        seed=0, n_timesteps=BUDGET, num_episode_test=1, logging_config=EVALUATE
    )
    bias = np.asarray(rows["Eval/mean bias"])
    assert bias.size == BUDGET // 32
    if name in AVERAGE_REWARD:
        assert np.isnan(bias).all()
    else:
        assert np.isfinite(bias).all()
    rate = np.asarray(rows["Eval/mean average reward"])
    assert np.isfinite(rate).all() == (name in MEASURES_REWARD_RATE)
    assert np.isnan(rate).all() == (name not in MEASURES_REWARD_RATE)


def _normalising_sac() -> SAC:
    net = ("8", "relu")
    return SAC(
        "Pendulum-v1",
        normalize_observations=True,
        learning_starts=8,
        batch_size=8,
        actor_architecture=net,
        critic_architecture=net,
    )


def test_a_logged_evaluation_is_handed_the_agents_observation_statistics(monkeypatch):
    """An on-policy agent with normalize_observations stores its
    observations normalised: the evaluations a run logs normalise theirs
    through its controller with the training statistics, not raw."""
    handed = []
    evaluate = log.evaluate

    def evaluating(*args, controller=None, agent_state=None, **kwargs):
        handed.append((controller, agent_state.collector_state.obs_norm_info))
        return evaluate(*args, controller=controller, agent_state=agent_state, **kwargs)

    monkeypatch.setattr(log, "evaluate", evaluating)
    logging = LoggingConfig(config={}, log_frequency=16, use_wandb=False)
    net = ("8", "relu")
    agent = PPO(
        "Pendulum-v1",
        normalize_observations=True,
        n_steps=16,
        n_epochs=1,
        actor_architecture=net,
        critic_architecture=net,
    )
    agent.train(seed=0, n_timesteps=32, num_episode_test=1, logging_config=logging)
    assert handed
    assert all(c.normalize_obs and stats is not None for c, stats in handed)


def test_an_evaluation_keeps_the_actors_own_statistics(monkeypatch):
    """A trained replay agent normalises in its networks: evaluating its
    actor (no env statistics) acts on observations normalised by the actor's
    statistics, which the evaluation does not drop."""
    state, _ = _normalising_sac().train(seed=0, n_timesteps=32)
    actor = jax.tree.map(lambda x: x[0], state.actor_state)
    assert actor.obs_norm_info is not None
    read = []
    get_pi = evaluate_module.get_pi

    def acting(actor_state, *args, **kwargs):
        read.append(getattr(actor_state, "obs_norm_info", None))
        return get_pi(actor_state, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "get_pi", acting)
    env, params = gymnax.make("Pendulum-v1")
    evaluate_module.evaluate(env, actor, 1, jax.random.PRNGKey(0), params)
    assert read and all(info is not None for info in read)


def test_an_evaluation_normalises_with_the_collectors_statistics(monkeypatch):
    """A trained on-policy agent stores its observations normalised:
    evaluating it through its run's controller normalises the evaluation's
    observations with the statistics its collector holds, which the
    evaluation does not drop."""
    net = ("8", "relu")
    agent = PPO(
        "Pendulum-v1",
        normalize_observations=True,
        n_steps=16,
        n_epochs=1,
        actor_architecture=net,
        critic_architecture=net,
    )
    state, _ = agent.train(seed=0, n_timesteps=32)
    state = jax.tree.map(lambda x: x[0], state)
    assert state.collector_state.obs_norm_info is not None
    read = []
    normalized_obs = controller_module.normalized_obs

    def normalising(obs, info, *args):
        read.append(info)
        return normalized_obs(obs, info, *args)

    monkeypatch.setattr(controller_module, "normalized_obs", normalising)
    env_args = agent.env_args
    evaluate_module.evaluate(
        env_args.env,
        state.actor_state,
        1,
        jax.random.PRNGKey(0),
        env_args.env_params,
        agent_state=state,
        controller=controller_module.make_run_controller(env_args),
    )
    assert read and all(info is not None for info in read)
