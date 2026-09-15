"""End-to-end tests for the APG agent and its contextual-controller configuration."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax import APG
from ajax.agents.APG import CurriculumStage, PIDHeadConfig, train_curriculum
from ajax.agents.APG.train_APG import evaluate_apg
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.system_class import FixedSystem, UniformPerturbation
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig

HORIZON = 16
N_ENVS = 4


def angle(obs):
    return jnp.arctan2(obs[1], obs[0])


def tracking_env(horizon=HORIZON, min_duration=4, max_duration=8):
    """Gravity-free pendulum (a double integrator, so angle references are
    trackable with unit torque) under a model-reference tracking wrapper."""
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(g=0.0)
    reference = StepReference(
        horizon=horizon,
        min_value=-0.5,
        max_value=0.5,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    wrapped = ModelReferenceWrapper(
        env, reference, LinearReferenceModel.first_order(), output_fn=angle
    )
    return wrapped, params


@pytest.fixture
def env_and_class():
    env, params = tracking_env()
    return env, UniformPerturbation(params, fields=("m", "l"), scale=0.1)


def make_agent(env, system_class, **kwargs):
    defaults = {
        "n_envs": N_ENVS,
        "horizon": HORIZON,
        "learning_rate": 3e-3,
        "actor_architecture": ("16", "relu"),
        "system_class": system_class,
        "env_params": system_class.nominal,
    }
    defaults.update(kwargs)
    return APG(env, **defaults)


def test_rejects_non_differentiable_env():
    with pytest.raises(ValueError, match="transition gradients"):
        APG("CartPole-v1")


def test_initial_state_has_no_critic_and_flows_through_env(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc)
    assert agent.env_args.env._env.transition_gradients_enabled
    state, aux = agent.train(seed=0, n_timesteps=N_ENVS * HORIZON)
    assert state.critic_state is None
    assert not state.actor_state.recurrent
    assert aux.loss.shape == (1, 1)
    assert jnp.isfinite(aux.loss).all() and jnp.isfinite(aux.m_rmse).all()
    assert int(state.collector_state.timestep[0]) == N_ENVS * HORIZON


def test_training_reduces_matching_loss():
    env, params = tracking_env(horizon=32, min_duration=8, max_duration=16)
    sc = UniformPerturbation(params, fields=("m", "l"), scale=0.1)
    n_envs, horizon, n_updates = 8, 32, 80
    agent = make_agent(env, sc, n_envs=n_envs, horizon=horizon, learning_rate=1e-2)
    _, aux = agent.train(seed=1, n_timesteps=n_updates * n_envs * horizon)
    losses = aux.matching_loss[0]
    assert losses.shape == (n_updates,)
    assert losses[-10:].mean() < 0.7 * losses[:10].mean()


def test_multiple_seeds_are_vmapped(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc)
    state, aux = agent.train(seed=[0, 1, 2], n_timesteps=2 * N_ENVS * HORIZON)
    assert aux.loss.shape == (3, 2)
    assert state.actor_state.params["params"]["out"]["kernel"].shape[0] == 3


def test_contextual_controller_configuration(env_and_class):
    env, sc = env_and_class
    agent = APG.contextual_controller(
        env,
        sc,
        n_envs=2,
        horizon=8,
        n_layers=1,
        n_heads=2,
        d_model=8,
        context=8,
        warmup_steps=2,
        weight_decay=0.01,
    )
    assert agent.network_args.memory == MemoryConfig(
        kind="transformer", hidden_size=8, num_layers=1, num_heads=2, window=8
    )
    assert agent.pid == PIDHeadConfig() and agent.lr_schedule == "warmup_cosine"
    assert agent.actor_optimizer_args.weight_decay == 0.01
    state, aux = agent.train(seed=0, n_timesteps=4 * 2 * 8)
    assert state.actor_state.recurrent
    params = state.actor_state.params["params"]
    assert "memory_cell" in params and "pid_head" in params
    assert jnp.isfinite(aux.loss).all()


def test_memory_only_controller_trains(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc, memory=MemoryConfig(kind="gru", hidden_size=8))
    state, aux = agent.train(seed=0, n_timesteps=3 * N_ENVS * HORIZON)
    assert state.actor_state.recurrent and jnp.isfinite(aux.loss).all()


def test_warmup_cosine_schedule_is_wired_and_validated(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc, lr_schedule="warmup_cosine", warmup_steps=2)
    _, aux = agent.train(seed=0, n_timesteps=4 * N_ENVS * HORIZON)
    assert jnp.isfinite(aux.loss).all()
    with pytest.raises(ValueError, match="lr_schedule"):
        make_agent(env, sc, lr_schedule="linear").train(seed=0, n_timesteps=64)


def test_resume_keeps_params_and_resets_optimizer(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc)
    state1, _ = agent.train(seed=0, n_timesteps=3 * N_ENVS * HORIZON)
    assert int(state1.actor_state.step[0]) == 3
    # resume donates the incoming state's buffers: copy before reusing it
    state1_copy = jax.tree.map(jnp.copy, state1)
    state2, _ = agent.train(
        seed=0, n_timesteps=2 * N_ENVS * HORIZON, initial_state=state1_copy
    )
    assert int(state2.actor_state.step[0]) == 2  # optimizer restarted
    assert int(state2.collector_state.timestep[0]) == 5 * N_ENVS * HORIZON
    state3, _ = agent.train(
        seed=0,
        n_timesteps=2 * N_ENVS * HORIZON,
        initial_state=state1,
        reset_optimizer_on_resume=False,
    )
    assert int(state3.actor_state.step[0]) == 5
    # parameters were carried over (not re-initialised)
    fresh, _ = agent.train(seed=0, n_timesteps=N_ENVS * HORIZON)
    k3 = state3.actor_state.params["params"]["out"]["kernel"]
    assert not jnp.allclose(k3, fresh.actor_state.params["params"]["out"]["kernel"])


def test_curriculum_chains_stages(env_and_class):
    env, sc = env_and_class
    _, params = tracking_env()
    stage1 = CurriculumStage(
        make_agent(env, FixedSystem(params)), 2 * N_ENVS * HORIZON, "nominal"
    )
    stage2 = CurriculumStage(make_agent(env, sc), 3 * N_ENVS * HORIZON, "class")
    results = train_curriculum([stage1, stage2], seed=0)
    assert len(results) == 2
    state_final = results[-1][0]
    assert int(state_final.collector_state.timestep[0]) == 5 * N_ENVS * HORIZON
    assert results[-1][1].loss.shape == (1, 3)


def test_curriculum_rejects_incompatible_stages(env_and_class):
    env, sc = env_and_class
    a = make_agent(env, sc)
    with pytest.raises(ValueError, match="n_envs"):
        train_curriculum(
            [CurriculumStage(a, 64), CurriculumStage(make_agent(env, sc, n_envs=2), 64)]
        )
    with pytest.raises(ValueError, match="network"):
        train_curriculum(
            [
                CurriculumStage(a, 64),
                CurriculumStage(make_agent(env, sc, actor_architecture=("8",)), 64),
            ]
        )
    with pytest.raises(ValueError, match="at least one"):
        train_curriculum([])


def test_evaluate_reports_m_rmse_on_fresh_systems(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc)
    state, _ = agent.train(seed=0, n_timesteps=N_ENVS * HORIZON)
    single = jax.tree.map(lambda x: x[0], state)
    metrics = evaluate_apg(
        single, jax.random.PRNGKey(0), agent.env_args, sc, HORIZON, 5, stateful=False
    )
    assert set(metrics) == {"Eval/episodic mean reward", "Eval/m_rmse"}
    assert jnp.isfinite(metrics["Eval/m_rmse"]) and metrics["Eval/m_rmse"] >= 0
    assert jnp.allclose(
        metrics["Eval/m_rmse"],
        jnp.sqrt(-metrics["Eval/episodic mean reward"] / HORIZON),
    )


@dataclass(frozen=True)
class Counter(Extension):
    name: str = "counter"

    def init_state(self, agent_state, rng):
        return jnp.asarray(0)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1


def test_extensions_fold_post_update(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc, extensions=[Counter()])
    state, _ = agent.train(seed=0, n_timesteps=4 * N_ENVS * HORIZON)
    assert int(state.ext_state[0][0]) == 4
