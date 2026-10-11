"""End-to-end tests for the APG agent and its contextual-controller configuration."""

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax import APG
from ajax.agents.APG import (
    Controller,
    CurriculumStage,
    PIDHeadConfig,
    train_curriculum,
)
from ajax.agents.APG.train_APG import (
    evaluate_apg,
    init_APG,
    make_policy_step,
    reset_optimizer,
    rollout_returns,
)
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.system_class import FixedSystem, UniformPerturbation
from ajax.networks.memory import MemoryConfig
from ajax.wrappers import InitialStateWrapper
from tests.agents.extension_stubs import CounterExt

HORIZON = 16
N_ENVS = 4


def angle(obs):
    return jnp.arctan2(obs[1], obs[0])


def tracking_env():
    """Gravity-free pendulum (a double integrator, so angle references are
    trackable with unit torque) under the paper's model-reference tracking
    wrapper."""
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(g=0.0)
    reference = StepReference(
        horizon=HORIZON, min_value=-0.5, max_value=0.5, min_duration=4, max_duration=8
    )
    model = LinearReferenceModel.first_order()
    wrapped = ModelReferenceWrapper(env, reference, model, output_fn=angle)
    return wrapped, params


def tracking_class(params):
    return UniformPerturbation(params, fields=("m", "l"), scale=0.1)


@pytest.fixture
def env_and_class():
    env, params = tracking_env()
    return env, tracking_class(params)


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


@pytest.fixture(scope="module")
def default_run():
    """The default agent's one-update run, shared by the tests that only
    read it: ``(agent, system_class, state, aux)``."""
    env, params = tracking_env()
    sc = tracking_class(params)
    agent = make_agent(env, sc)
    state, aux = agent.train(seed=0, n_timesteps=N_ENVS * HORIZON)
    return agent, sc, state, aux


def test_gradient_clipping_is_on_by_default_and_can_be_disabled(env_and_class):
    env, sc = env_and_class
    assert make_agent(env, sc).actor_optimizer_args.max_grad_norm is not None
    off = make_agent(env, sc, max_grad_norm=None)
    assert off.actor_optimizer_args.max_grad_norm is None


def test_rejects_non_differentiable_env():
    with pytest.raises(ValueError, match="transition gradients"):
        APG("CartPole-v1")


def test_initial_state_has_no_critic_and_flows_through_env(default_run):
    agent, _, state, aux = default_run
    assert agent.env_args.env._env.transition_gradients_enabled
    assert state.critic_state is None
    assert not state.actor_state.recurrent
    assert aux.loss.shape == (1, 1)
    assert jnp.isfinite(aux.loss).all() and jnp.isfinite(aux.m_rmse).all()
    assert int(state.collector_state.timestep[0]) == N_ENVS * HORIZON


class _FixedStepReference(StepReference):
    """Same reference every reset (makes the objective deterministic)."""

    def sample(self, rng):
        del rng
        return StepReference.sample(self, jax.random.PRNGKey(0))


def test_training_reduces_matching_loss():
    """Gradient descent through the simulator lowers the matching cost.

    Deterministic set-up so the assertion is about optimisation, not about
    the luck of sampled references / initial states (a stochastic version
    of this test passed on macOS and failed on the Linux CI runner): one
    fixed system, one fixed reference, a pinned initial state, a reference
    model slow enough to be reachable at Pendulum's 50 ms step, and the
    default gradient clipping disabled (it only slows this 32-step problem).
    """
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(g=0.0)
    env = InitialStateWrapper(
        env,
        lambda key, state, _: state.replace(
            theta=jnp.asarray(0.3), theta_dot=jnp.asarray(0.0)
        ),
    )
    reference = _FixedStepReference(
        horizon=32, min_value=-0.5, max_value=0.5, min_duration=8, max_duration=16
    )
    model = LinearReferenceModel.first_order(a=0.9, b=0.1, c=1.0, d=0.0)
    task = ModelReferenceWrapper(env, reference, model, output_fn=angle)
    n_envs, horizon, n_updates = 2, 32, 60
    agent = APG(
        task,
        n_envs=n_envs,
        horizon=horizon,
        learning_rate=1e-2,
        max_grad_norm=None,
        actor_architecture=("16", "relu"),
        system_class=FixedSystem(params),
        env_params=params,
    )
    _, aux = agent.train(seed=1, n_timesteps=n_updates * n_envs * horizon)
    losses = aux.matching_loss[0]
    assert losses.shape == (n_updates,)
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-5:].mean() < 0.5 * losses[:5].mean()


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
    assert agent.pid == PIDHeadConfig()  # identity init, see APG._DEFAULT_PID
    assert agent.lr_schedule == "warmup_cosine"
    assert agent.actor_optimizer_args.max_grad_norm == 1.0
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
    # A warmup longer than the run is clamped, leaving one decay step:
    # warmup_cosine_schedule rejects warmup_steps >= total_steps when the
    # train function is built, before anything is traced.
    short = make_agent(env, sc, lr_schedule="warmup_cosine", warmup_steps=50)
    short.get_make_train()(
        env_args=short.env_args,
        actor_optimizer_args=short.actor_optimizer_args,
        critic_optimizer_args=short.critic_optimizer_args,
        network_args=short.network_args,
        agent_config=short.agent_config,
        total_timesteps=2 * N_ENVS * HORIZON,
        num_episode_test=1,
    )


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


def test_the_validation_set_is_drawn_apart_from_training(env_and_class):
    """``eval_rng`` is split from the init key apart from ``rng``, which
    every training draw descends from: the validation systems are not
    training ones."""
    env, sc = env_and_class
    agent = make_agent(env, sc)
    args = (agent.env_args, agent.actor_optimizer_args, agent.network_args)
    state = init_APG(jax.random.PRNGKey(0), *args, agent.pid, agent.squash)
    assert not jnp.array_equal(state.eval_rng, state.rng)


def test_reset_optimizer_keeps_the_parameters_and_restarts_adam(env_and_class):
    """What resuming with ``reset_optimizer_on_resume`` (the default) does:
    the same parameters, bit for bit, under a fresh optimizer (step 0, zero
    moments). That a resume carries the parameters over is pinned by
    ``tests/probing/test_p4_split_run.py`` (a split APG run equals an
    unsplit one)."""
    env, sc = env_and_class
    agent = make_agent(env, sc)
    args = (agent.env_args, agent.actor_optimizer_args, agent.network_args)
    state = init_APG(jax.random.PRNGKey(0), *args, agent.pid, agent.squash)
    actor = state.actor_state
    grads = jax.tree.map(jnp.ones_like, actor.params)
    state = state.replace(actor_state=actor.apply_gradients(grads=grads))
    assert int(state.actor_state.step) == 1
    reset = reset_optimizer(state, agent.actor_optimizer_args).actor_state
    assert int(reset.step) == 0
    same = jax.tree.map(jnp.array_equal, reset.params, state.actor_state.params)
    assert jax.tree.all(same)
    fresh = reset.tx.init(reset.params)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, reset.opt_state, fresh))


def test_curriculum_chains_stages(env_and_class):
    env, sc = env_and_class
    stage1 = CurriculumStage(
        make_agent(env, FixedSystem(sc.nominal)), 2 * N_ENVS * HORIZON, "nominal"
    )
    stage2 = CurriculumStage(make_agent(env, sc), 3 * N_ENVS * HORIZON, "class")
    results = train_curriculum([stage1, stage2], seed=0)
    assert len(results) == 2
    # earlier stage states stay readable (a copy is handed to the next stage)
    assert jnp.all(
        jnp.isfinite(results[0][0].actor_state.params["params"]["out"]["kernel"])
    )
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


def test_evaluate_reports_m_rmse_on_fresh_systems(default_run):
    agent, sc, state, _ = default_run
    single = jax.tree.map(lambda x: x[0], state)
    metrics = evaluate_apg(
        single, jax.random.PRNGKey(0), agent.env_args, sc, HORIZON, 5
    )
    assert set(metrics) == {"Eval/episodic mean reward", "Eval/m_rmse"}
    assert jnp.isfinite(metrics["Eval/m_rmse"]) and metrics["Eval/m_rmse"] >= 0
    assert jnp.allclose(
        metrics["Eval/m_rmse"],
        jnp.sqrt(-metrics["Eval/episodic mean reward"] / HORIZON),
    )


def test_extensions_fold_post_update(env_and_class):
    env, sc = env_and_class
    agent = make_agent(env, sc, extensions=[CounterExt()])
    state, _ = agent.train(seed=0, n_timesteps=4 * N_ENVS * HORIZON)
    assert int(state.ext_state[0][0]) == 4


def test_the_downstream_training_and_evaluation_contract(env_and_class):
    """What TargetFoundation does (tmdp/experiments, tmdp/evaluate.py): a
    contextual controller from a factory, trained without a logging config,
    returns every update's aux per seed; init_APG and rollout_returns rebuild
    and evaluate it positionally, with ``stateful=True``."""
    env, sc = env_and_class

    def factory(*, env_args, network_args, action_dim, pid, squash):
        del env_args
        return Controller((), action_dim, network_args.memory, pid, squash)

    agent = APG.contextual_controller(
        env,
        sc,
        n_envs=2,
        horizon=HORIZON,
        n_layers=1,
        n_heads=1,
        d_model=8,
        context=8,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=1,
        env_params=sc.nominal,
        controller_factory=factory,
        max_grad_norm=1.0,
        extensions=[CounterExt()],
    )
    state, aux = agent.train(seed=0, n_timesteps=3 * 2 * HORIZON)
    curve = jax.device_get(-aux.matching_loss[0] / HORIZON).tolist()
    assert len(curve) == 3 and jnp.isfinite(aux.loss).all()
    single = jax.tree.map(lambda x: x[0], state)
    fresh = init_APG(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.network_args,
        agent.pid,
        agent.squash,
        controller_factory=agent.controller_factory,
    )
    assert jax.tree.structure(fresh.actor_state.params) == jax.tree.structure(
        single.actor_state.params
    )
    returns, _ = rollout_returns(
        single,
        single.actor_state.params,
        jax.random.PRNGKey(1),
        agent.env_args,
        sc,
        HORIZON,
        5,
        stateful=True,
    )
    assert returns.shape == (5,) and jnp.isfinite(returns).all()


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "MountainCarContinuous-v0"])
def test_the_controller_output_is_mapped_onto_the_torque_bounds(env_id):
    """Inside the differentiable rollout the controller's [-1, 1] output is
    mapped onto the env's bounds: on Pendulum's [-2, 2] the executed torque
    and its gradient are twice the output's; on MountainCarContinuous's
    [-1, 1] they are the output's own (to float rounding: the two programs
    compile apart)."""
    agent = APG(env_id, n_envs=4, horizon=2, pid=None, actor_architecture=("8", "tanh"))
    state, _ = agent.train(seed=0, n_timesteps=0)
    state = jax.tree.map(lambda x: x[0], state)
    key = jax.random.PRNGKey(1)

    def executed(params):
        _, rollout = rollout_returns(state, params, key, agent.env_args, None, 2, 4)
        return rollout.action[0, :, 0].sum(), rollout

    params = state.actor_state.params
    (total, rollout), grad_executed = jax.value_and_grad(executed, has_aux=True)(params)

    def own(params):
        step = make_policy_step(state.actor_state, params, False)
        return step(None, rollout.obs[0], rollout.resets[0])[0][:, 0].sum()

    scale = 2.0 if env_id == "Pendulum-v1" else 1.0
    out, grad_own = jax.value_and_grad(own)(params)
    assert jnp.allclose(total, scale * out, rtol=1e-6)
    close = jax.tree.map(
        lambda a, b: jnp.allclose(a, scale * b, rtol=1e-5, atol=1e-7),
        grad_executed,
        grad_own,
    )
    assert jax.tree.all(close)
