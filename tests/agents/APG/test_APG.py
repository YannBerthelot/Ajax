"""End-to-end tests for the APG agent and its contextual-controller configuration."""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gymnax import make as make_gymnax_env

from ajax import APG
from ajax.agents.APG import (
    Controller,
    CurriculumStage,
    PIDHeadConfig,
    train_curriculum,
)
from ajax.agents.APG.APG import (
    compute_rollout_returns,
    make_agent_state,
    make_policy_step,
    reset_optimizer,
)
from ajax.checkpoint import load_checkpoint, save_checkpoint
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.system_class import FixedSystem, UniformPerturbation
from ajax.networks.memory import MemoryCell, MemoryConfig, init_carry
from ajax.wrappers import InitialStateWrapper
from tests.agents.extension_stubs import CounterExt
from tests.environments import paid_action

HORIZON = 16
N_ENVS = 4
#: APG's module (the package's APG is the class).
APG_FILE = importlib.import_module("ajax.agents.APG.APG")


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


def test_adam_eps_is_torchs(env_and_class):
    """The reference's AdamW keeps torch's eps (in-context-controller
    transformer_onestep_CL.py:86), not Ajax's shared 1e-5."""
    env, sc = env_and_class
    assert make_agent(env, sc).actor_optimizer_args.eps == 1e-8


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


def test_an_update_logs_its_gradient_norm_before_the_clip(default_run):
    """The first update's norm is that of the matching loss's gradient at
    the initial controller, on the rollout key the update draws."""
    agent, sc, _, aux = default_run
    init_key = jax.random.split(jax.random.PRNGKey(0))[0]  # fresh_state's
    args = (agent.env_args, agent.actor_optimizer_args, agent.network_args)
    state = make_agent_state(init_key, *args, agent.pid, agent.squash)
    roll_key = jax.random.split(state.rng, 3)[1]

    def loss(params):
        returns, _ = compute_rollout_returns(
            state, params, roll_key, agent.env_args, sc, HORIZON, N_ENVS
        )
        return -returns.mean()

    grads = jax.grad(loss)(state.actor_state.params)
    expected = float(optax.global_norm(grads))
    np.testing.assert_allclose(aux.grad_norm[0, 0], expected, rtol=1e-4)
    assert expected > agent.actor_optimizer_args.max_grad_norm  # the clip binds


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
        kind="transformer",
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        window=8,
        init="nanogpt",
    )
    assert agent.pid == PIDHeadConfig()  # identity init, see APG._DEFAULT_PID
    assert agent.lr_schedule == "warmup_cosine"
    # The reference's AdamW: torch's eps, no gradient clipping.
    assert agent.actor_optimizer_args.max_grad_norm is None
    assert agent.actor_optimizer_args.eps == 1e-8
    assert agent.actor_optimizer_args.weight_decay == 0.01
    state, aux = agent.train(seed=0, n_timesteps=4 * 2 * 8)
    assert state.actor_state.recurrent
    params = state.actor_state.params["params"]
    assert "memory_cell" in params and "pid_head" in params
    assert jnp.isfinite(aux.loss).all()


def test_contextual_controller_rebuilds_a_transformer_without_its_final_norm(
    env_and_class,
):
    """final_norm reaches the transformer, so a contextual controller saved
    before the core ended with a LayerNorm restores into the network it was
    trained with (memory= would clash with the one the method builds)."""
    env, sc = env_and_class
    kw = {"n_layers": 1, "n_heads": 2, "d_model": 8, "context": 8}
    agent = APG.contextual_controller(env, sc, final_norm=False, **kw)
    assert agent.network_args.memory == MemoryConfig(
        kind="transformer",
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        window=8,
        final_norm=False,
        init="nanogpt",
    )


def test_a_transformer_controller_starts_as_the_reference_gpt(env_and_class):
    """in-context-controller @7eb094c (transformer_onestep.py:162-166,
    :189-195): normal(0.02) kernels, the residual projections at
    0.02/sqrt(2 n_layer), zero biases. An explicit init overrides it, and
    another kind keeps flax's."""
    env, sc = env_and_class
    kw = {"n_layers": 2, "n_heads": 2, "d_model": 64, "context": 8}
    memory = APG.contextual_controller(env, sc, **kw).network_args.memory
    x, resets = jnp.ones((3, 1, 5)), jnp.zeros((3, 1), bool)
    carry = init_carry(memory, jax.random.PRNGKey(0), 1)
    params = MemoryCell(memory).init(jax.random.PRNGKey(0), carry, x, resets)
    core = params["params"]["core"]
    for name in ("q_0", "mlp_up_1", "attn_out_0", "mlp_down_1"):
        residual = name.startswith(("attn_out", "mlp_down"))
        std = 0.02 / np.sqrt(4) if residual else 0.02
        assert float(jnp.std(core[name]["kernel"])) == pytest.approx(std, rel=0.15)
        assert (core[name]["bias"] == 0).all()
    flax_init = MemoryConfig("transformer", 8, num_heads=2, window=8, init="flax")
    assert APG(env, memory=flax_init).network_args.memory.init == "flax"
    assert APG(env, memory=MemoryConfig("gru", 8)).network_args.memory.init is None


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
    # Refused at construction, before a train call starts its logging worker.
    with pytest.raises(ValueError, match="lr_schedule"):
        make_agent(env, sc, lr_schedule="linear")
    with pytest.raises(ValueError, match="needs a float learning_rate"):
        make_agent(env, sc, lr_schedule="warmup_cosine", learning_rate=lambda s: 1e-3)
    # A warmup longer than the run is clamped, leaving one decay step:
    # warmup_cosine_schedule rejects warmup_steps >= total_steps when the
    # train function is built, before anything is traced.
    short = make_agent(env, sc, lr_schedule="warmup_cosine", warmup_steps=50)
    short.get_make_train()(
        env_args=short.env_args,
        actor_optimizer_args=short.actor_optimizer_args,
        critic_optimizer_args=short.critic_optimizer_args,
        network_args=short.network_args,
        total_timesteps=2 * N_ENVS * HORIZON,
        num_episode_test=1,
    )


def test_the_schedule_steps_the_matrices_and_the_vectors_keep_the_peak(env_and_class):
    """As in the reference, which sets its first parameter group's rate alone
    (buswayne/in-context-controller at 7eb094ce17: train_onestep_lin_CL.py:241).
    Under a constant gradient each Adam step is its rate."""
    env, sc = env_and_class
    agent = make_agent(
        env, sc, lr_schedule="warmup_cosine", warmup_steps=2, max_grad_norm=None
    )
    peak = agent.actor_optimizer_args.learning_rate
    context = agent.make_context(4 * N_ENVS * HORIZON)[1]
    context, _ = agent.configure_call(context, n_iterations=4)
    optimizer_args = context.actor_optimizer_args
    args = (agent.env_args, optimizer_args, agent.network_args)
    state = make_agent_state(
        jax.random.PRNGKey(0), *args, PIDHeadConfig(), agent.squash
    )
    params = state.actor_state.params
    tx, gradients = state.actor_state.tx, jax.tree.map(jnp.ones_like, params)
    opt_state = tx.init(params)
    for count in range(4):
        updates, opt_state = tx.update(gradients, opt_state, params)
        schedule = optimizer_args.learning_rate(count)
        for update in jax.tree.leaves(updates):
            rate = schedule if update.ndim >= 2 else peak
            np.testing.assert_allclose(update, -rate, rtol=1e-4)
    assert optimizer_args.learning_rate(0) == 0.0  # the schedule moves


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


def test_a_checkpoint_loaded_with_the_runs_options_resumes_as_its_state(
    env_and_class, tmp_path
):
    """Contract (load_checkpoint, the user-facing fixes plan, section 3): a
    run that keeps its optimiser on resume runs the warmup-cosine schedule
    of the call that built its state, over the run's total_timesteps.
    Loaded with the run's train options, a checkpoint resumes exactly as
    the state it saved resumes; loaded without them, its schedule spans the
    skeleton's call instead, and the resumed run ends elsewhere."""
    env, sc = env_and_class
    schedule = {"lr_schedule": "warmup_cosine", "warmup_steps": 1}
    agent = make_agent(env, sc, **schedule)
    step, total = N_ENVS * HORIZON, 4 * N_ENVS * HORIZON
    options = {"total_timesteps": total, "reset_optimizer_on_resume": False}
    half, _ = agent.train(seed=0, n_timesteps=2 * step, **options)
    path = str(tmp_path / "half.pkl")
    save_checkpoint(half, path)

    def resume(state):
        resumed, _ = agent.train(
            seed=0, n_timesteps=2 * step, initial_state=state, **options
        )
        return jax.tree.leaves(resumed.actor_state.params)

    # Copied: a resume donates the state's buffers.
    want = resume(jax.tree.map(jnp.copy, half))
    loaded = load_checkpoint(make_agent(env, sc, **schedule), path, **options)
    for got, expected in zip(resume(loaded), want, strict=True):
        np.testing.assert_array_equal(got, expected)
    loaded = load_checkpoint(make_agent(env, sc, **schedule), path)
    elsewhere = zip(resume(loaded), want, strict=True)
    assert any(not np.array_equal(got, expected) for got, expected in elsewhere)


def test_the_validation_set_is_drawn_apart_from_training(env_and_class):
    """``eval_rng`` is split from the init key apart from ``rng``, which
    every training draw descends from: the validation systems are not
    training ones."""
    env, sc = env_and_class
    agent = make_agent(env, sc)
    args = (agent.env_args, agent.actor_optimizer_args, agent.network_args)
    state = make_agent_state(jax.random.PRNGKey(0), *args, agent.pid, agent.squash)
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
    state = make_agent_state(jax.random.PRNGKey(0), *args, agent.pid, agent.squash)
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
    metrics = agent.evaluate(single, jax.random.PRNGKey(0), agent.make_context(1)[1], 5)
    assert set(metrics) == {
        "Eval/episodic mean reward",
        "Eval/m_rmse",
        "Eval/episodic mean expert reward",
        "Eval/expert bias",
    }
    assert jnp.isnan(metrics["Eval/episodic mean expert reward"])  # no expert
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
    returns every update's aux per seed; make_agent_state and
    compute_rollout_returns rebuild and evaluate it positionally, with
    ``stateful=True``."""
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
    fresh = make_agent_state(
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
    returns, _ = compute_rollout_returns(
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
        _, rollout = compute_rollout_returns(
            state, params, key, agent.env_args, None, 2, 4
        )
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


def test_apg_returns_its_evaluations_without_a_logging_backend():
    """A logging config without a backend still evaluates, and the
    evaluations come back on ``aux`` rather than being dropped: NaN on the
    updates that do not evaluate. Without a logging config there are none."""
    from ajax.logging.wandb_logging import LoggingConfig

    agent = APG("Pendulum-v1", n_envs=2, horizon=4, actor_architecture=("8", "relu"))
    logging = LoggingConfig(log_frequency=16, backends=())
    _, aux = agent.train(
        seed=0, n_timesteps=64, num_episode_test=1, logging_config=logging
    )
    reward = np.asarray(aux.evaluations["Eval/episodic mean reward"])
    assert reward.shape == aux.loss.shape == (1, 8)
    evaluated = np.isfinite(reward[0])
    np.testing.assert_array_equal(evaluated, [False, True] * 4)
    _, plain = agent.train(seed=0, n_timesteps=64)
    assert plain.evaluations is None


def test_the_evaluation_rolls_out_the_training_horizon(monkeypatch):
    """APG's evaluation episodes are rollouts of the training horizon, whose
    length the RMSE divides by."""
    seen = []

    def rollout(agent_state, params, key, env_args, system_class, horizon, n, **_):
        seen.append(horizon)
        return -jnp.full((n,), 4.0 * horizon), None

    monkeypatch.setattr(APG_FILE, "compute_rollout_returns", rollout)
    state = SimpleNamespace(actor_state=SimpleNamespace(params=None))
    agent = APG("Pendulum-v1", horizon=16)
    context = SimpleNamespace(env_args=None)
    out = agent.evaluate(state, jax.random.PRNGKey(0), context, 2)
    assert seen == [16]
    np.testing.assert_allclose(out["Eval/m_rmse"], 2.0)


def test_apg_refuses_a_dataset():
    """It learns through its own differentiable rollouts."""
    with pytest.raises(ValueError, match="cannot train offline"):
        APG("Pendulum-v1").train_offline(dataset=None, n_updates=1)


@pytest.mark.parametrize("option", ["expert_policy", "eval_expert_policy"])
def test_the_evaluation_logs_its_experts_return(option):
    """The expert acting alone on the evaluation's systems: on a task
    paying the action of each one-step episode, an expert taking 1 (env
    units) returns the horizon, 8, and the bias is the agent's return minus
    that."""
    env, params = paid_action.make()

    def one(obs):
        return jnp.ones(obs.shape[:-1] + (1,))

    agent = APG(env, env_params=params, n_envs=2, horizon=8, **{option: one})
    _, context = agent.make_context(16)
    state = agent.init(jax.random.PRNGKey(0), context)
    out = agent.evaluate(state, jax.random.PRNGKey(1), context, 3)
    np.testing.assert_allclose(out["Eval/episodic mean expert reward"], 8.0)
    np.testing.assert_allclose(
        out["Eval/expert bias"], out["Eval/episodic mean reward"] - 8.0, rtol=1e-6
    )
