import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.train_PPO import _force_reset, _normalisation_info
from ajax.environments.create import strip_ajax_wrappers
from ajax.environments.interaction import collect_experience, reset
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig, zeros_carry_like
from ajax.networks.networks import get_initialized_actor_critic
from tests.probing import readouts


def test_PPO_train_playground_smoke():
    """Smoke test: PPO trains briefly on a playground env (covers playground
    code path end-to-end including seed vmap and eval)."""
    agent = PPO(env_id="PendulumSwingup", n_envs=2, n_steps=16, n_epochs=1)
    agent.train(seed=42, n_timesteps=50)


@pytest.mark.parametrize("env_id, mode", [("Pendulum-v1", "gymnax"), ("fast", "brax")])
def test_a_forced_reset_keeps_the_normaliser_statistics(env_id, mode):
    """num_resets_per_eval resets the envs, not the normaliser's running
    statistics (each gymnax env restarted its own from the reset): the reset
    observation is normalised with the statistics before it."""
    agent = PPO(
        env_id=env_id,
        n_envs=2,
        n_steps=8,
        n_epochs=1,
        normalize_observations=True,
        normalize_rewards=True,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=32)[0])
    saved = _normalisation_info(state.collector_state.env_state, mode)
    after = _force_reset(state, agent.env_args, mode).collector_state
    kept = _normalisation_info(after.env_state, mode)
    jax.tree.map(np.testing.assert_array_equal, kept, saved)

    key = jax.random.split(state.rng)[0]
    keys = jax.random.split(key, 2) if mode == "gymnax" else key
    task = strip_ajax_wrappers(agent.env_args.env)[0]
    raw, _ = reset(keys, task, mode, agent.env_args.env_params)
    std = np.sqrt(np.asarray(saved.obs.var) + 1e-8).mean(axis=0)
    want = (raw - np.asarray(saved.obs.mean).mean(axis=0)) / std
    np.testing.assert_allclose(after.last_obs, want.reshape(raw.shape), atol=1e-4)


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_a_forced_reset_marks_an_episode_start_for_a_recurrent_policy(kind):
    """A forced reset flags every env truncated, the flag a recurrent actor
    and critic restart their carry on (collection and the rollout's first
    reset flag), for every memory kind. With ``num_evals=2`` the one reset
    falls on the last update; the run stays finite. The next collection
    step acts as from a fresh carry on the reset observation, read through
    the step after it (a transformer's cache keeps the entries a restart
    masks out)."""
    agent = PPO(
        env_id="Pendulum-v1",
        n_envs=2,
        n_steps=8,
        num_minibatches=2,
        bptt_length=4,
        memory=MemoryConfig(kind=kind, hidden_size=8, window=8),
        num_resets_per_eval=1,
        num_evals=2,
    )
    state, _ = agent.train(seed=0, n_timesteps=48)
    np.testing.assert_array_equal(state.collector_state.last_truncated, 1.0)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert np.isfinite(leaf).all()

    state = jax.tree.map(lambda x: x[0], state)
    actor, reset_obs = state.actor_state, state.collector_state.last_obs
    after, _ = collect_experience(
        state, None, recurrent=True, mode="gymnax", env_args=agent.env_args
    )
    starts, goes_on = jnp.ones((1, 2), bool), jnp.zeros((1, 2), bool)
    fresh = zeros_carry_like(actor.hidden_state, 2)
    _, want = actor.apply(actor.params, reset_obs[None], fresh, starts)
    key = jax.random.PRNGKey(0)
    got, wanted = (
        actor.apply(actor.params, reset_obs[None], h, goes_on)[0].sample(seed=key)
        for h in (after.actor_state.hidden_state, want)
    )
    np.testing.assert_allclose(got, wanted, rtol=1e-5, atol=1e-5)


def test_a_split_run_resets_on_the_one_calls_cadence():
    """brax's ``num_resets_per_eval`` divides the run's updates: a run of 96
    steps (7 updates of 16, whole rollouts plus one) with ``num_evals=3``
    resets every 4 updates, once. Split in two, it resets as the one call
    does, on update 4, its second leg's first: the first leg is told the
    run's horizon, the second ends where the run does (its own end is the
    horizon) and counts the run's updates, not its own."""

    def agent():
        return PPO(
            "Pendulum-v1",
            n_envs=2,
            n_steps=8,
            n_epochs=1,
            num_resets_per_eval=1,
            num_evals=3,
        )

    one, _ = agent().train(seed=0, n_timesteps=96)
    first, _ = agent().train(seed=0, n_timesteps=32, total_timesteps=96)
    np.testing.assert_array_equal(first.n_updates, 3)
    split, _ = agent().train(seed=0, n_timesteps=96 - 48, initial_state=first)
    # The two runs are separately compiled programs: some CI runners round them
    # 1.67e-6 apart (the env's last torque). A wrong cadence changes the keys,
    # the pendulum's state by 1 to 6 and the weights by 9e-3 or more.
    assert not readouts.differing_leaves(one, split, atol=1e-4)


def test_ppo_defaults_to_its_reference_log_std():
    """One learned log-std per action dimension, starting at 0 and never
    clipped, as Baselines, CleanRL and PureJaxRL learn it; the run config
    records the bounds the network is built with."""
    agent = PPO("Pendulum-v1")
    args = agent.network_args
    assert args.log_std_state_independent and args.log_std_init == 0.0
    assert args.log_std_bounds is None and agent.config["log_std_bounds"] is None


def _gain(kernel) -> float:
    """The gain of an orthogonal kernel: sqrt of its Gram matrix's diagonal,
    after checking the Gram matrix is that times the identity."""
    k = np.asarray(kernel)
    gram = k.T @ k if k.shape[0] >= k.shape[1] else k @ k.T
    gain2 = float(np.mean(np.diag(gram)))
    np.testing.assert_allclose(gram, gain2 * np.eye(len(gram)), atol=1e-4 * gain2)
    return float(np.sqrt(gain2))


@pytest.mark.parametrize("memory", [None, MemoryConfig("gru", 16)], ids=["ff", "gru"])
@pytest.mark.parametrize(
    "env_id, continuous", [("CartPole-v1", False), ("Pendulum-v1", True)]
)
def test_ppo_starts_its_layers_as_baselines_ppo2(env_id, continuous, memory):
    """Baselines' ppo2 @ea25b9e: hidden layers orthogonal with gain sqrt(2)
    (common/models.py:96), the policy's output 0.01 (common/policies.py:49),
    the value 1 (:63, a2c/utils.py:58), zero biases; the run config records
    that init. Recurrent PPO's reference, PureJaxRL @31756b1, starts the
    layers before its memory the same way (ppo_rnn.py:53-55) and keeps
    flax's GRU (:35): recurrent kernels orthogonal with gain 1, per gate."""
    from ajax.agents.PPO.PPO import PPO_HIDDEN_INIT

    agent = PPO(env_id, memory=memory)
    assert agent.config["encoder_kernel_init"] == PPO_HIDDEN_INIT
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=continuous,
        num_critics=1,
    )
    actor = actor_state.params["params"]
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    out = actor["mean"] if continuous else actor["model"]["layers_0"]
    gains = {
        "actor hidden 0": (actor["encoder"]["network"]["layers_0"], np.sqrt(2)),
        "actor hidden 1": (actor["encoder"]["network"]["layers_2"], np.sqrt(2)),
        "policy output": (out, 0.01),
        "critic hidden 0": (critic["encoder"]["network"]["layers_0"], np.sqrt(2)),
        "critic hidden 1": (critic["encoder"]["network"]["layers_2"], np.sqrt(2)),
        "value": (critic["model"], 1.0),
    }
    for name, (layer, gain) in gains.items():
        np.testing.assert_allclose(
            _gain(layer["kernel"]), gain, rtol=1e-4, err_msg=name
        )
        np.testing.assert_array_equal(layer["bias"], 0.0, err_msg=name)
    if memory is not None:
        gru = actor["memory_cell"]["core"]["gru_0"]
        for gate in ("hr", "hz", "hn"):
            np.testing.assert_allclose(_gain(gru[gate]["kernel"]), 1.0, rtol=1e-4)
