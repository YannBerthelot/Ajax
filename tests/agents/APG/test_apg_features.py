"""APG's observation features (ajax.extensions.observe): its differentiable
rollout appends them to each observation the controller reads, restarting
their memory at each episode start, and the return's gradient runs through
what they read, the expert's action included."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ajax import APG
from ajax.agents.APG.APG import compute_rollout_returns
from ajax.agents.loop import with_horizon
from ajax.extensions.base import Extension
from ajax.extensions.observe import (
    ExpertActionInput,
    PIDTermsInput,
    TrainingProgressInput,
)

N, HORIZON, LIMIT = 3, 10, 4  # systems, rollout steps, episode length


def push(obs: jax.Array) -> jax.Array:
    """An expert in Pendulum's torque units: against the angular velocity."""
    return -0.5 * obs[..., 2:3]


def rollout_of(extensions: tuple, timestep: int = 0) -> tuple:
    """APG on Pendulum with episodes of LIMIT steps, its fresh state at
    timestep of a 1000-step run: (agent, state, rollout)."""
    agent = APG(
        "Pendulum-v1",
        n_envs=N,
        horizon=HORIZON,
        actor_architecture=("8", "tanh"),
        expert_policy=push,
        extensions=extensions,
    )
    env_args = agent.env_args.replace(
        env_params=agent.env_args.env_params.replace(max_steps_in_episode=LIMIT)
    )
    _, context = agent.make_context(1000, env_args=env_args)
    state = with_horizon(agent.init(jax.random.PRNGKey(0), context), env_args, 1000)
    collector = state.collector_state.replace(timestep=jnp.asarray(timestep))
    state = state.replace(collector_state=collector)
    _, rollout = compute_rollout_returns(
        state,
        state.actor_state.params,
        jax.random.PRNGKey(1),
        env_args,
        None,
        HORIZON,
        N,
        expert_policy=agent.expert,
    )
    return agent, state, rollout, env_args


def test_the_controller_reads_the_observation_and_its_features():
    """Each step's observation, then the training progress at the
    collector's timestep, which the rollout holds, and the expert's action
    there (agent units: the torque over its bound 2), raw, so last."""
    extensions = (ExpertActionInput(), TrainingProgressInput())
    _, _, rollout, _ = rollout_of(extensions, timestep=250)
    seen = np.asarray(rollout.policy_observation)
    np.testing.assert_array_equal(seen[..., :3], rollout.obs)
    np.testing.assert_allclose(seen[..., 3], 0.25)
    np.testing.assert_allclose(seen[..., 4:], push(rollout.obs) / 2.0, rtol=1e-6)


def test_the_pid_terms_restart_at_each_episode_start():
    """(e, I, D) of e = obs[1] - obs[0], recomputed from the observations
    and the episode starts (every LIMIT steps): I sums e, D differences it,
    both afresh at a start."""
    _, _, rollout, _ = rollout_of((PIDTermsInput((0,), (1,)),))
    seen = np.asarray(rollout.policy_observation)[..., 3:]
    error = np.asarray(rollout.obs[..., 1] - rollout.obs[..., 0])
    integral, difference = np.zeros_like(error), np.zeros_like(error)
    for t in range(HORIZON):
        start = t % LIMIT == 0
        integral[t] = error[t] + (0.0 if start else integral[t - 1])
        difference[t] = 0.0 if start else error[t] - error[t - 1]
    np.testing.assert_array_equal(np.asarray(rollout.resets[::LIMIT]), True)
    np.testing.assert_allclose(seen[..., 0], error, rtol=1e-6)
    np.testing.assert_allclose(seen[..., 1], integral, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(seen[..., 2], difference, rtol=1e-5, atol=1e-6)


def test_the_return_differentiates_through_the_features():
    """The features keep their gradient: the expert's action and the PID
    terms move with the controller's parameters through the dynamics, and
    the gradient of the return and of each feature's sum is finite and
    nonzero."""
    extensions = (ExpertActionInput(), PIDTermsInput((0,), (1,)))
    agent, state, _, env_args = rollout_of(extensions)

    def sums(params):
        returns, rollout = compute_rollout_returns(
            state,
            params,
            jax.random.PRNGKey(1),
            env_args,
            None,
            HORIZON,
            N,
            expert_policy=agent.expert,
        )
        features = rollout.policy_observation
        return returns.sum(), features[..., 3].sum(), features[..., 4:].sum()

    for index, name in enumerate(("return", "expert action", "PID terms")):
        grads = jax.grad(lambda p, i=index: sums(p)[i])(state.actor_state.params)
        norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))))
        assert np.isfinite(norm) and norm > 0, name


@dataclass(frozen=True)
class SeesObservations(Extension):
    """Records the widths of the actor batch's observations, at trace time."""

    name: str = "sees_observations"

    def actor_loss(self, agent_state, ext_state, batch, context):
        widths = (batch["observations"].shape[-1], batch["raw_observations"].shape[-1])
        SEEN.append(widths)
        return 0.0, {}


SEEN: list = []


def test_the_actor_batch_holds_what_the_controller_read():
    """An actor-loss extension reads the observations with their features
    (Pendulum's 3 and the progress) and the env's own as raw_observations."""
    SEEN.clear()
    agent = APG(
        "Pendulum-v1",
        n_envs=2,
        horizon=4,
        actor_architecture=("8", "tanh"),
        extensions=(TrainingProgressInput(), SeesObservations()),
    )
    agent.train(seed=0, n_timesteps=8)
    assert set(SEEN) == {(4, 3)}
