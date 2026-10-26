"""UDRL's collector wired as the shared one is: each observation of its
segments carries the observation features before the command, and the
action stage runs on its steps, the segment keeping the choice the actor
fits (the expert's own action, or a residual of 0, where the stage handed
the expert the step)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import UDRL
from ajax.agents.UDRL.UDRL import UDRLState
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy, StorePolicyAction
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.observe import (
    ExpertActionInput,
    PIDTermsInput,
    TrainingProgressInput,
)

LIMIT, STEPS, N_ENVS = 5, 16, 2  # Pendulum's episode length here, segment steps
STARTS = np.arange(STEPS) % LIMIT == 0  # Pendulum's episode starts
#: The expert acts the first step of each episode, the policy the others.
FIRST_STEP = JSRLCurriculum(episode_length=1, decay_frac=1e6)


def torque(obs: jax.Array) -> jax.Array:
    """1 in Pendulum's units, half its bound: 0.5 in the agent's."""
    return jnp.ones(obs.shape[:-1] + (1,))


def push_toward_lean(obs: jax.Array) -> jax.Array:
    """CartPole's expert: push toward the pole's lean, an int (n, 1) column."""
    return (obs[..., 2:3] > 0).astype(jnp.int32)


def first_segment(env_id: str, total: int = STEPS, **kwargs) -> tuple:
    """UDRL trained for total steps per env (episodes of LIMIT steps on
    Pendulum): its first segment's (obs, actions, dones), each (STEPS,
    N_ENVS, ...)."""
    if env_id == "Pendulum-v1":
        params = UDRL(env_id).env_args.env_params
        kwargs["env_params"] = params.replace(max_steps_in_episode=LIMIT)
    agent = UDRL(
        env_id,
        n_envs=N_ENVS,
        n_steps=STEPS,
        batch_size=8,
        buffer_capacity=4,
        n_updates_per_iter=1,
        actor_architecture=("8", "tanh"),
        **kwargs,
    )
    state, _ = agent.train(seed=0, n_timesteps=total * N_ENVS)
    assert isinstance(state, UDRLState)
    buffer = jax.tree.map(lambda x: np.asarray(x[0]), state.buffer)
    return buffer.obs[0], buffer.actions[0], buffer.dones[0, ..., 0]


def test_a_segment_holds_the_progress_and_pid_terms_before_the_command():
    """[obs (3), progress, e, I, D, command (2)]: the progress at the
    timestep of the step that observed it (0 at the reset), as the shared
    collector's, and the PID terms of e = obs[1] - obs[0] recomputed from
    the observations, afresh at each episode start (every LIMIT steps)."""
    extensions = (PIDTermsInput((0,), (1,)), TrainingProgressInput())
    obs, _, dones = first_segment("Pendulum-v1", extensions=extensions)
    assert obs.shape[-1] == 3 + 1 + 3 + 2
    timesteps = np.maximum(np.arange(STEPS) - 1, 0) * N_ENVS
    np.testing.assert_allclose(
        obs[..., 3], np.stack([timesteps / (STEPS * N_ENVS)] * N_ENVS, -1)
    )
    error = obs[..., 1] - obs[..., 0]
    integral, difference = np.zeros_like(error), np.zeros_like(error)
    for t in range(STEPS):
        start = t % LIMIT == 0
        integral[t] = error[t] + (0.0 if start else integral[t - 1])
        difference[t] = 0.0 if start else error[t] - error[t - 1]
    np.testing.assert_array_equal(dones[LIMIT - 1 :: LIMIT], 1.0)
    np.testing.assert_allclose(obs[..., 4], error, rtol=1e-6)
    np.testing.assert_allclose(obs[..., 5], integral, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(obs[..., 6], difference, rtol=1e-5, atol=1e-6)


def test_a_discrete_experts_action_enters_one_hot():
    """On CartPole the expert's index at each observation, one-hot over the
    two actions, before the command."""
    obs, _, _ = first_segment(
        "CartPole-v1", expert_policy=push_toward_lean, extensions=(ExpertActionInput(),)
    )
    assert obs.shape[-1] == 4 + 2 + 2
    lean = np.asarray(push_toward_lean(obs[..., :4]))[..., 0]
    np.testing.assert_array_equal(obs[..., 4:6], np.eye(2)[lean])


def test_jsrl_hands_the_expert_each_episodes_first_step():
    """Pendulum: the segment keeps the expert's action, 0.5, at each episode
    start, the policy's samples elsewhere. CartPole: the expert's index at
    each start (after a done), in the segment's int column."""
    _, actions, _ = first_segment(
        "Pendulum-v1", expert_policy=torque, extensions=(FIRST_STEP,)
    )
    np.testing.assert_array_equal(actions[STARTS, :, 0], 0.5)
    assert (actions[~STARTS, :, 0] != 0.5).all()
    obs, actions, dones = first_segment(
        "CartPole-v1", expert_policy=push_toward_lean, extensions=(FIRST_STEP,)
    )
    starts = np.concatenate([np.ones((1, N_ENVS), bool), dones[:-1] > 0])
    lean = np.asarray(push_toward_lean(obs[..., :4]))[..., 0]
    assert actions.dtype == np.int32
    np.testing.assert_array_equal(actions[..., 0][starts], lean[starts])


def test_under_a_residual_the_segment_keeps_zero_where_the_expert_acted():
    """The residual executing the expert's action is 0."""
    extensions = (ResidualPolicy(), FIRST_STEP)
    _, actions, _ = first_segment(
        "Pendulum-v1", expert_policy=torque, extensions=extensions
    )
    np.testing.assert_array_equal(actions[STARTS, :, 0], 0.0)
    assert (actions[~STARTS, :, 0] != 0.0).all()


def test_store_policy_action_keeps_the_policys_own_choice():
    """The ablation: where JSRL handed the expert the step, the segment
    keeps the policy's sample, not the expert's action."""
    extensions = (FIRST_STEP, StorePolicyAction())
    _, actions, _ = first_segment(
        "Pendulum-v1", expert_policy=torque, extensions=extensions
    )
    assert (actions[..., 0] != 0.5).all()


@pytest.mark.parametrize("fixed_prob, share", [(0.0, 0.0), (1.0, 1.0)])
def test_edge_s_fixed_gate_hands_the_expert_its_share(fixed_prob, share):
    """EDGE's fixed gate reads no critic, so it runs beside UDRL's V(s)."""
    edge = EDGEExploration(gate="fixed", fixed_prob=fixed_prob, decay_frac=0.0)
    _, actions, _ = first_segment(
        "Pendulum-v1", total=2 * STEPS, expert_policy=torque, extensions=(edge,)
    )
    assert np.mean(actions[..., 0] == 0.5) == share
