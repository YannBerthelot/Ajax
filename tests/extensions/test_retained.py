"""RLPD and retained demonstrations on a replay agent.

Contract, from the papers: RLPD (Ball et al. 2023) keeps the logged rows in a
buffer of their own, unchanged, draws half of each batch from it, runs on
LayerNorm critics and may bootstrap on a random subset Z of its target critics
(Algorithm 1, lines 12-16); DDPGfD (Vecerik et al. 2017) keeps its
demonstrations forever. The update ratio and the critics stay the agent's.
On the identity task (tests/environments/identity_env.py) the reward is the
executed action, so a row reads its own action back.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import REDQ, SAC, TD3
from ajax.buffers import dataset as ds
from ajax.buffers.utils import TwoBufferState
from ajax.checkpoint import load_checkpoint, save_checkpoint
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.observe import PIDTermsInput
from ajax.extensions.pretrain import CloningPretrain
from ajax.extensions.retained import RLPD, RetainedDemonstrations, bootstrap_on_subset
from tests.environments.identity_env import IdentityEnv

LAYER_NORM = ("32", "layernorm", "relu", "32", "layernorm", "relu")
SMALL = ("32", "relu", "32", "relu")
KWARGS: dict[str, Any] = {"n_envs": 1, "batch_size": 8, "learning_starts": 16}


def expert(obs: jax.Array) -> jax.Array:
    """Acts c, the observation's second column, in the env's units."""
    return obs[..., 1:]


@pytest.fixture(scope="module")
def logged() -> ds.TransitionDataset:
    env_args = TD3(IdentityEnv(), n_envs=1).env_args
    return ds.record(expert, env_args, jax.random.PRNGKey(0), 24)


def train(ext: Any, n_timesteps: int = 0, agent: Any = SAC, **kwargs: Any) -> Any:
    kwargs = {**KWARGS, "critic_architecture": LAYER_NORM, **kwargs}
    run = agent(IdentityEnv(), extensions=(ext,), **kwargs)
    return jax.tree.map(lambda x: x[0], run.train(seed=0, n_timesteps=n_timesteps)[0])


def test_logged_rows_stay_as_logged_in_a_buffer_of_their_own(logged) -> None:
    """RLPD changes no logged tuple, and its rows take no slot of the online
    buffer, whose first write is the agent's own row."""
    replay = train(RLPD(0, dataset=logged)).collector_state.buffer_state
    n = len(logged.reward)
    assert isinstance(replay, TwoBufferState)
    assert int(replay.current_index) == 0
    assert int(replay.retained.current_index) == n
    kept = replay.retained.experience
    np.testing.assert_array_equal(kept["obs"][0, :n], logged.obs)
    np.testing.assert_array_equal(kept["action"][0, :n], logged.action)
    np.testing.assert_array_equal(kept["reward"][0, :n, 0], logged.reward)


def test_the_expert_fills_the_buffer_with_the_runs_features() -> None:
    """The study's actors read the PID terms: rows from the agent's expert
    hold them, as its own rows do, or the critics would read another input."""
    terms = PIDTermsInput(current_idx=(0,), target_idx=(1,))
    agent = SAC(
        IdentityEnv(),
        expert_policy=expert,
        extensions=(RLPD(n_steps=12), terms),
        critic_architecture=LAYER_NORM,
        **KWARGS,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    kept = state.collector_state.buffer_state.retained
    assert int(kept.current_index) == 12
    assert kept.experience["obs"].shape[-1] == 2 + 3
    # The error c - t of the row's own observation, feature column 0.
    obs = kept.experience["obs"][0, :12]
    np.testing.assert_allclose(obs[:, 2], obs[:, 1] - obs[:, 0], rtol=1e-6)
    # The expert acted c on Box(-2, 2): in [-1, 1] units, c / 2.
    np.testing.assert_allclose(
        kept.experience["action"][0, :12, 0], obs[:, 1] / 2, rtol=1e-6
    )


def test_the_run_keeps_the_rows_while_its_own_arrive(logged) -> None:
    """Training past the online ring's length evicts the agent's rows only."""
    n = len(logged.reward)
    state = train(RLPD(0, dataset=logged), 200, buffer_size=64, learning_starts=300)
    replay = state.collector_state.buffer_state
    assert bool(replay.is_full)
    np.testing.assert_array_equal(
        replay.retained.experience["reward"][0, :n, 0], logged.reward
    )


def test_a_seeded_critic_moves_on_both_kinds_of_row(logged) -> None:
    """An update reads the mixed batch: with share 1 only logged rows reach
    it, so its target differs from share 0's on the same keys."""
    only = train(RLPD(0, dataset=logged, share=1.0), 24)
    none = train(RLPD(0, dataset=logged, share=0.0), 24)
    moved = jax.tree.leaves(
        jax.tree.map(jnp.not_equal, only.critic_state.params, none.critic_state.params)
    )
    assert any(bool(jnp.any(leaf)) for leaf in moved)


def test_the_method_fixes_neither_ratio_nor_ensemble(logged) -> None:
    """The backbone's own settings reach the run, the paper's included."""
    state = train(
        RLPD(0, dataset=logged, critic_subset=2),
        20,
        num_critics=10,
        num_critic_updates=20,
    )
    assert jax.tree.leaves(state.critic_state.params)[0].shape[0] == 10
    # The first target update is its own proof the 20 steps ran: Adam's count.
    counts = [
        int(np.max(x))
        for x in jax.tree.leaves(state.critic_state.opt_state)
        if np.ndim(x) == 0 or np.size(x) == 1
    ]
    assert max(counts) == 20 * 5  # steps 16 to 20: five updates


def test_redq_runs_the_original_configuration(logged) -> None:
    """Ten LayerNorm critics, a subset of two, 20 steps: REDQ draws a fresh
    batch per step and reads their mean in the actor, as RLPD's code does."""
    state = train(
        RLPD(0, dataset=logged),
        20,
        agent=REDQ,
        num_critics=10,
        subset_size=2,
        num_critic_updates=20,
    )
    assert isinstance(state.collector_state.buffer_state, TwoBufferState)


def test_critics_without_layer_norm_are_refused_unless_asked(logged) -> None:
    """RLPD diverged without LayerNorm on narrow data (their figure 7): an
    arm that forgets it is another method, so it stops at the start."""
    with pytest.raises(ValueError, match="LayerNorm"):
        train(RLPD(0, dataset=logged), critic_architecture=SMALL)
    train(RLPD(0, dataset=logged, critic_layer_norm=False), critic_architecture=SMALL)


def test_a_subset_is_refused_where_the_agent_has_its_own(logged) -> None:
    with pytest.raises(ValueError, match="subset_size"):
        train(
            RLPD(0, dataset=logged, critic_subset=2),
            agent=REDQ,
            num_critics=4,
            subset_size=2,
        )
    with pytest.raises(ValueError, match="of 2 critics"):
        train(RLPD(0, dataset=logged, critic_subset=3))


def test_bootstrap_on_subset_is_the_papers_target() -> None:
    """y = r + gamma (min_{i in Z} Q_i(s', a') - alpha log pi), by hand, from
    SAC's y over all critics. Critics 2, 5, 3 at one row and 4, 1, 6 at the
    next, r = 1, alpha log pi = 0, gamma = 0.5. Over all critics y = 1 + 0.5
    * 2 = 2 and 1 + 0.5 * 1 = 1.5. Over Z = {0, 2}, min = 2 and 4, so y = 2
    and 3; over a terminal row y stays r."""
    next_values = jnp.array([[[2.0], [4.0]], [[5.0], [1.0]], [[3.0], [6.0]]])
    target = jnp.array([[2.0], [1.5]])
    dones = jnp.zeros((2, 1))
    subset = jnp.array([0, 2])
    got = bootstrap_on_subset(target, next_values, subset, 0.5, dones)
    np.testing.assert_allclose(got, [[2.0], [3.0]])
    cut = bootstrap_on_subset(jnp.ones((2, 1)), next_values, subset, 0.5, dones + 1)
    np.testing.assert_allclose(cut, jnp.ones((2, 1)))


def test_the_subset_moves_the_target_only_when_set(logged) -> None:
    """A subset of every critic is all of them: the on_target pass leaves the
    run where SAC's own target puts it, up to the extra key it draws."""
    all_of_them = train(RLPD(0, dataset=logged, critic_subset=2), 24)
    none = train(RLPD(0, dataset=logged), 24)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        all_of_them.critic_state.params,
        none.critic_state.params,
    )


def test_a_resumed_run_keeps_the_retained_rows(logged, tmp_path) -> None:
    """A checkpoint holds both rings: losing the retained one would resume as
    plain SAC with no error, half of every batch gone from the method."""
    agent = SAC(
        IdentityEnv(),
        extensions=(RLPD(0, dataset=logged),),
        critic_architecture=LAYER_NORM,
        **KWARGS,
    )
    path = str(tmp_path / "run.pkl")
    save_checkpoint(agent.train(seed=[0], n_timesteps=20)[0], path)
    state = load_checkpoint(agent, path)
    resumed = agent.train(seed=[0], n_timesteps=24, initial_state=state)[0]
    replay = jax.tree.map(lambda x: x[0], resumed).collector_state.buffer_state
    assert isinstance(replay, TwoBufferState)
    n = len(logged.reward)
    np.testing.assert_array_equal(
        replay.retained.experience["reward"][0, :n, 0], logged.reward
    )


def test_one_source_of_rows_is_asked_for(logged) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RLPD(0)
    with pytest.raises(ValueError, match="exactly one"):
        RetainedDemonstrations(8, dataset=logged)
    with pytest.raises(ValueError, match="part of a batch"):
        RLPD(8, share=1.5)


def test_demonstrations_are_drawn_over_all_rows_by_default() -> None:
    """Without priorities a demonstration weighs as much as any row, so the
    demonstrations' share is theirs of the rows held; 0.5 pins Rainbow-
    DemoRL's prefill."""
    assert RetainedDemonstrations(8).share is None
    assert RLPD(8).share == 0.5
    agent = SAC(
        IdentityEnv(),
        expert_policy=expert,
        extensions=(RetainedDemonstrations(8),),
        critic_architecture=SMALL,
        **KWARGS,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    replay = state.collector_state.buffer_state
    assert replay.share is None and int(replay.retained.current_index) == 8


def test_cloning_then_prefill_is_rainbows_baseline() -> None:
    """The BC start plus prefill arm is a stack of two modules: the clone
    fits the actor, the demonstrations stay for the run."""
    stack = (
        CloningPretrain(expert_policy=expert, n_steps=12, epochs=1, batch_size=4),
        RetainedDemonstrations(8, share=0.5),
    )
    agent = SAC(
        IdentityEnv(),
        expert_policy=expert,
        extensions=stack,
        critic_architecture=SMALL,
        **KWARGS,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=24)[0])
    assert int(state.collector_state.buffer_state.retained.current_index) == 8


def test_an_agent_without_a_replay_refuses_them(logged) -> None:
    from ajax import PPO

    with pytest.raises(MissingCapability, match="replay"):
        PPO(IdentityEnv(), extensions=(RLPD(0, dataset=logged),))


def test_two_rings_of_retained_rows_are_refused(logged) -> None:
    """A second store would silently replace the first one's rows."""
    agent = SAC(
        IdentityEnv(),
        extensions=(
            RLPD(0, dataset=logged, critic_layer_norm=False),
            RetainedDemonstrations(0, dataset=logged, name="second"),
        ),
        critic_architecture=SMALL,
        **KWARGS,
    )
    with pytest.raises(ValueError, match="one ring"):
        agent.train(seed=0, n_timesteps=0)
