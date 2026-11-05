"""PORL: clone an expert, warm up a fresh critic, fine-tune.

Contract, from Xiao et al. 2025 (arXiv 2505.16856), Algorithm 1, section 4 and
section 5.2, not from the code:

- Requires a pre-trained policy pi alone: the replay D starts empty and the
  critic random.
- Pre-sample stage, steps 1 to T: pi is frozen. Act epsilon-greedily around it
  (pi's action with probability 1 - epsilon, a random action with probability
  epsilon), store the row, then M TD updates of the critic. The stage holds no
  other update.
- From step T + 1: SAC. Act with pi; M TD updates, then a policy step, whose
  temperature step SAC takes with it.
- T = 0 is a pre-trained actor on a random critic (section 5.2).

Assumed where the paper is silent: the exploiting action is pi's mean (the
paper's "greedy", and its figure 9 compares deterministic sampling). The update
ratio M and the critics are the backbone's, not this arm's (the study's rule),
and Table 1's values must stay reachable by configuration.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import REDQ, SAC
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionContext
from ajax.extensions.expert import ResidualPolicy
from ajax.extensions.porl import EpsilonGreedyPresample, porl
from ajax.extensions.pretrain import CloningPretrain

SMALL: dict[str, Any] = {"actor_architecture": ("16", "relu")}
SMALL.update({"critic_architecture": ("16", "relu"), "n_envs": 2})
SMALL.update({"batch_size": 8, "buffer_size": 256})
UNIFORM_STD = 1.0 / 3.0**0.5  # of a uniform draw on [-1, 1]


def push(obs: jax.Array) -> jax.Array:
    """A constant expert: 0.5 in env units, 0.25 of Pendulum's action range."""
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def cloning() -> CloningPretrain:
    return CloningPretrain(push, n_steps=32, epochs=200, batch_size=16)


def agent(
    presample_steps: int, epsilon: float = 0.1, agent_class: Any = SAC, **kwargs: Any
) -> Any:
    arm = porl(cloning(), presample_steps, epsilon)
    return agent_class("Pendulum-v1", **{**SMALL, **arm, **kwargs})


def run(n_timesteps: int, *args: Any, **kwargs: Any) -> Any:
    """One seed's state after n_timesteps steps of agent(*args, **kwargs)."""
    state, _ = agent(*args, **kwargs).train(seed=0, n_timesteps=n_timesteps)
    return jax.tree.map(lambda leaf: leaf[0], state)


def same(left: Any, right: Any) -> bool:
    equal = jax.tree.map(lambda a, b: np.array_equal(a, b), left, right)
    return all(jax.tree.leaves(equal))


def test_the_actor_stays_frozen_through_the_presample_and_trains_after_it():
    """The gate is exact, at T = 8 and two envs: the fourth tick's update (its
    last row is pre-sampled) leaves the clone, and the fifth's steps it. A gate
    at T unfreezes the actor one tick early; none lets it drift off the clone
    while the critic is still random, which is what PORL exists to prevent."""
    first_tick = run(2, 8).actor_state.params
    last_presample_tick = run(8, 8).actor_state.params
    first_online_tick = run(10, 8).actor_state.params

    assert same(first_tick, last_presample_tick)
    assert not same(last_presample_tick, first_online_tick)


def test_the_presample_acts_on_the_clone():
    """The frozen actor is the expert's clone, not the actor as initialised."""
    state = run(8, 8)
    actor = state.actor_state
    obs = jnp.zeros((1, 3))
    assert get_pi(actor, actor.params, obs)[0].mean() == pytest.approx(0.25, abs=0.05)


def test_the_clones_actions_hold_through_the_presample():
    """Frozen means the policy, not the weights: the actor reads normalised
    observations, and rows of random actions would move the statistics under
    it, so the critic would warm up on a policy drifting away from the clone."""
    obs = jax.random.normal(jax.random.PRNGKey(0), (16, 3))
    actors = [run(n_timesteps, 8).actor_state for n_timesteps in (2, 8)]
    means = [get_pi(actor, actor.params, obs)[0].mean() for actor in actors]
    np.testing.assert_array_equal(*means)


def test_the_statistics_count_again_once_fine_tuning_starts():
    """Held past the stage they would stay the clone's for the whole run, on
    states the policy no longer visits."""
    held = run(8, 8).collector_state.obs_norm_info.count
    counting = run(10, 8).collector_state.obs_norm_info.count
    assert (counting > held).all()


def test_the_critic_takes_its_updates_from_the_first_step():
    """M updates on each step of the stage, on a critic nothing pretrained.
    Algorithm 1 has no learning_starts: a default one would leave the critic
    untrained through the whole warm-up, the failure PORL names."""
    state = run(8, 8, num_critic_updates=3)
    assert state.critic_state.step == 3 * 4  # M updates, four ticks of two envs
    assert not same(run(2, 8).critic_state.params, state.critic_state.params)


def test_the_temperature_waits_with_the_actor():
    """The stage's only update is the critic's: a temperature learning towards
    the frozen clone's entropy would start fine-tuning from another alpha."""
    assert same(run(2, 8).alpha.params, run(8, 8).alpha.params)
    assert not same(run(8, 8).alpha.params, run(10, 8).alpha.params)


def test_the_replay_holds_only_what_the_run_collected():
    """No dataset: the clone's rollouts do not seed D."""
    state = run(8, 8)
    assert state.collector_state.buffer_state.current_index == 8 // 2


def decide(epsilon: float, step: int, rows: int = 8192) -> tuple[jax.Array, ...]:
    """The extension's choice on rows of observations at step, then the
    clone's mean there: the stage's call, whose dict holds the agent's own
    action as a value no real action takes."""
    state = run(2, 8)
    keys = jax.random.split(jax.random.PRNGKey(0))
    observations = jax.random.normal(keys[0], (rows, 3))
    own_action = jnp.full((rows, 1), 7.0)
    context = ExtensionContext(step=jnp.asarray(step), rng=keys[1], total_steps=64)
    module = EpsilonGreedyPresample(8, epsilon)
    step_dict = {"observations": observations, "post_warmup_action": own_action}
    chosen = module.action(state, None, step_dict, keys[1], context)
    actor = state.actor_state
    return chosen, get_pi(actor, actor.params, observations)[0].mean()


def test_epsilon_is_the_share_of_random_actions_and_the_rest_are_the_clones():
    """An action sampled from the clone instead of its mean, or one draw for
    every env, fails here. Tolerance: 4 binomial standard errors on 8192 rows,
    and on the mean and standard deviation of the 819 random draws."""
    chosen, greedy = decide(0.1, step=0)

    random_rows = np.asarray(chosen != greedy)[:, 0]
    tolerance = 4 * (0.1 * 0.9 / random_rows.size) ** 0.5
    assert random_rows.mean() == pytest.approx(0.1, abs=tolerance)
    random_actions = np.asarray(chosen)[random_rows]
    standard_error = UNIFORM_STD / random_actions.size**0.5
    assert np.abs(random_actions).max() <= 1.0
    assert random_actions.mean() == pytest.approx(0.0, abs=4 * standard_error)
    assert random_actions.std() == pytest.approx(UNIFORM_STD, abs=8 * standard_error)


def test_no_action_is_random_at_epsilon_zero_and_all_are_at_one():
    """The two ends of the rule."""
    chosen, greedy = decide(0.0, step=0)
    np.testing.assert_array_equal(chosen, greedy)
    chosen, greedy = decide(1.0, step=0)
    assert np.asarray(chosen != greedy).all()


@pytest.mark.parametrize("step, is_presample", [(7, True), (8, False)])
def test_exploration_ends_with_the_presample(step: int, is_presample: bool):
    """Step T is the stage's last, counted from 1: the collector's step index
    T - 1. Exploring past it would noise every later row of a fine-tuned run."""
    chosen, _ = decide(0.1, step=step)
    assert (np.asarray(chosen) == 7.0).all() != is_presample


def test_the_replay_stores_the_presampled_actions():
    """On the real collection, each row against the clone's mean at its
    observation: at epsilon 0 they agree, and at 1 the rows are a uniform draw's.
    A module the stage bypasses leaves the policy's samples at both. Tolerance:
    each step acts with its own row counted, one more than the 64 held."""
    gaps, spreads = {}, {}
    for epsilon in (0.0, 1.0):
        state = run(64, 64, epsilon)
        rows = state.collector_state.buffer_state.experience
        actor = state.actor_state
        action = np.asarray(rows["action"])[:, : 64 // 2]
        mean = get_pi(actor, actor.params, rows["obs"][:, : 64 // 2])[0].mean()
        gaps[epsilon] = np.abs(action - np.asarray(mean)).max()
        spreads[epsilon] = action.std()

    assert gaps[0.0] < 0.05
    assert spreads[1.0] == pytest.approx(UNIFORM_STD, abs=0.1)


def test_zero_presample_steps_fine_tune_at_once():
    """Section 5.2's ablation: the stage gone, the actor trains from step 1."""
    assert not same(run(2, 0).actor_state.params, run(4, 0).actor_state.params)


@pytest.mark.parametrize("agent_class", [SAC, REDQ], ids=["SAC", "REDQ"])
def test_the_papers_settings_are_reachable_by_configuration(agent_class: Any):
    """The update ratio and the critics come from the backbone: PORL sets
    neither, and Table 1's run on it (REDQ's fresh batches and subset among the
    ways to read it)."""
    layer = ("16", "layernorm", "relu")
    state = run(
        4,
        2,
        agent_class=agent_class,
        critic_architecture=(*layer, *layer, *layer, *layer),
        num_critics=10,
        num_critic_updates=16,
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=1e-4,
    )
    critic_axes = {leaf.shape[0] for leaf in jax.tree.leaves(state.critic_state.params)}
    assert critic_axes == {10}
    assert state.critic_state.step == 16 * 2  # two ticks


@pytest.mark.parametrize("settings", [{"num_critics": 7}, {"num_critic_updates": 5}])
def test_the_arm_leaves_the_backbones_settings_alone(settings: dict[str, Any]):
    """A shared backbone's values survive the preset, whichever comes first."""
    name, value = next(iter(settings.items()))
    assert getattr(agent(8, **settings), name) == value
    assert name not in porl(cloning())


def test_an_action_transform_is_refused():
    """Under a residual the actor's mean is not what the step executes: the
    clone's actions would run through the residual, with no error."""
    arm = porl(cloning(), features=(ResidualPolicy(),))
    residual = SAC("Pendulum-v1", expert_policy=push, **{**SMALL, **arm})
    with pytest.raises(ValueError, match="no action transform"):
        residual.train(seed=0, n_timesteps=2)
