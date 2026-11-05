"""CHEQ, from its paper (Cramer et al. 2024, eq. 2, 4 and 5, Algorithm 2 and
appendix A.1), on the set-point task (tests/modules/setpoint_env.py), where
x' = x + B u reads the executed action back from a row: (x' - x) / B.

The contract: the executed action blends the expert's and the RL agent's by
lambda_t; lambda_t follows the critics' disagreement between clip limits; the
actor and the critics read it; the replay holds the RL action, not the blend,
and a row's next observation holds the row's lambda_t.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, REDQ, SAC
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionContext
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.cheq import (
    CHEQ,
    adapted_weight,
    blend,
    disagreement,
    interquartile_range,
    relative_uncertainty,
    row_masks,
)
from ajax.extensions.expert import JSRLCurriculum
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.observe import ExpertStateInput
from ajax.extensions.pretrain import MCPretrain
from ajax.extensions.target_mods import ValueBox
from tests.modules import setpoint_env
from tests.modules.setpoint_env import TERMS, B

SMALL = ("32", "relu", "32", "relu")
LENGTH = setpoint_env.LENGTH
KWARGS: dict[str, Any] = {
    "n_envs": 1,
    "batch_size": 8,
    "learning_starts": 20,
    "actor_architecture": SMALL,
    "critic_architecture": SMALL,
    "buffer_size": 1000,
}


def expert(obs: jax.Array) -> jax.Array:
    """Half the error, a prior of its own: (r - x) / 2 / B, deadbeat at 2."""
    return (obs[..., 1:] - obs[..., :1]) / (2.0 * B)


class StatefulExpert:
    """A prior with memory: it counts its calls."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return expert(obs), state + 1


def build(stack: tuple, policy: Any = expert, agent: Any = SAC, **kwargs: Any) -> Any:
    env, params = setpoint_env.make()
    return agent(
        env,
        env_params=params,
        expert_policy=policy,
        extensions=stack,
        **{**KWARGS, **kwargs},
    )


def train(agent: Any, n_timesteps: int) -> Any:
    return agent.train(seed=0, n_timesteps=n_timesteps)[0]


def one_seed(state: Any) -> Any:
    return jax.tree.map(lambda x: x[0], state)


def run(ext: CHEQ, n_timesteps: int, policy: Any = expert, **kwargs: Any) -> Any:
    agent = build((ext,), policy, **kwargs)
    return agent, one_seed(train(agent, n_timesteps))


def rows_of(state: Any) -> dict:
    replay = state.collector_state.buffer_state
    n = int(replay.current_index)
    return {k: np.asarray(v[0, :n]) for k, v in replay.experience.items()}


@pytest.mark.parametrize(
    ("u", "weight"),
    [
        (0.0, 1.0),
        (0.03, 1.0),  # at u_min, lambda_max
        (0.09, 0.6),  # halfway: 0.2 + 0.5 * 0.8
        (0.15, 0.2),  # at u_max, lambda_min
        (0.5, 0.2),
    ],
)
def test_the_weight_follows_eq_5(u: float, weight: float) -> None:
    """The paper's numbers (appendix A.1), by hand."""
    got = adapted_weight(jnp.asarray(u), 0.03, 0.15, 0.2, 1.0)
    np.testing.assert_allclose(got, weight, rtol=1e-5)


def test_more_disagreement_never_raises_the_weight() -> None:
    weights = adapted_weight(jnp.linspace(0.0, 1.0, 101), 0.03, 0.15, 0.2, 1.0)
    assert bool(jnp.all(jnp.diff(weights) <= 0.0))


@pytest.mark.parametrize(
    ("weight", "mix"), [(0.0, 0.4), (1.0, -0.8), (0.25, 0.75 * 0.4 + 0.25 * -0.8)]
)
def test_the_blend_is_eq_2(weight: float, mix: float) -> None:
    got = blend(jnp.array([0.4]), jnp.array([-0.8]), jnp.array([weight]))
    np.testing.assert_allclose(got, [mix], rtol=1e-6)


def test_disagreement_is_the_critics_standard_deviation() -> None:
    """Critics at 1 and 3 on a row, 2 and 6 on the next: eq. 4's population
    deviations are 1 and 2, a sample deviation would give sqrt(2) and
    2 sqrt(2)."""
    _, state = run(CHEQ(), 0)
    critic = state.critic_state.replace(
        apply_fn=lambda *_: jnp.array([[[1.0], [2.0]], [[3.0], [6.0]]]),
        obs_norm_info=None,
    )
    got = disagreement(critic, jnp.zeros((2, 3)), jnp.zeros((2, 1)))
    np.testing.assert_allclose(got, [[1.0], [2.0]])


def test_the_spread_means_the_same_at_any_reward_scale() -> None:
    """Rewards scaled by c and shifted move the bootstrapped values alike and
    the critics' disagreement by c alone: u / spread stays, so thresholds carry
    across tasks whose rewards differ by 1e8. A share of the mean would not."""
    targets = jnp.arange(256.0)[:, None]
    u = 12.0
    base = u / interquartile_range(targets)
    for scale, shift in ((1e-4, 0.0), (6e4, 0.0), (1.0, 1e3)):
        got = scale * u / interquartile_range(scale * targets + shift)
        np.testing.assert_allclose(got, base, rtol=1e-3)


def test_one_trip_in_a_batch_leaves_the_spread() -> None:
    """A trip costs up to 1e12 once: a deviation-based scale would jump by
    1e10 for 1 row of 256, the weight then sitting at lambda_max until it
    decayed. The middle half of the rows does not see it."""
    targets = jnp.arange(256.0)[:, None]
    tripped = targets.at[0].set(-1e12)
    spread, moved = interquartile_range(targets), interquartile_range(tripped)
    np.testing.assert_allclose(moved, spread, rtol=0.01)


def test_the_spread_tracks_what_the_targets_bootstrap() -> None:
    """Values 0 to 4 have quartiles 1 and 3, a spread of 2, taken whole at the
    first target. Times 3 and shifted by 100 their spread is 6, and rate 0.5
    tracks it to 2 + 0.5 * (6 - 2) = 4. A scale on the mean would read the
    shift; the target itself passes through."""
    ext = CHEQ(scale_rate=0.5)
    context = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    state = ext.init_state(None, jax.random.PRNGKey(0))
    values, target = jnp.arange(5.0)[:, None], jnp.ones((5, 1))
    same, _, state = ext.on_target(
        None, state, {"next_values": values}, target, context
    )
    np.testing.assert_array_equal(same, target)
    np.testing.assert_allclose(state.value_spread, 2.0, rtol=1e-6)
    batch = {"next_values": 3.0 * values + 100.0}
    _, _, state = ext.on_target(None, state, batch, target, context)
    np.testing.assert_allclose(state.value_spread, 4.0, rtol=1e-5)


def test_masks_are_a_rows_own_and_each_critics_own() -> None:
    """Algorithm 2 stores m_t^e with the row: a resampled mask would average
    out over updates and leave the critics fitting one dataset. kappa = 0.8
    over 4 critics and 512 rows: standard error 0.01, so 0.05 is 5 of them."""
    key = jax.random.PRNGKey(0)
    rows = jax.random.normal(jax.random.PRNGKey(1), (512, 4))
    masks = row_masks(key, rows, 4, 0.8)
    assert masks.shape == (4, 512, 1)
    np.testing.assert_array_equal(masks, row_masks(key, rows, 4, 0.8))
    np.testing.assert_array_equal(masks[:, :8], row_masks(key, rows[:8], 4, 0.8))
    assert abs(float(masks.mean()) - 0.8) < 0.05
    assert bool(jnp.any(masks[0] != masks[1]))


def test_a_masked_row_leaves_its_critics_loss() -> None:
    """With the term the extension adds, the agent's mean squared error sums
    only the rows a critic keeps. Errors 1 on every row: the loss is the kept
    share of (critic, row) pairs, and it is a mean over all of them."""
    _, state = run(CHEQ(mask_rate=0.5), 0)
    ext = CHEQ(mask_rate=0.5)
    observations, actions = jnp.ones((16, 3)), jnp.ones((16, 1))
    observations = observations * jnp.arange(16.0)[:, None]
    critic = state.critic_state.replace(
        apply_fn=lambda *_: jnp.ones((2, 16, 1)), obs_norm_info=None
    )
    ext_state = state.ext_state[0]
    term = ext.critic_loss(
        state,
        ext_state,
        {
            "observations": observations,
            "actions": actions,
            "critic_state": critic,
            "critic_params": critic.params,
            "targets": jnp.zeros((16, 1)),
        },
        ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0)),
    )
    x = jnp.concatenate([observations, actions], axis=-1)
    kept = row_masks(ext_state.mask_key, x, 2, 0.5)
    np.testing.assert_allclose(1.0 + term, jnp.mean(kept), rtol=1e-6)


def test_the_executed_action_is_the_blend_and_the_replay_holds_the_rl_action() -> None:
    """The heart of the contextual MDP (Algorithm 2's D <- (s, a_RL, lambda,
    s', r)): on every row, warm-up included, the plant moved by the blend of
    the expert's action and the stored one, at the row's weight."""
    _, state = run(CHEQ(adaptation_start=0), 40)
    rows = rows_of(state)
    x, r, weight = rows["obs"][:, 0], rows["obs"][:, 1], rows["obs"][:, 2]
    executed = (rows["next_obs"][:, 0] - x) / B
    prior = (r - x) / (2.0 * B)
    want = (1.0 - weight) * prior + weight * rows["action"][:, 0]
    np.testing.assert_allclose(executed, want, rtol=1e-4, atol=1e-5)
    # What the rows hold is not the executed action: the weight is below 1.
    assert np.abs(rows["action"][:, 0] - executed).max() > 0.1


def test_warm_up_rows_hold_the_uniform_draw_not_a_sample_of_the_policy() -> None:
    """Their warm-up samples a_RL uniformly. Each row must hold that draw: the
    untrained actor's sample, stored in its place, would pair the row with an
    action that never entered the blend. So the warm-up's rows, read from the
    plant as above, obey the blend with what they hold."""
    _, state = run(CHEQ(), 3 * LENGTH, learning_starts=200)
    rows = rows_of(state)
    ends = rows["truncated"][:, 0] > 0
    x, r, weight = rows["obs"][:, 0], rows["obs"][:, 1], rows["obs"][:, 2]
    executed = (rows["next_obs"][:, 0] - x) / B
    want = (1.0 - weight) * (r - x) / (2.0 * B) + weight * rows["action"][:, 0]
    assert ends.sum() == 3
    np.testing.assert_allclose(executed, want, rtol=1e-4, atol=1e-5)


def test_weights_start_at_lambda_min_and_the_warm_up_keeps_them_in_its_band() -> None:
    """Algorithm 2's lambda_0 = lambda_min, appendix A.1's [0.2, 0.3] before
    adaptation: every row's weight stays there and episodes differ."""
    _, state = run(CHEQ(), 3 * LENGTH, learning_starts=200)
    weight = rows_of(state)["obs"][:, 2]
    assert weight.min() >= 0.2 - 1e-6 and weight.max() <= 0.3 + 1e-6
    np.testing.assert_allclose(weight[0], 0.2, rtol=1e-6)
    by_episode = weight.reshape(3, LENGTH)
    np.testing.assert_allclose(by_episode.std(axis=1), 0.0, atol=1e-6)
    assert len(set(np.round(by_episode[:, 0], 5))) == 3


def test_a_rows_next_observation_holds_the_rows_weight() -> None:
    """Algorithm 2 bootstraps at (s', lambda) with the sampled lambda, and
    their code injects the same weight in both: the critic's target must not
    read lambda_{t+1}, which the row's action did not run under."""
    _, state = run(CHEQ(adaptation_start=0, u_low=0.0, u_high=1.0), 60)
    rows = rows_of(state)
    # Past the warm-up, an episode's weights move: the check below has teeth.
    assert np.ptp(rows["obs"][20:30, 2]) > 1e-3
    np.testing.assert_allclose(rows["obs"][:, 2], rows["next_obs"][:, 2], rtol=1e-6)


def test_the_weight_follows_the_critics_after_each_update() -> None:
    """lambda_{t+1} = Lambda(u(s_t, a_RL, lambda_t)): the next observation
    holds eq. 5 of the newest row, at the disagreement as a share of the
    tracked spread, on the updated critics."""
    ext = CHEQ(adaptation_start=0, u_low=0.0, u_high=1.0)
    _, state = run(ext, 40 + LENGTH // 2)
    rows, collector = rows_of(state), state.collector_state
    critic = state.critic_state
    u = relative_uncertainty(
        critic, state.ext_state[0], rows["obs"][-1:], rows["action"][-1:]
    )
    want = adapted_weight(u, 0.0, 1.0, 0.2, 1.0)
    np.testing.assert_allclose(collector.last_obs[0, 2], want[0, 0], rtol=1e-5)
    assert float(state.ext_state[0].value_spread) > 0.0
    assert rows["obs"][:, 2].max() > 0.2 + 1e-3  # it left lambda_min


def test_the_logged_weight_reads_no_episodes_first_step() -> None:
    """A run's evaluations often land on an episode's end, where the next
    observation holds lambda_min: a log of that column reports lambda_min
    whatever the weights were. This run ends so, on weights above it."""
    ext = CHEQ(adaptation_start=0, u_low=0.0, u_high=1.0, scale_rate=0.5)
    _, state = run(ext, 40)
    np.testing.assert_allclose(state.collector_state.last_obs[0, 2], 0.2, rtol=1e-5)
    context = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0))
    logged = ext.eval_metrics(state, state.ext_state[0], context.rng, context)
    assert float(logged["cheq/weight"]) > 0.2 + 1e-3


def test_a_new_episode_starts_at_lambda_min_once_adapting() -> None:
    """Algorithm 2 restarts lambda at each episode; so do their schedulers
    (episode_weight_reset): an uncertain start must lean on the prior."""
    ext = CHEQ(adaptation_start=0, u_low=0.0, u_high=1e6)
    _, state = run(ext, 40)
    obs = rows_of(state)["obs"]
    starts = obs[::LENGTH, 2]
    np.testing.assert_allclose(starts, 0.2, rtol=1e-5)
    assert obs[21 : LENGTH * 3, 2].max() > 0.2 + 1e-3


OBS = jnp.array([[0.5, -0.25], [-0.75, 0.5]])


def evaluated_action(agent: Any, state: Any) -> jax.Array:
    """What the deployed policy executes at OBS, as evaluation acts."""
    policy = agent.policy(state)
    return policy.act(OBS, policy.reset(2))[0]


def actor_mean(actor: Any, weight: jax.Array) -> jax.Array:
    observed = jnp.concatenate([OBS, weight], axis=-1)
    return get_pi(actor, actor.params, observed)[0].mean()


def test_at_weight_0_evaluation_executes_the_expert() -> None:
    ext = CHEQ(lambda_min=0.0, lambda_max=0.0, warmup_lambda_max=0.0)
    agent = build((ext,))
    state = train(agent, 24)
    np.testing.assert_allclose(evaluated_action(agent, state), expert(OBS), rtol=1e-5)


def test_at_weight_1_evaluation_executes_the_actor_on_its_own_column() -> None:
    """Their weight 1 is the RL agent in full control, and an actor-only score
    reads the same column, so it does not ask the actor about a blend."""
    ext = CHEQ(lambda_min=1.0, lambda_max=1.0, warmup_lambda_max=1.0)
    agent = build((ext,))
    state = train(agent, 24)
    actor = one_seed(state).actor_state
    want = actor_mean(actor, jnp.ones((2, 1)))
    np.testing.assert_allclose(evaluated_action(agent, state), want, rtol=1e-5)


def test_evaluation_blends_at_the_weight_the_critics_give_at_this_state() -> None:
    """Their evaluation adapts the weight as training does. It reads eq. 4 and
    5 at the actor's action on its own column, then the actor at that weight."""
    ext = CHEQ(u_low=0.0, u_high=1.0, adaptation_start=0)
    agent = build((ext,))
    state = train(agent, 24)
    seed = one_seed(state)
    actor, critic = seed.actor_state, seed.critic_state
    first = actor_mean(actor, jnp.ones((2, 1)))
    probe = jnp.concatenate([OBS, jnp.ones((2, 1))], axis=-1)
    u = relative_uncertainty(critic, seed.ext_state[0], probe, first)
    weight = adapted_weight(u, 0.0, 1.0, 0.2, 1.0)
    assert 0.2 < float(weight.min()) and float(weight.max()) < 1.0
    want = blend(expert(OBS), actor_mean(actor, weight), weight)
    np.testing.assert_allclose(evaluated_action(agent, state), want, rtol=1e-4)


def test_a_prior_with_memory_needs_it_in_the_observation() -> None:
    """The paper assumes a prior without internal state: an integrator it does
    not observe makes the same (s, a_RL, lambda) lead to different blends."""
    with pytest.raises(ValueError, match="PIDTermsInput or ExpertStateInput"):
        run(CHEQ(), 0, policy=StatefulExpert())
    for feature in (TERMS, ExpertStateInput.for_expert(StatefulExpert())):
        agent = build((CHEQ(), feature), policy=StatefulExpert())
        rows = one_seed(train(agent, 4)).collector_state.buffer_state
        width = 2 + 1 + feature.observation_width(None)
        assert rows.experience["obs"].shape[-1] == width


def test_the_weight_column_is_found_among_other_features() -> None:
    """Read from a PID column the weight would blend by an integral: rows of
    the two features together must still obey the blend."""
    agent = build((TERMS, CHEQ()), n_envs=1)
    rows = rows_of(one_seed(train(agent, 8)))
    columns = rows["obs"].shape[1]
    weight = rows["obs"][:, columns - 1]
    assert ((weight >= 0.2 - 1e-6) & (weight <= 0.3 + 1e-6)).all()
    x, r = rows["obs"][:, 0], rows["obs"][:, 1]
    executed = (rows["next_obs"][:, 0] - x) / B
    want = (1.0 - weight) * (r - x) / (2.0 * B) + weight * rows["action"][:, 0]
    np.testing.assert_allclose(executed, want, rtol=1e-4, atol=1e-5)


def test_it_refuses_what_it_cannot_run_on() -> None:
    env, params = setpoint_env.make()
    with pytest.raises(MissingCapability, match="replay"):
        PPO(env, env_params=params, expert_policy=expert, extensions=(CHEQ(),))
    with pytest.raises(MissingCapability, match="critic_ensemble"):
        build((CHEQ(),), num_critics=1)
    with pytest.raises(ValueError, match="lambda_min"):
        CHEQ(lambda_min=0.9, lambda_max=0.5)
    with pytest.raises(ValueError, match="u_low"):
        CHEQ(u_low=0.2, u_high=0.1)


@pytest.mark.parametrize(
    "stack",
    [
        (JSRLCurriculum(episode_length=LENGTH, decay_frac=0.5),),
        (EDGEExploration(gate="fixed", fixed_prob=0.5, decay_frac=1.0),),
        (MCPretrain(expert, 200, 4, 20, online_light_steps=5), ValueBox()),
    ],
    ids=["JSRL", "EDGE", "ValueBox"],
)
def test_the_other_acting_extensions_still_store_what_they_execute(stack) -> None:
    """Without replay_action a row holds the executed action, read from the
    plant, whoever chose it: the stage's new key reaches CHEQ's rows alone."""
    agent = build(stack, learning_starts=8, n_envs=1)
    rows = rows_of(one_seed(train(agent, 2 * LENGTH)))
    executed = (rows["next_obs"][:, 0] - rows["obs"][:, 0]) / B
    np.testing.assert_allclose(rows["action"][:, 0], executed, rtol=1e-4, atol=1e-5)


def test_redq_runs_the_papers_ensemble() -> None:
    """Five critics, a random pair in the target and the critics' mean in the
    actor are REDQ's own: the module adds the weight, unchanged."""
    agent = build((CHEQ(adaptation_start=0),), agent=REDQ, num_critics=5, subset_size=2)
    rows = rows_of(one_seed(train(agent, 24)))
    assert rows["obs"].shape == (24, 3)
