"""TD-MPC2's episodic mode (``config.episodic``; deviation T29).

The latest code's support for terminating tasks
(``nicklashansen/tdmpc2@eece801``, as at ``e9f5932``): the termination head
and its loss, ``1 - terminated`` in the TD target, the termination-masked
value estimate of the planner, the variable-length episode replay, and a
training-level known answer on a terminating chain.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2
from ajax.agents.base import ActorCritic
from ajax.agents.TDMPC2 import core, planner
from ajax.agents.TDMPC2.buffer import EpisodicBuffer, EpisodicBufferState
from ajax.agents.TDMPC2.networks import make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.agents.TDMPC2.train_TDMPC2 import EpisodicSchedule
from ajax.checkpoint import restore_into, save_checkpoint, skeleton_of
from ajax.environments.row_collector import Row

from .common import random_heads, td_target_of
from .toy_envs import ChainEnv

EPISODIC = TDMPC2Config(
    latent_dim=16,
    enc_dim=32,
    mlp_dim=32,
    num_q=5,
    dropout=0.0,
    num_samples=64,
    num_elites=8,
    num_pi_trajs=6,
    episodic=True,
    termination_coef=0.7,
)
PAPER_ERA = EPISODIC.replace(episodic=False)
OBS, ACT, B = 5, 2, 8
GAMMA = 0.9


def _with_termination_bias(params: dict, bias: float) -> dict:
    """A head that predicts the same logit ``bias`` everywhere."""
    params = jax.tree_util.tree_map(lambda x: x, params)
    out = params["termination"]["out"]
    out["kernel"] = jnp.zeros_like(out["kernel"])
    out["bias"] = jnp.full_like(out["bias"], bias)
    return params


@pytest.fixture(scope="module")
def state() -> core.TDMPC2UpdateState:
    state = core.create_update_state(jax.random.PRNGKey(0), EPISODIC, OBS, ACT)
    wm = state.world_model_state
    # Non-zero reward and Q heads, so values and targets differ by sample.
    reward_key, q_key = jax.random.split(jax.random.PRNGKey(1))
    params = random_heads(wm.params, reward_key, q_key, 0.3)
    wm = wm.replace(params=params, target_params=params["q"])
    return state.replace(world_model_state=wm)


@pytest.fixture(scope="module")
def batch() -> core.TDMPC2Batch:
    k = jax.random.split(jax.random.PRNGKey(2), 4)
    h = EPISODIC.horizon
    return core.TDMPC2Batch(
        obs=jax.random.normal(k[0], (h + 1, B, OBS)),
        action=jax.random.uniform(k[1], (h, B, ACT), minval=-1, maxval=1),
        reward=jax.random.normal(k[2], (h, B)),
        terminated=(jax.random.uniform(k[3], (h, B)) < 0.3).astype(jnp.float32),
    )


@pytest.fixture(scope="module")
def noise() -> core.UpdateNoise:
    return core.draw_update_noise(jax.random.PRNGKey(3), EPISODIC, B, ACT)


# ------------------------------------------------------------------- the head


def test_the_termination_head_exists_only_in_the_episodic_mode():
    """``mlp([z], 2 * [mlp_dim], 1)`` with a ``N(0, 0.02)`` output layer, built
    only when episodic; every other parameter is the paper-era model's."""
    x, a = jnp.zeros((1, OBS)), jnp.zeros((1, ACT))
    key = jax.random.PRNGKey(4)
    on = make_world_model(EPISODIC).init(key, x, a)["params"]
    off = make_world_model(PAPER_ERA).init(key, x, a)["params"]
    assert set(on) == set(off) | {"termination"}
    jax.tree.map(np.testing.assert_array_equal, {k: on[k] for k in off}, off)
    head = on["termination"]
    assert head["out"]["kernel"].shape == (EPISODIC.mlp_dim, 1)
    assert 0.005 < float(jnp.std(head["out"]["kernel"])) < 0.05  # not zeroed
    np.testing.assert_array_equal(head["out"]["bias"], 0.0)
    assert set(head["trunk"]) == set(on["reward"]["trunk"])  # 2 hidden layers
    z = jnp.ones((3, 4, EPISODIC.latent_dim)) / 8
    logits = make_world_model(EPISODIC).apply(
        {"params": on}, z, method="termination_logits"
    )
    assert logits.shape == (3, 4)


# ------------------------------------------------------------------- the loss


def test_the_termination_loss_is_the_bce_at_the_rollout_latents(state, batch):
    """``mean_{H,B} BCE(T(z_{t+1}), terminated_t)`` on the dynamics' latents
    ``z_1 .. z_H``, weighted by ``termination_coef`` into the total, which is
    otherwise the paper-era loss; only the head's loss reaches its weights."""
    wm = state.world_model_state
    args = {
        "wm_apply": wm.apply_fn,
        "batch": batch,
        "next_z": jnp.zeros((EPISODIC.horizon, B, EPISODIC.latent_dim)),
        "td_targets": batch.reward,
        "dropout_key": None,
    }
    total, (terms, zs) = core.world_model_loss(wm.params, config=EPISODIC, **args)
    base, (base_terms, _) = core.world_model_loss(wm.params, config=PAPER_ERA, **args)
    logits = wm.apply_fn({"params": wm.params}, zs[1:], method="termination_logits")
    p = jax.nn.sigmoid(logits)
    t = batch.terminated
    bce = -jnp.mean(t * jnp.log(p) + (1 - t) * jnp.log(1 - p))
    np.testing.assert_allclose(terms["termination_loss"], bce, rtol=1e-5)
    np.testing.assert_allclose(total, base + 0.7 * bce, rtol=1e-5)
    assert set(terms) == set(base_terms) | {
        "termination_loss",
        "termination_rate",
        "termination_f1",
    }
    np.testing.assert_allclose(terms["termination_rate"], jnp.mean(t[-1]))

    def head_loss(head):
        params = {**wm.params, "termination": head}
        return core.world_model_loss(params, config=EPISODIC, **args)[0]

    grads = jax.grad(head_loss)(wm.params["termination"])
    assert all(np.any(np.asarray(g)) for g in jax.tree.leaves(grads))


def test_termination_statistics_are_the_latest_codes():
    logits = jnp.array([2.0, -1.0, 3.0, -2.0, 0.5])
    target = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    stats = core.termination_statistics(logits, target)
    # tp = 2, fn = 1, fp = 1: precision = recall = 2 / 3.
    np.testing.assert_allclose(stats["termination_rate"], 0.6)
    np.testing.assert_allclose(stats["termination_f1"], 2 / 3, rtol=1e-6)


# ------------------------------------------------------------------ TD target


def test_the_td_target_drops_the_bootstrap_after_a_termination(state, batch, noise):
    """``y = r + gamma (1 - terminated) Q``: the reward alone where the next
    state is terminal, the paper-era target elsewhere."""
    online_q = state.world_model_state.params["q"]

    def target(terminated):
        return td_target_of(
            state,
            batch,
            noise,
            EPISODIC,
            gamma=GAMMA,
            target_q=online_q,
            dropout_key=None,
            terminated=terminated,
        )[0]

    paper_era = target(None)
    assert not np.allclose(paper_era, batch.reward)
    np.testing.assert_array_equal(target(jnp.zeros_like(batch.reward)), paper_era)
    np.testing.assert_array_equal(target(jnp.ones_like(batch.reward)), batch.reward)
    t = batch.terminated
    np.testing.assert_array_equal(target(t), jnp.where(t == 1, batch.reward, paper_era))


def test_an_episodic_update_trains_the_head_and_masks_its_targets(state, batch, noise):
    """The update logs the termination terms, steps the head, and its TD
    targets read ``batch.terminated`` (a batch that terminates everywhere
    gives another value loss); the paper-era update ignores it."""
    step = jax.jit(
        lambda s, b, c: core.update(s, b, noise, config=c, gamma=GAMMA),
        static_argnums=2,
    )
    new, aux = step(state, batch, EPISODIC)
    assert {"termination_loss", "termination_rate", "termination_f1"} <= set(aux)
    head = state.world_model_state.params["termination"]
    moved = jax.tree.map(
        lambda a, b: bool(np.any(a != b)),
        new.world_model_state.params["termination"],
        head,
    )
    assert all(jax.tree.leaves(moved))
    everywhere = batch.replace(terminated=jnp.ones_like(batch.reward))
    _, ended = step(state, everywhere, EPISODIC)
    assert float(ended["value_loss"]) != float(aux["value_loss"])
    _, a = step(state, batch, PAPER_ERA)
    _, b = step(state, everywhere, PAPER_ERA)
    assert "termination_loss" not in a
    assert float(a["value_loss"]) == float(b["value_loss"])


def test_the_episodic_mode_is_single_task(state, batch):
    wm = state.world_model_state
    task = core.TaskContext(ids=jnp.zeros(B, jnp.int32), mask=jnp.ones((B, ACT)))
    with pytest.raises(ValueError, match="single-task"):
        core.world_model_loss(
            wm.params,
            wm_apply=wm.apply_fn,
            batch=batch,
            next_z=jnp.zeros((EPISODIC.horizon, B, EPISODIC.latent_dim)),
            td_targets=batch.reward,
            dropout_key=None,
            config=EPISODIC,
            task=task,
        )


# ------------------------------------------------------------------ planning


def _estimate(params, config, z, actions, noise):
    return planner.estimate_value(
        params[0],
        params[1],
        z,
        actions,
        noise.terminal_eps[0],
        noise.q_pair[0],
        noise.q_dropout[0],
        config=config,
        gamma=GAMMA,
    )


def test_the_value_estimate_stops_at_a_predicted_termination(state):
    """The latest ``_estimate_value``: ``G = sum_t gamma^t alive_t r_t +
    gamma^H alive_H Q`` with ``alive_{t+1} = alive_t [sigmoid(T(z_{t+1}))
    <= 0.5]``: the reward of the terminating step counts, later ones and
    the terminal Q do not. A head that never terminates gives the
    paper-era estimate; one that always does, the first reward."""
    wm, pi = state.world_model_state.params, state.actor_state.params
    n, h = EPISODIC.num_samples, EPISODIC.horizon
    k = jax.random.split(jax.random.PRNGKey(8), 3)
    z = jax.nn.softmax(jax.random.normal(k[0], (n, EPISODIC.latent_dim)))
    actions = jax.random.uniform(k[1], (h, n, ACT), minval=-1, maxval=1)
    noise = planner.draw_plan_noise(k[2], EPISODIC, ACT)
    apply = make_world_model(EPISODIC).apply

    never = _estimate(
        (_with_termination_bias(wm, -9.0), pi), EPISODIC, z, actions, noise
    )
    paper_era = _estimate((wm, pi), PAPER_ERA, z, actions, noise)
    np.testing.assert_array_equal(never, paper_era)
    always = _estimate(
        (_with_termination_bias(wm, 9.0), pi), EPISODIC, z, actions, noise
    )
    first = EPISODIC.two_hot.decode(
        apply({"params": wm}, z, actions[0], method="reward_logits")
    )
    np.testing.assert_allclose(always, first, rtol=1e-6)

    # A head that terminates some candidates at some steps, against the
    # latest code's loop written out step by step.
    mixed = jax.tree_util.tree_map(lambda x: x, wm)
    mixed["termination"]["out"]["kernel"] = 3.0 * jax.random.normal(
        jax.random.PRNGKey(9), mixed["termination"]["out"]["kernel"].shape
    )
    got = _estimate((mixed, pi), EPISODIC, z, actions, noise)
    full = _estimate((mixed, pi), PAPER_ERA, z, actions, noise)
    value, termination, latent, discount = 0.0, jnp.zeros(n), z, 1.0
    for t in range(h):
        reward = EPISODIC.two_hot.decode(
            apply({"params": mixed}, latent, actions[t], method="reward_logits")
        )
        latent = apply({"params": mixed}, latent, actions[t], method="next")
        value = value + discount * (1 - termination) * reward
        discount = discount * GAMMA
        p = jax.nn.sigmoid(
            apply({"params": mixed}, latent, method="termination_logits")
        )
        termination = jnp.minimum(termination + (p > 0.5), 1.0)
    alive = 1 - termination
    assert 0 < float(jnp.mean(alive)) < 1  # some candidates terminate
    # The terminal Q term is the paper-era estimate's, masked.
    terminal_q = (full - _rewards_only((mixed, pi), z, actions)) / discount
    np.testing.assert_allclose(
        got, value + discount * alive * terminal_q, rtol=1e-5, atol=1e-5
    )


def _rewards_only(params, z, actions):
    """The paper-era discounted reward sum, without the terminal Q."""
    apply = make_world_model(PAPER_ERA).apply
    value, discount = 0.0, 1.0
    for t in range(actions.shape[0]):
        reward = PAPER_ERA.two_hot.decode(
            apply({"params": params[0]}, z, actions[t], method="reward_logits")
        )
        z = apply({"params": params[0]}, z, actions[t], method="next")
        value, discount = value + discount * reward, discount * GAMMA
    return value


# -------------------------------------------------------------------- replay

# Per env, episodes as (steps, ends by a termination), then a running one.
# H = 2 needs episodes of at least 2 steps (3 rows).
EPISODES = (
    ((4, True), (1, True), (3, False), (1, None)),  # A, B (too short), C, D running
    ((6, False), (2, True), (2, None)),  # E (time limit), F, G running
)
TICKS = 13  # both envs: 5 + 2 + 4 + 2 and 7 + 3 + 3 rows


def _fill(buffer: EpisodicBuffer) -> EpisodicBufferState:
    """Write the ``EPISODES`` row by row; each row's observation is
    ``[env, episode, step]`` and its action ``10 * episode + step``."""
    rows = []
    for env, episodes in enumerate(EPISODES):
        out = []
        for episode, (steps, terminated) in enumerate(episodes):
            for step in range(steps + 1):
                ends = terminated is not None and step == steps
                out.append(([env, episode, step], ends, ends and terminated))
        rows.append(out[:TICKS])
    state = buffer.init()
    for tick in range(TICKS):
        columns, last, terminal = zip(
            *(rows[env][tick] for env in range(2)), strict=False
        )
        obs = jnp.asarray(columns, jnp.float32)
        row = Row(
            obs=obs,
            reward=obs[:, 2],
            is_first=obs[:, 2] == 0,
            is_last=jnp.asarray(last),
            is_terminal=jnp.asarray(terminal),
            action=(10 * obs[:, 1] + obs[:, 2])[:, None],
        )
        state = buffer.add(state, row, obs[:, 2].astype(jnp.int32), tick)
    return state


def _draws(buffer: EpisodicBuffer, n: int = 4000) -> core.TDMPC2Batch:
    state = _fill(buffer)
    return jax.jit(buffer.sample, static_argnums=(3, 4))(
        state, jax.random.PRNGKey(0), TICKS - 1, n, 2
    )


def test_the_replay_draws_whole_episodes_uniformly_then_a_uniform_offset():
    """The finished episodes of at least H + 1 rows (A, C, E, F: not the
    one-step B, not the running D and G) are drawn uniformly, then an offset
    uniformly inside each; a slice never leaves its episode, its actions and
    rewards are the reference's (``a_s .. a_{s+H-1}``, ``r_s .. r_{s+H-1}``)
    and ``terminated`` is 1 only on a terminating episode's last row, not
    after a time limit."""
    buffer = EpisodicBuffer.create(buffer_size=32, n_envs=2, obs_dim=3, action_dim=1)
    assert buffer.capacity == 16
    assert int(buffer.num_episodes(_fill(buffer), TICKS - 1, 2)) == 4
    batch = _draws(buffer)
    obs = np.asarray(batch.obs)  # [H + 1, n, 3]
    env, episode, step = obs[0].T
    assert np.all(obs[:, :, :2] == obs[:1, :, :2])  # one episode per slice
    assert np.all(obs[:, :, 2] - step == np.arange(3)[:, None])  # consecutive
    np.testing.assert_array_equal(
        batch.action[..., 0], 10 * obs[:-1, :, 1] + obs[:-1, :, 2]
    )
    np.testing.assert_array_equal(batch.reward, obs[1:, :, 2])  # r_{t} on row t + 1
    lengths = {(0, 0): 4, (0, 2): 3, (1, 0): 6, (1, 1): 2}
    terminating = {(0, 0), (1, 1)}
    keys = list(zip(env.astype(int), episode.astype(int), strict=False))
    counts = {k: keys.count(k) for k in lengths}
    assert set(keys) == set(lengths)
    assert all(abs(c - 1000) < 150 for c in counts.values()), counts
    for (e, ep), steps in lengths.items():
        mine = (env == e) & (episode == ep)
        offsets = np.bincount(step[mine].astype(int), minlength=steps - 1)
        assert len(offsets) == steps - 1  # offsets 0 .. steps - H
        expected = mine.sum() / (steps - 1)
        assert np.all(np.abs(offsets - expected) < 5 * np.sqrt(expected)), offsets
        ends = obs[-1, mine, 2] == steps
        terminated = np.asarray(batch.terminated)[:, mine]
        np.testing.assert_array_equal(terminated[-1], ends & ((e, ep) in terminating))
        assert not terminated[:-1].any()


def test_an_overwritten_episode_keeps_its_remaining_rows():
    """With 10 rows per env the ring holds ticks 3 .. 12: A keeps 2 rows and
    E 4 (rows 3 .. 6), so E's slices start at its rows 3 or 4 and A is no
    longer sampleable."""
    buffer = EpisodicBuffer(n_envs=2, capacity=10, obs_dim=3, action_dim=1)
    assert int(buffer.num_episodes(_fill(buffer), TICKS - 1, 2)) == 3  # C, E, F
    obs = np.asarray(_draws(buffer, 600).obs)
    env, episode, step = obs[0].T
    assert not np.any((env == 0) & (episode == 0))
    e = (env == 1) & (episode == 0)
    assert e.any() and set(step[e].astype(int)) == {3, 4}


@pytest.mark.parametrize("carried, capacity", [(13, 16), (20, 16), (16, 10)])
def test_adopted_rings_hold_and_draw_what_rings_written_from_the_start_do(
    carried, capacity
):
    """Rings carried to a resume at tick 13 (13, 20 or 16 rows per env)
    and adopted by rings of another length hold the rows, zeros included,
    and draw the slices of rings of that length written from the start."""
    rings = EpisodicBuffer(n_envs=2, capacity=capacity, obs_dim=3, action_dim=1)
    old = EpisodicBuffer(n_envs=2, capacity=carried, obs_dim=3, action_dim=1)
    adopted, written = rings.adopt(_fill(old), TICKS), _fill(rings)
    jax.tree.map(np.testing.assert_array_equal, adopted, written)
    for state in (adopted, written):
        batch = rings.sample(state, jax.random.PRNGKey(0), TICKS - 1, 64, 2)
        np.testing.assert_array_equal(batch.obs, _draws(rings, 64).obs)


def test_rings_that_lost_a_row_the_adopting_ones_read_are_refused():
    """Rings of 10 rows hold ticks 3 .. 12 only: rings of 16 resuming at
    tick 13 would read ticks 0 .. 2 too."""
    rings = EpisodicBuffer(n_envs=2, capacity=16, obs_dim=3, action_dim=1)
    old = EpisodicBuffer(n_envs=2, capacity=10, obs_dim=3, action_dim=1)
    with pytest.raises(ValueError, match="smaller buffer_size"):
        rings.adopt(_fill(old), TICKS)


# ------------------------------------------------------------ the whole agent


def test_the_episodic_schedule_counts_ticks_and_stepped_envs():
    """Random actions up to the seed tick ``max(S div n, T)``, the burst of
    S on it, then one update per env the tick stepped; ``n_timesteps``
    counts rows."""
    schedule = EpisodicSchedule(n_envs=4, episode_length=10, seed_steps=100)
    assert schedule.seed_tick == 25 and schedule.num_ticks(400) == 100
    ticks = np.arange(40)
    np.testing.assert_array_equal(schedule.random_phase(ticks), ticks <= 25)
    got = np.asarray(schedule.n_updates(ticks, 3))
    np.testing.assert_array_equal(got, np.where(ticks > 25, 3, 0) + 100 * (ticks == 25))
    assert (
        EpisodicSchedule(n_envs=16, episode_length=10, seed_steps=100).seed_tick == 10
    )
    with pytest.raises(ValueError, match="n_envs"):
        EpisodicSchedule(n_envs=0, episode_length=10, seed_steps=100)


@pytest.mark.parametrize(
    "ratio, per_stepped",
    [
        (1.0, [0, 1, 2, 3, 4]),  # one per env step
        (0.5, [0, 1, 1, 2, 2]),  # at least one; round(1.5) = 2, round(2.5) = 2
        (2.0, [0, 2, 4, 6, 8]),
    ],
)
def test_the_episodic_schedule_honours_updates_per_step(ratio, per_stepped):
    """``max(1, round(stepped r))`` updates on a tick that stepped
    ``stepped > 0`` envs (T28), none on a tick that stepped none; the
    burst and the seed phase are unchanged."""
    schedule = EpisodicSchedule(
        n_envs=4, episode_length=10, seed_steps=100, updates_per_step=ratio
    )
    stepped = jnp.arange(5)
    np.testing.assert_array_equal(schedule.n_updates(30, stepped), per_stepped)
    np.testing.assert_array_equal(schedule.n_updates(25, stepped), 100)
    np.testing.assert_array_equal(schedule.n_updates(20, stepped), 0)
    with pytest.raises(ValueError, match="updates_per_step"):
        EpisodicSchedule(
            n_envs=4, episode_length=10, seed_steps=100, updates_per_step=0.0
        )


# The terminating chain s0 -0-> s1 -0-> s2 -1-> s3 (terminal), time limit 10:
# the values are gamma^2, gamma, 1 at s0, s1, s2. The paper-era agent does
# not see the termination (gymnax auto-resets to s0 mid-slot), so it learns
# the continuing chain with reward 1 every third step, gamma^(2 - t) / (1 -
# gamma^3). Calibrated on seeds 0-3 (1072 updates): the episodic values
# within 0.008 of the truth and the termination probability below 0.002 or
# above 0.999; the paper-era values within 0.035 of the continuing chain's.
CHAIN_GAMMA = 0.8
TRUTH = CHAIN_GAMMA ** np.array([2.0, 1.0, 0.0])
CONTINUING = TRUTH / (1 - CHAIN_GAMMA**3)
CHAIN = {
    "n_envs": 2,
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "num_q": 2,
    "batch_size": 32,
    "num_samples": 64,
    "num_elites": 8,
    "num_pi_trajs": 4,
    "iterations": 4,
    "gamma": CHAIN_GAMMA,
    "tau": 0.1,
    "learning_rate": 1e-3,
    "seed_steps": 100,
}
CHAIN_SEEDS = [0, 1]
ROWS = 1400  # 700 ticks of 2 envs


@pytest.fixture(scope="module")
def episodic_chain():
    agent = TDMPC2(ChainEnv(), episodic=True, buffer_size=4000, **CHAIN)
    state, _ = agent.train(seed=CHAIN_SEEDS, n_timesteps=ROWS, num_episode_test=1)
    return agent, state


def _chain_values(agent, state):
    """Per seed: Q at s0, s1, s2 (every head, action 0) and the termination
    probability after one step from each."""
    out = []
    for i in range(len(CHAIN_SEEDS)):
        wm = jax.tree.map(lambda x, i=i: x[i], state.world_model_state)
        apply = functools.partial(wm.apply_fn, {"params": wm.params})
        z = apply(jax.nn.one_hot(jnp.arange(3), 4), method="encode")
        a = jnp.zeros((3, 1))
        q = agent.agent_config.two_hot.decode(
            core.q_logits(wm.apply_fn, wm.params, z, a)
        )
        p = None
        if agent.agent_config.episodic:
            p = jax.nn.sigmoid(
                apply(apply(z, a, method="next"), method="termination_logits")
            )
        out.append((np.asarray(q), p))
    return out


def test_the_episodic_agent_learns_the_terminating_chains_values(episodic_chain):
    agent, state = episodic_chain
    for q, p in _chain_values(agent, state):
        np.testing.assert_allclose(q, np.broadcast_to(TRUTH, q.shape), atol=0.05)
        np.testing.assert_array_equal(np.asarray(p) > 0.5, [False, False, True])


def test_the_paper_era_agent_learns_a_continuing_chain_instead():
    """Without the episodic mode the agent refuses the chain after training
    (T10); the base class's ``train`` skips that check, to read what it
    learned: the values of a chain that never ends."""
    agent = TDMPC2(ChainEnv(), **CHAIN)
    budget = ROWS * 3 // 4  # about as many updates as the episodic run
    buffer = agent._replay_buffer(budget)
    state, _ = ActorCritic.train(
        agent,
        seed=CHAIN_SEEDS,
        n_timesteps=budget,
        num_episode_test=1,
        buffer=buffer,
        start_tick=0,
    )
    assert np.all(np.asarray(state.collector_state.n_offschedule_dones) > 0)
    for q, _ in _chain_values(agent, state):
        np.testing.assert_allclose(q, np.broadcast_to(CONTINUING, q.shape), atol=0.1)
        assert np.all(q > TRUTH + 0.5)


def test_an_episodic_run_updates_once_per_env_step(episodic_chain):
    """Both envs run the 3-step chain in lockstep: 4 ticks an episode, the
    last one held. After the burst of S = 100 on the seed tick
    ``max(100 // 2, 10) = 50``, each tick runs one update per env it
    stepped; the run takes terminations without raising (T10 is the
    paper-era agent's)."""
    agent, state = episodic_chain
    ticks = np.arange(agent.schedule.num_ticks(ROWS))
    assert agent.schedule.seed_tick == 50 and ticks.size == 700
    stepped = np.where(ticks % 4 == 3, 0, 2)
    expected = 100 + stepped[ticks > 50].sum()
    for count in (state.n_updates, state.world_model_state.step):
        np.testing.assert_array_equal(count, expected)
    cs = state.collector_state
    np.testing.assert_array_equal(cs.timestep, ROWS)  # rows
    np.testing.assert_array_equal(cs.env_steps, 2 * stepped.sum() // 2)
    np.testing.assert_array_equal(state.n_terminations, 2 * (700 // 4))
    assert "termination_loss" in state.update_metrics


def test_the_episodic_ring_holds_the_collected_episodes(episodic_chain):
    """The rings hold the run's 700 rows per env, ``min(buffer_size,
    rows) / n_envs`` (the reference's clamp to the run), not 2000. Every
    row's observation is ``one_hot(step)``; the last rows are the terminal
    ``s3`` after 3 steps; the reset rows have no reward."""
    agent, state = episodic_chain
    assert agent.replay_capacity == ROWS
    assert state.buffer_state.obs.shape[-2] == ROWS // 2
    replay = jax.tree.map(lambda x: np.asarray(x[0]), state.buffer_state)  # seed 0
    np.testing.assert_array_equal(replay.obs, np.eye(4)[replay.step])
    np.testing.assert_array_equal(replay.last, replay.step == 3)
    np.testing.assert_array_equal(replay.terminal, replay.last)
    np.testing.assert_array_equal(replay.reward, replay.step == 3)
    np.testing.assert_array_equal(replay.action[replay.last], 0.0)


def test_an_episodic_run_takes_its_updates_per_step():
    """Two updates per env step in an episodic run: the chain's ticks step
    both envs except every fourth (held), after the burst of S = 20 on the
    seed tick ``max(20 // 2, 10) = 10``."""
    agent = TDMPC2(
        ChainEnv(),
        episodic=True,
        buffer_size=200,
        updates_per_step=2.0,
        **{**CHAIN, "seed_steps": 20},
    )
    state, _ = agent.train(seed=0, n_timesteps=120, num_episode_test=1)
    ticks = np.arange(60)
    stepping = (ticks > 10) & (ticks % 4 != 3)
    expected = 20 + 4 * int(stepping.sum())
    for count in (state.n_updates, state.world_model_state.step):
        np.testing.assert_array_equal(count, expected)


def test_an_episodic_run_resumes_as_the_uninterrupted_run(tmp_path):
    """The first call's rings hold its 30 rows per env. A new agent restores
    its checkpoint into its skeleton (``skeleton_of``: rings of one row,
    ``ajax.checkpoint``'s flow); the resumed call adopts the 30 rows into
    rings of 50 (``buffer_size`` 100 over 2 envs) and continues the ticks:
    it computes what one call does. The replayed rows
    and every count are equal; the parameters agree up to the float32
    rounding of the differently compiled programs, which the updates
    amplify, far below one Adam step (``learning_rate`` 1e-3). Rings of 90
    (``buffer_size`` 200) cannot resume the full run's, which lost ticks
    0 .. 9."""
    small = {**CHAIN, "seed_steps": 20}
    agent = TDMPC2(ChainEnv(), episodic=True, buffer_size=100, **small)
    first, _ = agent.train(seed=0, n_timesteps=60, num_episode_test=1)
    assert first.buffer_state.obs.shape[-2] == 30
    path = str(tmp_path / "first.pkl")
    save_checkpoint(first, path)
    resumer = TDMPC2(ChainEnv(), episodic=True, buffer_size=100, **small)
    skeleton = skeleton_of(resumer, 0, num_episode_test=1)
    assert skeleton.buffer_state.obs.shape[-2] == 1
    resumed, _ = resumer.train(
        seed=0,
        n_timesteps=60,
        num_episode_test=1,
        initial_state=restore_into(skeleton, path),
    )
    full, _ = agent.train(seed=0, n_timesteps=120, num_episode_test=1)
    np.testing.assert_array_equal(resumed.n_updates, full.n_updates)
    assert int(full.n_updates[0]) > 20
    for name in ("obs", "reward", "terminal", "last", "step"):
        np.testing.assert_array_equal(
            getattr(resumed.buffer_state, name), getattr(full.buffer_state, name)
        )
    for a, b in (
        (resumed.world_model_state.params, full.world_model_state.params),
        (resumed.actor_state.params, full.actor_state.params),
    ):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-4)
    with pytest.raises(ValueError, match="smaller buffer_size"):
        TDMPC2(ChainEnv(), episodic=True, buffer_size=200, **small).train(
            seed=0, n_timesteps=60, num_episode_test=1, initial_state=full
        )
