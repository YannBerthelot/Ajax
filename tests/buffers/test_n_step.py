"""SAC's n-step returns: replay rows that sum n rewards before they bootstrap.

Contract (Sutton and Barto 2018, eq. 7.1; Hessel et al. 2018, "Multi-step
learning"; SAC's docstring). For the row of step t, m is n or the steps to the
episode's end, that step included, whichever is less:

- its reward is the sum over k < m of gamma^k r_{t+k};
- its state and action stay step t's;
- its next state, flags and next expert action are step t+m-1's;
- its target bootstraps at gamma^m, unless step t+m-1 terminated;
- nothing past an episode's end, or past the last step of a contiguous run of
  rows (the replay's write position, a prefill's or a dataset's last row),
  enters it.

Oracles: hand arithmetic, and a Python loop written from the definition.
"""

from itertools import product
from typing import Any, ClassVar

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import SAC
from ajax.buffers import dataset as ds
from ajax.buffers.utils import (
    empty_ring_like,
    get_batch_from_buffer,
    get_buffer,
    n_step_rows,
    with_retained_rows,
)
from ajax.extensions.base import Extension
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.centering import ValueCentering
from ajax.extensions.expert import ResidualPolicy
from ajax.extensions.pretrain import BellmanPretrain, ExpertPrefill
from ajax.extensions.retained import RLPD
from ajax.extensions.target_mods import IBRL
from ajax.networks.memory import MemoryConfig
from ajax.types import NormalizationInfo

SCHEMA = {
    field: jnp.zeros(1)
    for field in (
        "obs",
        "action",
        "reward",
        "terminated",
        "truncated",
        "next_obs",
        "next_a_expert",
    )
}
GAMMA = 0.5
KIND = {"go": (0.0, 0.0), "terminated": (1.0, 0.0), "truncated": (0.0, 1.0)}


def expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


def window(ends: tuple[str, ...], rewards: tuple[float, ...]) -> dict:
    """One window of a batch: step k has reward rewards[k], state 10 k, next
    state 10 k + 5 and next expert action 100 k, and ends as ends[k] says."""
    steps = jnp.arange(float(len(ends)))[None, :, None]
    flags = jnp.array([KIND[end] for end in ends])[None]
    return {
        "obs": 10 * steps,
        "action": -steps,
        "reward": jnp.array(rewards)[None, :, None],
        "terminated": flags[..., :1],
        "truncated": flags[..., 1:],
        "next_obs": 10 * steps + 5,
        "next_a_expert": 100 * steps,
    }


def fold(ends: tuple[str, ...], rewards: tuple[float, ...]) -> dict:
    rows = n_step_rows(window(ends, rewards), GAMMA)
    return {name: float(value[0, 0]) for name, value in rows.items()}


def test_a_window_with_no_end_sums_every_reward() -> None:
    """Three steps at gamma 0.5: 1 + 0.5 * 2 + 0.25 * 4 = 3, and the target
    looks 3 steps ahead, at 0.125. A sum left undiscounted reads 7, and a
    bootstrap kept at gamma overvalues every state."""
    row = fold(("go", "go", "go"), (1.0, 2.0, 4.0))
    assert row["reward"] == pytest.approx(3.0)
    assert row["bootstrap_discount"] == pytest.approx(0.125)
    assert row["next_obs"] == 25.0
    assert row["next_a_expert"] == 200.0


def test_a_row_keeps_its_first_steps_state_and_action() -> None:
    """The critics fit Q(s_t, a_t): an update on another step's pair would
    credit the action with rewards that came before it."""
    row = fold(("go", "go", "go"), (1.0, 2.0, 4.0))
    assert (row["obs"], row["action"]) == (0.0, 0.0)


def test_a_termination_stops_the_sum_and_cuts_the_bootstrap() -> None:
    """The second step terminates: 1 + 0.5 * 2 = 2. The third reward belongs
    to the next episode and must not enter (it would make a terminal state
    look worth what the reset brings). Terminated is set, so the target has
    no bootstrap."""
    row = fold(("go", "terminated", "go"), (1.0, 2.0, 1000.0))
    assert row["reward"] == pytest.approx(2.0)
    assert row["terminated"] == 1.0
    assert row["bootstrap_discount"] == pytest.approx(0.25)
    assert row["next_obs"] == 15.0


def test_a_time_limit_stops_the_sum_and_bootstraps_there() -> None:
    """A time limit is not a terminal state: the target goes on bootstrapping,
    at gamma^1, on the final observation, which that row keeps."""
    row = fold(("truncated", "go", "go"), (1.0, 1000.0, 1000.0))
    assert row["reward"] == pytest.approx(1.0)
    assert (row["terminated"], row["truncated"]) == (0.0, 1.0)
    assert row["bootstrap_discount"] == pytest.approx(0.5)
    assert row["next_obs"] == 5.0


@pytest.mark.parametrize("ends", list(product(KIND, repeat=3)))
def test_every_way_a_window_can_end_matches_the_definition(ends) -> None:
    """Differential oracle: the definition as a plain loop, on all 27 windows
    of three steps."""
    rewards = (3.0, 5.0, 7.0)
    total, discount, last = 0.0, 1.0, 0
    for k, end in enumerate(ends):
        total, last = total + discount * rewards[k], k
        discount *= GAMMA
        if end != "go":
            break
    row = fold(ends, rewards)
    assert row["reward"] == pytest.approx(total)
    assert row["bootstrap_discount"] == pytest.approx(discount)
    assert row["next_obs"] == 10.0 * last + 5.0
    assert row["terminated"] == float(ends[last] == "terminated")


def written(buffer: Any, state: Any, steps: Any, n_envs: int = 1) -> Any:
    """state after rows of steps: obs i, reward i, next_obs i + 1, one add per
    row, every env alike."""
    for i in steps:
        row = {k: jnp.full((n_envs, *v.shape), float(i)) for k, v in SCHEMA.items()}
        row = {
            **row,
            "terminated": jnp.zeros((n_envs, 1)),
            "truncated": jnp.zeros((n_envs, 1)),
            "next_obs": jnp.full((n_envs, 1), float(i + 1)),
        }
        state = buffer.add(state, row)
    return state


def sampled(buffer: Any, state: Any, draws: int = 32) -> dict:
    """draws batches, end to end."""
    batches = [
        buffer.sample(state, jax.random.PRNGKey(seed)).experience
        for seed in range(draws)
    ]
    return {name: np.concatenate([b[name] for b in batches]) for name in batches[0]}


def assert_steps_in_order(rows: dict, expected_starts: set) -> None:
    """Every row of ``rows`` is three steps of written, from one of
    expected_starts, in the order they were written."""
    starts = rows["obs"][:, 0]
    assert set(starts.tolist()) == expected_starts
    np.testing.assert_allclose(
        rows["reward"][:, 0], starts + GAMMA * (starts + 1) + GAMMA**2 * (starts + 2)
    )
    np.testing.assert_array_equal(rows["next_obs"][:, 0], starts + 3)
    np.testing.assert_allclose(rows["bootstrap_discount"], GAMMA**3)


def test_a_window_stays_behind_the_write_position_before_the_ring_fills() -> None:
    """Of steps 0 to 5, windows of 3 start at 0 to 3: a start at 4 would read
    an empty row as the next step."""
    buffer = get_buffer(16, 4, n_step=3, gamma=GAMMA)
    state = written(buffer, buffer.init(SCHEMA), range(6))
    assert_steps_in_order(sampled(buffer, state), {0.0, 1.0, 2.0, 3.0})


def test_a_window_never_crosses_the_write_position_of_a_full_ring() -> None:
    """A ring of 8 after 13 steps holds 5 to 12: a window across the write
    position would splice the newest step onto the oldest, a step that never
    followed it. Windows start at 5 to 10, wrapped ones included."""
    buffer = get_buffer(8, 4, n_step=3, gamma=GAMMA)
    state = written(buffer, buffer.init(SCHEMA), range(13))
    assert_steps_in_order(sampled(buffer, state), {5.0, 6.0, 7.0, 8.0, 9.0, 10.0})


def test_both_rings_of_a_two_buffer_replay_give_whole_windows() -> None:
    """RLPD on n-step SAC: half of each batch from the kept rows, each row
    summing the steps of one ring, not a mix of both."""
    buffer = get_buffer(128, 64, n_step=3, gamma=GAMMA)
    ring = empty_ring_like(buffer.init(SCHEMA), 32)
    kept = written(buffer, empty_ring_like(ring, 7), range(100, 106))
    state = with_retained_rows(written(buffer, ring, range(6)), kept, 0.5)
    rows = sampled(buffer, state, draws=1)
    assert_steps_in_order(
        {name: value[:32] for name, value in rows.items()},
        {100.0, 101.0, 102.0, 103.0},
    )
    assert_steps_in_order(
        {name: value[32:] for name, value in rows.items()},
        {0.0, 1.0, 2.0, 3.0},
    )


def test_a_normaliser_divides_the_sum_as_it_divides_a_reward() -> None:
    """The replay keeps the env's rewards and divides at sampling: a sum over
    rows divided by the same std is the normalised rewards' own sum."""
    buffer = get_buffer(16, 8, n_step=3, gamma=GAMMA)
    state = written(buffer, buffer.init(SCHEMA), range(6))
    key = jax.random.PRNGKey(0)
    # 4 rows of variance 4: a std of 2, up to the normaliser's 1e-8.
    stats = NormalizationInfo(
        var=jnp.full((1, 1), 4.0),
        count=jnp.full((1, 1), 4.0),
        mean=jnp.zeros((1, 1)),
        mean_2=jnp.full((1, 1), 16.0),
    )
    plain = get_batch_from_buffer(buffer, state, key)
    divided = get_batch_from_buffer(buffer, state, key, stats)
    np.testing.assert_allclose(divided.reward, plain.reward / 2.0, rtol=1e-6)
    np.testing.assert_array_equal(divided.bootstrap_discount, plain.bootstrap_discount)


def by_ring_row(rows: dict, ring: dict) -> dict:
    """The sampled rows, keyed by the ring's row each started from."""
    found: dict[int, dict] = {}
    for i, value in enumerate(rows["obs"]):
        matches = np.all(ring["obs"][0] == value, axis=-1)
        found[int(np.argmax(matches))] = {k: v[i] for k, v in rows.items()}
    return found


def test_a_row_stops_at_the_end_of_the_prefill() -> None:
    """The prefill's last step is followed in the ring by the first step
    collected, from another episode. Reading on credits the expert's action
    with rewards it did not cause. With the replay holding rows 0 to 3 of the
    prefill then the run's: row 2 sums two rewards, row 3 its own alone."""
    gamma = 0.9
    agent = SAC(
        "Pendulum-v1",
        n_step=2,
        gamma=gamma,
        learning_starts=2,
        batch_size=64,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        expert_policy=expert,
        extensions=(ExpertPrefill(n_steps=4),),
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=4)[0])
    replay = state.collector_state.buffer_state
    ring = {name: np.asarray(value) for name, value in replay.experience.items()}
    rows = by_ring_row(sampled(agent.buffer, replay, draws=4), ring)
    reward = ring["reward"][0]

    assert rows[2]["reward"] == pytest.approx(reward[2, 0] + gamma * reward[3, 0])
    assert rows[2]["bootstrap_discount"] == pytest.approx(gamma**2)
    assert rows[3]["reward"] == pytest.approx(reward[3, 0])
    assert rows[3]["bootstrap_discount"] == pytest.approx(gamma)
    np.testing.assert_array_equal(rows[3]["next_obs"], ring["next_obs"][0, 3])


def test_a_row_stops_where_a_datasets_rows_leave_an_env() -> None:
    """One episode of 4 rows on 2 envs: rows 0 and 1 go to the first env, and
    what follows row 1 there is the first step collected, not row 2."""
    n_rows = 4
    ends = np.zeros(n_rows, bool)
    ends[-1] = True
    data = ds.TransitionDataset(
        obs=np.arange(n_rows, dtype=np.float32)[:, None],
        action=np.zeros((n_rows, 1), np.float32),
        reward=np.arange(n_rows, dtype=np.float32),
        terminated=np.zeros(n_rows, bool),
        truncated=ends,
        next_obs=np.arange(1, n_rows + 1, dtype=np.float32)[:, None],
        is_expert=np.zeros(n_rows, bool),
        a_expert=np.zeros((n_rows, 1), np.float32),
        gamma=0.99,
    )
    schema = {k: v for k, v in SCHEMA.items() if k != "next_a_expert"}
    schema["seam"] = jnp.zeros(1)
    buffer = get_buffer(64, 8, n_envs=2, n_step=2, gamma=GAMMA)
    state = ds.fill_replay(buffer.init(schema), data)
    next_step = {k: jnp.full((2, 1), 50.0) for k in schema}
    next_step["seam"] = jnp.zeros((2, 1))
    state = buffer.add(state, next_step)
    state = buffer.add(state, next_step)

    rows = sampled(buffer, state)
    last_of_the_env = rows["obs"][:, 0] == 1.0
    assert last_of_the_env.any()
    np.testing.assert_allclose(rows["reward"][last_of_the_env], 1.0)
    np.testing.assert_allclose(rows["bootstrap_discount"][last_of_the_env], GAMMA)


class ConstantReward(Pendulum):
    """Every step pays 1, so a row's return is a sum of powers of gamma."""

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        obs, state, _, done, info = super().step_env(key, state, action, params)
        return obs, state, jnp.ones_like(done, jnp.float32), done, info


class TargetTap(Extension):
    """Reads what the critics' target was built from."""

    name: str = "target_tap"
    batch_keys: ClassVar = {
        "on_target": frozenset({"rewards", "gamma", "dones", "next_values"})
    }

    def __init__(self, seen: list) -> None:
        self.seen = seen

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def on_target(
        self, agent_state: Any, ext_state: Any, batch: Any, target: Any, context: Any
    ) -> Any:
        del agent_state, ext_state, context

        def keep(*arrays: Any) -> None:
            self.seen.append(
                tuple(np.broadcast_to(a, np.shape(arrays[0])) for a in arrays)
            )

        jax.debug.callback(
            keep,
            target,
            batch["rewards"],
            batch["gamma"],
            batch["dones"],
            batch["next_values"],
        )
        return target


def test_the_critics_target_bootstraps_as_many_steps_ahead_as_it_sums() -> None:
    """On a run with episodes of 4 steps, every step paying 1: a row m steps
    from its end sums 1 + 0.9 + ... and bootstraps at 0.9^m, for m up to 3.
    Keeping gamma in the bootstrap, the realistic slip, would value the state
    n - 1 steps too far ahead."""
    gamma = 0.9
    seen: list = []
    params = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=4)
    agent = SAC(
        ConstantReward(),
        env_params=params,
        n_step=3,
        gamma=gamma,
        learning_starts=4,
        batch_size=32,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        extensions=(TargetTap(seen),),
    )
    agent.train(seed=0, n_timesteps=20)
    jax.effects_barrier()

    target, rewards, discounts, dones, next_values = (
        np.concatenate(arrays) for arrays in zip(*seen, strict=True)
    )
    steps = np.rint(np.log(discounts) / np.log(gamma))
    assert set(steps.ravel().tolist()) == {1.0, 2.0, 3.0}
    np.testing.assert_allclose(discounts, gamma**steps, rtol=1e-6)
    np.testing.assert_allclose(rewards, (1 - gamma**steps) / (1 - gamma), rtol=1e-5)
    assert not dones.any()
    np.testing.assert_allclose(target, rewards + discounts * next_values, rtol=1e-5)


def critic_of(agent: SAC, n_timesteps: int) -> list:
    state = agent.train(seed=0, n_timesteps=n_timesteps)[0]
    return [np.asarray(leaf) for leaf in jax.tree.leaves(state.critic_state.params)]


def test_the_first_update_waits_for_a_window() -> None:
    """With no warm-up and n_step 3, the first update comes at the third step:
    an earlier one would read rows not yet written as steps of the episode. No
    uniform step comes with the wait, which a run asking for no warm-up
    (RLPD's, a prefill's) would pay for."""
    agent = SAC(
        "Pendulum-v1",
        n_step=3,
        learning_starts=0,
        batch_size=4,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
    )
    before = critic_of(agent, 2)
    after = critic_of(agent, 3)
    untrained = critic_of(agent, 0)

    assert all(np.array_equal(a, b) for a, b in zip(before, untrained, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(after, untrained, strict=True))
    assert agent.uniform_steps() == 0


@pytest.mark.parametrize("n_step", [0, -1])
def test_a_return_of_no_steps_is_refused(n_step: int) -> None:
    with pytest.raises(ValueError, match="n_step"):
        SAC("Pendulum-v1", n_step=n_step)


def test_a_run_with_memory_refuses_n_steps() -> None:
    """A run with memory replays windows of its own, which this fold does not
    read: left alone, its targets would stay one-step while n_step said more."""
    with pytest.raises(ValueError, match="n_step"):
        SAC("Pendulum-v1", n_step=2, memory=MemoryConfig("gru", 4))


def test_an_extension_shifting_a_one_step_target_refuses_n_steps() -> None:
    """ValueCentering subtracts its estimate once per step of target: on an
    n-step return it would be wrong by the rest of the sum."""
    with pytest.raises(MissingCapability, match="one_step_target"):
        SAC("Pendulum-v1", n_step=2, extensions=(ValueCentering(),))


STACKS = {
    "RLPD": (RLPD(n_steps=8, critic_layer_norm=False),),
    "IBRL": (IBRL(expert_policy=expert),),
    "residual": (ResidualPolicy(),),
    "bellman": (ExpertPrefill(n_steps=8), BellmanPretrain(n_steps=2)),
}


@pytest.mark.parametrize("name", list(STACKS))
def test_the_study_stacks_run_on_n_step_rows(name: str) -> None:
    """These read a row's next fields (the expert's next action, the raw next
    observation) or the target's gamma: a shape or field they could not take
    from an n-step row raises here, or turns the critics to NaN."""
    agent = SAC(
        "Pendulum-v1",
        n_step=2,
        learning_starts=4,
        batch_size=4,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        expert_policy=expert,
        extensions=STACKS[name],
    )
    assert all(np.isfinite(leaf).all() for leaf in critic_of(agent, 12))
