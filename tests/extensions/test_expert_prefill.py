"""The expert prefill writes its rows as the live collector does
(replay_row): every field the replay holds, from the expert's steps, and
under an action transform what collection stores when the expert acts.
ExpertPrefill runs it before training on every replay agent."""

import warnings
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax import IQL, PPO, SAC, TD3
from ajax.buffers.utils import get_buffer, init_buffer
from ajax.environments.interaction import replay_row, reward_statistics
from ajax.extensions.base import Extension, ExtensionStack
from ajax.extensions.capabilities import MissingCapability, NotWired
from ajax.extensions.expert import ExpertGains, ImitationLoss, ResidualPolicy
from ajax.extensions.pretrain import (
    EXPERT_PREFILL_KEY_TAG,
    ExpertPrefill,
    collect_and_store_expert_transitions,
)
from tests.agents.test_agent_contract import CASES
from tests.agents.test_gain_mode import OneGainTwoActions, TwoGains
from tests.agents.test_offline import episodes
from tests.environments import paid_action
from tests.environments.identity_env import LIMIT, IdentityEnv

REPLAY_AGENTS = ("SAC", "REDQ", "ASAC", "TQC", "TD3", "CrossQ", "IQL")


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def prefill(agent: Any = None, n_steps: int = 4, **fields: Any) -> dict:
    """The first n_steps rows of each env of the agent's replay (by default
    2-env SAC on Pendulum with push), holding the expert fields and the
    fields asked, after the expert's prefill."""
    agent = agent or SAC("Pendulum-v1", n_envs=2, expert_policy=push)
    buffer = get_buffer(64, 8, n_envs=2)
    state = init_buffer(buffer, agent.env_args, include_expert_fields=True, **fields)
    key = jax.random.PRNGKey(0)
    state = collect_and_store_expert_transitions(
        agent.expert_policy, agent.env_args, buffer, state, key, n_steps
    )
    return {
        name: np.asarray(value)[:, :n_steps] for name, value in state.experience.items()
    }


def test_the_prefill_fills_the_policy_action_and_the_next_action():
    """The expert chose its action, and the next one is unknown."""
    rows = prefill(include_policy_action=True, include_next_action=True)
    np.testing.assert_array_equal(rows["policy_action"], rows["action"])
    assert np.isnan(rows["next_action"]).all()


@pytest.mark.parametrize("expert", [OneGainTwoActions(), TwoGains()])
def test_gain_mode_prefill_rows_store_the_anchor_gains(expert):
    """Collection stores the gains chosen; at 0, the anchor gains, the expert
    acts as itself, its env action in a_expert."""
    env_width = 2 if isinstance(expert, OneGainTwoActions) else 1
    env, params = paid_action.make(length=8, dim=env_width)
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        expert_policy=expert,
        extensions=(ExpertGains(),),
    )
    rows = prefill(agent)
    assert rows["action"].shape[-1] == len(expert.learnable_fields)
    assert (rows["action"] == 0).all()
    anchor_action, _ = expert(expert.init_state(1), paid_action.OBS[None])
    np.testing.assert_allclose(
        rows["a_expert"], np.broadcast_to(anchor_action / 2, rows["a_expert"].shape)
    )


def test_a_residual_prefill_stores_a_zero_residual_and_the_experts_action():
    """IQL fits its stored residuals: the expert's rows are residual 0, and
    the executed action is the expert's (0.25 in agent units)."""
    agent = IQL(
        "Pendulum-v1", n_envs=2, expert_policy=push, extensions=(ResidualPolicy(),)
    )
    rows = prefill(agent, include_policy_action=True)
    assert (rows["policy_action"] == 0).all()
    np.testing.assert_allclose(rows["action"], 0.25)


def test_a_terminated_prefill_row_keeps_the_next_observation_collection_keeps():
    """After a termination the replay keeps the reset observation, the next
    row's (ASAC's target continues from it), and after a time limit the
    final one, as the live collector does (bootstrap_obs)."""
    agent = SAC(IdentityEnv(), n_envs=2, expert_policy=lambda obs: obs[..., 1:])
    rows = prefill(agent, n_steps=20)
    terminated, truncated = (
        rows["terminated"][..., 0] > 0,
        rows["truncated"][..., 0] > 0,
    )
    assert terminated.any() and truncated.any()
    continuing = ~truncated[:, :-1]
    np.testing.assert_array_equal(
        rows["next_obs"][:, :-1][continuing], rows["obs"][:, 1:][continuing]
    )
    assert (rows["next_obs"][truncated][:, 0] == LIMIT).all()


def test_the_prefill_refuses_a_recurrent_actors_stored_carry():
    with pytest.raises(ValueError, match="actor_carry"):
        prefill(actor_carry_dim=4)


def test_a_row_names_the_field_it_cannot_fill():
    zeros = jnp.zeros((2, 1))
    fields = "obs action reward terminated truncated next_obs raw_obs is_expert"
    fields += " a_expert next_a_expert raw_next_obs policy_action"
    with pytest.raises(ValueError, match="'episode'"):
        replay_row({"episode": zeros}, **dict.fromkeys(fields.split(), zeros))


def run(name: str, extensions: tuple = (), **kwargs: Any) -> Any:
    """The agent, 2 envs on Pendulum with push as its expert, after two
    warm-up steps (no update yet): one seed's state."""
    case = CASES[name]
    cls: Any = case.cls  # each agent's own keywords
    agent = cls(
        env_id="Pendulum-v1",
        n_envs=2,
        expert_policy=push,
        extensions=extensions,
        **{**case.kwargs, **kwargs},
    )
    state, _ = agent.train(seed=0, n_timesteps=4)
    return jax.tree.map(lambda leaf: leaf[0], state)


@pytest.mark.parametrize("name", REPLAY_AGENTS)
def test_the_first_rows_of_each_env_are_the_experts(name):
    """Push acts 0.5 on Pendulum's [-2, 2]: 0.25 in agent units."""
    state = run(name, (ExpertPrefill(n_steps=4),))
    replay = state.collector_state.buffer_state
    assert int(replay.current_index) == 4 + 2  # the prefill, then two live steps
    rows = {key: np.asarray(value)[:, :4] for key, value in replay.experience.items()}
    np.testing.assert_allclose(rows["action"], 0.25)
    for key in ("a_expert", "next_a_expert"):
        if key in rows:
            np.testing.assert_allclose(rows[key], 0.25)
    if "is_expert" in rows:
        assert (rows["is_expert"] == 1).all()


@pytest.mark.parametrize("name", ["SAC", "TD3"])
def test_the_prefill_counts_in_the_normalisers_and_no_timestep(name):
    """Its rows count once each in the observation statistics and in the
    reward statistics (each env's copy of them), as PriorData's do, beside
    the live steps; the timestep counts the live steps alone."""
    norm = {"normalize_observations": True, "normalize_rewards": True}
    # SAC adds a prefill of its own by default beside an expert.
    no_prefill = {"expert_buffer_n_steps": 0} if name == "SAC" else {}
    plain = run(name, **norm, **no_prefill).collector_state
    prefilled = run(name, (ExpertPrefill(n_steps=4),), **norm).collector_state
    assert int(prefilled.timestep) == int(plain.timestep) == 4
    rows = 4 * 2  # four prefill steps in each of two envs
    for counts in (
        (plain.obs_norm_info.count, prefilled.obs_norm_info.count),
        (
            reward_statistics(plain.env_state).count,
            reward_statistics(prefilled.env_state).count,
        ),
    ):
        np.testing.assert_array_equal(counts[1], counts[0] + rows)


def test_sac_with_expert_prefill_fills_its_replay_once():
    """Beside an ExpertPrefill, SAC's expert_buffer_n_steps left unset builds
    no prefill of its own."""
    state = run("SAC", (ExpertPrefill(n_steps=4),), expert_fraction=0.0)
    assert int(state.collector_state.buffer_state.current_index) == 4 + 2


def quiet_sac(**kwargs: Any) -> SAC:
    """2-env SAC on Pendulum with push as its expert, its deprecation
    warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return SAC("Pendulum-v1", **{**CASES["SAC"].kwargs, "n_envs": 2, **kwargs})


def test_sacs_keyword_builds_the_prefill_last_and_trains_the_same():
    """expert_buffer_n_steps builds ExpertPrefill(n_steps) after the
    caller's extensions, so they keep their keys: the same run, bitwise, as
    the extension passed there, through eight updates. Left unset with an
    expert, it builds a prefill of 20 000 steps."""
    imitation = ImitationLoss(push, coef=1.0)
    by_keyword = quiet_sac(
        expert_policy=push, extensions=(imitation,), expert_buffer_n_steps=4
    )
    by_extension = quiet_sac(
        expert_policy=push, extensions=(imitation, ExpertPrefill(n_steps=4))
    )
    states = [
        agent.train(seed=0, n_timesteps=24)[0] for agent in (by_keyword, by_extension)
    ]
    for old, new in zip(*(jax.tree.leaves(state) for state in states), strict=True):
        np.testing.assert_array_equal(old, new)
    built = quiet_sac(expert_policy=push).extension_stack.extensions
    assert built == (ExpertPrefill(n_steps=20_000),)
    assert not quiet_sac(expert_buffer_n_steps=4).extension_stack  # no expert


def test_sacs_keyword_warns_and_raises_beside_the_extension():
    """Built from the default it warns that the default will go, a value
    passed warns that the keyword will go, the extension alone stays quiet
    and a value next to it raises."""
    with pytest.warns(FutureWarning, match=r"adds ExpertPrefill\(n_steps=20000\)"):
        SAC("Pendulum-v1", expert_policy=push)
    with pytest.warns(FutureWarning, match="expert_buffer_n_steps will be removed"):
        SAC("Pendulum-v1", expert_policy=push, expert_buffer_n_steps=0)
    prefill = (ExpertPrefill(n_steps=4),)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", ".*(ExpertPrefill|expert_buffer_n_steps)", FutureWarning
        )
        SAC("Pendulum-v1", expert_policy=push, extensions=prefill)
    both = {"expert_buffer_n_steps": 0, "extensions": prefill}
    with pytest.raises(ValueError, match="not both"):
        SAC("Pendulum-v1", expert_policy=push, **both)


def test_the_prefill_key_is_no_split_of_the_pretrain_key():
    """CloningPretrain and MCPretrain draw split(key, n): the prefill's
    fold_in(key, tag) must not be one of their keys."""
    key = jax.random.PRNGKey(0)
    prefill_key = np.asarray(jax.random.fold_in(key, EXPERT_PREFILL_KEY_TAG))
    for n in range(1, 65):
        splits = np.asarray(jax.random.split(key, n))
        assert not (splits == prefill_key).all(axis=-1).any()


@struct.dataclass
class Recorded:
    ext_state: tuple = ()


class Writes(Extension):
    """Records, on its pretrain, its name in the order the stack runs it."""

    def __init__(self, name: str, order: list):
        self.name, self.order = name, order

    def pretrain(self, agent_state: Any, ext_state: Any, context: Any) -> tuple:
        self.order.append(self.name)
        return agent_state, ext_state


class Seeds(Writes):
    provides: ClassVar[frozenset[str]] = frozenset({"seeded_replay"})


def test_the_replays_seeders_pretrain_first_in_stack_order():
    order: list[str] = []
    writers = [
        Writes("a", order),
        Seeds("seed1", order),
        Writes("b", order),
        Seeds("seed2", order),
    ]
    stack = ExtensionStack(writers)
    state = Recorded(ext_state=((), (), (), ()))
    stack.apply_pretrain(state, jnp.asarray(0), jax.random.PRNGKey(0), 1)
    assert order == ["seed1", "seed2", "a", "b"]


def test_an_agent_lacking_what_the_prefill_needs_refuses_it():
    """PPO has no replay (inherent); DQN takes no expert_policy yet (a gap)."""
    with pytest.raises(MissingCapability, match="replay"):
        PPO("Pendulum-v1", extensions=(ExpertPrefill(n_steps=4),))
    with pytest.raises(NotWired, match="expert_policy"):
        CASES["DQN"].cls("CartPole-v1", extensions=(ExpertPrefill(n_steps=4),))


def test_offline_training_refuses_a_prefill_overwriting_the_dataset():
    """16 dataset rows and 2 x 32 prefill rows outnumber a replay of 64."""
    agent = TD3(
        "Pendulum-v1",
        n_envs=2,
        buffer_size=64,
        batch_size=8,
        expert_policy=push,
        extensions=(ExpertPrefill(n_steps=32),),
    )
    with pytest.raises(ValueError, match="overwrite dataset rows"):
        agent.train_offline(episodes(3, discrete=False), 2)
