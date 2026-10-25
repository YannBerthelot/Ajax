"""ExpertWarmup: the agent's expert takes a share of the warm-up steps, a
controller phase (choose_warmup_action) with no state and no key in the
learner, on every agent with a uniform warm-up. SAC's expert_fraction builds
it and warns, its None default standing for 0.7 with an expert until B7."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import ASAC, AVG, DQN, IQL, PPO, REDQ, SAC, TD3, TDMPC2, TQC, CrossQ
from ajax.agents.TD3.TD3 import make_default_action_pipeline
from ajax.extensions.capabilities import MissingCapability, NotWired
from ajax.extensions.expert import ExpertGains, ExpertWarmup, ResidualPolicy
from ajax.extensions.pretrain import PriorData
from tests.agents.test_gain_mode import TwoGains
from tests.agents.test_offline import episodes

SMALL: dict[str, Any] = {"actor_architecture": ("16", "relu")}
SMALL.update({"critic_architecture": ("16", "relu"), "n_envs": 2})
SMALL.update({"learning_starts": 8, "batch_size": 8, "buffer_size": 64})
SMALL.update({"expert_buffer_n_steps": 0})


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def sac(**kwargs: Any) -> SAC:
    """SAC on Pendulum with push as its expert, its warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return SAC("Pendulum-v1", **{**SMALL, "expert_policy": push, **kwargs})


def train(agent: SAC) -> Any:
    """Six steps per env, four of them the warm-up: one seed's state."""
    state, _ = agent.train(seed=0, n_timesteps=12)
    return jax.tree.map(lambda leaf: leaf[0], state)


@pytest.mark.parametrize("fraction", [1.0, 0.0])
def test_the_expert_takes_its_share_of_the_warmup_and_is_flagged(fraction):
    """At 1 every warm-up row holds the expert's action (push: 0.25 in
    agent units) and is flagged; at 0 none is. No row after the warm-up is
    flagged."""
    state = train(sac(extensions=(ExpertWarmup(fraction=fraction),)))
    rows = state.collector_state.buffer_state.experience
    warmup = {key: np.asarray(value)[:, :4] for key, value in rows.items()}
    took = warmup["is_expert"][..., 0] == 1
    assert took.all() if fraction == 1.0 else not took.any()
    expert_rows = np.isclose(warmup["action"][..., 0], 0.25)
    np.testing.assert_array_equal(expert_rows, took)
    assert (np.asarray(rows["is_expert"])[:, 4:6] == 0).all()


def test_the_keyword_and_the_extension_train_the_same():
    """expert_fraction, and its None default with an expert, build the
    extension: bitwise the same runs."""
    pairs = [
        (sac(expert_fraction=0.5), sac(extensions=(ExpertWarmup(fraction=0.5),))),
        (sac(), sac(extensions=(ExpertWarmup(fraction=0.7),))),
    ]
    for keyword, extension in pairs:
        by_keyword = jax.tree.leaves(train(keyword))
        by_extension = jax.tree.leaves(train(extension))
        for old, new in zip(by_keyword, by_extension, strict=True):
            np.testing.assert_array_equal(old, new)


def test_the_keyword_warns_and_raises_beside_the_extension():
    with pytest.warns(
        FutureWarning, match="adds ExpertWarmup.fraction=0.7. by default"
    ):
        SAC("Pendulum-v1", **SMALL, expert_policy=push)
    with pytest.warns(FutureWarning, match="expert_fraction will be removed"):
        SAC("Pendulum-v1", **SMALL, expert_policy=push, expert_fraction=0.3)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", ".*(ExpertWarmup|expert_fraction)", FutureWarning
        )
        warmup = (ExpertWarmup(fraction=0.3),)
        agent = SAC("Pendulum-v1", **SMALL, expert_policy=push, extensions=warmup)
    assert agent.env_args.warmups == warmup and not agent.extension_stack
    both = {"expert_fraction": 0.3, "extensions": warmup}
    with pytest.raises(ValueError, match="not both"):
        SAC("Pendulum-v1", **SMALL, expert_policy=push, **both)


def test_sac_builds_no_default_warmup_where_it_never_mixed():
    """Without warm-up steps, under an action transform or without an
    expert there is no mix, and nothing is built."""
    assert sac(learning_starts=0).env_args.warmups == ()
    assert sac(extensions=(ResidualPolicy(),)).env_args.warmups == ()
    assert sac(expert_policy=None).env_args.warmups == ()


def test_sac_with_an_expert_and_no_warmup_steps_builds_and_trains():
    """The default mix needs warm-up steps; without them SAC builds none,
    also beside PriorData (whose rows need learning_starts=0). A replay with
    an expert's columns takes no dataset yet, so that one only builds."""
    train(sac(learning_starts=0))
    data = PriorData(episodes(3, discrete=False))
    assert sac(learning_starts=0, extensions=(data,)).env_args.warmups == ()


def test_the_warmup_is_refused_where_it_cannot_act():
    warmup = ExpertWarmup(fraction=0.5)
    with pytest.raises(MissingCapability, match="warmup_steps"):
        sac(learning_starts=0, extensions=(warmup,))
    with pytest.raises(MissingCapability, match="agent_expert"):
        sac(expert_policy=None, extensions=(warmup,))
    with pytest.raises(ValueError, match="one warm-up"):
        sac(extensions=(warmup, ExpertWarmup(fraction=0.2)))
    # DQN takes no expert yet; TD-MPC2's own collector runs no chooser yet.
    with pytest.raises(NotWired, match="takes no expert_policy"):
        DQN("CartPole-v1", extensions=(warmup,))
    with pytest.raises(NotWired, match="choose_warmup_action"):
        TDMPC2("Pendulum-v1", extensions=(warmup,))
    # A user's pipeline replaces TD3's default one, which runs the chooser.
    pipeline = make_default_action_pipeline(False, 0.1)
    with pytest.raises(ValueError, match="never sees the warm-up"):
        TD3(
            "Pendulum-v1",
            expert_policy=push,
            action_pipeline=pipeline,
            extensions=(warmup,),
        )


NO_WARMUP = frozenset({"warmup", "warmup_steps"})


@pytest.mark.parametrize(
    ("cls", "lacks"),
    [
        (PPO, NO_WARMUP),  # on policy: every step is the policy's
        (TQC, NO_WARMUP),  # acts with its policy from step 0, as its authors do
        (AVG, NO_WARMUP),  # no replay: warm-up steps would feed no update
        (IQL, frozenset({"warmup_steps"})),  # its learning_starts is 0
    ],
)
def test_the_warmup_is_refused_where_there_is_none(cls, lacks):
    warmup = ExpertWarmup(fraction=0.5)
    with pytest.raises(MissingCapability) as refusal:
        cls("Pendulum-v1", expert_policy=push, extensions=(warmup,))
    assert refusal.value.missing == {"ExpertWarmup": lacks}


ARCHITECTURES = {
    "actor_architecture": ("16", "relu"),
    "critic_architecture": ("16", "relu"),
}
#: Each agent with a uniform warm-up but SAC, with its small networks.
WARMUP_AGENTS: dict[str, tuple[Any, dict[str, Any]]] = {
    "TD3": (TD3, ARCHITECTURES),
    "CrossQ": (CrossQ, {"actor_width": 16, "critic_width": 16}),
    "REDQ": (REDQ, ARCHITECTURES),
    "ASAC": (ASAC, ARCHITECTURES),
    "IQL": (IQL, ARCHITECTURES),
}


def warmup_rows(cls: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """One seed's replay after ten steps per env, the first eight the
    warm-up, push its expert: per field, [env, row, ...]."""
    agent = cls(
        "Pendulum-v1",
        n_envs=2,
        learning_starts=16,
        batch_size=8,
        buffer_size=64,
        **kwargs,
    )
    state, _ = agent.train(seed=0, n_timesteps=20)
    rows = state.collector_state.buffer_state.experience
    return {field: np.asarray(value)[0] for field, value in rows.items()}


@pytest.mark.parametrize("name", sorted(WARMUP_AGENTS))
def test_the_expert_takes_its_share_of_every_agents_warmup(name):
    """The warm-up steps the expert took hold its action (push: 0.25 in
    agent units) in every env, the others uniform ones. The replay flags
    them where it keeps is_expert, and no row after the warm-up; IQL keeps
    none, so its rows are read by their action."""
    cls, networks = WARMUP_AGENTS[name]
    warmup = (ExpertWarmup(fraction=0.5),)
    rows = warmup_rows(cls, expert_policy=push, extensions=warmup, **networks)
    took = np.isclose(rows["action"][:, :8, 0], 0.25)
    # One decision per step for every env, both outcomes within eight steps.
    np.testing.assert_array_equal(took[0], took[1])
    assert took.any() and not took.all()
    assert ("is_expert" in rows) == (name != "IQL")
    if "is_expert" in rows:
        np.testing.assert_array_equal(rows["is_expert"][:, :8, 0] == 1, took)
        assert (rows["is_expert"][:, 8:10] == 0).all()


@pytest.mark.parametrize("name", ["SAC", "TD3", "IQL"])
def test_the_expert_warmup_stores_the_policy_action_that_executes_the_expert(name):
    """Where the transform acts after the choice (gain mode on SAC and TD3,
    IQL's residual), the warm-up draws the policy's actions: the expert's
    steps store the one executing its own action, 0 (the anchor gains, a
    zero residual), and IQL's executed action is the expert's."""
    warmup = ExpertWarmup(fraction=1.0)
    if name == "IQL":
        transformed = {"expert_policy": push, "extensions": (ResidualPolicy(), warmup)}
    else:
        transformed = {
            "expert_policy": TwoGains(),
            "extensions": (ExpertGains(), warmup),
        }
    if name == "SAC":
        transformed.update({"expert_buffer_n_steps": 0})
    cls, networks = (SAC, ARCHITECTURES) if name == "SAC" else WARMUP_AGENTS[name]
    rows = warmup_rows(cls, **transformed, **networks)
    chosen = rows["policy_action"] if name == "IQL" else rows["action"]
    np.testing.assert_array_equal(chosen[:, :8], 0.0)
    if name == "IQL":
        np.testing.assert_allclose(rows["action"][:, :8], 0.25)
    else:
        assert (rows["is_expert"][:, :8] == 1).all()
